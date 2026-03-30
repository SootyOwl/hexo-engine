# Phase 2: Gumbel AlphaZero Training Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete self-play + training pipeline using Gumbel MCTS — the agent plays games against itself, collects improved policy targets, and trains the Phase 1 GNN on the resulting data.

**Architecture overview:** Gumbel MCTS searches from each game position, producing an improved policy distribution and selecting an action. Self-play generates full games, assigning the game outcome as each position's value target. Training examples (graph, policy target, value target) are stored in a replay buffer and sampled for gradient updates. The loop alternates: generate N self-play games → train K gradient steps → repeat.

**Key design decisions:**
- **Sequential two-move search:** Each MCTS call decides one hex placement. The self-play loop calls MCTS once per placement, not once per turn. This is the simplest approach from the master plan; joint two-move search is a Phase 4 ablation.
- **Player-aware backup:** HeXO has two placements per turn (same player), so the MCTS backup negates the value only when the current player changes between parent and child, NOT at every tree level. Each MCTS node stores the current player.
- **Single-threaded self-play initially:** One game at a time, one network eval per MCTS simulation. Parallel self-play with batched inference is a future optimization. With defaults (`n_simulations=16`, `m_actions=16`, `games_per_iteration=25`), expect ~10-20 min per iteration on CPU with small game configs.
- **Pre-computed graphs in replay buffer:** Training examples store PyG `Data` objects (not raw `GameState`), avoiding re-computing graphs during training. Memory budget: ~50KB per mid-game position × 100K capacity ≈ 5GB. Acceptable for 128GB machine; reduce `buffer_capacity` for smaller machines.
- **L2 regularisation via weight_decay:** The optimizer handles L2; not in the loss function.
- **Policy loss: KL divergence (paper's Eq. 12):** `KL(pi_improved || network_policy)` trains all actions, extracting more signal from each search than the simpler `-log(pi(a_chosen))` (Eq. 9). Implementation: `KL(p||q) = sum(p * log(p)) - sum(p * log(q))`. The entropy term `sum(p * log(p))` is constant w.r.t. network params, so the gradient is identical to cross-entropy. We implement the full KL to match the paper. The entropy term uses `torch.special.xlogy(target, target)` to handle the `0 * log(0) = 0` case safely. This is a modification to `loss.py` (or a new `kl_policy_loss()` function alongside the existing `alphazero_loss()`).
- **Sigma function (paper's Eq. 8):** `sigma(q) = (c_visit + max_b N(b)) * c_scale * q` where `c_visit=50, c_scale=1.0`. `max_b N(b)` is the maximum visit count among the **children of the node being scored from** — at the root this is the root's children (candidates); at an interior node this is that node's children. Visit-count-dependent: the Q-value influence grows as the tree deepens, progressively overriding the prior.
- **Q-value normalization (paper's Appendix F):** Q-values are normalized to [0,1] via min-max scaling before being passed to sigma: `q_normalized = (q - q_min) / (q_max - q_min)` where `q_min, q_max` are the minimum and maximum empirical Q-values among **visited** children only. Unvisited actions' completed Q (v_mix) is then also normalized using this same [q_min, q_max] range. If all visited Q-values are equal (or only one visited), normalization maps everything to 0, and the improved policy reduces to `softmax(logits)` (the prior).
- **Paper's Section 5 formula for non-root selection:** Interior nodes use the deterministic formula: `pi' = softmax(logits + sigma(completed_Q))`, then `select a = argmax_a [pi'(a) - N(a) / (1 + sum_b N(b))]`. This drives visit counts to match the improved policy. **Completed Q-values (paper's Appendix D, Eq. 33):** uses `v_mix`, an interpolation of the value network estimate and the prior-weighted empirical Q-values: `v_mix = (v_hat + total_N / sum_visited_prior * sum_visited(pi(a)*Q(a))) / (1 + total_N)` where `total_N = sum_b N(b)`, `sum_visited_prior = sum_{b:N(b)>0} pi(b)`, and the inner sum is `sum_{a:N(a)>0} pi(a)*Q(a)`. Note: Q-values are weighted by **prior pi(a)**, not visit count N(a). Visited actions use empirical Q(a); unvisited actions use v_mix. Same formulation used at root for the improved policy target (REQ-8). No PUCT, no `c_puct` hyperparameter.
- **Exploration in self-play (paper's Appendix F):** For the first `exploration_moves` (default 60 placements, matching the paper's 30 turns for HeXO's 2-placement turns) of each self-play game, the action is sampled proportionally to the visit counts of the root's children (as in the paper). With `n=m=16`, this is uniform over the 16 Gumbel-selected candidates. After `exploration_moves`, action selection reverts to `argmax[g(a) + logits(a) + sigma(Q(a))]`.
- **Defaults match the paper:** `n_simulations=16`, `m_actions=16`. With the Sequential Halving formula `max(1, ceil(n / (ceil(log2(m)) * num_considered)))`, this gives 1 sim/candidate in a single halving round. The paper intentionally uses this aggressive budget — the Gumbel trick provides policy improvement guarantees even with minimal search. Larger budgets (e.g., n=64, m=8 for multi-round halving) can be explored in Phase 4 ablations.

**Tech Stack:** Same as Phase 1 — Python 3.13, PyTorch 2.6+, PyG 2.7+, `hexo-rs`, `uv`, `pytest`. All new code goes in `hexo-a0/`.

---

## Requirements

### Training Configuration

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-1 | All training hyperparameters in a single `TrainingConfig` dataclass with sensible defaults | AC-1a: Default config instantiates without arguments. AC-1b: Fields include MCTS params (`n_simulations=16`, `m_actions=16`, `c_scale=1.0`, `c_visit=50`), self-play params (`games_per_iteration=25`, `exploration_moves=60`), training params (`batch_size=256`, `lr=2e-4`, `weight_decay=1e-4`, `train_steps_per_iteration=200`, `max_grad_norm=1.0`), replay buffer params (`buffer_capacity=100_000`), and a `seed` field (default `None` for non-deterministic). AC-1c: Custom values can be passed at construction. AC-1d: `TrainingConfig` and `ModelConfig` are independent — neither inherits from or contains the other. AC-1e: No `max_moves` field — game termination is handled by `GameConfig` in the Rust engine | T1 |

### MCTS Tree

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-2 | MCTS node stores visit count, total value, prior probability, children, associated action (coordinate), current player, and network value estimate V | AC-2a: Fresh node has `visit_count=0`, `value_sum=0.0`, empty children dict. AC-2b: `q_value` property returns `value_sum / visit_count` (or 0.0 if unvisited). AC-2c: Node tracks the action (coord) that led to it. AC-2d: Node stores which player is to move at this position. AC-2e: Node stores `value_estimate` (the network's value prediction at this state), used as `V_parent` for completed Q-values in child selection | T2 |
| REQ-3 | Tree backup propagates the evaluation value from leaf to root, negating only when the current player changes between child and parent (not at every level) | AC-3a: After backup with value `v`, leaf's visit count increments by 1 and value_sum increases by `v`. AC-3b: When parent and child have different current players, parent's value_sum increases by `-v` (sign flip). AC-3c: When parent and child have the same current player (mid-turn, e.g. moves_remaining goes from 2 to 1), parent's value_sum increases by `+v` (no flip). AC-3d: All ancestors' visit counts increment by 1. AC-3e: Test case: a 3-node path where middle node has same player as leaf (two-placement turn) — verify only one sign flip at the grandparent | T2 |
| REQ-4 | Non-root child selection uses the paper's Section 5 formula: compute `pi' = softmax(logits + sigma(completed_Q))`, then `select a = argmax_a [pi'(a) - N(a) / (1 + sum_b N(b))]`. Completed Q uses `v_mix` (paper's Eq. 33): visited actions use `Q(a)`, unvisited use `v_mix` (interpolation of value estimate and empirical Q) | AC-4a: With equal completed Q-values, selects the action with highest prior. AC-4b: Unvisited actions get `completedQ = v_mix` (not 0), so they are explored proportionally to their prior. AC-4c: After many visits, visit counts approximate `pi'`. AC-4d: The formula handles all-unvisited children (first visit to a node) without error — when all children are unvisited, `v_mix = value_estimate` | T2 |

### Network Evaluator

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-5 | A function converts a `GameState` to network outputs: prior logits over legal moves, value estimate, and the corresponding legal move coordinates | AC-5a: Returns `(logits, value, coords)` where `logits` is 1-D with length = number of legal moves, `value` is a scalar in [-1, 1], `coords` is a list of `(q, r)` tuples sorted by `(q, r)` — matching the order produced by `game_to_graph()`'s legal-node ordering. AC-5b: Works on both CPU and GPU. AC-5c: Network is in `eval` mode with `torch.no_grad()` during inference | T3 |

### Gumbel MCTS

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-6 | Gumbel-Top-k selects `m` candidate actions from legal moves using Gumbel noise + log-prior scores | AC-6a: Returns exactly `min(m, num_legal)` candidates. AC-6b: Each candidate has an associated Gumbel sample `g(a)`. AC-6c: Candidates are the top-m by `g(a) + log_prior(a)` | T3 |
| REQ-7 | Sequential Halving allocates simulation budget across candidates, eliminating the bottom half by Q-value each round | AC-7a: Total simulations used does not exceed `n_simulations`. AC-7b: Each round eliminates roughly half the remaining candidates. AC-7c: Exactly one candidate survives (or ties are broken arbitrarily). AC-7d: Each surviving candidate has received at least 1 simulation. AC-7e: With `n_simulations=16` and `m_actions=16`, exactly 1 halving round occurs (1 sim per candidate, then budget exhausted). AC-7f: When `n_simulations=0`, Sequential Halving is skipped entirely — all candidates retain Q=0. AC-7g: When only 1 candidate exists (forced move), Sequential Halving is a no-op | T3 |
| REQ-8 | Improved policy target uses completed Q-values (with `v_mix`, paper's Eq. 33) over ALL legal moves: `pi' = softmax(logits + sigma(completedQ))`. Q-values are min-max normalized to [0,1] before sigma. Unvisited actions get `completedQ = v_mix` | AC-8a: Output is a probability distribution summing to 1.0 over legal moves. AC-8b: Searched actions with higher Q-values get higher probability than their prior. AC-8c: Unsearched actions get `completedQ = v_mix` (not 0 or raw value estimate). AC-8d: Computation uses log-space: `log_score(a) = logits(a) + sigma(normalized_completedQ(a))`, then `softmax(log_scores)`. No raw `exp()` calls. AC-8e: `sigma(q) = (c_visit + max_b N(b)) * c_scale * q` where `max_b N(b)` is over siblings (paper's Eq. 8). AC-8f: Q-values are normalized to [0,1] via min-max before sigma | T3 |
| REQ-9 | Action selection at root: `argmax_a [g(a) + logits(a) + sigma(Q(a))]` over the final surviving candidates | AC-9a: Returns a single `(q, r)` coordinate. AC-9b: With zero simulations (pure prior), reduces to Gumbel-argmax of the prior | T3 |
| REQ-10 | Full `gumbel_mcts(state, network, config, device)` returns the selected action and the improved policy target | AC-10a: Returns `(action_coord, improved_policy)`. AC-10b: `improved_policy` has length equal to number of legal moves. AC-10c: On a terminal state, raises `ValueError` | T3 |

### MCTS Simulation

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-11 | A single MCTS simulation: select a path from root to leaf using Section 5 selection, expand the leaf (or handle terminal leaf), and backup | AC-11a: If the leaf is non-terminal, it is expanded with children using network-predicted priors, and the node's `value_estimate` is set from the network. The network value is backed up. AC-11b: If the leaf is terminal, no child node is created. The backup value is computed from the parent node's perspective: +1 if parent's current_player won, -1 if they lost, 0 for draw. Backup starts from the parent node (not the non-existent terminal child). AC-11c: The backup value propagates with player-aware sign flips (REQ-3). AC-11d: The root's visit count increases by 1 after one simulation | T3 |

### Self-Play

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-12 | Self-play generates a complete game trajectory of `(state, improved_policy)` pairs, then assigns value targets from the perspective of each position's current player. For the first `exploration_moves` placements, the action is sampled proportionally to root visit counts; after that, deterministic Gumbel-argmax is used | AC-12a: Trajectory ends when the game is terminal (win or draw via `GameConfig.max_moves`). AC-12b: Value target is `+1` if current player won, `-1` if current player lost, `0` for draw. AC-12c: Each position's improved_policy has length matching that position's legal move count. AC-12d: For a two-placement turn, both placements in the trajectory have the same current player and the same value target sign. AC-12e: A game that ends mid-turn (win on first placement of a two-move turn) produces a trajectory where the last entry has `moves_remaining > 1`. AC-12f: With `exploration_moves=60`, the first 60 placements sample from root visit counts (stochastic); subsequent placements use Gumbel-argmax (deterministic) | T4 |
| REQ-13 | Training examples are `(PyG Data, policy_target_tensor, value_target_scalar)` tuples ready for the replay buffer | AC-13a: PyG Data is produced by `game_to_graph()`. AC-13b: Policy target tensor has the same length as the number of True entries in `data.legal_mask` and is in the same order as `data.coords[data.legal_mask]`. AC-13c: Value target is a float scalar | T4 |

### Replay Buffer

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-14 | Fixed-capacity replay buffer stores training examples and samples uniformly at random | AC-14a: Adding beyond capacity evicts the oldest examples (FIFO). AC-14b: `sample(n)` returns `n` examples drawn uniformly without replacement when `n <= len`, with replacement when `n > len`. AC-14c: `len(buffer)` returns current size | T5 |

### Training Step

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-15 | A single training step: sample a batch from the replay buffer, collate into a PyG Batch, run forward + loss + backward + optimizer step with gradient clipping | AC-15a: Loss decreases over multiple steps on a fixed dataset (overfitting test). AC-15b: Gradients reach all model parameters. AC-15c: Uses `Batch.from_data_list()` for collation. AC-15d: Policy targets of different lengths are handled correctly in the loss (per-graph, not batched). AC-15e: Gradient norm is clipped to `max_grad_norm` before the optimizer step | T6 |
| REQ-16 | Loss is computed per-graph and averaged across the batch (not concatenated), since policy targets have different lengths per graph | AC-16a: Batch loss is the mean of individual per-graph losses. AC-16b: Individual policy_loss and value_loss components are tracked for logging | T6 |

### Training Loop

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-17 | Training iteration: generate `games_per_iteration` self-play games → add to buffer → run `train_steps_per_iteration` gradient steps → log metrics | AC-17a: Buffer grows by the total number of positions from self-play games. AC-17b: Model parameters change after training steps. AC-17c: Iteration can be repeated multiple times. AC-17d: Each iteration logs: iteration number, buffer size, mean total/policy/value loss, mean self-play game length, wall-clock time via Python `logging` module | T7 |
| REQ-18 | Checkpointing: save and load model weights, optimizer state, iteration count, and training config. `GameConfig` is stored as a plain dict (not the PyO3 object, which is not picklable) | AC-18a: Loading a checkpoint restores training to the saved state. AC-18b: Checkpoint is a single `.pt` file. AC-18c: Training can resume from a checkpoint with restored optimizer momentum | T7 |

---

## File Structure

```
hexo-a0/
├── src/
│   └── hexo_a0/
│       ├── __init__.py              # Updated exports
│       ├── config.py                # ModelConfig (Phase 1)
│       ├── training_config.py       # TrainingConfig (NEW)
│       ├── graph.py                 # game_to_graph (Phase 1)
│       ├── model.py                 # HeXONet (Phase 1)
│       ├── loss.py                  # alphazero_loss (Phase 1)
│       ├── mcts.py                  # MCTSNode + gumbel_mcts + evaluate_state (NEW)
│       ├── self_play.py             # self_play_game, TrainingExample (NEW)
│       ├── replay_buffer.py         # ReplayBuffer (NEW)
│       └── trainer.py               # Trainer, train_step (NEW)
└── tests/
    ├── conftest.py                  # Shared fixtures (Phase 1 + new)
    ├── test_training_config.py      # TrainingConfig tests (NEW)
    ├── test_mcts.py                 # MCTS tree + Gumbel MCTS tests (NEW)
    ├── test_self_play.py            # Self-play tests (NEW)
    ├── test_replay_buffer.py        # Replay buffer tests (NEW)
    ├── test_trainer.py              # Training step + loop tests (NEW)
    └── test_training_integration.py # End-to-end training integration (NEW)
```

**Implementation notes:**
- **Two-move turn backup:** HeXO players place 2 stones per turn (except P1's first). Consecutive MCTS tree nodes can have the same `current_player`. The backup must check `node.current_player != parent.current_player` to decide whether to negate. Test this explicitly with a path containing same-player consecutive nodes.
- **Terminal leaves in MCTS:** During simulation, applying a move may result in a terminal state. Do NOT create a child node for the terminal state. Instead, compute the backup value from the parent node's perspective: +1 if `winner() == parent.current_player`, -1 if the opponent won, 0 for draw. `game.current_player()` returns `None` on terminal states, so perspective must come from the parent. Backup starts from the parent, not from a non-existent terminal child.
- **Policy target ordering:** `game_to_graph()` sorts legal-move nodes by `(q, r)`. `evaluate_state()` returns coords in this same sorted order. The improved policy from MCTS must also be stored in this sorted coordinate order. The training loss pairs `policy_logits[i]` with `policy_target[i]` positionally — any ordering mismatch corrupts training.
- **Numerical stability:** The improved policy computation works in log-space: `log_score[a] = logits[a] + sigma(completedQ[a])`. Then `softmax(log_scores)` produces the probability distribution. Never compute raw `exp(logits)`.
- **Memory budget:** Each mid-game training example stores ~50KB (PyG Data with ~200-400 nodes). At `buffer_capacity=100_000`, expect ~5GB RAM for the buffer. Acceptable for 128GB machine; reduce capacity for smaller machines.
- **Non-root selection (Section 5):** Interior nodes compute `pi' = softmax(logits + sigma(normalized_completedQ))` then select `argmax_a [pi'(a) - N(a)/(1 + sum N)]`. Completed Q-values use `v_mix` (Eq. 33): visited = empirical Q(a), unvisited = v_mix. Q-values are min-max normalized to [0,1] before sigma. Each node stores its network-predicted `value_estimate` for computing v_mix.
- **PyO3 API note:** `game.apply_move(q, r)` takes two separate `i32` arguments, NOT a tuple. Write `game.apply_move(coord[0], coord[1])`, not `game.apply_move(coord)`.
- **MCTS tree lifecycle:** The tree persists across all simulations within a single `gumbel_mcts()` call — simulations progressively build up the tree. A fresh tree is created for each move (no tree reuse between moves). The root node is expanded (children created with network priors) before Sequential Halving begins.
- **Simulation targeting:** During Sequential Halving, each simulation allocated to a candidate action is forced to take that action as the first move from the root. Subsequent moves within the simulation use the Section 5 selection formula. This ensures simulations explore the subtree of the target candidate.
- **GameState in MCTS nodes:** Each MCTS node stores a `GameState` clone representing the board at that node. On expansion, the parent's state is cloned and the child's action applied. This avoids replaying moves from root on every simulation. Memory cost is modest: `GameState.clone()` is ~460ns and states are small.
- **Model mode management:** `self_play_game()` sets `model.eval()` at the start and leaves it in eval mode. `train_step()` sets `model.train()` at the start. Callers (the Trainer) are responsible for mode transitions between self-play and training phases.
- Gumbel samples are drawn once per MCTS call and reused throughout Sequential Halving. They are NOT resampled each round.
- `evaluate_state()` is a private helper in `mcts.py` (not a separate module). It calls `game_to_graph()` and `model.forward()` inside `torch.no_grad()`. It does NOT toggle model mode — it assumes the caller has set eval mode.
- The `GameConfig` (win_length, placement_radius, max_moves) controls game termination. Self-play uses whatever `GameConfig` the `GameState` was created with. For curriculum learning (Phase 3), the caller passes different `GameConfig`s.
- Logging uses Python's `logging` module at INFO level. Per-game progress is logged at DEBUG level. No external dependencies (tensorboard/wandb deferred to Phase 3).

---

## Task 1: Training Configuration

**Files:** Create `src/hexo_a0/training_config.py`, create `tests/test_training_config.py`

- [ ] Write failing tests for `TrainingConfig`:
  - Default values match AC-1b
  - Custom values override correctly (AC-1c)
  - `TrainingConfig` and `ModelConfig` are independent types (AC-1d)
  - No `max_moves` field exists (AC-1e)
- [ ] Run tests — verify they fail
- [ ] Implement `TrainingConfig` as a `dataclass`
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 2: MCTS Node and Tree Operations

**Files:** Create `src/hexo_a0/mcts.py` (MCTSNode + tree ops only), create `tests/test_mcts.py` (started here, extended in T4)

- [ ] Write failing tests for `MCTSNode`:
  - Fresh node has correct initial values (AC-2a)
  - `q_value` returns 0.0 for unvisited node, correct ratio otherwise (AC-2b)
  - Node stores its action coordinate (AC-2c)
  - Node stores current player (AC-2d)
- [ ] Write failing tests for `backup()`:
  - Leaf visit count and value_sum update correctly (AC-3a)
  - Different-player parent gets negated value (AC-3b)
  - Same-player parent (mid-turn) gets un-negated value (AC-3c)
  - All ancestors' visit counts increment (AC-3d)
  - Three-node path with same-player middle: only one sign flip at grandparent (AC-3e)
- [ ] Write failing tests for `select_child()` (Section 5 non-root selection):
  - Equal completed Q-values → selects highest prior action (AC-4a)
  - Unvisited actions get completedQ = v_mix (not 0); when all unvisited, v_mix = value_estimate (AC-4b, AC-4d)
  - All-unvisited children handled without error (AC-4d)
- [ ] Run tests — verify they fail
- [ ] Write failing tests for `sigma()`, `v_mix()`, and `normalize_q()`:
  - sigma scales with visit count: grows as max_b N(b) increases (AC-8e)
  - sigma scope: max_b N(b) is over children of the scored node (AC-8e)
  - v_mix: with all unvisited, equals value_estimate; with some visited, uses pi-weighted Q (Eq. 33)
  - v_mix: concrete numerical test against hand-computed Eq. 33 reference values
  - Q normalization: maps to [0,1] over visited Q range; handles all-equal Q (returns 0) (AC-8f)
- [ ] Run tests — verify they fail
- [ ] Implement `MCTSNode` with `q_value` property, `value_estimate` field, `backup(path, value)` with player-aware sign flips, and `select_child(node, config)` using Section 5 formula. Also implement `sigma()`, `v_mix()`, and `normalize_q()` as pure functions (shared by T2 select_child and T3a compute_improved_policy)
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 3a: Network Evaluator and Gumbel-Top-k

**Files:** Modify `src/hexo_a0/mcts.py`, modify `tests/test_mcts.py`

Stateless functions: bridge game state to network inference, and sample candidate actions via the Gumbel trick.

- [ ] Write failing tests for `evaluate_state()`:
  - Returns `(logits, value, coords)` with correct shapes (AC-5a)
  - `logits` length matches `len(game.legal_moves())` (AC-5a)
  - `value` is a scalar in [-1, 1] (AC-5a)
  - `coords` are sorted by `(q, r)` matching `game_to_graph()` order (AC-5a)
  - No gradients are computed (AC-5c)
- [ ] Write failing tests for `gumbel_top_k()`:
  - Returns `min(m, num_legal)` candidates (AC-6a)
  - Each candidate has an associated Gumbel sample (AC-6b)
  - Candidates are top-m by `g(a) + log_prior(a)` — verify with seeded RNG (AC-6c)
- [ ] Write failing tests for `compute_improved_policy()`:
  - Output sums to 1.0 (AC-8a)
  - Searched actions with higher Q get boosted relative to prior (AC-8b)
  - Unsearched actions get completedQ = v_mix, not 0 (AC-8c)
  - Uses log-space with sigma function: no NaN/inf for large logit values (AC-8d)
  - Q-values are min-max normalized to [0,1] before sigma (AC-8f)
- [ ] (sigma, v_mix, normalize_q are already implemented and tested in T2 — reuse them here)
- [ ] Run tests — verify they fail
- [ ] Implement `evaluate_state()` as a module-level helper, `gumbel_top_k()`, and `compute_improved_policy()`. Use a mock/stub network for unit tests (returns fixed priors and values)
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 3b: MCTS Simulation and Sequential Halving

**Files:** Modify `src/hexo_a0/mcts.py`, modify `tests/test_mcts.py`

Stateful tree traversal: single simulation (select → expand → evaluate → backup) and the Sequential Halving loop that allocates simulations to candidates. Budget formula: `max(1, ceil(n / (ceil(log2(m)) * num_considered)))` visits per action per phase (Karnin et al., 2013).

- [ ] Write failing tests for `simulate()` (single MCTS simulation):
  - Non-terminal leaf is expanded with children (AC-11a)
  - Terminal leaf uses game outcome from parent's perspective, no expansion (AC-11b)
  - Value is backed up with player-aware sign flips (AC-11c)
  - Root visit count increments (AC-11d)
  - Child GameState is independent from parent (apply move on child doesn't affect parent)
- [ ] Write failing tests for `sequential_halving()`:
  - Total simulations ≤ budget (AC-7a)
  - Roughly halves candidates each round (AC-7b)
  - One candidate survives (AC-7c)
  - Each surviving candidate has ≥ 1 simulation (AC-7d)
  - With n=16, m=16, exactly 1 halving round occurs (AC-7e)
  - With n_simulations=0, halving is skipped (AC-7f)
  - With 1 candidate, halving is a no-op (AC-7g)
- [ ] Run tests — verify they fail
- [ ] Implement `simulate()` and `sequential_halving()`. Simulations are forced to start with the designated candidate action from the root
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 3c: Gumbel MCTS Integration

**Files:** Modify `src/hexo_a0/mcts.py`, modify `tests/test_mcts.py`

Compose all components into the `gumbel_mcts()` entry point.

- [ ] Write failing tests for `gumbel_mcts()`:
  - Returns `(action, improved_policy)` (AC-10a)
  - `improved_policy` length equals number of legal moves (AC-10b)
  - Terminal state raises ValueError (AC-10c)
  - With n_simulations=0, reduces to Gumbel-argmax of prior (AC-9b)
  - Action selection uses `argmax[g(a) + logits(a) + sigma(Q(a))]` (AC-9a)
- [ ] Run tests — verify they fail
- [ ] Implement `gumbel_mcts(state, network, config, device)` composing evaluate_state, gumbel_top_k, sequential_halving, compute_improved_policy, and action selection
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 4: Self-Play

**Files:** Create `src/hexo_a0/self_play.py`, create `tests/test_self_play.py`

- [ ] Write failing tests for `TrainingExample` dataclass:
  - Holds `(data, policy_target, value_target)` with correct types (AC-13a, AC-13b, AC-13c)
- [ ] Write failing tests for `self_play_game()`:
  - Returns a non-empty list of `TrainingExample`s (AC-12a)
  - Game terminates (trajectory ends at terminal state) (AC-12a)
  - Winner's positions have value_target=+1, loser's have -1, draw has 0 (AC-12b)
  - Each policy_target length matches legal_mask.sum() in its Data (AC-12c)
  - Each policy_target sums to ~1.0 (AC-12c)
  - Two-placement turn: both placements have same value target sign (AC-12d)
  - Win mid-turn: last entry has `moves_remaining > 1` in the game state (AC-12e)
  - Policy target ordering matches `data.coords[data.legal_mask]` (AC-13b)
  - With exploration_moves=60, the first 60 placements sample from root visit counts (AC-12f)
- [ ] Run tests — verify they fail. Use a tiny network (hidden_dim=16, num_layers=1) and tiny config (n_simulations=8, m_actions=4) for fast tests. Use GameConfig(win_length=4, placement_radius=4, max_moves=50) for short games
- [ ] Implement `self_play_game(network, training_config, game_config, device)` that plays one full game using `gumbel_mcts()` for each placement, collects the trajectory, and converts to `TrainingExample`s
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 5: Replay Buffer

**Files:** Create `src/hexo_a0/replay_buffer.py`, create `tests/test_replay_buffer.py`

- [ ] Write failing tests for `ReplayBuffer`:
  - Adding examples increases length (AC-14c)
  - Capacity is enforced — oldest evicted (AC-14a)
  - `sample(n)` returns exactly `n` examples (AC-14b)
  - Sampling without replacement when n <= len, with replacement when n > len (AC-14b)
  - Sampling from an empty buffer raises ValueError
- [ ] Run tests — verify they fail
- [ ] Implement `ReplayBuffer(capacity)` backed by `collections.deque(maxlen=capacity)`
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 6: Training Step

**Files:** Modify `src/hexo_a0/loss.py` (add KL loss), create `src/hexo_a0/trainer.py` (training step portion), create `tests/test_trainer.py`

- [ ] Write failing tests for `kl_policy_loss()` in `test_loss.py`:
  - KL divergence matches hand-computed reference value
  - Uses `torch.special.xlogy` for the entropy term (no NaN when target=0)
  - Gradient-equivalent to cross-entropy (same parameter gradients)
- [ ] Write failing tests for `train_step()`:
  - Loss decreases over multiple steps on fixed data (AC-15a overfitting test)
  - Gradients reach all model parameters (AC-15b)
  - Handles variable-length policy targets correctly (AC-15d)
  - Gradient norm is clipped (AC-15e)
  - Returns loss components dict with `policy_loss`, `value_loss`, `total_loss` means (AC-16b)
  - Batch loss is mean of per-graph losses (AC-16a)
- [ ] Run tests — verify they fail
- [ ] Implement `train_step(model, optimizer, examples, device, max_grad_norm)`:
  - Collate PyG Data objects via `Batch.from_data_list()` (AC-15c)
  - Run `model.forward_batch(batch)`
  - Compute per-graph KL policy loss + MSE value loss and average
  - `clip_grad_norm_` then optimizer step
  - Return mean loss components
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 7: Training Loop and Checkpointing

**Files:** Modify `src/hexo_a0/trainer.py`, modify `tests/test_trainer.py`

- [ ] Write failing tests for `save_checkpoint()` and `load_checkpoint()`:
  - Round-trip: save then load restores model weights (AC-18a)
  - Checkpoint is a single `.pt` file (AC-18b)
  - Optimizer state is restored (AC-18c)
  - Iteration count and config are restored (AC-18a)
  - GameConfig is stored as plain dict and reconstructed on load (AC-18a)
- [ ] Write failing tests for `train_iteration()`:
  - Buffer grows after self-play (AC-17a)
  - Model parameters change after training (AC-17b)
  - Can run multiple iterations (AC-17c)
  - Returns metrics dict with loss components and game length (AC-17d)
- [ ] Run tests — verify they fail
- [ ] Implement `Trainer` class with `train_iteration()`, `save_checkpoint()`, `load_checkpoint()`, and `logging`-based metrics output. Uses constant LR (scheduling deferred to Phase 3)
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 8: End-to-End Integration Tests

**Files:** Create `tests/test_training_integration.py`, update `src/hexo_a0/__init__.py`

- [ ] Write integration test: create a tiny model + config with GameConfig(win_length=4, placement_radius=4, max_moves=30), run 2 training iterations (2 self-play games each, 5 train steps each), verify:
  - No crashes
  - Loss values are finite
  - Buffer has expected number of examples
  - Checkpoint round-trip works
- [ ] Write integration test: save checkpoint after iteration 1, load it, continue for iteration 2, verify training continues normally
- [ ] Write ordering consistency test: for a given game state, verify that `game_to_graph().coords[legal_mask]`, `evaluate_state()` coords, and a policy target from `gumbel_mcts()` all agree on coordinate ordering
- [ ] Update `__init__.py` to export `TrainingConfig`, `ReplayBuffer`, `Trainer` (or key public functions)
- [ ] Run all tests (Phase 1 + Phase 2) — all pass
- [ ] Commit

---

## Summary

| Task | What | Requirements |
|------|------|-------------|
| 1 | TrainingConfig dataclass | REQ-1 |
| 2 | MCTS node + tree ops (backup, Section 5 select, sigma, v_mix, Q normalization) | REQ-2, REQ-3, REQ-4 |
| 3a | Network evaluator, Gumbel-Top-k, improved policy | REQ-5, REQ-6, REQ-8 |
| 3b | MCTS simulation + Sequential Halving | REQ-7, REQ-11 |
| 3c | Gumbel MCTS integration | REQ-9, REQ-10 |
| 4 | Self-play game generation | REQ-12, REQ-13 |
| 5 | Replay buffer | REQ-14 |
| 6 | Training step (batch collation, per-graph loss, grad clipping) | REQ-15, REQ-16 |
| 7 | Training loop, checkpointing, logging (constant LR) | REQ-17, REQ-18 |
| 8 | End-to-end integration tests + public exports | All |

Every REQ has labeled ACs; every AC maps to at least one test.

**Dependency graph:**
```
T1 (config) ──→ T3a (evaluator, top-k, improved policy) → T3b (simulate, halving) → T3c (gumbel_mcts) → T4 (self-play) → T7 (loop) → T8
T2 (MCTS node + sigma/v_mix) → T3a
T5 (replay buffer) → T6 (train step) ─────────────────────────────────────────────────→ T7
```
Critical path: T1+T2→T3a→T3b→T3c→T4→T7→T8. T1 and T2 can be done in parallel; both feed T3a. T5 is independent. T6 depends on T5 only.
