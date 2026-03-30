# Phase 3: Curriculum Learning — Master Plan

**Goal:** Train through progressively harder HeXO variants (4→5→6-in-a-row), with evaluation infrastructure, LR scheduling, enhanced monitoring, and performance optimisations to support multi-day training runs.

**Architecture:** Five independent subsystems, each testable in isolation, decomposed into separate sub-phase plans. Each sub-phase can be implemented and reviewed independently before composition.

**Tech Stack:** Python 3.13, PyTorch 2.6+, PyG 2.7+, Rust 2024 + PyO3 0.28, `hexo-rs`, `uv`, `pytest`, `criterion`. All Python code in `hexo-a0/`. Rust MCTS extensions in `hexo-rs/`.

## Sub-Phase Decomposition

| Sub-Phase | Plan Document | Scope | Dependencies |
|-----------|--------------|-------|--------------|
| 3A | `phase3a-evaluation.md` | Evaluation infrastructure: greedy/random players, play_game, evaluate vs baseline/checkpoint | None |
| 3B | `phase3b-lr-scheduling.md` | LR scheduling: cosine decay, new config fields, checkpoint integration | None |
| 3C | `phase3c-curriculum.md` | Curriculum management: CurriculumScheduler, stage definitions, TOML config, Trainer integration | 3A, 3B |
| 3D | `phase3d-monitoring.md` | Enhanced monitoring: policy entropy, Elo, win rates, LR, stage tracking in tensorboard | 3A, 3B, 3C |
| 3E | `phase3e-rust-mcts.md` | Rust MCTS: tree ops in Rust, PyO3 bridge, batched eval callback, benchmarks | None (parallel with 3A-3D) |

**Execution order:** 3A + 3B + 3E in parallel → 3C → 3D → integration test across all sub-phases.

The remainder of this document defines the requirements and acceptance criteria shared across all sub-phases. Each sub-phase plan contains its own task breakdown, file structure, and tests.

---

## TrainingConfig Field Additions (coordination)

Fields added across sub-phases, in implementation order:

| Sub-Phase | Field | Type | Default | Used By |
|-----------|-------|------|---------|---------|
| 3B | `total_train_steps` | int | 100_000 | Cosine LR scheduler T_max |
| 3B | `lr_min` | float | 0.0 | Cosine LR scheduler eta_min |
| 3B | `eval_games` | int | 20 | Evaluation game count (used in 3D) |
| 3B | `eval_interval` | int | 5 | Evaluate every N iterations (used in 3D) |
| 3E | `use_rust_mcts` | bool | True | Dispatch to Rust or Python MCTS |

No field name collisions. 3B fields must be added before 3C/3D can use them.

---

## Requirements

### Evaluation

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-1 | Greedy policy player: select `argmax(logits)` over legal moves — no Gumbel noise, no MCTS | AC-1a: Given a game state, returns the single best action. AC-1b: Deterministic — same state always produces same action | T1 |
| REQ-2 | Random policy player: select uniformly at random from legal moves | AC-2: Each legal move has equal probability of selection | T1 |
| REQ-3 | Play a full evaluation game between two policy functions, returning outcome from P1's perspective | AC-3a: Returns +1 (P1 win), -1 (P1 loss), 0 (draw). AC-3b: Handles terminal games correctly. AC-3c: Both sides play as P1 and P2 across multiple games | T1 |
| REQ-4 | Evaluate current network: play `eval_games` between greedy-network and random baseline, split evenly between P1/P2 sides | AC-4a: Returns dict with `wins`, `losses`, `draws`, `win_rate`. AC-4b: Each side plays `eval_games // 2` as P1 and P2. AC-4c: Logs results to tensorboard | T1 |
| REQ-5 | Evaluate network vs previous checkpoint: play `eval_games` between current greedy and checkpoint greedy | AC-5a: Returns same dict format as REQ-4. AC-5b: Loads checkpoint model independently (doesn't affect training model). AC-5c: Both networks in eval mode with `torch.no_grad()` | T1 |

### LR Scheduling

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-6 | Cosine decay from `lr` to `lr_min` over `total_train_steps` gradient steps | AC-6a: LR starts at `config.lr` on step 0. AC-6b: LR reaches `lr_min` at step `total_train_steps`. AC-6c: Follows cosine curve: `lr_min + 0.5*(lr - lr_min)*(1 + cos(pi * step / total))`. AC-6d: `scheduler.step()` called once per gradient step | T2 |
| REQ-7 | Scheduler state persisted in checkpoints | AC-7a: `save_checkpoint` includes scheduler state_dict. AC-7b: `load_checkpoint` restores scheduler to the exact step. AC-7c: LR after load matches LR before save | T2 |
| REQ-8 | New TrainingConfig fields: `total_train_steps` (default 100_000), `lr_min` (default 0.0), `eval_games` (default 20), `eval_interval` (default 5) | AC-8: Fields exist with defaults, round-trip through TOML | T2 |

### Curriculum Management

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-9 | `CurriculumScheduler` manages stage transitions (A→B→C) based on convergence criteria | AC-9a: Tracks current stage. AC-9b: Advances to next stage when `advance()` is called. AC-9c: Returns current `GameConfig` via `current_config()`. AC-9d: Serialisable to/from dict for checkpointing | T3 |
| REQ-10 | Stage definitions loaded from TOML config | AC-10a: Each stage has `win_length`, `placement_radius`, `max_moves`, and `name`. AC-10b: Stages are ordered (A before B before C). AC-10c: Defaults match the master plan variants | T3 |
| REQ-11 | Trainer integrates curriculum: uses `CurriculumScheduler.current_config()` for self-play, advances stage on operator command or when convergence criteria are met | AC-11a: Self-play uses the current stage's GameConfig. AC-11b: Buffer is optionally cleared on stage transition (configurable). AC-11c: Checkpoint includes curriculum state | T3 |
| REQ-12 | CLI supports curriculum: `hexo-a0 train --config config.toml` with `[curriculum]` section | AC-12a: Stages defined in TOML. AC-12b: `--stage` flag to start from a specific stage. AC-12c: Stage transitions logged at INFO level | T3 |

### Enhanced Monitoring

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-13 | Policy entropy logged per iteration: `H(pi) = -sum(pi * log(pi))` averaged over self-play positions | AC-13: Tensorboard scalar `self_play/policy_entropy` decreases as the network becomes more decisive | T4 |
| REQ-14 | Win rate vs random logged per evaluation | AC-14: Tensorboard scalar `eval/win_rate_vs_random` | T4 |
| REQ-15 | Win rate vs previous checkpoint logged per evaluation | AC-15: Tensorboard scalar `eval/win_rate_vs_prev` | T4 |
| REQ-16 | Current LR logged per iteration | AC-16: Tensorboard scalar `training/lr` | T4 |
| REQ-17 | Current curriculum stage logged | AC-17: Tensorboard text or scalar `curriculum/stage` | T4 |
| REQ-18 | Win/draw/loss counts per evaluation logged | AC-18: Tensorboard scalars `eval/wins`, `eval/draws`, `eval/losses` | T4 |

### Rust MCTS (Performance)

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-19 | Rust implementation of MCTS tree: MCTSNode, select_child, backup, sigma, v_mix, normalize_q, Sequential Halving | AC-19a: All tree operations run in Rust. AC-19b: Python callback for batched network evaluation. AC-19c: Results match Python implementation on identical inputs (property test) | T5, T6 |
| REQ-20 | Rust `gumbel_mcts` exposed via PyO3: `hexo_rs.gumbel_mcts(game, eval_fn, config)` | AC-20a: Returns `(action, improved_policy)` matching Python version. AC-20b: `eval_fn` is a Python callable that takes a batch of game states and returns `(logits_list, values)`. AC-20c: 10x+ speedup on tree operations vs Python | T7 |
| REQ-21 | Fallback: Python MCTS still works if Rust MCTS is not available | AC-21: `gumbel_mcts` in Python remains functional. Trainer can be configured to use either | T7 |

---

## File Structure

### Python (hexo-a0/)

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/evaluation.py` | **NEW** — Greedy/random players, play_game, evaluate vs baseline/checkpoint |
| `src/hexo_a0/curriculum.py` | **NEW** — CurriculumScheduler, stage definitions |
| `src/hexo_a0/training_config.py` | **MODIFY** — Add `total_train_steps`, `lr_min`, `eval_games`, `eval_interval` |
| `src/hexo_a0/trainer.py` | **MODIFY** — Add LR scheduler, evaluation hooks, curriculum integration, enhanced tensorboard |
| `src/hexo_a0/config_io.py` | **MODIFY** — Add `[curriculum]` and `[evaluation]` TOML sections |
| `src/hexo_a0/cli.py` | **MODIFY** — Add `--stage` flag, evaluation subcommand |
| `tests/test_evaluation.py` | **NEW** — Evaluation tests |
| `tests/test_curriculum.py` | **NEW** — Curriculum scheduler tests |
| `tests/test_lr_scheduling.py` | **NEW** — LR scheduling tests |
| `tests/test_monitoring.py` | **NEW** — Enhanced monitoring tests |

### Rust (hexo-rs/)

| File | Responsibility |
|------|---------------|
| `src/mcts/mod.rs` | **NEW** — Rust MCTS module: node, tree ops |
| `src/mcts/node.rs` | **NEW** — MCTSNode struct |
| `src/mcts/select.rs` | **NEW** — select_child, sigma, v_mix, normalize_q |
| `src/mcts/simulate.rs` | **NEW** — simulate, simulate_batch |
| `src/mcts/halving.rs` | **NEW** — sequential_halving, gumbel_top_k |
| `src/mcts/gumbel.rs` | **NEW** — gumbel_mcts entry point |
| `src/python.rs` | **MODIFY** — Expose Rust MCTS via PyO3 |
| `tests/mcts_tests.rs` | **NEW** — Rust MCTS unit tests |
| `benches/mcts.rs` | **NEW** — Criterion benchmarks for Rust MCTS |

---

## Task 1: Evaluation Infrastructure

**Files:** Create `src/hexo_a0/evaluation.py`, `tests/test_evaluation.py`

- [ ] Write failing tests for `greedy_action(network, game, device)`:
  - Returns a valid legal move coordinate
  - Deterministic: same state → same action
  - Returns argmax of network logits (test with a mock network that returns known logits)
- [ ] Write failing tests for `random_action(game)`:
  - Returns a valid legal move coordinate
  - Uniform distribution (statistical test over many calls)
- [ ] Write failing tests for `play_game(p1_policy, p2_policy, game_config, device)`:
  - Returns +1, -1, or 0
  - Game terminates (doesn't hang)
  - With deterministic policies, produces deterministic outcomes
- [ ] Write failing tests for `evaluate_vs_random(network, game_config, device, n_games)`:
  - Returns dict with `wins`, `losses`, `draws`, `win_rate` keys
  - `wins + losses + draws == n_games`
  - Each side plays as P1 and P2 (`n_games // 2` each)
- [ ] Write failing tests for `evaluate_vs_checkpoint(network, checkpoint_path, model_config, game_config, device, n_games)`:
  - Returns same dict format
  - Loads checkpoint into separate model (original model unchanged)
  - Both models in eval mode
- [ ] Run tests — verify they fail
- [ ] Implement all evaluation functions
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- `greedy_action` calls `evaluate_state` from `mcts.py` to get logits, returns `coords[argmax(logits)]`
- `play_game` takes two callables `(game, device) -> (q, r)` and alternates turns. Uses `game.apply_move(q, r)` — TWO args.
- `evaluate_vs_random` creates a fresh game per match, alternates who plays P1.
- `evaluate_vs_checkpoint` creates a second `HeXONet`, loads checkpoint weights, uses `greedy_action` for both.

---

## Task 2: LR Scheduling + Config Fields

**Files:** Modify `src/hexo_a0/training_config.py`, modify `src/hexo_a0/trainer.py`, modify `src/hexo_a0/config_io.py`, create `tests/test_lr_scheduling.py`

- [ ] Write failing tests for new TrainingConfig fields:
  - `total_train_steps=100_000`, `lr_min=0.0`, `eval_games=20`, `eval_interval=5` exist with defaults
  - Round-trip through TOML
- [ ] Write failing tests for cosine LR scheduler:
  - LR starts at `config.lr` (2e-4)
  - LR at step `total_train_steps` is approximately `lr_min`
  - LR follows cosine curve (check midpoint: should be ~`(lr + lr_min) / 2`)
  - Scheduler state is saved and restored in checkpoints
  - After checkpoint load, LR matches the value at the saved step
- [ ] Run tests — verify they fail
- [ ] Add fields to `TrainingConfig`
- [ ] Add cosine scheduler to `Trainer.__init__`: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=lr_min)`
- [ ] Call `scheduler.step()` once per gradient step in `train_step` (NOT per iteration)
- [ ] Add scheduler state to `save_checkpoint` / `load_checkpoint`
- [ ] Update `config_io.py` to read/write new fields in `[training]` section
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 3: Curriculum Management

**Files:** Create `src/hexo_a0/curriculum.py`, `tests/test_curriculum.py`, modify `src/hexo_a0/config_io.py`, modify `src/hexo_a0/trainer.py`, modify `src/hexo_a0/cli.py`

- [ ] Write failing tests for `CurriculumStage` dataclass:
  - Has `name`, `win_length`, `placement_radius`, `max_moves`
  - Can convert to/from `GameConfig` dict
- [ ] Write failing tests for `CurriculumScheduler`:
  - Initialises with list of stages, starts at stage 0
  - `current_stage()` returns current `CurriculumStage`
  - `current_config()` returns a `GameConfig`-compatible dict
  - `advance()` moves to next stage, returns True. At last stage, returns False.
  - `to_dict()` / `from_dict()` round-trip
  - Default stages match master plan (A: 4/4, B: 5/6, C: 6/8)
- [ ] Write failing tests for TOML integration:
  - `[curriculum]` section with `stages` array and `clear_buffer_on_advance` flag
  - Missing `[curriculum]` section uses defaults
- [ ] Write failing tests for Trainer integration:
  - `Trainer` accepts optional `CurriculumScheduler`
  - Self-play uses `scheduler.current_config()` for GameConfig
  - `advance_stage()` method on Trainer triggers scheduler advance
  - Checkpoint includes curriculum state
- [ ] Write failing tests for CLI:
  - `--stage A` flag starts from stage A (default)
  - Stage name logged on startup
- [ ] Run tests — verify they fail
- [ ] Implement `CurriculumStage`, `CurriculumScheduler`, TOML integration, Trainer integration, CLI flag
- [ ] Run tests — all pass
- [ ] Commit

**TOML format:**
```toml
[curriculum]
clear_buffer_on_advance = true

[[curriculum.stages]]
name = "A"
win_length = 4
placement_radius = 4
max_moves = 80

[[curriculum.stages]]
name = "B"
win_length = 5
placement_radius = 6
max_moves = 120

[[curriculum.stages]]
name = "C"
win_length = 6
placement_radius = 8
max_moves = 200
```

---

## Task 4: Enhanced Monitoring

**Files:** Modify `src/hexo_a0/trainer.py`, create `tests/test_monitoring.py`

- [ ] Write failing tests for policy entropy computation:
  - `compute_policy_entropy(improved_policy)` returns scalar
  - Uniform policy → max entropy (`log(n)`)
  - Deterministic policy (one-hot) → entropy ~0
- [ ] Write failing tests for enhanced tensorboard logging:
  - After `train_iteration`, tensorboard has scalars for: `training/lr`, `self_play/policy_entropy`, `curriculum/stage`
  - After evaluation, tensorboard has: `eval/win_rate_vs_random`, `eval/wins`, `eval/draws`, `eval/losses`
- [ ] Run tests — verify they fail
- [ ] Add `compute_policy_entropy` helper to trainer.py
- [ ] Modify `self_play_game` to return policy entropy alongside `TrainingExample`s (or compute from improved_policy stored in examples)
- [ ] Modify `train_iteration` to:
  - Log `training/lr` (from scheduler)
  - Log `self_play/policy_entropy` (averaged over positions)
  - Log `curriculum/stage` (if curriculum scheduler present)
  - Run evaluation every `config.eval_interval` iterations
  - Log evaluation results to tensorboard
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 5: Rust MCTS — Core Tree Operations

**Files:** Create `src/mcts/` module in hexo-rs, create `tests/mcts_tests.rs`

- [ ] Write failing Rust tests for `MCTSNode`:
  - Fresh node has visit_count=0, value_sum=0.0
  - `q_value()` returns 0.0 for unvisited, ratio otherwise
  - Node stores prior, action, current_player, value_estimate
  - `is_expanded` / `_child_priors` / `get_or_create_child` (lazy expansion)
- [ ] Write failing tests for `backup`:
  - Player-aware sign flips (same tests as Python: same-player mid-turn, different-player switch)
  - 3-node path with HeXO two-placement turn
- [ ] Write failing tests for pure functions:
  - `sigma(q, max_visits, c_visit, c_scale)` — matches Python output
  - `normalize_q(q_values, q_min, q_max)` — clamped to [0,1]
  - `v_mix(value_estimate, children)` — Eq. 33, matches Python on reference inputs (priors=[0.8,0.2], N=[3,1], Q=[1.0,0.0], v_hat=0.5 → 0.74)
- [ ] Write failing tests for `select_child`:
  - Equal Q → highest prior wins
  - Unvisited get v_mix, not 0
  - All-unvisited handled without error
- [ ] Run tests — verify they fail
- [ ] Implement MCTSNode, backup, sigma, normalize_q, v_mix, select_child
- [ ] Run tests — all pass
- [ ] Run `cargo bench` to establish Rust baseline
- [ ] Commit

**Implementation notes:**
- Use `HashMap<(i32,i32), Box<MCTSNode>>` for children
- `_child_priors: HashMap<(i32,i32), f32>` for lazy expansion
- `GameState` stored as `Option<GameState>` on each node
- All math uses f64 for numerical consistency with Python

---

## Task 6: Rust MCTS — Simulation and Sequential Halving

**Files:** Add to `src/mcts/` module, extend `tests/mcts_tests.rs`

- [ ] Write failing tests for `simulate`:
  - Non-terminal leaf gets expanded (priors stored)
  - Terminal leaf backs up game outcome
  - Root visit count increments
  - Forced action works
- [ ] Write failing tests for `simulate_batch`:
  - Multiple leaves evaluated in one call
  - Returns list of (leaf_state, leaf_index) pairs for Python to evaluate
  - After Python provides results, expansion and backup complete
- [ ] Write failing tests for `sequential_halving`:
  - Total sims ≤ budget
  - Eliminates by Gumbel score (not raw Q)
  - Budget formula uses original n
  - With n=m=16, one round
- [ ] Write failing tests for `gumbel_top_k`:
  - Returns min(m, n) candidates
  - Deterministic with seeded RNG
- [ ] Run tests — verify they fail
- [ ] Implement simulate, simulate_batch, sequential_halving, gumbel_top_k
- [ ] Run tests — all pass
- [ ] Benchmark: Rust simulate vs Python simulate on same game state
- [ ] Commit

**Key design for simulate_batch:**
The Rust side does selection for all forced actions, collecting leaf nodes. It then returns the leaf game states to Python for network evaluation. Python calls `forward_batch`, returns logits+values. Rust then expands all leaves and backs up. This is a two-phase protocol:

```
Rust: select N leaves → return N game states to Python
Python: evaluate_state_batch(game_states) → (logits_list, values)
Rust: expand N leaves with priors, backup N paths
```

---

## Task 7: Rust MCTS — PyO3 Bridge and Integration

**Files:** Modify `src/python.rs`, create Python tests for Rust MCTS

- [ ] Write failing Python tests for `hexo_rs.gumbel_mcts`:
  - Same signature as Python version: `gumbel_mcts(game, eval_fn, config_dict)` → `(action, improved_policy_list)`
  - `eval_fn` is a Python callable: `eval_fn(game_states: list[GameState]) -> tuple[list[list[float]], list[float]]`
  - Returns same action as Python version on identical inputs (seeded)
  - 10x+ speedup (benchmark)
- [ ] Write failing tests for `hexo_rs.MCTSConfig`:
  - Wraps n_simulations, m_actions, c_visit, c_scale
- [ ] Write failing tests for fallback:
  - Python `gumbel_mcts` still works
  - Trainer can switch between Rust and Python MCTS via config flag
- [ ] Run tests — verify they fail
- [ ] Implement PyO3 bridge:
  - `hexo_rs.MCTSConfig` pyclass
  - `hexo_rs.gumbel_mcts` pyfn that:
    1. Builds Rust MCTS tree
    2. During simulate_batch, calls back to Python's `eval_fn`
    3. Returns (action_tuple, improved_policy_as_list)
- [ ] Add `use_rust_mcts: bool = True` to TrainingConfig
- [ ] Modify `self_play_game` to dispatch to Rust or Python MCTS based on config
- [ ] Run tests — all pass
- [ ] Benchmark: end-to-end iteration time with Rust vs Python MCTS
- [ ] Commit

---

## Task 8: End-to-End Integration Tests

**Files:** Create `tests/test_phase3_integration.py`, update `src/hexo_a0/__init__.py`

- [ ] Write integration test: full curriculum training
  - Start at Stage A with tiny config (2 games/iter, 3 train steps, 2 eval games)
  - Run 3 iterations at Stage A
  - Advance to Stage B
  - Run 2 iterations at Stage B
  - Verify: curriculum state correct, buffer optionally cleared, losses finite, eval results logged
- [ ] Write integration test: checkpoint round-trip with LR scheduler and curriculum
  - Save after iteration 2 at Stage A
  - Load into new Trainer
  - Verify: LR matches, curriculum stage matches, training continues
- [ ] Write integration test: Rust MCTS produces same improved policy direction as Python MCTS
  - On the same game state with seeded RNG
  - The actions don't need to be identical (Gumbel noise), but policy distributions should be similar
- [ ] Write integration test: evaluation produces reasonable results
  - Untrained network vs random: win_rate should be roughly 0.5 (± large margin since untrained)
  - Results dict has all required keys
- [ ] Update `__init__.py` with Phase 3 exports: `evaluate_vs_random`, `evaluate_vs_checkpoint`, `CurriculumScheduler`
- [ ] Run all tests (Phase 1 + 2 + 3) — all pass
- [ ] Commit

---

## Summary

| Task | What | Requirements | Depends on |
|------|------|-------------|------------|
| 1 | Evaluation infrastructure | REQ-1–5 | — |
| 2 | LR scheduling + config fields | REQ-6–8 | — |
| 3 | Curriculum management | REQ-9–12 | T1, T2 |
| 4 | Enhanced monitoring | REQ-13–18 | T1, T2, T3 |
| 5 | Rust MCTS core (tree, select, backup) | REQ-19 | — |
| 6 | Rust MCTS simulation + halving | REQ-19 | T5 |
| 7 | Rust MCTS PyO3 bridge | REQ-20–21 | T6 |
| 8 | End-to-end integration | All | T1–T7 |

**Dependency graph:**
```
T1 (evaluation) ──→ T3 (curriculum) → T4 (monitoring) → T8 (integration)
T2 (LR scheduling) → T3
T5 (Rust core) → T6 (Rust sim) → T7 (Rust PyO3) ──→ T8
```

Parallelisation: T1, T2, and T5 are fully independent. T3 depends on T1+T2. T6 depends on T5. T4 depends on T1+T2+T3. T7 depends on T6. T8 depends on everything.

**Critical path:** T5 → T6 → T7 → T8 (Rust MCTS is the longest chain at ~3 days).

**Architecture constant:** The GNN model architecture (6 layers, 128 hidden, 4 heads) remains unchanged across all curriculum stages. The GATv2Conv message-passing network handles variable-size graphs natively — no architecture change needed when board size increases.

**Questions for the human (noted, not blocking):**
1. **Stage transition trigger:** Should advancement be manual (operator command), automatic (convergence metric), or both? Currently planned as manual via `trainer.advance_stage()` — auto-advance can be added later.
2. **Buffer clearing on stage advance:** Should the replay buffer be cleared when changing game variants? Old positions have different legal-move counts. Planned as configurable (`clear_buffer_on_advance = true`).
3. **Rust MCTS parity:** Should the Rust MCTS produce bit-identical results to Python, or just statistically equivalent? Bit-identical requires matching float arithmetic and RNG. Planned as statistically equivalent with a property test.
4. **Mixed-variant training:** The master plan mentions an alternative where variant is sampled per game with shifting weights. Should this be in Phase 3 scope, or deferred? Planned as deferred — implement hard transitions first.
5. **Model architecture changes on stage transition:** The GNN handles variable graph sizes natively, so no architecture change is needed. But should we increase `num_layers` for larger boards (Stage C needs receptive field ≥ 6)? Planned as same architecture throughout — the Phase 2 model already has 6 layers.
