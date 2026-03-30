# Phase 3E: Rust MCTS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MCTS tree operations from Python to Rust for 10-50x speedup, enabling practical training on larger boards (radius 6-8).

**Architecture:** Rust owns the MCTS tree (nodes, selection, backup, Sequential Halving). When leaf nodes need network evaluation, Rust batches the requests and calls a Python callback. Python runs `forward_batch` on the GPU and returns logits+values. Rust then expands leaves and backs up.

```
Rust (PyO3)                         Python
┌───────────────────────┐           ┌──────────────┐
│ MCTSNode tree         │  batch    │              │
│ select_child()        │  game     │ HeXONet      │
│ backup()              │  states   │ .forward()   │
│ sequential_halving()  │ ───────→  │              │
│ gumbel_mcts()         │ ←───────  │ logits+vals  │
│ sigma/v_mix/normalize │  results  │              │
└───────────────────────┘           └──────────────┘
```

**Tech Stack:** Rust 2024, PyO3 0.28, `rand` crate for Gumbel sampling, `criterion` for benchmarks. Tests in both Rust (`cargo test`) and Python (`pytest`).

**Algorithm invariant:** Gumbel-top-k sampling is used ONLY at the root to select m candidate actions before Sequential Halving. Non-root nodes use select_child (Section 5, Eq. 14). This is a fundamental invariant of the algorithm.

**Depends on:** Nothing — can be developed in parallel with 3A-3D. Integration tested in Phase 3 integration task.

---

## Requirements

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-19 | Rust MCTS tree ops: node, select, backup, sigma, v_mix, normalize_q, Sequential Halving | AC-19a: All tree ops in Rust. AC-19b: Python callback for batched network eval. AC-19c: Property test: Rust and Python produce same backup values on identical inputs | T1, T2 |
| REQ-20 | PyO3 entry point: `hexo_rs.gumbel_mcts(game, eval_fn, config)` | AC-20a: Returns `(action_tuple, improved_policy_list)`. AC-20b: `eval_fn` is a Python callable. AC-20c: 10x+ speedup on tree ops vs Python | T3 |
| REQ-21 | Fallback to Python MCTS | AC-21a: Python `gumbel_mcts` remains functional. AC-21b: `use_rust_mcts` config flag controls dispatch | T3 |

---

## File Structure

### Rust (hexo-rs/)

| File | Responsibility |
|------|---------------|
| `src/mcts/mod.rs` | **NEW** — Module declarations |
| `src/mcts/node.rs` | **NEW** — `MCTSNode` struct with lazy expansion |
| `src/mcts/scoring.rs` | **NEW** — `sigma`, `v_mix`, `normalize_q`, `softmax` |
| `src/mcts/backup.rs` | **NEW** — Player-aware backup |
| `src/mcts/select.rs` | **NEW** — `select_child` (Section 5 formula) |
| `src/mcts/simulate.rs` | **NEW** — `simulate`, `simulate_batch` |
| `src/mcts/halving.rs` | **NEW** — `sequential_halving`, `gumbel_top_k` |
| `src/mcts/gumbel_mcts.rs` | **NEW** — `gumbel_mcts` entry point |
| `src/python.rs` | **MODIFY** — Expose `gumbel_mcts` + `MCTSConfig` via PyO3 |
| `src/lib.rs` | **MODIFY** — Add `pub mod mcts;` |

### Python (hexo-a0/)

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/training_config.py` | **MODIFY** — Add `use_rust_mcts: bool = True` |
| `src/hexo_a0/self_play.py` | **MODIFY** — Dispatch to Rust or Python MCTS |
| `tests/test_rust_mcts.py` | **NEW** — Python-side tests for Rust MCTS |

### Benchmarks

| File | Responsibility |
|------|---------------|
| `benches/mcts.rs` | **NEW** — Criterion benchmarks for Rust MCTS tree ops |

---

## Task 1: Rust MCTS Core — Node + Scoring + Backup

**Files:** Create `src/mcts/` module with `mod.rs`, `node.rs`, `scoring.rs`, `backup.rs`

- [ ] Write failing Rust tests for `MCTSNode`:
  - Fresh node: visit_count=0, value_sum=0.0, no children, not expanded
  - `q_value()`: 0.0 if unvisited, ratio otherwise
  - Stores prior (f64), action ((i32,i32)), current_player (enum), value_estimate (f64)
  - `is_expanded()`: false until `expand()` called
  - `child_priors`: HashMap<(i32,i32), f64>
  - `get_or_create_child(action)`: creates child lazily from GameState.clone() + apply_move
- [ ] Write failing tests for scoring functions:
  - `sigma(q, max_visits, c_visit, c_scale)` — numerical match with Python: sigma(0.5, 10, 50, 1.0) == 30.0
  - `normalize_q(q_map, q_min, q_max)` — clamped to [0,1], NaN-safe (`q_min == q_max` → 0)
  - `v_mix(v_hat, children)` — Eq. 33 reference: priors=[0.8,0.2], N=[3,1], Q=[1.0,0.0], v_hat=0.5 → 0.74
  - `v_mix` with 3+ actions where only a subset are visited — verify `sum_visited_prior` uses only visited children's priors
  - `softmax(logits)` — standard numerically-stable softmax
- [ ] Write failing tests for `backup`:
  - Same-player path: no sign flip
  - Different-player: sign flip
  - 3-node HeXO path (P1→P2→P2): one flip at grandparent
  - All visit_counts increment
  - Terminal draw (winner=None) → value=0.0. Terminal win (current player won) → value=+1.0. Terminal loss (opponent won) → value=-1.0
- [ ] Run tests — verify they fail
- [ ] Implement all structs and functions
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- `MCTSNode` uses `HashMap<(i32,i32), Box<MCTSNode>>` for children (boxed for recursive ownership)
- Player is represented as a Rust enum `Player { P1, P2 }` internally. At the PyO3 boundary, it converts to/from Python strings `'P1'`/`'P2'`.
- Lazy expansion invariant: after expansion, `node.is_expanded() == true` AND `node.child_priors` is non-empty. `select_child` panics if called on an unexpanded node.
- All scoring math uses f64 for consistency with Python
- `GameState` stored as `Option<GameState>` on each node (owned)
- `get_or_create_child` clones the parent's GameState and applies the move

---

## Task 2: Rust MCTS — Selection, Simulation, Sequential Halving

**Files:** Create `select.rs`, `simulate.rs`, `halving.rs`

- [ ] Write failing tests for `select_child`:
  - Equal Q → highest prior wins
  - Unvisited get v_mix, not 0
  - All-unvisited → degrades to prior ranking
  - c_visit and c_scale parameters are threaded through
- [ ] Write failing tests for `simulate`:
  - Non-terminal leaf gets expanded (priors stored, lazy)
  - Terminal leaf backs up game outcome
  - Forced action works
  - Returns list of leaf game states that need network evaluation (for batching)
- [ ] Write failing tests for `simulate_batch`:
  - Given N forced actions, returns N leaves
  - After providing eval results, all leaves expanded and backed up
  - Root visit_count increments by N
- [ ] Write failing tests for `gumbel_top_k`:
  - Returns min(m, n) candidates
  - Seeded RNG → deterministic
- [ ] Write failing tests for `sequential_halving`:
  - Total sims ≤ budget
  - Eliminates by Gumbel score (g + logits + sigma(Q))
  - Budget uses original n (not decrementing)
  - n=m=16 → one round
  - `num_phases = ceil(log2(m_actions))`. Formula: `sims_per_action = max(1, floor(n / (num_phases * len(remaining))))`
- [ ] Run tests — verify they fail
- [ ] Implement all functions
- [ ] Run tests — all pass
- [ ] Benchmark with criterion: `select_child`, `simulate`, `sequential_halving` on a realistic game state
- [ ] Benchmark `GameState::clone` latency on radius 2, 4, 6, 8 boards separately
- [ ] Commit

**Design for simulate_batch (two-phase protocol):**

Phase 1 (Rust): For each forced action, select a path to a leaf node. Collect all non-terminal leaf GameStates. Return them as a vec.

Phase 2 (Rust, after Python provides results): For each leaf, store priors from the provided logits, set value_estimate. Backup each path.

The Python bridge (Task 3) connects these phases by calling `eval_fn(game_states)`.

---

## Task 3: PyO3 Bridge + Python Integration

**Files:** Modify `src/python.rs`, create `tests/test_rust_mcts.py` in hexo-a0, modify `self_play.py`, modify `training_config.py`

- [ ] Write failing Rust tests for `gumbel_mcts`:
  - Composes top_k, sequential_halving, compute_improved_policy
  - Returns (action, improved_policy)
  - Terminal state returns error
- [ ] Write failing Python tests for `hexo_rs.gumbel_mcts`:
  - `hexo_rs.gumbel_mcts(game, eval_fn, mcts_config)` → `((q,r), [floats])`
  - `eval_fn` signature: `eval_fn(game_states: list[GameState]) -> tuple[list[list[float]], list[float]]`
  - Returns valid legal move
  - improved_policy sums to ~1.0
  - improved_policy length == len(game.legal_moves())
  - 10x speedup vs Python (benchmark, not assertion)
- [ ] Write failing Python tests for fallback:
  - `use_rust_mcts = False` → Python MCTS used
  - `use_rust_mcts = True` → Rust MCTS used (default)
- [ ] Run tests — verify they fail
- [ ] Implement `gumbel_mcts` in Rust (composes T1+T2 components)
- [ ] Implement PyO3 bridge:
  - `#[pyclass] MCTSConfig` wrapping n_simulations, m_actions, c_visit, c_scale
  - `#[pyfunction] fn gumbel_mcts(game: &PyGameState, eval_fn: PyObject, config: &MCTSConfig) -> PyResult<(...)>`
  - Inside: build tree, during simulate_batch call `eval_fn.call1(py, (game_states_list,))?` to get Python eval results. `eval_fn` exceptions are caught and re-raised as `PyErr`. Output shapes are validated: `logits_per_state[i].len()` must equal `legal_moves(states[i]).len()`. Values outside `[-1, 1]` are clamped.
- [ ] Add `use_rust_mcts: bool = True` to `TrainingConfig`
- [ ] Modify `self_play_game` dispatch:
  ```python
  if config.use_rust_mcts:
      action, policy = hexo_rs.gumbel_mcts(game, eval_fn, mcts_config)
  else:
      action, policy = gumbel_mcts(game, network, config, device)
  ```
  Where `eval_fn` is a closure that wraps `evaluate_state_batch`
- [ ] Add `use_rust_mcts` to TOML config_io
- [ ] Run all tests — Rust and Python pass
- [ ] Run benchmark: compare iteration time with Rust vs Python MCTS
- [ ] Commit

**PyO3 bridge details:**

The `eval_fn` callback protocol:
```python
def eval_fn(game_states: list[hexo_rs.GameState]) -> tuple[list[list[float]], list[float]]:
    """Evaluate a batch of game states.

    Returns:
        (logits_per_state, values) where:
        - logits_per_state[i] is a list of floats (one per legal move in state i)
        - values[i] is a float in [-1, 1]
    """
    # Uses game_to_graph + Batch.from_data_list + model.forward_batch
    ...
```

The Rust side calls this once per `simulate_batch` invocation (one Python→GPU round-trip per batch of 16 leaves).

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|-------------|
| 1 | Rust core: node + scoring + backup | ~15 | REQ-19 (partial) |
| 2 | Rust sim + halving + benchmarks | ~15 | REQ-19 (complete) |
| 3 | PyO3 bridge + Python integration | ~10 | REQ-20, REQ-21 |

Total: ~40 tests across 3 tasks. ~2-3 days of work. Independent of 3A-3D.

**Expected performance:**
- Tree operations (select, backup, expand): <1ms per move (vs ~200ms Python at radius 4)
- Network eval (Python callback): ~15ms per batch (unchanged — this is the GPU)
- `game_to_graph`: ~100ms per batch (unchanged — Python)
- Net speedup per move: ~350ms → ~120ms (tree ops eliminated, graph+forward remain)
- Further wins: move `game_to_graph` to Rust too (future Phase 4 optimisation)
