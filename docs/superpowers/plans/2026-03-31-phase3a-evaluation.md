# Phase 3A: Evaluation Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play evaluation games between the current network and baselines (random policy, previous checkpoint) to measure training progress.

**Architecture:** Two policy functions (greedy network, random), a generic `play_game` driver, and two evaluation functions that aggregate results across multiple games with P1/P2 symmetry. All are stateless functions — no modification to Trainer required (that's in 3C/3D).

**Tech Stack:** Python 3.13, PyTorch 2.6+, `hexo-rs`, `pytest`.

---

## Requirements

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-1 | Greedy policy: `argmax(logits)` over legal moves, no noise, no MCTS | AC-1a: Returns valid legal move. AC-1b: Deterministic — same state → same action. AC-1c: Uses `evaluate_state` from mcts.py, network in eval mode with `torch.no_grad()` | T1 |
| REQ-2 | Random policy: uniform random from legal moves | AC-2: Each legal move has equal probability | T1 |
| REQ-3 | `play_game(p1_fn, p2_fn, game_config, device)` → outcome from P1's perspective | AC-3a: Returns +1 (P1 win), -1 (P1 loss), 0 (draw). Returns outcome from **P1's perspective** (+1 P1 win, -1 P2 win, 0 draw). Evaluation aggregators translate to network perspective. AC-3b: Game always terminates. AC-3c: apply_move uses TWO args: `game.apply_move(q, r)` | T1 |
| REQ-4 | `evaluate_vs_random(network, game_config, device, n_games)` → results dict | AC-4a: Returns `{"wins": int, "losses": int, "draws": int, "win_rate": float}`. AC-4b: `n_games` split evenly — half as P1, half as P2. AC-4c: `wins + losses + draws == n_games` | T2 |
| REQ-5 | `evaluate_vs_checkpoint(network, ckpt_path, model_config, game_config, device, n_games)` → results dict | AC-5a: Same dict format as REQ-4. AC-5b: Loads checkpoint into separate model instance (training model unaffected). AC-5c: Both models in eval mode | T2 |

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/evaluation.py` | **NEW** — greedy_action, random_action, play_game, evaluate_vs_random, evaluate_vs_checkpoint |
| `tests/test_evaluation.py` | **NEW** — All evaluation tests |

---

## Task 1: Policy Functions and Game Driver

**Files:** Create `src/hexo_a0/evaluation.py`, create `tests/test_evaluation.py`

- [ ] Write failing tests for `greedy_action(network, game, device)`:
  - Returns a `(q, r)` tuple that is in `game.legal_moves()`
  - Deterministic: calling twice on same state returns same action
  - With a mock network returning known logits `[0.1, 0.9, 0.5]`, returns the coord corresponding to index 1 (highest logit)
- [ ] Write failing tests for `random_action(game, rng=None)`:
  - Returns a `(q, r)` tuple in `game.legal_moves()`
  - With seeded `random.Random`, deterministic
  - Over 1000 calls, all legal moves are selected at least once (statistical uniformity)
- [ ] Write failing tests for `play_game(p1_fn, p2_fn, game_config, device)`:
  - Returns an int in `{+1, -1, 0}`
  - With two random players, game terminates (no hang)
  - With `max_moves=2`, game always draws (not enough moves to win)
  - P1 function is called when P1 is to move, P2 function when P2 is to move
- [ ] Run tests — verify they fail
- [ ] Implement `greedy_action`, `random_action`, `play_game`
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- `greedy_action` uses `evaluate_state(network, game, device)` from `mcts.py` — returns `coords[argmax(logits)]`. `greedy_action` assumes the caller has set eval mode and torch.no_grad — it does NOT toggle model mode itself (matching evaluate_state's contract).
- `random_action` uses `random.choice(sorted(game.legal_moves()))` (sorted for consistency with graph ordering)
- `play_game` creates `GameState(game_config)`, loops until terminal, calls the appropriate policy function based on `game.current_player()`

---

## Task 2: Evaluation Aggregators

**Files:** Modify `src/hexo_a0/evaluation.py`, extend `tests/test_evaluation.py`

- [ ] Write failing tests for `evaluate_vs_random`:
  - Returns dict with correct keys
  - `wins + losses + draws == n_games`
  - With `n_games=4`, plays 2 games as P1 and 2 as P2
  - Untrained network: win_rate is roughly 0.5 (wide tolerance ±0.4 since only 4 games)
- [ ] Write failing tests for `evaluate_vs_checkpoint`:
  - Returns dict with correct keys
  - Original model weights unchanged after evaluation
  - Checkpoint model loaded separately
  - With same checkpoint as current model: win_rate ≈ 0.5 (symmetric)
- [ ] Run tests — verify they fail
- [ ] Implement both functions
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- `evaluate_vs_random` runs `n_games // 2` games with network as P1 (vs random P2) and `n_games // 2` with network as P2 (vs random P1). Counts wins/losses/draws from the **network's** perspective, not P1's. When the network plays P2 and wins, that's a 'win'.
- `evaluate_vs_checkpoint` creates a second `HeXONet(model_config)`, loads `torch.load(ckpt_path, weights_only=True)["model_state_dict"]`, uses `greedy_action` for both.
- Both functions set networks to eval mode and use `torch.no_grad()`.

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|-------------|
| 1 | Policy functions + game driver | ~8 | REQ-1, REQ-2, REQ-3 |
| 2 | Evaluation aggregators | ~6 | REQ-4, REQ-5 |

Total: ~14 tests across 2 tasks. No dependencies on other sub-phases.
