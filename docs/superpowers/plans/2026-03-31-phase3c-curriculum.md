# Phase 3C: Curriculum Management — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manage stage transitions (A→B→C) through progressively harder HeXO variants, with TOML config, Trainer integration, and checkpoint persistence.

**Architecture:** `CurriculumStage` dataclass defines each variant (name, win_length, placement_radius, max_moves). `CurriculumScheduler` holds an ordered list of stages and tracks the current position. The Trainer queries `scheduler.current_config()` for the active GameConfig during self-play. Stage advancement is manual (operator calls `trainer.advance_stage()`) — automatic advancement based on convergence metrics is deferred.

**Tech Stack:** Python 3.13, `hexo-rs`, `pytest`.

**Depends on:** Phase 3A (evaluation functions) and Phase 3B (LR scheduling) — specifically, the `eval_interval` and `eval_games` config fields added by 3B must exist in TrainingConfig.

---

## Requirements

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-9 | `CurriculumScheduler` manages stage transitions | AC-9a: Tracks current stage index. AC-9b: `advance()` moves to next, returns True; at last stage returns False. AC-9c: `current_config()` returns dict compatible with `hexo_rs.GameConfig(**d)`. AC-9d: `to_dict()` / `from_dict()` round-trip | T1 |
| REQ-10 | Stage definitions from TOML | AC-10a: `[[curriculum.stages]]` array in TOML. AC-10b: Stages are ordered by array position. AC-10c: Defaults match master plan: A(4,4,80), B(5,6,120), C(6,8,200) | T1 |
| REQ-11 | Trainer integration | AC-11a: Self-play uses current stage's GameConfig. AC-11b: `clear_buffer_on_advance` option (default True). AC-11c: Checkpoint includes curriculum state (current stage index). AC-11d: After load, curriculum resumes at saved stage | T2 |
| REQ-12 | CLI support | AC-12a: `--stage A` starts from named stage (default: first stage). AC-12b: Stage name logged at INFO on startup and on advance. AC-12c: `hexo-a0 train --config config.toml --stage B` skips Stage A | T2 |

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/curriculum.py` | **NEW** — CurriculumStage, CurriculumScheduler, DEFAULT_STAGES |
| `src/hexo_a0/config_io.py` | **MODIFY** — Parse `[curriculum]` TOML section |
| `src/hexo_a0/trainer.py` | **MODIFY** — Accept optional CurriculumScheduler, use for GameConfig, checkpoint state |
| `src/hexo_a0/cli.py` | **MODIFY** — Add `--stage` flag |
| `tests/test_curriculum.py` | **NEW** — Curriculum tests |

---

## Task 1: CurriculumScheduler + TOML

**Files:** Create `src/hexo_a0/curriculum.py`, create `tests/test_curriculum.py`, modify `src/hexo_a0/config_io.py`

- [ ] Write failing tests for `CurriculumStage`:
  - Has `name: str`, `win_length: int`, `placement_radius: int`, `max_moves: int`
  - `to_game_config_dict()` returns `{"win_length": ..., "placement_radius": ..., "max_moves": ...}`
- [ ] Write failing tests for `CurriculumScheduler`:
  - Initialises with list of stages, starts at index 0
  - `current_stage()` returns the first `CurriculumStage`
  - `current_config()` returns `{"win_length": int, "placement_radius": int, "max_moves": int}` — usable with `hexo_rs.GameConfig(**d)`
  - `current_stage().name` returns `"A"` (no `current_name()` method — use `current_stage().name` instead)
  - `advance()` → True, now at stage 1. `advance()` again → True, now at 2. `advance()` → False (no more stages)
  - Single-stage curriculum: `advance()` returns False immediately
  - `stage_index` property returns current index
  - `to_dict()` returns `{"stage_index": N, "stages": [...]}`
  - `from_dict(d)` reconstructs scheduler at the saved stage
  - Round-trip: `from_dict(to_dict())` preserves stage index
- [ ] Write failing tests for `DEFAULT_STAGES`:
  - 3 stages: A(4,4,80), B(5,6,120), C(6,8,200)
- [ ] Write failing tests for TOML loading:
  - TOML with `[[curriculum.stages]]` array produces matching stages
  - Missing `[curriculum]` section uses `DEFAULT_STAGES`
  - `clear_buffer_on_advance` flag (default True)
  - `clear_buffer_on_advance=True` → buffer empty after `advance()`. `clear_buffer_on_advance=False` → buffer unchanged after `advance()`.
- [ ] Run tests — verify they fail
- [ ] Implement `CurriculumStage`, `CurriculumScheduler`, `DEFAULT_STAGES`
- [ ] Add `[curriculum]` parsing: implement `load_curriculum(path) -> CurriculumScheduler` as a new function in `config_io.py` that reads only the `[curriculum]` section from a TOML file. `load_config()` return type is unchanged. CLI calls both.
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

## Task 2: Trainer + CLI Integration

**Files:** Modify `src/hexo_a0/trainer.py`, modify `src/hexo_a0/cli.py`, extend `tests/test_curriculum.py`

- [ ] Write failing tests for Trainer with curriculum:
  - `Trainer` accepts EITHER `game_config: dict` (no curriculum) OR `curriculum: CurriculumScheduler` (curriculum mode). If curriculum is provided, game_config is derived from `curriculum.current_config()`. Both cannot be None.
  - When curriculum is present, `self_play_game` uses `curriculum.current_config()` instead of `self.game_config`
  - `trainer.advance_stage()` calls `curriculum.advance()`, optionally clears buffer, logs stage change
  - Checkpoint includes `"curriculum": curriculum.to_dict()`
  - After load, `curriculum.current_stage().name` matches saved stage
- [ ] Write failing tests for CLI:
  - `--stage B` sets scheduler to stage index 1 (skipping A)
  - Default (no `--stage` flag) starts at stage 0 (A)
  - Invalid stage name raises error
- [ ] Run tests — verify they fail
- [ ] Modify `Trainer.__init__` to accept EITHER `game_config: dict` OR `curriculum: CurriculumScheduler` (not both None)
- [ ] Modify `train_iteration`: use `self.curriculum.current_config()` if present, else `self.game_config`
- [ ] Add `advance_stage()` method to Trainer
- [ ] Add curriculum state to `save_checkpoint` / `load_checkpoint`
- [ ] Add `--stage` flag to CLI `train` subparser
- [ ] In `_run_train`, build CurriculumScheduler from config, apply `--stage` flag
- [ ] Run tests — all pass
- [ ] Commit

**Design decision:** `advance_stage()` is manual — the operator decides when to advance (watching tensorboard for convergence). Automatic advancement can be added later by checking metrics in `train_iteration`.

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|-------------|
| 1 | CurriculumScheduler + TOML | ~12 | REQ-9, REQ-10 |
| 2 | Trainer + CLI integration | ~8 | REQ-11, REQ-12 |

Total: ~20 tests across 2 tasks. Depends on 3A and 3B being complete.
