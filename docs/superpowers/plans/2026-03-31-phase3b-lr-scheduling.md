# Phase 3B: LR Scheduling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cosine learning rate decay to the training loop, with new config fields and checkpoint integration.

**Architecture:** Wrap the existing Adam optimiser with `CosineAnnealingLR`. Add `total_train_steps`, `lr_min`, `eval_games`, `eval_interval` to `TrainingConfig`. Persist scheduler state in checkpoints. Step the scheduler once per gradient step, not per iteration.

**Tech Stack:** Python 3.13, PyTorch 2.6+, `pytest`.

---

## Requirements

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-6 | Cosine decay from `lr` to `lr_min` over `total_train_steps` | AC-6a: LR starts at `config.lr`. AC-6b: LR at final step ≈ `lr_min`. AC-6c: Midpoint LR ≈ `(lr + lr_min) / 2`. AC-6d: `scheduler.step()` per gradient step | T1 |
| REQ-7 | Scheduler state in checkpoints | AC-7a: save includes scheduler state_dict. AC-7b: load restores to exact step. AC-7c: LR after load matches pre-save value | T1 |
| REQ-8 | New config fields with defaults | AC-8a: `total_train_steps=100_000`. AC-8b: `lr_min=0.0`. AC-8c: `eval_games=20`. AC-8d: `eval_interval=5`. AC-8e: Round-trip through TOML | T1 |

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/training_config.py` | **MODIFY** — Add 4 new fields |
| `src/hexo_a0/trainer.py` | **MODIFY** — Add scheduler, step per gradient step, save/load scheduler state |
| `src/hexo_a0/config_io.py` | **MODIFY** — Add new fields to `[training]` TOML section |
| `tests/test_lr_scheduling.py` | **NEW** — All LR scheduling tests |

---

## Task 1: Config Fields + Cosine Scheduler + Checkpointing

**Files:** Modify `training_config.py`, modify `trainer.py`, modify `config_io.py`, create `tests/test_lr_scheduling.py`

- [ ] Write failing tests for new TrainingConfig fields:
  - Default `total_train_steps == 100_000`
  - Default `lr_min == 0.0`
  - Default `eval_games == 20`
  - Default `eval_interval == 5`
  - Custom values override
  - TOML round-trip (write then read) preserves all new fields
- [ ] Write failing tests for cosine scheduler behaviour:
  - At step 0, LR == `config.lr` (2e-4)
  - At step `total_train_steps`, LR ≈ `lr_min` (tolerance 1e-8)
  - At step `total_train_steps // 2`, LR ≈ `(lr + lr_min) / 2` (tolerance 1e-6)
  - After N gradient steps, LR has decreased (not constant)
- [ ] Write failing tests for checkpoint integration:
  - Save checkpoint after 100 gradient steps
  - Load checkpoint into fresh Trainer
  - Verify `scheduler.get_last_lr()` matches the pre-save value
  - After loading, run 10 more gradient steps and verify LR continues the cosine curve from where it was saved (not from the start)
- [ ] Write failing test for empty-buffer iteration:
  - When buffer is empty (no training steps in an iteration), scheduler is NOT stepped — scheduler stays in sync with actual gradient steps
- [ ] Run tests — verify they fail
- [ ] Add fields to `TrainingConfig`
- [ ] Add TOML read/write for new fields in `config_io.py`
- [ ] Add `CosineAnnealingLR` to `Trainer.__init__` as `self.scheduler`: created with `T_max=config.total_train_steps`, `eta_min=config.lr_min`
- [ ] In `train_iteration` (NOT `train_step`): call `self.scheduler.step()` after each `train_step()` call inside the training loop
  - Note: `train_step` remains a pure function — the scheduler is stepped by `train_iteration` after each call, keeping `train_step` free of side effects.
- [ ] In `save_checkpoint`: add `"scheduler_state_dict": self.scheduler.state_dict()`
- [ ] In `load_checkpoint`: restore `self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])`
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- The scheduler is a Trainer attribute (`self.scheduler`): `CosineAnnealingLR(self.optimizer, T_max=config.total_train_steps, eta_min=config.lr_min)`
- `self.scheduler.step()` is called inside `train_iteration` after each `train_step()` call, NOT inside `train_step()`. This keeps `train_step` as a pure function.
- Implementation detail: Trainer tracks `self.global_step: int` counting total gradient steps. This is used internally by the scheduler and saved in checkpoints as part of the scheduler state_dict. It is NOT a separate requirement — the scheduler's `load_state_dict` handles step restoration.
- Trainer also stores `self.last_checkpoint_path: str | None = None`, updated in `save_checkpoint()`. This is used by Phase 3D's evaluation-vs-checkpoint feature.
- `eval_games` and `eval_interval` are forward-compatible fields for Phase 3D — not used in this sub-phase.

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|-------------|
| 1 | Config fields + scheduler + checkpoints | ~12 | REQ-6, REQ-7, REQ-8 |

Single task — all pieces are tightly coupled (config defines scheduler params, scheduler must be in checkpoints).
