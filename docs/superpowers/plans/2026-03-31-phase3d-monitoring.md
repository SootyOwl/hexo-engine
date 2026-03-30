# Phase 3D: Enhanced Monitoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend tensorboard logging with policy entropy, evaluation win rates, LR tracking, and curriculum stage — everything needed to monitor multi-day curriculum training runs.

**Architecture:** Modify `Trainer.train_iteration` to compute and log additional metrics. Evaluation runs every `config.eval_interval` iterations. Policy entropy is computed from the improved policies already stored in `TrainingExample`. All metrics go to the existing `SummaryWriter`.

**Tech Stack:** Python 3.13, PyTorch 2.6+, `pytest`.

**Depends on:** Phase 3A (evaluation functions), 3B (LR scheduler + config fields), 3C (curriculum scheduler).

---

## Requirements

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-13 | Policy entropy per iteration | AC-13: `self_play/policy_entropy` scalar in tensorboard, computed as `-sum(pi * log(pi))` averaged over positions | T1 |
| REQ-14 | Win rate vs random per evaluation | AC-14: `eval/win_rate_vs_random` scalar logged every `eval_interval` iterations | T1 |
| REQ-15 | Win rate vs previous checkpoint | AC-15: `eval/win_rate_vs_prev` scalar logged. Previous checkpoint = last saved checkpoint. | T1 |
| REQ-16 | Current LR per iteration | AC-16: `training/lr` scalar matches scheduler's current LR | T1 |
| REQ-17 | Curriculum stage per iteration | AC-17: `curriculum/stage` logged (as scalar index or text) | T1 |
| REQ-18 | Win/draw/loss counts per evaluation | AC-18: `eval/wins`, `eval/draws`, `eval/losses` scalars | T1 |

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/hexo_a0/trainer.py` | **MODIFY** — Add evaluation hooks, entropy computation, enhanced tensorboard logging |
| `tests/test_monitoring.py` | **NEW** — Monitoring tests |

---

## Task 1: Enhanced Tensorboard Logging

**Files:** Modify `src/hexo_a0/trainer.py`, create `tests/test_monitoring.py`

- [ ] Write failing tests for `compute_policy_entropy(policy_tensor)`:
  - Uniform distribution over N actions → entropy = log(N)
  - One-hot distribution → entropy ≈ 0
  - Uses `xlogy` for NaN safety (same as KL loss): `entropy = -torch.special.xlogy(pi, pi).sum()` — xlogy(p,p) returns p*log(p) which is ≤0, so negating gives positive entropy ≥0
- [ ] Write failing tests for evaluation hooks in Trainer:
  - After `eval_interval` iterations, evaluation runs automatically
  - Evaluation results are in the returned metrics dict: `eval_win_rate_vs_random`
  - `eval_interval=0` disables evaluation entirely — no eval functions called
  - `train_iteration` works correctly when writer is None (no tensorboard) — returns metrics dict, no crash
- [ ] Write failing tests for enhanced tensorboard:
  - After `train_iteration`, writer has scalars for `training/lr`, `self_play/policy_entropy`
  - After evaluation iteration, writer has scalars for `eval/win_rate_vs_random`, `eval/wins`, `eval/draws`, `eval/losses`
  - If curriculum scheduler present, `curriculum/stage` is logged
- [ ] Run tests — verify they fail
- [ ] Implement `compute_policy_entropy` as a helper function
- [ ] Modify `self_play_game` or `train_iteration` to compute mean entropy from improved policies
  - Simplest: iterate over the examples returned by `self_play_game`, compute entropy of each `example.policy_target`, average
- [ ] Modify `train_iteration` to:
  - Log `training/lr` from `self.scheduler.get_last_lr()[0]` (if scheduler exists)
  - Log `self_play/policy_entropy`
  - Log `curriculum/stage` (if curriculum exists): `self.curriculum.stage_index`
  - Every `config.eval_interval` iterations (when `self.iteration % config.eval_interval == 0`):
    - Run `evaluate_vs_random(self.model, game_config, self.device, config.eval_games)`
    - Log all eval results to tensorboard
    - Optionally: if a previous checkpoint path is known, run `evaluate_vs_checkpoint`
  - Include eval metrics in the returned metrics dict
- [ ] Run tests — all pass
- [ ] Commit

**Implementation notes:**
- `compute_policy_entropy(pi)`: `entropy = -torch.special.xlogy(pi, pi).sum()` — xlogy(p,p) returns p*log(p) which is ≤0, so negating gives positive entropy ≥0. Same NaN-safe approach as KL loss.
- Evaluation at interval: only runs when `self.iteration % config.eval_interval == 0 and config.eval_interval > 0`
- The "previous checkpoint" for `evaluate_vs_checkpoint` is the most recently saved checkpoint file. Trainer stores `self.last_checkpoint_path: str | None = None`, set in `save_checkpoint()`. Used by evaluation to run `evaluate_vs_checkpoint` when available.
- Policy entropy is averaged across ALL positions from ALL self-play games in the iteration — one scalar per iteration.
- `evaluate_vs_random` returns dict with key `'win_rate'`. `evaluate_vs_checkpoint` returns same format. The tensorboard keys `eval/win_rate_vs_random` and `eval/win_rate_vs_prev` come from the monitoring code, not the evaluation functions.

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|-------------|
| 1 | Enhanced tensorboard + evaluation hooks | ~10 | REQ-13–18 |

Single task — all monitoring additions are tightly coupled to `train_iteration`. Depends on 3A (evaluation functions), 3B (scheduler for LR logging), 3C (curriculum for stage logging).
