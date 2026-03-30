# Phase 0: HeXO Game Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast, correct HeXO game engine in pure Rust with comprehensive tests.

**Architecture:** Library crate with small, focused modules. `GameState` is the public-facing type that composes internal `Board`, `TurnState`, and win-detection logic. Sparse `HashMap<Coord, Player>` board handles the infinite grid. TurnState is a state machine encoding the two-move-per-turn rule. Legal moves are computed on demand by scanning a radius around placed stones. Output of `legal_moves()` is sorted for determinism.

**Tech Stack:** Rust 2024 edition, `criterion` for benchmarks, no other dependencies.

---

## Requirements

### Data Structures

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-1 | Axial coordinate system using (q, r) pairs of i32 | AC-1: `Coord` is defined as `(i32, i32)` and used throughout the codebase | T1 |
| REQ-2 | Sparse board representation using `HashMap<Coord, Player>` | AC-2: `Board` stores stones in a `HashMap`; empty cells consume no memory | T3 |
| REQ-3 | Hex distance calculated as `max(\|dq\|, \|dr\|, \|dq + dr\|)` | AC-3: `hex_distance` passes tests for same-point (0), adjacent (1), along-axis, diagonal, symmetric, and boundary (8/9) cases | T1 |

### Turn Rules

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-4 | P1's first move is hardcoded at (0,0); game starts with P2 to move | AC-4a: `GameState::new()` places P1 at origin and sets current player to P2. AC-4b: `Board::new()` contains exactly one stone: P1 at (0,0) | T3, T6 |
| REQ-5 | Each turn consists of 2 hex placements (except P1's implicit first move) | AC-5: `moves_remaining_this_turn()` returns 2 at start of turn, 1 after first placement, then player switches after second placement | T2, T6 |
| REQ-6 | Turn alternates between P2 and P1 after each 2-move turn | AC-6: Full turn cycle test — P2(2)→P2(1)→P1(2)→P1(1)→P2(2) — passes | T2 |

### Legal Moves

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-7 | A new hex must be placed on an empty cell within hex-distance ≤ radius of any existing stone | AC-7a: `legal_moves()` returns only empty cells within the configured radius. AC-7b: `apply_move()` returns `OutOfRange` for cells beyond radius. AC-7c: No occupied cells appear in legal moves | T5, T6 |
| REQ-8 | Legal move set expands as stones are placed further from origin | AC-8: After placing a stone at the edge of the radius, `legal_moves()` returns more cells than before (minus the newly occupied cell) | T5 |
| REQ-9 | Legal moves are deterministically ordered | AC-9a: No duplicate coordinates. AC-9b: Output is sorted lexicographically by (q, r) ascending, producing identical results across calls on the same state | T5 |

### Win Detection

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-10 | First player to connect N of their hexes in a straight line wins (N = `win_length`) | AC-10a: N consecutive same-player stones along any axis triggers win. AC-10b: N-1 in a row does not trigger win. AC-10c: N+1 in a row also triggers win | T4, T7 |
| REQ-11 | Win detection checks all 3 hex-grid axes: (1,0), (0,1), (1,-1) | AC-11: Win is detected along q-axis, r-axis, and diagonal axis independently | T4 |
| REQ-12 | Win on the first placement of a 2-move turn ends the game immediately (no second placement) | AC-12: After a winning first move, `is_terminal()` is true and `apply_move()` returns `GameOver` | T6, T7 |
| REQ-13 | Win detection only counts the current player's stones | AC-13: N P1 stones in a row does not register as a P2 win | T4 |
| REQ-14 | Non-line arrangements do not trigger false wins | AC-14: 6+ same-player stones in an L-shape or scattered pattern do not trigger a win | T7 |

### Game State API

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-15 | `GameState::new()` initialises with P1 at origin, P2 to move | AC-15: Tested in unit tests for initial state and origin stone | T6 |
| REQ-16 | `apply_move()` validates occupancy, range, and game-over state | AC-16a: `CellOccupied` on occupied cell. AC-16b: `OutOfRange` on distant cell. AC-16c: `GameOver` after terminal state | T6 |
| REQ-17 | `clone()` produces an independent copy (for MCTS rollback) | AC-17: Mutating a clone does not affect the original | T3, T6 |
| REQ-18 | `placed_stones()` returns all stones with their owning player | AC-18: Stone count grows correctly after each placement | T6 |

### Curriculum Support

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-19 | Win length, placement radius, and max moves are configurable per game | AC-19a: `GameConfig` allows setting `win_length`, `placement_radius`, and `max_moves`. AC-19b: 4-in-a-row variant works correctly. AC-19c: Smaller radius produces fewer legal moves. AC-19d: `FULL_HEXO` constant uses win_length=6, placement_radius=8, max_moves=200 | T4, T6 |

### Game Termination

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-20 | Game ends in a draw after a configurable move limit (default 200) | AC-20a: `is_terminal()` returns true when move limit reached. AC-20b: `winner()` returns `None` for draws. AC-20c: `apply_move()` returns `GameOver` after limit | T6, T7 |

### Robustness

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-21 | No panics under random play | AC-21: 500 random games (4-in-a-row) + 100 random games (full HeXO) complete without panics | T7 |

### Performance

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-22 | `legal_moves()` completes in <100μs for ~50-100 placed stones | AC-22: Criterion benchmark median < 100μs | T8 |
| REQ-23 | `apply_move()` completes in <10μs | AC-23: Criterion benchmark median < 10μs | T8 |

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/lib.rs` | Module declarations, re-exports of public API |
| `src/types.rs` | `Player` enum, `Coord` type alias, `HEX_DIRS` and `WIN_AXES` constants |
| `src/hex.rs` | `hex_distance(a, b)` function |
| `src/turn.rs` | `TurnState` enum and state transitions |
| `src/board.rs` | `Board` struct — stone storage, placement, adjacency queries |
| `src/win.rs` | `check_win(board, coord, player, win_length)` — N-in-a-row detection |
| `src/legal_moves.rs` | `legal_moves(board, radius)` — all empty cells within radius, sorted |
| `src/game.rs` | `GameState` + `GameConfig` + `MoveError` — public API composing all of the above |
| `tests/integration.rs` | Integration tests: full games, edge cases, fuzz |
| `benches/engine.rs` | Criterion benchmarks for `legal_moves` and `apply_move` |

---

## Task 1: Core Types and Hex Distance

**Files:** Create `src/lib.rs`, `src/types.rs`, `src/hex.rs`

- [ ] Delete `src/main.rs`, create `src/lib.rs` with module declarations
- [ ] Define `Coord = (i32, i32)`, `Player` enum with `opponent()` and `Display`, `HEX_DIRS` (6 neighbor offsets), `WIN_AXES` (3 axis directions) in `src/types.rs`
- [ ] Write failing tests for `hex_distance`: same-point, all 6 adjacents, along-axis, diagonal, symmetry, negative coords, boundary 8/9
- [ ] Run tests — verify they fail (`todo!()` panics)
- [ ] Implement `hex_distance(a, b) -> i32` as `max(|dq|, |dr|, |dq+dr|)`
- [ ] Run tests — all 8 pass
- [ ] Commit

---

## Task 2: Turn State Machine

**Files:** Create `src/turn.rs`, modify `src/lib.rs`

- [ ] Define `TurnState` enum: `P1Turn { moves_left: u8 }`, `P2Turn { moves_left: u8 }`, `GameOver`
- [ ] Write failing tests for: `current_player()`, `moves_remaining()`, `advance(winner)` — covering normal flow (2→1→switch), winner ends game mid-turn, full cycle P2→P1→P2
- [ ] Run tests — verify they fail
- [ ] Implement methods. `advance()` must use exhaustive match arms (no `_ =>` catch-all) — `GameOver` returns itself, invalid `moves_left` values hit `unreachable!()`
- [ ] Run tests — all 12 pass
- [ ] Commit

---

## Task 3: Board — Stone Storage and Placement

**Files:** Create `src/board.rs`, modify `src/lib.rs`

- [ ] Define `Board` struct wrapping `HashMap<Coord, Player>`, derive `Clone`
- [ ] Define `PlaceError::CellOccupied` error type
- [ ] Write failing tests for: `new()` has P1 at origin, `place()` on empty succeeds, `place()` on occupied fails, `get()` on empty returns None, `stones()` returns all, `clone()` is independent
- [ ] Run tests — verify they fail
- [ ] Implement: `new()` inserts P1 at (0,0), `place()` checks occupancy then inserts, `get()`, `stones()`, `stone_count()`
- [ ] Run tests — all 6 pass
- [ ] Commit

---

## Task 4: Win Detection

**Files:** Create `src/win.rs`, modify `src/lib.rs`

- [ ] Write failing tests for `count_consecutive(board, coord, dir, player)`: empty, 3-in-a-row, stops at opponent, stops at empty gap
- [ ] Write failing tests for `check_win(board, coord, player, win_length)`: no win on single stone, win-6 along each of the 3 axes, 5-is-not-enough, 7-also-wins, wrong-player-no-win, win-4 (curriculum), 3-not-enough-for-4
- [ ] Run tests — verify they fail
- [ ] Implement: `count_consecutive` walks in one direction counting matching stones. `check_win` scans each axis bidirectionally, returns true if `1 + forward + backward >= win_length`
- [ ] Run tests — all 12 pass
- [ ] Commit

---

## Task 5: Legal Move Generation

**Files:** Create `src/legal_moves.rs`, modify `src/lib.rs`

- [ ] Write failing tests: initial board has 216 legal moves (radius 8), all moves within radius, no occupied cells in moves, no duplicates, second stone expands set, smaller radius (4) yields 60 moves
- [ ] Run tests — verify they fail
- [ ] Implement: iterate all stones, for each scan the hex-circle within radius, collect empty cells into a `HashSet`, convert to `Vec`, sort, return
- [ ] Run tests — all 6 pass
- [ ] Commit

---

## Task 6: GameState — Public API

**Files:** Create `src/game.rs`, modify `src/lib.rs` (add re-exports)

- [ ] Define `GameConfig` with `win_length: u8`, `placement_radius: i32`, `max_moves: u32`, and `FULL_HEXO` constant (6, 8, 200)
- [ ] Define `MoveError` enum: `GameOver`, `CellOccupied`, `OutOfRange`
- [ ] Define `GameState` struct composing `Board`, `TurnState`, `GameConfig`, `move_count: u32`, `winner: Option<Player>`
- [ ] Write failing tests for:
  - Initial state: P2 to move, 2 moves remaining, not terminal, no winner, P1 at origin
  - `apply_move`: valid move decrements moves, two moves switch player, occupied fails, out-of-range fails, game-over fails
  - Win detection: P2 4-in-a-row along q-axis triggers game over; win on first of two moves ends immediately
  - Draw: game ends when `move_count` reaches `max_moves`, `winner()` is `None`
  - Legal moves: placed cell removed from legal moves
  - `placed_stones()` grows after each move
  - `clone()` is independent
  - Custom config changes legal move count
- [ ] Run tests — verify they fail
- [ ] Implement: `with_config()`, `apply_move()` (validate → place → check win → increment move_count → check draw → advance turn), delegating methods
- [ ] Run tests — all 12 pass
- [ ] Commit

---

## Task 7: Integration Tests — Edge Cases and Fuzz Testing

**Files:** Create `tests/integration.rs`

- [ ] Write integration tests:
  - `full_game_p2_wins_4_in_a_row` — complete game with known sequence, P2 wins along r-axis
  - `win_mid_turn_stops_game` — P2 wins on first of two moves, no second move allowed. **P1 must play scattered to avoid accidental 4-in-a-row on q-axis**
  - `p1_can_win` — P1 completes 4-in-a-row on diagonal axis (1,-1). P2 plays scattered
  - `no_false_positive_win_l_shape` — 5+ P2 stones in an L-shape, verify no win triggered
  - `legal_moves_all_in_range` — every legal move is within radius of at least one stone
  - `fuzz_random_games_no_panics` — 500 random 4-in-a-row games (radius 4), deterministic seeding, assert no panics
  - `fuzz_random_games_full_hexo` — 100 random full HeXO games, assert no panics
- [ ] Run tests — all 7 pass
- [ ] Commit

---

## Task 8: Performance Benchmarks

**Files:** Modify `Cargo.toml`, create `benches/engine.rs`

- [ ] Add `criterion = { version = "0.5", features = ["html_reports"] }` as dev-dependency with `[[bench]]` section
- [ ] Write benchmarks:
  - `legal_moves_50_stones` — benchmark `legal_moves()` on a 50-stone game state
  - `legal_moves_100_stones` — same at 100 stones
  - `apply_move_50_stones` — benchmark single `apply_move()` using `iter_batched`
  - Helper `build_game_state(n, config)` constructs deterministic game states via seeded pseudo-random play
- [ ] Run `cargo bench` — verify:
  - `legal_moves` benchmarks < 100μs median
  - `apply_move` benchmark < 10μs median
  - If targets missed, profile and create follow-up optimization task
- [ ] Commit

---

## Summary

| Task | What | Tests | Requirements |
|------|------|-------|--------------|
| 1 | Core types + hex distance | 8 unit | REQ-1, REQ-3 |
| 2 | Turn state machine | 12 unit | REQ-5, REQ-6 |
| 3 | Board struct | 6 unit | REQ-2, REQ-4 |
| 4 | Win detection | 12 unit | REQ-10, REQ-11, REQ-13, REQ-19 |
| 5 | Legal move generation | 6 unit | REQ-7, REQ-8, REQ-9 |
| 6 | GameState public API | 12 unit | REQ-4, REQ-12, REQ-15–20 |
| 7 | Integration + fuzz | 7 integration | REQ-12, REQ-14, REQ-20, REQ-21 |
| 8 | Performance benchmarks | 3 benchmarks | REQ-22, REQ-23 |

Total: ~63 tests + 3 benchmarks across 8 tasks. Every REQ has at least one AC; every AC maps to at least one test.
