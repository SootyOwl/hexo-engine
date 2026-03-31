# hexo-engine

A Rust game engine for **HeXO** — infinite hex tic-tac-toe played on an unbounded hexagonal grid.

Two players alternate placing stones (2 moves per turn) on a hex grid, competing to form a line of N consecutive stones. The board starts with a single P1 stone at the origin, and legal moves extend dynamically around existing stones.

## Game rules

- P1 opens with a stone at `(0, 0)`, then P2 moves first (2 moves per turn)
- Players alternate in pairs: P2(2), P1(2), P2(2), ...
- Legal moves: any empty hex within a configurable radius of any placed stone
- Win: form N-in-a-row along any of the 3 hex axes
- Draw: move limit reached with no winner

The default configuration (`FULL_HEXO`) uses 6-in-a-row, radius 8, and a 200-move limit.

## Usage

```rust
use hexo_engine::{GameState, GameConfig, Player};

// Standard 6-in-a-row game
let mut game = GameState::new();

// Or configure a smaller variant
let config = GameConfig { win_length: 4, placement_radius: 4, max_moves: 100 };
let mut game = GameState::with_config(config);

// P2 moves first
assert_eq!(game.current_player(), Some(Player::P2));

game.apply_move((1, 0)).unwrap();
game.apply_move((0, 1)).unwrap();

// Now it's P1's turn
assert_eq!(game.current_player(), Some(Player::P1));

// Query game state
let moves = game.legal_moves();         // sorted (q, r) pairs
let count = game.legal_move_count();     // no allocation
let done = game.is_terminal();
let winner = game.winner();
```

## Crates

This workspace contains two crates:

### `hexo-engine`

Core game logic — board representation, move validation, win detection, legal move generation. No dependencies beyond `std`.

- **Sparse board**: `HashMap<Coord, Player>` — grows with the game, no fixed bounds
- **Incremental legal moves**: cached `HashSet` updated on each move, no full recomputation
- **Efficient win detection**: checks only the 3 hex axes through the last-placed stone
- **Configurable**: win length, placement radius, and move limit

### `hexo-mcts`

Gumbel AlphaZero MCTS implementation with graph construction for GNN evaluation.

- **Gumbel MCTS** with Sequential Halving (Danihelka et al., 2022)
- **Batched neural network evaluation** via a pluggable `eval_fn`
- **Graph construction**: converts game states to node/edge tensors (8-dim node features, hex adjacency edges, legal/stone masks)
- **PyO3 bindings** (optional `python` feature) for integration with Python ML pipelines

## Interactive TUI

Play HeXO in the terminal:

```bash
cargo run --bin play -p hexo-engine --features tui
```

Controls: click to place, `t` for theme toggle, `r` to restart, `q` to quit.

## Coordinates

HeXO uses axial hex coordinates `(q, r)` with 6 neighbor directions. Hex distance is `max(|dq|, |dr|, |dq + dr|)`.

## License

MIT
