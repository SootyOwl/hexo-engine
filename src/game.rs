use crate::board::Board;
use crate::legal_moves::legal_moves;
use crate::turn::TurnState;
use crate::types::{Coord, Player};
use crate::win::check_win;

/// Configuration parameters for a HeXO game.
pub struct GameConfig {
    /// Number of consecutive stones needed to win.
    pub win_length: u8,
    /// Hex-distance radius within which moves are legal.
    pub placement_radius: i32,
    /// Maximum total stone placements before the game is a draw.
    pub max_moves: u32,
}

impl GameConfig {
    /// Standard HeXO configuration: 6-in-a-row, radius 8, 200 total moves.
    pub const FULL_HEXO: GameConfig = GameConfig {
        win_length: 6,
        placement_radius: 8,
        max_moves: 200,
    };
}

/// Errors returned by `GameState::apply_move`.
#[derive(Debug, PartialEq, Eq)]
pub enum MoveError {
    /// The game has already ended (win or draw).
    GameOver,
    /// The target cell is already occupied.
    CellOccupied,
    /// The target cell is outside the placement radius.
    OutOfRange,
}

/// The full game state: board, whose turn it is, move counter, and outcome.
pub struct GameState {
    board: Board,
    turn: TurnState,
    config: GameConfig,
    move_count: u32,
    winner: Option<Player>,
}

impl GameState {
    /// Creates a new game with the standard FULL_HEXO configuration.
    pub fn new() -> Self {
        Self::with_config(GameConfig::FULL_HEXO)
    }

    /// Creates a new game with a custom configuration.
    ///
    /// P1's first stone is placed at (0,0) by `Board::new()`.
    /// The game starts with P2 to move with 2 moves remaining.
    pub fn with_config(config: GameConfig) -> Self {
        GameState {
            board: Board::new(),
            turn: TurnState::P2Turn { moves_left: 2 },
            config,
            move_count: 0,
            winner: None,
        }
    }

    /// Attempts to place the current player's stone at `coord`.
    ///
    /// Steps:
    /// 1. Reject if game is already over.
    /// 2. Reject if `coord` is outside the placement radius.
    /// 3. Reject if `coord` is already occupied.
    /// 4. Increment `move_count`.
    /// 5. Check for a win.
    /// 6. Check for a draw (no win and move_count >= max_moves).
    /// 7. Advance the turn state.
    pub fn apply_move(&mut self, coord: Coord) -> Result<(), MoveError> {
        // 1. Game already over?
        if self.is_terminal() {
            return Err(MoveError::GameOver);
        }

        // 2. Within placement radius of any existing stone?
        let legal = legal_moves(&self.board, self.config.placement_radius);
        if !legal.contains(&coord) {
            // Distinguish occupied vs out-of-range.
            if self.board.get(coord).is_some() {
                return Err(MoveError::CellOccupied);
            }
            return Err(MoveError::OutOfRange);
        }

        // 3. Place stone (should succeed because legal_moves only includes empty cells).
        let player = self
            .turn
            .current_player()
            .expect("turn has current player when not terminal");
        self.board
            .place(coord, player)
            .map_err(|_| MoveError::CellOccupied)?;

        // 4. Increment move count.
        self.move_count += 1;

        // 5. Check for win.
        let won = check_win(&self.board, coord, player, self.config.win_length);
        if won {
            self.winner = Some(player);
        }

        // 6. Check for draw.
        let draw = !won && self.move_count >= self.config.max_moves;

        // 7. Advance turn.
        self.turn = self.turn.advance(won || draw);

        Ok(())
    }

    /// Returns all legal moves for the current position.
    pub fn legal_moves(&self) -> Vec<Coord> {
        if self.is_terminal() {
            return Vec::new();
        }
        legal_moves(&self.board, self.config.placement_radius)
    }

    /// Returns `true` when the game has ended (win or draw).
    pub fn is_terminal(&self) -> bool {
        self.turn == TurnState::GameOver
    }

    /// Returns the winner, or `None` if there is no winner (game ongoing or draw).
    pub fn winner(&self) -> Option<Player> {
        self.winner
    }

    /// Returns the player who should move next, or `None` if the game is over.
    pub fn current_player(&self) -> Option<Player> {
        self.turn.current_player()
    }

    /// Returns how many moves the current player has left this turn.
    /// Returns 0 when the game is over.
    pub fn moves_remaining_this_turn(&self) -> u8 {
        self.turn.moves_remaining().unwrap_or(0)
    }

    /// Returns all placed stones as `(coord, player)` pairs.
    pub fn placed_stones(&self) -> Vec<(Coord, Player)> {
        self.board
            .stones()
            .iter()
            .map(|(&coord, &player)| (coord, player))
            .collect()
    }

    /// Returns the total number of moves made so far (not counting P1's opening stone).
    pub fn move_count(&self) -> u32 {
        self.move_count
    }

    /// Returns a reference to the game configuration.
    pub fn config(&self) -> &GameConfig {
        &self.config
    }
}

impl Clone for GameState {
    fn clone(&self) -> Self {
        GameState {
            board: self.board.clone(),
            turn: self.turn,
            config: GameConfig {
                win_length: self.config.win_length,
                placement_radius: self.config.placement_radius,
                max_moves: self.config.max_moves,
            },
            move_count: self.move_count,
            winner: self.winner,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Initial state
    // ------------------------------------------------------------------

    #[test]
    fn initial_state_p2_to_move() {
        let gs = GameState::new();
        assert_eq!(gs.current_player(), Some(Player::P2));
    }

    #[test]
    fn initial_state_two_moves_remaining() {
        let gs = GameState::new();
        assert_eq!(gs.moves_remaining_this_turn(), 2);
    }

    #[test]
    fn initial_state_not_terminal() {
        let gs = GameState::new();
        assert!(!gs.is_terminal());
    }

    #[test]
    fn initial_state_no_winner() {
        let gs = GameState::new();
        assert_eq!(gs.winner(), None);
    }

    #[test]
    fn initial_state_p1_at_origin() {
        let gs = GameState::new();
        let stones = gs.placed_stones();
        assert!(
            stones.contains(&((0, 0), Player::P1)),
            "expected P1 stone at origin, got {:?}",
            stones
        );
    }

    // ------------------------------------------------------------------
    // 2. apply_move basic cases
    // ------------------------------------------------------------------

    #[test]
    fn apply_move_valid_decrements_moves_remaining() {
        let mut gs = GameState::new();
        // P2 starts with 2 moves.
        gs.apply_move((1, 0)).unwrap();
        assert_eq!(gs.moves_remaining_this_turn(), 1);
    }

    #[test]
    fn apply_move_two_moves_switches_player_to_p1() {
        let mut gs = GameState::new();
        gs.apply_move((1, 0)).unwrap();
        gs.apply_move((2, 0)).unwrap();
        assert_eq!(gs.current_player(), Some(Player::P1));
    }

    #[test]
    fn apply_move_occupied_returns_error() {
        let mut gs = GameState::new();
        // (0,0) is P1's opening stone.
        let result = gs.apply_move((0, 0));
        assert_eq!(result, Err(MoveError::CellOccupied));
    }

    #[test]
    fn apply_move_out_of_range_returns_error() {
        let mut gs = GameState::new();
        // (100, 100) is far outside radius 8.
        let result = gs.apply_move((100, 100));
        assert_eq!(result, Err(MoveError::OutOfRange));
    }

    #[test]
    fn apply_move_after_game_over_returns_error() {
        // Build a win quickly with win_length=4 to keep it short.
        let config = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs = GameState::with_config(config);
        // P2 needs 4-in-a-row. Origin P1 is at (0,0).
        // Give P1 scattered moves to avoid accidental wins.
        // Turn structure: P2(2) -> P2(1) -> P1(2) -> P1(1) -> P2(2) -> ...
        // Move 1 (P2): (1,0)
        gs.apply_move((1, 0)).unwrap();
        // Move 2 (P2): (2,0)
        gs.apply_move((2, 0)).unwrap();
        // Move 3 (P1): scattered
        gs.apply_move((0, 3)).unwrap();
        // Move 4 (P1): scattered
        gs.apply_move((0, -3)).unwrap();
        // Move 5 (P2): (3,0)
        gs.apply_move((3, 0)).unwrap();
        // Move 6 (P2): (4,0) → P2 has (1,0),(2,0),(3,0),(4,0) = 4-in-a-row → win
        gs.apply_move((4, 0)).unwrap();
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), Some(Player::P2));
        // Now applying another move should return GameOver.
        let result = gs.apply_move((5, 0));
        assert_eq!(result, Err(MoveError::GameOver));
    }

    // ------------------------------------------------------------------
    // 3. Win detection
    // ------------------------------------------------------------------

    #[test]
    fn win_p2_four_in_a_row_along_q_axis() {
        let config = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs = GameState::with_config(config);
        // P2 places: (1,0), (2,0) then P1 scattered, then P2: (3,0),(4,0) → win
        gs.apply_move((1, 0)).unwrap(); // P2 move 1
        gs.apply_move((2, 0)).unwrap(); // P2 move 2
        gs.apply_move((0, 3)).unwrap(); // P1 move 1
        gs.apply_move((0, -3)).unwrap(); // P1 move 2
        gs.apply_move((3, 0)).unwrap(); // P2 move 1
        gs.apply_move((4, 0)).unwrap(); // P2 move 2 → win!
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), Some(Player::P2));
    }

    #[test]
    fn win_on_first_of_two_moves_ends_immediately() {
        // REQ-12: if P2's first move of a turn is winning, game ends right away.
        let config = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs = GameState::with_config(config);
        // Build up P2 stones: (1,0),(2,0) in first turn.
        gs.apply_move((1, 0)).unwrap(); // P2 move 1 (turn 1)
        gs.apply_move((2, 0)).unwrap(); // P2 move 2 (turn 1)
        gs.apply_move((0, 3)).unwrap(); // P1 move 1
        gs.apply_move((0, -3)).unwrap(); // P1 move 2
        // P2 now has their second turn; first move of that turn should win.
        gs.apply_move((3, 0)).unwrap(); // P2 move 1 of turn 2 — 3 in a row, not enough
        // (3,0) gives (1,0),(2,0),(3,0) = 3, still need 4
        // Now (0,0) is P1. Let's use a fresh setup to be sure.
        // P2 move 2 of turn 2 → (4,0): 4-in-a-row
        gs.apply_move((4, 0)).unwrap();
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), Some(Player::P2));

        // For win on FIRST move of a turn, we need a 3-stone setup + win on move1.
        let config2 = GameConfig {
            win_length: 4,
            placement_radius: 8,
            max_moves: 200,
        };
        let mut gs2 = GameState::with_config(config2);
        // Get 3 P2 stones without winning (win_length=4 so 3 is safe).
        gs2.apply_move((1, 0)).unwrap(); // P2 turn1 move1
        gs2.apply_move((2, 0)).unwrap(); // P2 turn1 move2
        gs2.apply_move((0, 3)).unwrap(); // P1 turn1 move1
        gs2.apply_move((0, -3)).unwrap(); // P1 turn1 move2
        gs2.apply_move((3, 0)).unwrap(); // P2 turn2 move1 → 3 in row, no win yet
        // P2 has moves_remaining = 1; game is not terminal yet
        assert!(!gs2.is_terminal());
        assert_eq!(gs2.moves_remaining_this_turn(), 1);
        // Now the FIRST move of P2's next block needs to win on first placement.
        // We need to get P2's turn2 move2 without winning first.
        gs2.apply_move((5, 1)).unwrap(); // P2 turn2 move2 — scattered, no win
        gs2.apply_move((0, 4)).unwrap(); // P1
        gs2.apply_move((0, -4)).unwrap(); // P1
        // Now P2 has turn3; (1,0),(2,0),(3,0) exist plus (5,1).
        // Place (4,0) as P2's FIRST move of turn3 → 4-in-a-row → win on first move.
        gs2.apply_move((4, 0)).unwrap(); // P2 turn3 move1 → WIN
        assert!(gs2.is_terminal(), "game should be terminal after win on first move");
        assert_eq!(gs2.winner(), Some(Player::P2));
        // apply_move should now fail with GameOver.
        assert_eq!(gs2.apply_move((6, 0)), Err(MoveError::GameOver));
    }

    // ------------------------------------------------------------------
    // 4. Draw
    // ------------------------------------------------------------------

    #[test]
    fn draw_when_move_count_reaches_max_moves() {
        let config = GameConfig {
            win_length: 6,
            placement_radius: 8,
            max_moves: 4, // small cap so the test is fast
        };
        let mut gs = GameState::with_config(config);
        // We need 4 moves without anyone winning.
        // Use coords that cannot form a line of 6.
        gs.apply_move((1, 0)).unwrap();  // P2 move 1
        gs.apply_move((2, 0)).unwrap();  // P2 move 2
        gs.apply_move((0, 3)).unwrap();  // P1 move 1
        gs.apply_move((0, -3)).unwrap(); // P1 move 2 → move_count == 4 → draw
        assert!(gs.is_terminal());
        assert_eq!(gs.winner(), None);
    }

    // ------------------------------------------------------------------
    // 5. Legal moves
    // ------------------------------------------------------------------

    #[test]
    fn placed_cell_removed_from_legal_moves() {
        let mut gs = GameState::new();
        let before = gs.legal_moves();
        assert!(before.contains(&(1, 0)));
        gs.apply_move((1, 0)).unwrap();
        let after = gs.legal_moves();
        assert!(!after.contains(&(1, 0)));
    }

    // ------------------------------------------------------------------
    // 6. placed_stones grows
    // ------------------------------------------------------------------

    #[test]
    fn placed_stones_grows_after_each_move() {
        let mut gs = GameState::new();
        // Initially 1 stone (P1 at origin).
        assert_eq!(gs.placed_stones().len(), 1);
        gs.apply_move((1, 0)).unwrap();
        assert_eq!(gs.placed_stones().len(), 2);
        gs.apply_move((2, 0)).unwrap();
        assert_eq!(gs.placed_stones().len(), 3);
    }

    // ------------------------------------------------------------------
    // 7. clone is independent
    // ------------------------------------------------------------------

    #[test]
    fn clone_is_independent() {
        let mut original = GameState::new();
        original.apply_move((1, 0)).unwrap();

        let mut cloned = original.clone();
        cloned.apply_move((2, 0)).unwrap();

        // Original should not have the stone at (2,0).
        assert!(!original
            .placed_stones()
            .iter()
            .any(|&(c, _)| c == (2, 0)));

        // Apply a different move to original.
        original.apply_move((0, -2)).unwrap();
        assert!(!cloned
            .placed_stones()
            .iter()
            .any(|&(c, _)| c == (0, -2)));
    }

    // ------------------------------------------------------------------
    // 8. Custom config changes legal move count
    // ------------------------------------------------------------------

    #[test]
    fn custom_config_changes_legal_move_count() {
        let config = GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 200,
        };
        let gs = GameState::with_config(config);
        // Radius 4 around (0,0): 61 cells total, 1 occupied → 60 legal moves.
        assert_eq!(gs.legal_moves().len(), 60);

        let gs_full = GameState::new(); // radius 8 → 216 legal moves
        assert_eq!(gs_full.legal_moves().len(), 216);
    }
}
