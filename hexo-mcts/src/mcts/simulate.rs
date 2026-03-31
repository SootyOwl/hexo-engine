use hexo_engine::{Coord, Player};

use super::backup::compute_backup_values;
use super::node::MCTSNode;
use super::select::select_child;

/// Result of selecting a path to a leaf node during simulation.
pub struct LeafSelection {
    /// Players at each depth along the path (root to leaf).
    pub path_players: Vec<Player>,
    /// Actions taken at each step (length = path_players.len() - 1).
    pub path_actions: Vec<Coord>,
    /// Whether the leaf is terminal.
    pub is_terminal: bool,
    /// If terminal: the game outcome from the leaf's perspective.
    /// +1.0 = leaf's player won, -1.0 = lost, 0.0 = draw.
    pub terminal_value: Option<f64>,
}

/// Select a path from root to an unexpanded leaf, optionally forcing the first action.
///
/// Returns a `LeafSelection` describing the path taken. The caller is responsible
/// for evaluating non-terminal leaves (via network) and calling `apply_backup`.
pub fn select_leaf(
    root: &mut MCTSNode,
    forced_action: Option<Coord>,
    c_visit: u32,
    c_scale: f64,
) -> LeafSelection {
    let mut path_players = vec![root.current_player];
    let mut path_actions: Vec<Coord> = Vec::new();

    // Navigate to the first child
    let first_action = if let Some(action) = forced_action {
        action
    } else if root.is_expanded() {
        select_child(root, c_visit, c_scale)
    } else {
        // Root not expanded and no forced action — return root as leaf
        return LeafSelection {
            path_players,
            path_actions,
            is_terminal: root.game_state.as_ref().is_some_and(|g| g.is_terminal()),
            terminal_value: None,
        };
    };

    // Walk down the tree using the standard mutable linked-list traversal pattern.
    // select_child takes &MCTSNode (shared reborrow), returns a Copy Coord, then
    // get_or_create_child takes &mut self. Reassigning `node` releases the old borrow.
    let mut node = root;
    let mut action = first_action;

    loop {
        let child = node.get_or_create_child(action);
        path_actions.push(action);
        path_players.push(child.current_player);

        if !child.is_expanded() {
            // Reached a leaf
            let is_terminal = child
                .game_state
                .as_ref()
                .is_some_and(|g| g.is_terminal());

            let terminal_value = if is_terminal {
                let game = child.game_state.as_ref().unwrap();
                Some(match game.winner() {
                    None => 0.0, // draw
                    Some(winner) => {
                        // Value from the leaf node's perspective
                        if winner == child.current_player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                })
            } else {
                None
            };

            return LeafSelection {
                path_players,
                path_actions,
                is_terminal,
                terminal_value,
            };
        }

        action = select_child(child, c_visit, c_scale);
        node = child;
    }
}

/// Apply backup values along a path through the tree.
///
/// `path_actions` contains the actions taken at each step. The function navigates
/// from root, applying visit_count and value_sum updates at each node.
pub fn apply_backup(
    root: &mut MCTSNode,
    path_actions: &[Coord],
    backup_values: &[f64],
) {
    debug_assert_eq!(
        backup_values.len(),
        path_actions.len() + 1,
        "backup_values length must be path_actions + 1 (root)"
    );
    // First value is for the root
    root.visit_count += 1;
    root.value_sum += backup_values[0];

    let mut node: &mut MCTSNode = root;
    for (i, &action) in path_actions.iter().enumerate() {
        let child = node.children.get_mut(&action).expect("backup path broken");
        child.visit_count += 1;
        child.value_sum += backup_values[i + 1];
        node = child;
    }
}

/// Run a full simulation: select leaf, compute backup values, apply them.
///
/// For non-terminal leaves, the caller must provide the network evaluation
/// via `leaf_value` and `leaf_priors`. For terminal leaves, the value is
/// computed from the game outcome.
///
/// Returns `Some((path_actions, leaf_index))` if the leaf needs network evaluation,
/// or `None` if the leaf was terminal (backup already applied).
pub fn simulate_select(
    root: &mut MCTSNode,
    forced_action: Option<Coord>,
    c_visit: u32,
    c_scale: f64,
) -> Option<LeafSelection> {
    let selection = select_leaf(root, forced_action, c_visit, c_scale);

    if selection.is_terminal {
        let value = selection.terminal_value.unwrap_or(0.0);
        let backup_vals = compute_backup_values(&selection.path_players, value);
        apply_backup(root, &selection.path_actions, &backup_vals);
        return None;
    }

    Some(selection)
}

/// After network evaluation, expand the leaf and apply backup.
///
/// Navigate to the leaf using `path_actions`, expand it with the provided priors
/// and value, then backup.
pub fn complete_simulation(
    root: &mut MCTSNode,
    selection: &LeafSelection,
    leaf_priors: std::collections::HashMap<Coord, f64>,
    leaf_value: f64,
) {
    // Navigate to the leaf
    let mut node: &mut MCTSNode = root;
    for &action in &selection.path_actions {
        node = node.children.get_mut(&action).expect("path broken");
    }

    // Expand the leaf
    node.expand(leaf_priors, leaf_value);

    // Backup
    let backup_vals = compute_backup_values(&selection.path_players, leaf_value);
    apply_backup(root, &selection.path_actions, &backup_vals);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, GameState};
    use std::collections::HashMap;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    fn make_root() -> MCTSNode {
        let game = GameState::with_config(small_config());
        let moves = game.legal_moves();
        let mut priors = HashMap::new();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }
        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);
        root
    }

    #[test]
    fn select_leaf_unexpanded_child() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        let selection = select_leaf(&mut root, Some(action), 50, 1.0);
        assert_eq!(selection.path_players.len(), 2); // root + child
        assert_eq!(selection.path_actions.len(), 1);
        assert_eq!(selection.path_actions[0], action);
        assert!(!selection.is_terminal);
    }

    #[test]
    fn select_leaf_terminal() {
        // Create a game that's almost won, then force the winning move
        let config = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config);
        // P2 plays (1,0) — this gives P2 two stones adjacent, winning with win_length=2?
        // Actually P1 is at (0,0), P2 goes first. P2 plays (1,0) — that's only 1 P2 stone.
        // We need win_length=2 and two P2 stones in a row.
        // P2 plays at (1,0), P2 plays at (2,0) — but wait, P2 has 2 moves per turn.
        game.apply_move((1, 0)).unwrap(); // P2 move 1

        // After this, (1,0) has P2. P1 is at (0,0). P2 needs one more adjacent.
        // If P2 plays at (2,0), that's 2 P2 stones in a row → win with win_length=2
        let moves = game.legal_moves();
        let winning_move = (2, 0);
        assert!(moves.contains(&winning_move));

        let mut priors = HashMap::new();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }

        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);

        let selection = select_leaf(&mut root, Some(winning_move), 50, 1.0);
        assert!(selection.is_terminal);
        assert!(selection.terminal_value.is_some());
    }

    #[test]
    fn apply_backup_updates_counts() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        // Create a child
        root.get_or_create_child(action);

        let backup_vals = vec![0.5, -0.5]; // root, child
        apply_backup(&mut root, &[action], &backup_vals);

        assert_eq!(root.visit_count, 1);
        assert!((root.value_sum - 0.5).abs() < 1e-10);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, 1);
        assert!((child.value_sum - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn simulate_select_returns_none_for_terminal() {
        let config = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config);
        game.apply_move((1, 0)).unwrap(); // P2 move 1

        let moves = game.legal_moves();
        let mut priors = HashMap::new();
        for &m in &moves {
            priors.insert(m, 1.0 / moves.len() as f64);
        }

        let mut root = MCTSNode::new(1.0, None, Player::P2);
        root.game_state = Some(game);
        root.expand(priors, 0.0);

        let result = simulate_select(&mut root, Some((2, 0)), 50, 1.0);
        assert!(result.is_none()); // terminal → backup already done
        assert_eq!(root.visit_count, 1); // root was backed up
    }

    #[test]
    fn simulate_select_returns_some_for_non_terminal() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();

        let result = simulate_select(&mut root, Some(moves[0]), 50, 1.0);
        assert!(result.is_some());
        assert_eq!(root.visit_count, 0); // not yet backed up
    }

    #[test]
    fn complete_simulation_expands_and_backs_up() {
        let mut root = make_root();
        let moves: Vec<Coord> = root.child_priors.keys().copied().collect();
        let action = moves[0];

        let selection = simulate_select(&mut root, Some(action), 50, 1.0).unwrap();

        // Simulate network eval: provide priors and value for the leaf
        let leaf_priors = HashMap::from([((0, 0), 0.5), ((1, 1), 0.5)]);
        complete_simulation(&mut root, &selection, leaf_priors, 0.3);

        assert_eq!(root.visit_count, 1);
        let child = root.children.get(&action).unwrap();
        assert_eq!(child.visit_count, 1);
        assert!(child.is_expanded());
    }
}
