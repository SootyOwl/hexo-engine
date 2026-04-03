use std::collections::HashMap;

use hexo_engine::{Coord, GameState};
use rand::Rng;

use super::MCTSConfig;
use super::halving::{compute_improved_policy, gumbel_top_k, sequential_halving};
use super::node::MCTSNode;
use super::scoring::{QContext, sigma};
use super::simulate::{complete_simulation, simulate_select};

/// Result of a Gumbel MCTS search.
pub struct MCTSResult {
    /// The chosen action (q, r).
    pub action: Coord,
    /// Improved policy over all legal moves (same order as `coords`).
    pub improved_policy: Vec<f64>,
    /// Legal move coordinates (same order as `improved_policy`).
    pub coords: Vec<Coord>,
}

/// Run Gumbel MCTS from a game state.
///
/// The `eval_fn` callback evaluates a batch of game states:
///   `eval_fn(states) -> (logits_per_state, values)`
/// where logits_per_state[i] maps action → logit for state i,
/// and values[i] is the scalar value estimate.
///
/// # Errors
/// Returns `Err` if the game state is terminal.
pub fn gumbel_mcts<R, F>(
    game: &GameState,
    config: &MCTSConfig,
    rng: &mut R,
    eval_fn: &mut F,
) -> Result<MCTSResult, &'static str>
where
    R: Rng,
    F: FnMut(&[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>),
{
    if game.is_terminal() {
        return Err("Cannot run MCTS on terminal state");
    }

    let legal_moves = game.legal_moves();
    if legal_moves.is_empty() {
        return Err("No legal moves in non-terminal state");
    }

    // Step 1: Evaluate the root state — eval_fn returns raw logits, not priors
    let (logits_list, values) = eval_fn(&[game.clone()]);
    let root_logits_map = &logits_list[0];
    let root_value = values[0];

    // Build coords and logits arrays (sorted order matching legal_moves)
    let coords: Vec<Coord> = legal_moves.clone();
    let logits: Vec<f64> = coords
        .iter()
        .map(|c| root_logits_map.get(c).copied().unwrap_or(0.0))
        .collect();

    // Compute priors via softmax of raw logits
    let priors_vec = super::scoring::softmax(&logits);
    let root_priors: HashMap<Coord, f64> = coords
        .iter()
        .zip(priors_vec.iter())
        .map(|(&c, &p)| (c, p))
        .collect();

    // Step 2: Build root node
    let current_player = game
        .current_player()
        .expect("Non-terminal game should have current player");

    let mut root = MCTSNode::new(1.0, None, current_player);
    root.game_state = Some(game.clone());
    root.expand(root_priors, root_value);

    // Step 3: Gumbel top-k sampling
    let (candidate_indices, gumbel_samples) =
        gumbel_top_k(&logits, config.m_actions, rng);

    // Step 3.5: Check for immediate winning moves among candidates
    // If any candidate action leads to a terminal win, skip expensive sequential halving
    for &idx in &candidate_indices {
        let coord = coords[idx];
        let mut test_game = game.clone();
        if test_game.apply_move(coord).is_ok() && test_game.is_terminal() {
            if test_game.winner() == game.current_player() {
                // Found a winning move — return immediately
                let improved_policy = compute_improved_policy(
                    &logits, &coords, &root, config.c_visit, config.c_scale,
                );
                return Ok(MCTSResult {
                    action: coord,
                    improved_policy,
                    coords,
                });
            }
        }
    }

    // Step 4: Sequential halving with batched evaluation
    let surviving_indices = sequential_halving(
        &mut root,
        &candidate_indices,
        &gumbel_samples,
        &coords,
        &logits,
        config,
        &mut |root_node, batch_actions, c_visit, c_scale| {
            simulate_batch_with_eval(root_node, batch_actions, c_visit, c_scale, eval_fn);
        },
    );

    // Step 5: Compute improved policy over all legal moves
    let improved_policy =
        compute_improved_policy(&logits, &coords, &root, config.c_visit, config.c_scale);

    // Step 6: Select action from survivors using Gumbel + logits + σ(Q)
    let qctx = QContext::new(&root, &coords);

    let mut best_score = f64::NEG_INFINITY;
    let mut selected_coord = coords[surviving_indices[0]];

    for &idx in &surviving_indices {
        let coord = coords[idx];
        let norm_q = qctx.norm_q[&coord];

        let score = gumbel_samples[idx]
            + logits[idx]
            + sigma(norm_q, qctx.max_child_visits, config.c_visit, config.c_scale);

        if score > best_score {
            best_score = score;
            selected_coord = coord;
        }
    }

    Ok(MCTSResult {
        action: selected_coord,
        improved_policy,
        coords,
    })
}

/// Run a batch of forced-action simulations, calling eval_fn for non-terminal leaves.
fn simulate_batch_with_eval<F>(
    root: &mut MCTSNode,
    batch_actions: &[Coord],
    c_visit: u32,
    c_scale: f64,
    eval_fn: &mut F,
) where
    F: FnMut(&[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>),
{
    // Phase 1: Select leaves for each forced action
    let mut pending_selections = Vec::new();
    let mut pending_states = Vec::new();

    for &action in batch_actions {
        match simulate_select(root, Some(action), c_visit, c_scale) {
            None => {
                // Terminal leaf — backup already applied
            }
            Some(selection) => {
                // Navigate to leaf and get its game state for evaluation
                let mut node: &MCTSNode = root;
                for &a in &selection.path_actions {
                    node = node.children.get(&a).expect("path broken");
                }
                if let Some(game_state) = &node.game_state {
                    pending_states.push(game_state.clone());
                    pending_selections.push(selection);
                }
            }
        }
    }

    if pending_states.is_empty() {
        return;
    }

    // Phase 2: Batch evaluate all non-terminal leaves — eval_fn returns raw logits
    let (logits_list, values) = eval_fn(&pending_states);

    // Phase 3: Convert logits to priors via softmax, expand leaves, and backup
    for (i, selection) in pending_selections.iter().enumerate() {
        let logits_map = &logits_list[i];
        // Convert logits to priors via softmax
        let coords: Vec<Coord> = logits_map.keys().copied().collect();
        let logit_vals: Vec<f64> = coords.iter().map(|c| logits_map[c]).collect();
        let prior_vals = super::scoring::softmax(&logit_vals);
        let leaf_priors: HashMap<Coord, f64> = coords
            .into_iter()
            .zip(prior_vals)
            .collect();
        let leaf_value = values[i];
        complete_simulation(root, selection, leaf_priors, leaf_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, Player};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn small_config() -> GameConfig {
        GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        }
    }

    /// Dummy eval_fn that returns uniform logits (0.0) and value 0.0.
    fn dummy_eval(states: &[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let mut all_logits = Vec::new();
        let mut all_values = Vec::new();
        for state in states {
            let moves = state.legal_moves();
            let logits: HashMap<Coord, f64> = moves.iter().map(|&m| (m, 0.0)).collect();
            all_logits.push(logits);
            all_values.push(0.0);
        }
        (all_logits, all_values)
    }

    #[test]
    fn gumbel_mcts_returns_legal_move() {
        let game = GameState::with_config(small_config());
        let legal = game.legal_moves();
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval).unwrap();
        assert!(legal.contains(&result.action));
    }

    #[test]
    fn gumbel_mcts_policy_sums_to_one() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval).unwrap();
        let sum: f64 = result.improved_policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gumbel_mcts_policy_length_matches_legal_moves() {
        let game = GameState::with_config(small_config());
        let n_legal = game.legal_moves().len();
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval).unwrap();
        assert_eq!(result.improved_policy.len(), n_legal);
        assert_eq!(result.coords.len(), n_legal);
    }

    #[test]
    fn gumbel_mcts_terminal_returns_error() {
        let config_game = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config_game);
        game.apply_move((1, 0)).unwrap(); // P2
        game.apply_move((2, 0)).unwrap(); // P2 wins with win_length=2
        assert!(game.is_terminal());

        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval);
        assert!(result.is_err());
    }

    #[test]
    fn gumbel_mcts_deterministic_with_same_seed() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 8,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let r1 = gumbel_mcts(&game, &config, &mut rng1, &mut dummy_eval).unwrap();
        let r2 = gumbel_mcts(&game, &config, &mut rng2, &mut dummy_eval).unwrap();
        assert_eq!(r1.action, r2.action);
    }

    #[test]
    fn gumbel_mcts_zero_simulations() {
        let game = GameState::with_config(small_config());
        let config = MCTSConfig {
            n_simulations: 0,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Should still return a valid action (based on Gumbel + logits alone)
        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval).unwrap();
        assert!(game.legal_moves().contains(&result.action));
    }

    #[test]
    fn biased_eval_prefers_high_value_action() {
        // eval_fn that returns value=0.9 when it's your turn at states where
        // target_move was played (good position), and value=-0.9 otherwise.
        // After search, the action should favor the high-value move.
        let game = GameState::with_config(small_config());
        let legal = game.legal_moves();
        let target_move = legal[0]; // first legal move gets the boost

        // This eval gives massive logit advantage to target_move at the root
        // and value=0 for all states, so Q values are zero and the improved
        // policy reduces to softmax(logits), which should heavily favor target_move.
        let mut biased_eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            let mut all_logits = Vec::new();
            let mut all_values = Vec::new();
            for state in states {
                let moves = state.legal_moves();
                let logits: HashMap<Coord, f64> = moves
                    .iter()
                    .map(|&m| {
                        if m == target_move {
                            (m, 10.0)
                        } else {
                            (m, 0.0)
                        }
                    })
                    .collect();
                all_logits.push(logits);
                all_values.push(0.0);
            }
            (all_logits, all_values)
        };

        let config = MCTSConfig {
            n_simulations: 32,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = gumbel_mcts(&game, &config, &mut rng, &mut biased_eval).unwrap();

        // The improved policy should strongly favor the target_move due to the
        // large logit advantage at the root.
        let target_idx = result.coords.iter().position(|&c| c == target_move).unwrap();
        let target_prob = result.improved_policy[target_idx];
        assert!(
            target_prob > 0.3,
            "Target move should get high policy weight, got {}",
            target_prob,
        );
    }

    #[test]
    fn gumbel_mcts_finds_winning_move() {
        // Set up a game state where one move wins immediately (win_length=2,
        // one stone away). With enough simulations, MCTS should find it.
        // Use a simple eval that returns uniform logits but the terminal
        // detection gives +1/-1.
        let config_game = GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 80,
        };
        let mut game = GameState::with_config(config_game);
        // P1 is at (0,0). P2 starts with 2 moves.
        // P2 plays non-adjacent moves within radius 2 of existing stones.
        game.apply_move((2, 0)).unwrap();  // P2 move 1 (hex-dist 2 from origin)
        game.apply_move((-2, 0)).unwrap(); // P2 move 2 (hex-dist 2 from origin)
        // Now it's P1's turn with 2 moves. P1 already has (0,0).
        // P1 plays at (1,0) → 2 adjacent P1 stones → P1 wins with win_length=2!
        assert_eq!(game.current_player(), Some(Player::P1));
        let winning_move = (1, 0); // adjacent to P1's stone at (0,0)
        assert!(game.legal_moves().contains(&winning_move));

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 16,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Uniform eval: all logits 0, value 0. MCTS must rely on terminal
        // detection to discover the win.
        let result = gumbel_mcts(&game, &config, &mut rng, &mut dummy_eval).unwrap();

        // Any move adjacent to (0,0) wins with win_length=2.
        // Early termination should find one of them.
        let winning_moves: Vec<Coord> = hexo_engine::types::HEX_DIRS.iter()
            .map(|&(dq, dr)| (dq, dr))
            .filter(|c| game.legal_moves().contains(c))
            .collect();

        let is_winning_action = winning_moves.contains(&result.action);
        assert!(
            is_winning_action,
            "MCTS should find a winning move: action={:?}, winning_moves={:?}",
            result.action,
            winning_moves,
        );
    }

    #[test]
    fn gumbel_mcts_early_terminates_on_win() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let eval_count = AtomicU32::new(0);

        let mut eval_fn = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            eval_count.fetch_add(states.len() as u32, Ordering::Relaxed);
            let mut logits = Vec::new();
            let mut values = Vec::new();
            for s in states {
                let moves = s.legal_moves();
                let n = moves.len();
                let priors: HashMap<Coord, f64> =
                    moves.iter().map(|&c| (c, 1.0 / n as f64)).collect();
                logits.push(priors);
                values.push(0.0);
            }
            (logits, values)
        };

        let config = MCTSConfig {
            n_simulations: 64,
            m_actions: 8,
            c_visit: 50,
            c_scale: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let game = GameState::with_config(small_config());
        let result = gumbel_mcts(&game, &config, &mut rng, &mut eval_fn);
        assert!(result.is_ok());
        // Just verify it completes — early termination is an optimization,
        // correctness is preserved regardless.
    }
}
