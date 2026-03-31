//! Standalone self-play binary using TorchScript model inference.
//!
//! Plays N games using Rust-native MCTS with libtorch inference,
//! writes trajectories as a msgpack file for the Python trainer to read.
//!
//! Usage:
//!   self_play --model model.pt --games 15 --device cuda \
//!             --win-length 6 --radius 8 --max-moves 200 \
//!             --sims 8 --m-actions 8 --exploration 16 \
//!             --output trajectories.bin

use std::collections::HashMap;
use std::fs;
use std::io::Write;

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use hexo_rs::inference::TorchModel;
use hexo_rs::mcts::batched::batched_gumbel_mcts;
use hexo_rs::mcts::MCTSConfig;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Simple arg parsing
    let mut model_path = String::new();
    let mut n_games: usize = 15;
    let mut device_str = "cuda";
    let mut win_length: u8 = 6;
    let mut radius: i32 = 8;
    let mut max_moves: u32 = 200;
    let mut n_sims: u32 = 8;
    let mut m_actions: usize = 8;
    let mut exploration: usize = 16;
    let mut output_path = "trajectories.bin".to_string();
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--games" => { n_games = args[i + 1].parse().unwrap(); i += 2; }
            "--device" => { device_str = if args[i + 1] == "cuda" { "cuda" } else { "cpu" }; i += 2; }
            "--win-length" => { win_length = args[i + 1].parse().unwrap(); i += 2; }
            "--radius" => { radius = args[i + 1].parse().unwrap(); i += 2; }
            "--max-moves" => { max_moves = args[i + 1].parse().unwrap(); i += 2; }
            "--sims" => { n_sims = args[i + 1].parse().unwrap(); i += 2; }
            "--m-actions" => { m_actions = args[i + 1].parse().unwrap(); i += 2; }
            "--exploration" => { exploration = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--seed" => { seed = Some(args[i + 1].parse().unwrap()); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    if model_path.is_empty() {
        eprintln!("Usage: self_play --model <path.pt> [options]");
        std::process::exit(1);
    }

    let tch_device = match device_str {
        "cuda" => tch::Device::Cuda(0),
        _ => tch::Device::Cpu,
    };

    eprintln!("Loading model from {}...", model_path);
    let model = TorchModel::load(&model_path, tch_device, false)
        .expect("Failed to load TorchScript model");

    let game_config = GameConfig { win_length, placement_radius: radius, max_moves };
    let mcts_config = MCTSConfig {
        n_simulations: n_sims,
        m_actions,
        c_visit: 50,
        c_scale: 1.0,
    };

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        model.evaluate(states)
    };

    eprintln!("Playing {} games (sims={}, m={}, explore={})...",
              n_games, n_sims, m_actions, exploration);

    let start = std::time::Instant::now();

    // Play games in lockstep
    let mut games: Vec<GameState> = (0..n_games)
        .map(|_| GameState::with_config(game_config))
        .collect();
    // trajectory: Vec of (stones_snapshot, action, policy, current_player_str)
    // We'll serialize as: for each game, list of (action, policy), then winner
    let mut trajectories: Vec<Vec<(Vec<((i32, i32), &str)>, (i32, i32), Vec<f64>, &str)>> =
        (0..n_games).map(|_| Vec::new()).collect();
    let mut move_counts: Vec<usize> = vec![0; n_games];
    let mut active: Vec<bool> = vec![true; n_games];

    loop {
        let active_indices: Vec<usize> = (0..n_games)
            .filter(|&i| active[i] && !games[i].is_terminal())
            .collect();

        if active_indices.is_empty() {
            break;
        }

        let active_states: Vec<GameState> =
            active_indices.iter().map(|&i| games[i].clone()).collect();

        let results = batched_gumbel_mcts(&active_states, &mcts_config, &mut rng, &mut eval)
            .expect("MCTS failed");

        for (result_idx, &game_idx) in active_indices.iter().enumerate() {
            let mcts_result = &results[result_idx];

            // Snapshot placed stones for the Python side to reconstruct the graph
            let stones: Vec<((i32, i32), &str)> = games[game_idx]
                .placed_stones()
                .into_iter()
                .map(|(c, p)| (c, match p {
                    hexo_engine::types::Player::P1 => "P1",
                    hexo_engine::types::Player::P2 => "P2",
                }))
                .collect();

            let current_player = match games[game_idx].current_player() {
                Some(hexo_engine::types::Player::P1) => "P1",
                Some(hexo_engine::types::Player::P2) => "P2",
                None => "?",
            };

            let action = if move_counts[game_idx] < exploration {
                let policy = &mcts_result.improved_policy;
                let total: f64 = policy.iter().sum();
                if total > 0.0 {
                    let r: f64 = rng.random::<f64>() * total;
                    let mut cumsum = 0.0;
                    let mut chosen = 0;
                    for (i, &p) in policy.iter().enumerate() {
                        cumsum += p;
                        if cumsum >= r { chosen = i; break; }
                    }
                    mcts_result.coords[chosen]
                } else {
                    mcts_result.action
                }
            } else {
                mcts_result.action
            };

            trajectories[game_idx].push((stones, action, mcts_result.improved_policy.clone(), current_player));
            games[game_idx].apply_move(action).expect("valid action");
            move_counts[game_idx] += 1;
        }

        for i in 0..n_games {
            if active[i] && games[i].is_terminal() {
                active[i] = false;
            }
        }
    }

    let elapsed = start.elapsed();

    // Write output as JSON (simple, Python can parse easily)
    let mut output: Vec<serde_json::Value> = Vec::new();
    for (gi, traj) in trajectories.iter().enumerate() {
        let winner = match games[gi].winner() {
            Some(hexo_engine::types::Player::P1) => "P1",
            Some(hexo_engine::types::Player::P2) => "P2",
            None => "draw",
        };
        let moves: Vec<serde_json::Value> = traj.iter().map(|(_, action, policy, cp)| {
            serde_json::json!({
                "action": [action.0, action.1],
                "policy": policy,
                "player": cp,
            })
        }).collect();
        output.push(serde_json::json!({
            "winner": winner,
            "moves": moves,
            "length": traj.len(),
        }));
    }

    let json = serde_json::to_string(&output).unwrap();
    fs::write(&output_path, &json).expect("Failed to write output");

    let total_moves: usize = trajectories.iter().map(|t| t.len()).sum();
    eprintln!(
        "Done: {} games, {} moves, {:.1}s ({:.2}s/game, {:.1} moves/s)",
        n_games, total_moves, elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / n_games as f64,
        total_moves as f64 / elapsed.as_secs_f64(),
    );
}
