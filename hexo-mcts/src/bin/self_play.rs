//! Standalone self-play binary using TorchScript model inference.
//!
//! Supports two modes:
//!   1. Batch mode (default): play N games, write output, exit.
//!   2. Continuous mode (--continuous): play games forever, writing
//!      completed games to output dir. Reloads model when it changes.
//!
//! Each thread loads its own model instance to avoid libtorch contention.
//! In continuous mode, games are streamed as JSONL to a shared output file.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::SystemTime;

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use hexo_rs::axis_graph::game_to_axis_graph_raw;
use hexo_rs::graph::game_to_graph_raw;
use hexo_rs::inference::{GraphType, TorchModel};
use hexo_rs::mcts::MCTSConfig;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Per-position data collected during self-play.
struct PositionData {
    /// Improved policy from MCTS (one float per legal move).
    policy: Vec<f64>,
    /// Current player at this position ("P1" or "P2").
    player: &'static str,
    /// Pre-built graph data (hex or axis) serialized as JSON.
    graph_json: serde_json::Value,
}

/// Completed game data.
struct GameResult {
    positions: Vec<PositionData>,
    winner: &'static str,
}

/// Serialize graph data to a JSON dict matching Python's `_raw_to_data` expectations.
fn graph_data_to_json(g: &hexo_rs::graph::GraphData) -> serde_json::Value {
    serde_json::json!({
        "features": g.features,
        "edge_src": g.edge_src,
        "edge_dst": g.edge_dst,
        "legal_mask": g.legal_mask,
        "stone_mask": g.stone_mask,
        "coords": g.coords,
        "num_nodes": g.num_nodes,
    })
}

/// Serialize axis graph data to a JSON dict matching Python's `_raw_to_axis_data` expectations.
fn axis_graph_data_to_json(g: &hexo_rs::axis_graph::AxisGraphData) -> serde_json::Value {
    serde_json::json!({
        "features": g.features,
        "edge_src": g.edge_src,
        "edge_dst": g.edge_dst,
        "edge_attr": g.edge_attr,
        "legal_mask": g.legal_mask,
        "stone_mask": g.stone_mask,
        "coords": g.coords,
        "num_nodes": g.num_nodes,
    })
}

/// Play one complete game. Returns None if MCTS fails (e.g. model error).
fn play_one_game(
    model: &TorchModel,
    game_config: GameConfig,
    mcts_config: &MCTSConfig,
    exploration: usize,
    graph_type: GraphType,
    rng: &mut ChaCha8Rng,
) -> Option<GameResult> {
    use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;

    let mut game = GameState::with_config(game_config);
    let mut positions = Vec::new();
    let mut move_count = 0;

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        model.evaluate(states)
    };

    while !game.is_terminal() {
        let current_player = match game.current_player() {
            Some(hexo_engine::types::Player::P1) => "P1",
            Some(hexo_engine::types::Player::P2) => "P2",
            None => "?",
        };

        // Build graph from current position BEFORE making the move
        let graph_json = match graph_type {
            GraphType::Axis => axis_graph_data_to_json(&game_to_axis_graph_raw(&game)),
            GraphType::Hex => graph_data_to_json(&game_to_graph_raw(&game)),
        };

        let result = match gumbel_mcts(&game, mcts_config, rng, &mut eval) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("MCTS failed: {e}, dropping game");
                return None;
            }
        };

        let action = if move_count < exploration {
            let policy = &result.improved_policy;
            let total: f64 = policy.iter().sum();
            if total > 0.0 {
                let r: f64 = rng.random::<f64>() * total;
                let mut cumsum = 0.0;
                let mut chosen = 0;
                for (i, &p) in policy.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= r { chosen = i; break; }
                }
                result.coords[chosen]
            } else {
                result.action
            }
        } else {
            result.action
        };

        positions.push(PositionData {
            policy: result.improved_policy,
            player: current_player,
            graph_json,
        });

        if let Err(e) = game.apply_move(action) {
            eprintln!("Invalid move from MCTS: {e:?}, dropping game");
            return None;
        }
        move_count += 1;
    }

    let winner = match game.winner() {
        Some(hexo_engine::types::Player::P1) => "P1",
        Some(hexo_engine::types::Player::P2) => "P2",
        None => "draw",
    };

    Some(GameResult { positions, winner })
}

/// Serialize one game to JSON: pre-built graph data with value targets.
fn game_to_json(result: &GameResult) -> serde_json::Value {
    let examples: Vec<serde_json::Value> = result.positions
        .iter()
        .map(|pos| {
            let value: f64 = if result.winner == "draw" {
                0.0
            } else if result.winner == pos.player {
                1.0
            } else {
                -1.0
            };
            let mut ex = pos.graph_json.clone();
            ex.as_object_mut().unwrap().insert("policy".to_string(), serde_json::json!(pos.policy));
            ex.as_object_mut().unwrap().insert("value".to_string(), serde_json::json!(value));
            ex
        })
        .collect();
    serde_json::json!({
        "length": result.positions.len(),
        "examples": examples,
    })
}

fn file_mtime(path: &str) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut n_games: usize = 15;
    let mut device_str = "cpu";
    let mut win_length: u8 = 6;
    let mut radius: i32 = 8;
    let mut max_moves: u32 = 200;
    let mut n_sims: u32 = 8;
    let mut m_actions: usize = 8;
    let mut exploration: usize = 16;
    let mut output_path = "trajectories.json".to_string();
    let mut output_dir: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut n_threads: usize = 0;
    let mut omp_threads: usize = 0;
    let mut continuous = false;
    let mut graph_type_str = "hex".to_string();
    let mut jsonl_filename = "games.jsonl".to_string();

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
            "--output-dir" => { output_dir = Some(args[i + 1].clone()); i += 2; }
            "--seed" => { seed = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--threads" => { n_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--omp" => { omp_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--continuous" => { continuous = true; i += 1; }
            "--graph-type" => { graph_type_str = args[i + 1].clone(); i += 2; }
            "--jsonl-filename" => { jsonl_filename = args[i + 1].clone(); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    if model_path.is_empty() {
        eprintln!("Usage: self_play --model <path.pt> [--continuous] [options]");
        std::process::exit(1);
    }

    // Auto-detect thread counts
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    if omp_threads == 0 {
        omp_threads = 1; // 1 OMP thread per game thread for max parallelism
    }
    if n_threads == 0 {
        n_threads = (num_cpus / omp_threads).saturating_sub(1).max(1);
    }

    // SAFETY: called before spawning threads.
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", omp_threads.to_string());
        std::env::set_var("MKL_NUM_THREADS", omp_threads.to_string());
        std::env::set_var("OMP_PROC_BIND", "CLOSE");
    }
    // Disable libtorch's internal thread pools so our game threads
    // are the only source of parallelism.
    tch::set_num_threads(omp_threads as i32);   // intraop pool
    tch::set_num_interop_threads(1);             // interop pool

    let tch_device = match device_str {
        "cuda" => tch::Device::Cuda(0),
        _ => tch::Device::Cpu,
    };

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            running.store(false, Ordering::Relaxed);
        }).expect("Failed to set Ctrl+C handler");
    }

    let graph_type = match graph_type_str.as_str() {
        "axis" => GraphType::Axis,
        _ => GraphType::Hex,
    };

    let game_config = GameConfig { win_length, placement_radius: radius, max_moves };
    let mcts_config = Arc::new(MCTSConfig {
        n_simulations: n_sims,
        m_actions,
        c_visit: 50,
        c_scale: 1.0,
    });

    if !continuous {
        // --- Batch mode: each thread loads own model, plays games, collect results ---
        eprintln!("Loading model from {}...", model_path);
        eprintln!(
            "Playing {} games (sims={}, m={}, explore={}, threads={}, omp={})...",
            n_games, n_sims, m_actions, exploration, n_threads, omp_threads,
        );
        let start = std::time::Instant::now();

        let all_games: Vec<_> = std::thread::scope(|s| {
            let games_per_thread = distribute(n_games, n_threads);
            let handles: Vec<_> = games_per_thread
                .into_iter()
                .enumerate()
                .map(|(ti, count)| {
                    let model_path = &model_path;
                    let mcts_config = &mcts_config;
                    s.spawn(move || {
                        let model = match TorchModel::load_with_graph_type(model_path, tch_device, graph_type) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("Thread {ti}: failed to load model: {e}");
                                return vec![];
                            }
                        };
                        let mut rng = make_rng(seed, ti as u64, 0);
                        let mut results = Vec::with_capacity(count);
                        for _ in 0..count {
                            if let Some(game) = play_one_game(
                                &model, game_config, mcts_config, exploration, graph_type, &mut rng,
                            ) {
                                results.push(game);
                            }
                        }
                        results
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });

        let elapsed = start.elapsed();
        let total_moves: usize = all_games.iter().map(|g| g.positions.len()).sum();

        let json: Vec<serde_json::Value> = all_games.iter()
            .map(|g| game_to_json(g))
            .collect();
        let tmp_path = format!("{}.tmp", output_path);
        fs::write(&tmp_path, serde_json::to_string(&json).unwrap())
            .and_then(|_| fs::rename(&tmp_path, &output_path))
            .expect("Failed to write output");

        eprintln!(
            "Done: {} games, {} moves, {:.1}s ({:.2}s/game, {:.1} games/s)",
            n_games, total_moves, elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / n_games as f64,
            n_games as f64 / elapsed.as_secs_f64(),
        );
    } else {
        // --- Continuous mode: threads play games, append JSONL to shared file ---
        let dir = output_dir.unwrap_or_else(|| "self_play_output".to_string());
        fs::create_dir_all(&dir).expect("Failed to create output directory");

        let jsonl_path = format!("{}/{}", dir, jsonl_filename);
        let jsonl_file = fs::File::create(&jsonl_path)
            .expect("Failed to create games.jsonl");
        let shared_writer = Arc::new(Mutex::new(BufWriter::new(jsonl_file)));

        eprintln!(
            "Continuous self-play: threads={}, omp={}, output={}",
            n_threads, omp_threads, jsonl_path,
        );

        let game_counter = Arc::new(AtomicU64::new(0));

        std::thread::scope(|s| {
            for ti in 0..n_threads {
                let running = running.clone();
                let game_counter = game_counter.clone();
                let shared_writer = shared_writer.clone();
                let model_path = &model_path;
                let mcts_config = &mcts_config;

                s.spawn(move || {
                    // Each thread loads its own model — no contention
                    let mut model = match TorchModel::load_with_graph_type(model_path, tch_device, graph_type) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("Thread {ti}: failed to load model: {e}");
                            return;
                        }
                    };
                    let mut model_mtime = file_mtime(model_path);
                    let mut rng = make_rng(seed, ti as u64, 0);

                    while running.load(Ordering::Relaxed) {
                        // Play one game (skip failures)
                        if let Some(result) = play_one_game(
                            &model, game_config, mcts_config, exploration, graph_type, &mut rng,
                        ) {
                            let json_val = game_to_json(&result);
                            if let Ok(line) = serde_json::to_string(&json_val) {
                                let mut writer = shared_writer.lock().unwrap();
                                if let Err(e) = writeln!(writer, "{}", line)
                                    .and_then(|_| writer.flush())
                                {
                                    eprintln!("Thread {ti}: failed to write game: {e}");
                                }
                            }
                            let count = game_counter.fetch_add(1, Ordering::Relaxed) + 1;
                            if ti == 0 && count % 10 == 0 {
                                eprintln!("{} games written", count);
                            }
                        }

                        // Check for model update (every game, cheap stat call)
                        let current_mtime = file_mtime(model_path);
                        if current_mtime != model_mtime {
                            match TorchModel::load_with_graph_type(model_path, tch_device, graph_type) {
                                Ok(new_model) => {
                                    model = new_model;
                                    model_mtime = current_mtime;
                                    if ti == 0 {
                                        eprintln!("Model reloaded.");
                                    }
                                }
                                Err(e) => {
                                    if ti == 0 {
                                        eprintln!("Model reload failed (will retry): {e}");
                                    }
                                    // Keep old model, try again next game
                                }
                            }
                        }
                    }
                });
            }
        });

        let total_games = game_counter.load(Ordering::Relaxed);
        eprintln!("Stopped after {} games.", total_games);
    }
}

fn distribute(total: usize, n: usize) -> Vec<usize> {
    let base = total / n;
    let remainder = total % n;
    (0..n).map(|i| base + if i < remainder { 1 } else { 0 }).collect()
}

fn make_rng(seed: Option<u64>, thread_idx: u64, batch_idx: u64) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s.wrapping_add(thread_idx * 1000 + batch_idx)),
        None => ChaCha8Rng::from_os_rng(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_graph_json_has_expected_keys() {
        let game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 50,
        });
        let g = game_to_graph_raw(&game);
        let json = graph_data_to_json(&g);
        let obj = json.as_object().unwrap();
        assert!(obj.contains_key("features"));
        assert!(obj.contains_key("edge_src"));
        assert!(obj.contains_key("edge_dst"));
        assert!(obj.contains_key("legal_mask"));
        assert!(obj.contains_key("stone_mask"));
        assert!(obj.contains_key("coords"));
        assert!(obj.contains_key("num_nodes"));
        // Should NOT have edge_attr (hex graph)
        assert!(!obj.contains_key("edge_attr"));

        let n = obj["num_nodes"].as_u64().unwrap() as usize;
        assert_eq!(obj["features"].as_array().unwrap().len(), n * 8);
        assert_eq!(obj["legal_mask"].as_array().unwrap().len(), n);
        assert_eq!(obj["stone_mask"].as_array().unwrap().len(), n);
        assert_eq!(obj["coords"].as_array().unwrap().len(), n * 2);
    }

    #[test]
    fn axis_graph_json_has_edge_attr() {
        let game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 50,
        });
        let g = game_to_axis_graph_raw(&game);
        let json = axis_graph_data_to_json(&g);
        let obj = json.as_object().unwrap();
        assert!(obj.contains_key("edge_attr"));
        let e = obj["edge_src"].as_array().unwrap().len();
        assert_eq!(obj["edge_attr"].as_array().unwrap().len(), e * 5);
    }

    #[test]
    fn game_to_json_value_targets() {
        // Simulate a 2-position game where P1 wins
        let game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 50,
        });
        let g = game_to_graph_raw(&game);

        let result = GameResult {
            positions: vec![
                PositionData {
                    policy: vec![0.5, 0.5],
                    player: "P1",
                    graph_json: graph_data_to_json(&g),
                },
                PositionData {
                    policy: vec![0.3, 0.7],
                    player: "P2",
                    graph_json: graph_data_to_json(&g),
                },
            ],
            winner: "P1",
        };

        let json = game_to_json(&result);
        let examples = json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0]["value"].as_f64().unwrap(), 1.0);  // P1 wins, P1's turn
        assert_eq!(examples[1]["value"].as_f64().unwrap(), -1.0); // P1 wins, P2's turn
        assert_eq!(json["length"].as_u64().unwrap(), 2);
        // Each example should have graph keys + policy + value
        assert!(examples[0].as_object().unwrap().contains_key("features"));
        assert!(examples[0].as_object().unwrap().contains_key("policy"));
    }

    #[test]
    fn game_to_json_draw_value() {
        let game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 50,
        });
        let g = game_to_graph_raw(&game);

        let result = GameResult {
            positions: vec![PositionData {
                policy: vec![1.0],
                player: "P1",
                graph_json: graph_data_to_json(&g),
            }],
            winner: "draw",
        };

        let json = game_to_json(&result);
        assert_eq!(json["examples"][0]["value"].as_f64().unwrap(), 0.0);
    }
}
