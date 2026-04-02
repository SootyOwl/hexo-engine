//! Standalone self-play binary using TorchScript model inference.
//!
//! Supports two modes:
//!   1. Batch mode (default): play N games, write output, exit.
//!   2. Continuous mode (--continuous): play games forever, writing
//!      completed games to output dir. Reloads model when it changes.
//!
//! Each thread loads its own model instance to avoid libtorch contention.
//! Games are written to disk as soon as each batch completes.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::SystemTime;

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use hexo_rs::inference::TorchModel;
use hexo_rs::mcts::MCTSConfig;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Play one complete game. Returns None if MCTS fails (e.g. model error).
fn play_one_game(
    model: &TorchModel,
    game_config: GameConfig,
    mcts_config: &MCTSConfig,
    exploration: usize,
    rng: &mut ChaCha8Rng,
) -> Option<(Vec<((i32, i32), Vec<f64>, &'static str)>, &'static str)> {
    use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;

    let mut game = GameState::with_config(game_config);
    let mut trajectory = Vec::new();
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

        trajectory.push((action, result.improved_policy, current_player));
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

    Some((trajectory, winner))
}

/// Serialize one game to JSON value.
fn game_to_json(
    trajectory: &[((i32, i32), Vec<f64>, &str)],
    winner: &str,
) -> serde_json::Value {
    let moves: Vec<serde_json::Value> = trajectory
        .iter()
        .map(|(action, policy, cp)| {
            serde_json::json!({
                "action": [action.0, action.1],
                "policy": policy,
                "player": cp,
            })
        })
        .collect();
    serde_json::json!({
        "winner": winner,
        "moves": moves,
        "length": trajectory.len(),
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
                        let model = match TorchModel::load(model_path, tch_device) {
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
                                &model, game_config, mcts_config, exploration, &mut rng,
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
        let total_moves: usize = all_games.iter().map(|(t, _)| t.len()).sum();

        let json: Vec<serde_json::Value> = all_games.iter()
            .map(|(t, w)| game_to_json(t, w))
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
        // --- Continuous mode: threads play games independently, write batches ---
        let dir = output_dir.unwrap_or_else(|| "self_play_output".to_string());
        fs::create_dir_all(&dir).expect("Failed to create output directory");

        eprintln!(
            "Continuous self-play: threads={}, omp={}, batch_size={}, output={}",
            n_threads, omp_threads, n_games, dir,
        );

        let batch_counter = Arc::new(AtomicU64::new(0));

        std::thread::scope(|s| {
            for ti in 0..n_threads {
                let running = running.clone();
                let batch_counter = batch_counter.clone();
                let model_path = &model_path;
                let mcts_config = &mcts_config;
                let dir = &dir;

                s.spawn(move || {
                    // Each thread loads its own model — no contention
                    let mut model = match TorchModel::load(model_path, tch_device) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("Thread {ti}: failed to load model: {e}");
                            return;
                        }
                    };
                    let mut model_mtime = file_mtime(model_path);
                    let mut rng = make_rng(seed, ti as u64, 0);
                    let mut local_batch: Vec<serde_json::Value> = Vec::new();

                    while running.load(Ordering::Relaxed) {
                        // Play one game (skip failures)
                        if let Some((trajectory, winner)) = play_one_game(
                            &model, game_config, mcts_config, exploration, &mut rng,
                        ) {
                            local_batch.push(game_to_json(&trajectory, winner));
                        }

                        // Flush batch when we have enough games
                        if local_batch.len() >= n_games {
                            let batch_idx = batch_counter.fetch_add(1, Ordering::Relaxed);
                            let batch_path = format!("{}/batch_{:06}.json", dir, batch_idx);
                            let tmp_path = format!("{}/batch_{:06}.json.tmp", dir, batch_idx);
                            if let Ok(json) = serde_json::to_string(&local_batch) {
                                // Atomic write: temp file then rename
                                if let Err(e) = fs::write(&tmp_path, &json)
                                    .and_then(|_| fs::rename(&tmp_path, &batch_path))
                                {
                                    eprintln!("Thread {ti}: failed to write batch: {e}");
                                }
                            }
                            if ti == 0 {
                                eprintln!("Batch {}: {} games written", batch_idx, local_batch.len());
                            }
                            local_batch.clear();
                        }

                        // Check for model update (every game, cheap stat call)
                        let current_mtime = file_mtime(model_path);
                        if current_mtime != model_mtime {
                            match TorchModel::load(model_path, tch_device) {
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

                    // Flush remaining games
                    if !local_batch.is_empty() {
                        let batch_idx = batch_counter.fetch_add(1, Ordering::Relaxed);
                        let batch_path = format!("{}/batch_{:06}.json", dir, batch_idx);
                        let tmp_path = format!("{}/batch_{:06}.json.tmp", dir, batch_idx);
                        if let Ok(json) = serde_json::to_string(&local_batch) {
                            fs::write(&tmp_path, &json)
                                .and_then(|_| fs::rename(&tmp_path, &batch_path))
                                .ok();
                        }
                    }
                });
            }
        });

        let total_batches = batch_counter.load(Ordering::Relaxed);
        eprintln!("Stopped after {} batches.", total_batches);
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
