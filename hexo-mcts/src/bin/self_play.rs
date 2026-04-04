//! Standalone self-play binary using TorchScript model inference.
//!
//! Supports two modes:
//!   1. Batch mode (default): play N games, write output, exit.
//!   2. Continuous mode (--continuous): play games forever, writing
//!      completed games to output dir. Reloads model when it changes.
//!
//! Uses a shared inference server thread: game threads build graphs on
//! CPU and submit them to a batching queue. The inference thread collects
//! a batch and runs a single forward pass, achieving better hardware
//! utilization (especially on GPU) than per-thread model instances.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime};

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use hexo_rs::axis_graph::game_to_axis_graph_raw;
use hexo_rs::graph::game_to_graph_raw;
use hexo_rs::inference::{GraphTensors, GraphType, TorchModel};
use hexo_rs::mcts::MCTSConfig;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Batched inference server
// ---------------------------------------------------------------------------

/// A request from a game thread to the inference server.
struct EvalRequest {
    /// Pre-built graph tensors (game threads build these on CPU).
    graphs: Vec<GraphTensors>,
    /// Channel to send results back to the requesting thread.
    response_tx: mpsc::Sender<(Vec<HashMap<Coord, f64>>, Vec<f64>)>,
}

/// Handle held by game threads to submit eval requests.
#[derive(Clone)]
struct InferenceClient {
    request_tx: mpsc::Sender<EvalRequest>,
}

impl InferenceClient {
    /// Submit graphs for evaluation and block until results arrive.
    fn evaluate(&self, graphs: Vec<GraphTensors>) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let (response_tx, response_rx) = mpsc::channel();
        self.request_tx
            .send(EvalRequest { graphs, response_tx })
            .expect("inference server gone");
        response_rx.recv().expect("inference server dropped response")
    }
}

/// Run the inference server loop. Collects requests into batches and
/// runs forward passes. Returns when all senders are dropped or `running`
/// is set to false.
fn inference_server(
    request_rx: mpsc::Receiver<EvalRequest>,
    model: &mut TorchModel,
    max_batch: usize,
    batch_timeout: Duration,
    running: &AtomicBool,
    // For continuous mode: poll for model updates
    model_path: Option<&str>,
    device: tch::Device,
    graph_type: GraphType,
) {
    let mut model_mtime = model_path.and_then(file_mtime);

    loop {
        // Block on first request (or check shutdown)
        let first = match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => req,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // Check for model reload during idle
                if let Some(path) = model_path {
                    try_reload_model(model, path, &mut model_mtime, device, graph_type);
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        // Collect more requests up to max_batch or timeout
        let mut requests = vec![first];
        let mut total_graphs: usize = requests[0].graphs.len();
        let deadline = Instant::now() + batch_timeout;

        while total_graphs < max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => {
                    total_graphs += req.graphs.len();
                    requests.push(req);
                }
                Err(_) => break,
            }
        }

        // Flatten all graphs into one batch
        let mut all_graphs = Vec::with_capacity(total_graphs);
        let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        for req in &mut requests {
            let start = all_graphs.len();
            all_graphs.extend(req.graphs.drain(..));
            slice_ranges.push((start, all_graphs.len()));
        }

        // Single forward pass
        let (all_logits, all_values) = model.forward_graphs(all_graphs);

        // Scatter results back
        for (i, req) in requests.into_iter().enumerate() {
            let (start, end) = slice_ranges[i];
            let logits = all_logits[start..end].to_vec();
            let values = all_values[start..end].to_vec();
            let _ = req.response_tx.send((logits, values));
        }

        // Check for model reload after processing a batch
        if let Some(path) = model_path {
            try_reload_model(model, path, &mut model_mtime, device, graph_type);
        }
    }
}

fn try_reload_model(
    model: &mut TorchModel,
    path: &str,
    cached_mtime: &mut Option<SystemTime>,
    device: tch::Device,
    graph_type: GraphType,
) {
    let current_mtime = file_mtime(path);
    if current_mtime != *cached_mtime {
        match TorchModel::load_with_graph_type(path, device, graph_type) {
            Ok(new_model) => {
                *model = new_model;
                *cached_mtime = current_mtime;
                eprintln!("Model reloaded.");
            }
            Err(e) => {
                eprintln!("Model reload failed (will retry): {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Game data types and serialization
// ---------------------------------------------------------------------------

/// A single training example written to the output file.
/// Contains pre-built graph data + targets. Serialized with msgpack.
#[derive(serde::Serialize)]
struct ExampleRecord {
    features: Vec<f32>,    // N×8 flat
    edge_src: Vec<i64>,
    edge_dst: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_attr: Option<Vec<f32>>,  // E×5 flat, axis only
    legal_mask: Vec<bool>,
    stone_mask: Vec<bool>,
    coords: Vec<i32>,      // N×2 flat
    num_nodes: usize,
    policy: Vec<f64>,
    value: f64,
}

/// One game's worth of examples, written as a single msgpack record.
#[derive(serde::Serialize)]
struct GameRecord {
    length: usize,
    examples: Vec<ExampleRecord>,
}

/// Per-position data collected during self-play (internal, not serialized).
struct PositionData {
    policy: Vec<f64>,
    player: &'static str,
    graph: GraphOutput,
}

/// Holds the raw graph data for serialization (avoids cloning into JSON).
enum GraphOutput {
    Hex(hexo_rs::graph::GraphData),
    Axis(hexo_rs::axis_graph::AxisGraphData),
}

/// Completed game data.
struct GameResult {
    positions: Vec<PositionData>,
    winner: &'static str,
}

impl GameResult {
    /// Convert to a serializable GameRecord, computing value targets.
    fn to_record(&self) -> GameRecord {
        let examples = self.positions.iter().map(|pos| {
            let value = if self.winner == "draw" {
                0.0
            } else if self.winner == pos.player {
                1.0
            } else {
                -1.0
            };
            match &pos.graph {
                GraphOutput::Hex(g) => ExampleRecord {
                    features: g.features.clone(),
                    edge_src: g.edge_src.clone(),
                    edge_dst: g.edge_dst.clone(),
                    edge_attr: None,
                    legal_mask: g.legal_mask.clone(),
                    stone_mask: g.stone_mask.clone(),
                    coords: g.coords.clone(),
                    num_nodes: g.num_nodes,
                    policy: pos.policy.clone(),
                    value,
                },
                GraphOutput::Axis(g) => ExampleRecord {
                    features: g.features.clone(),
                    edge_src: g.edge_src.clone(),
                    edge_dst: g.edge_dst.clone(),
                    edge_attr: Some(g.edge_attr.clone()),
                    legal_mask: g.legal_mask.clone(),
                    stone_mask: g.stone_mask.clone(),
                    coords: g.coords.clone(),
                    num_nodes: g.num_nodes,
                    policy: pos.policy.clone(),
                    value,
                },
            }
        }).collect();
        GameRecord {
            length: self.positions.len(),
            examples,
        }
    }
}

/// Magic bytes identifying the start of a game record.
const RECORD_MAGIC: &[u8; 4] = b"HX01";

/// Write a game as a length-prefixed binary record.
///
/// Format: `[4B magic "HX01"][4B LE record_size][4B LE num_examples][example...]`
///
/// Each example:
/// ```text
/// [2B num_nodes][2B num_edges][2B num_legal][1B has_edge_attr][4B value f32]
/// [num_nodes*8*4B features f32]
/// [num_edges*8B edge_src i64][num_edges*8B edge_dst i64]
/// [num_edges*5*4B edge_attr f32]  -- only if has_edge_attr
/// [num_nodes*1B legal_mask u8][num_nodes*1B stone_mask u8]
/// [num_nodes*2*4B coords i32]
/// [num_legal*8B policy f64]
/// ```
fn write_game_binary<W: Write>(writer: &mut W, result: &GameResult) -> std::io::Result<()> {
    // First pass: compute total size
    let mut size = 4u32; // num_examples
    for pos in &result.positions {
        let (n, e, nl, has_ea) = example_dims(pos);
        size += 2 + 2 + 2 + 1 + 4; // header (u16×3 + u8 + f32)
        size += (n * 8 * 4) as u32; // features (f32)
        size += (e * 2 * 2) as u32; // edge_src + edge_dst (u16)
        if has_ea { size += (e * 5 * 4) as u32; } // edge_attr (f32)
        size += (n * 2) as u32; // legal_mask + stone_mask (u8)
        size += (n * 2 * 2) as u32; // coords (i16)
        size += (nl * 4) as u32; // policy (f32)
    }

    writer.write_all(RECORD_MAGIC)?;
    writer.write_all(&size.to_le_bytes())?;
    writer.write_all(&(result.positions.len() as u32).to_le_bytes())?;

    for pos in &result.positions {
        let value: f64 = if result.winner == "draw" {
            0.0
        } else if result.winner == pos.player {
            1.0
        } else {
            -1.0
        };
        write_example(writer, pos, value)?;
    }
    writer.flush()
}

fn example_dims(pos: &PositionData) -> (usize, usize, usize, bool) {
    match &pos.graph {
        GraphOutput::Hex(g) => (g.num_nodes, g.edge_src.len(), pos.policy.len(), false),
        GraphOutput::Axis(g) => (g.num_nodes, g.edge_src.len(), pos.policy.len(), true),
    }
}

fn write_example<W: Write>(writer: &mut W, pos: &PositionData, value: f64) -> std::io::Result<()> {
    let (features, edge_src, edge_dst, edge_attr, legal_mask, stone_mask, coords) = match &pos.graph {
        GraphOutput::Hex(g) => (&g.features, &g.edge_src, &g.edge_dst, None, &g.legal_mask, &g.stone_mask, &g.coords),
        GraphOutput::Axis(g) => (&g.features, &g.edge_src, &g.edge_dst, Some(&g.edge_attr), &g.legal_mask, &g.stone_mask, &g.coords),
    };

    let num_nodes = legal_mask.len();
    let num_edges = edge_src.len();
    let num_legal = pos.policy.len();
    let has_ea = edge_attr.is_some();

    // Header
    writer.write_all(&(num_nodes as u16).to_le_bytes())?;
    writer.write_all(&(num_edges as u16).to_le_bytes())?;
    writer.write_all(&(num_legal as u16).to_le_bytes())?;
    writer.write_all(&[has_ea as u8])?;
    writer.write_all(&(value as f32).to_le_bytes())?;

    // Features (f32 LE)
    for &f in features { writer.write_all(&f.to_le_bytes())?; }
    // Edge src/dst (u16 LE)
    for &s in edge_src { writer.write_all(&(s as u16).to_le_bytes())?; }
    for &d in edge_dst { writer.write_all(&(d as u16).to_le_bytes())?; }
    // Edge attr (f32 LE, optional)
    if let Some(ea) = edge_attr {
        for &a in ea { writer.write_all(&a.to_le_bytes())?; }
    }
    // Masks (u8)
    for &b in legal_mask { writer.write_all(&[b as u8])?; }
    for &b in stone_mask { writer.write_all(&[b as u8])?; }
    // Coords (i16 LE)
    for &c in coords { writer.write_all(&(c as i16).to_le_bytes())?; }
    // Policy (f32 LE)
    for &p in &pos.policy { writer.write_all(&(p as f32).to_le_bytes())?; }

    Ok(())
}

/// Serialize a game to JSON (for batch mode / backwards compat).
fn game_to_json(result: &GameResult) -> serde_json::Value {
    let record = result.to_record();
    serde_json::to_value(&record).expect("serialization failed")
}

// ---------------------------------------------------------------------------
// Game playing
// ---------------------------------------------------------------------------

/// Build graph data and inference tensors for a position.
/// Returns (graph_output_for_serialization, tensors_for_inference).
fn build_position_graph(
    game: &GameState,
    graph_type: GraphType,
) -> (GraphOutput, GraphTensors) {
    match graph_type {
        GraphType::Axis => {
            let g = game_to_axis_graph_raw(game);
            let t = GraphTensors::from_axis(&g);
            (GraphOutput::Axis(g), t)
        }
        GraphType::Hex => {
            let g = game_to_graph_raw(game);
            let t = GraphTensors::from_hex(&g);
            (GraphOutput::Hex(g), t)
        }
    }
}

/// Play one complete game using batched inference via the client.
fn play_one_game(
    client: &InferenceClient,
    game_config: GameConfig,
    mcts_config: &MCTSConfig,
    exploration: &AtomicUsize,
    graph_type: GraphType,
    rng: &mut ChaCha8Rng,
) -> Option<GameResult> {
    use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;

    let mut game = GameState::with_config(game_config);
    let mut positions = Vec::new();
    let mut move_count = 0;

    while !game.is_terminal() {
        let current_player = match game.current_player() {
            Some(hexo_engine::types::Player::P1) => "P1",
            Some(hexo_engine::types::Player::P2) => "P2",
            None => "?",
        };

        // Build graph ONCE for this position — reuse for both output and root eval
        let (graph_output, root_tensors) = build_position_graph(&game, graph_type);

        // Eval closure: builds graphs on this thread, sends to inference server.
        // Root eval reuses the pre-built graph; leaf evals build fresh.
        let mut cached_root = Some(root_tensors);
        let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            let graphs = if let Some(gt) = cached_root.take() {
                debug_assert_eq!(states.len(), 1, "root eval should be single state");
                vec![gt]
            } else {
                // Leaf evals: build graphs on this game thread (CPU work)
                states.iter().map(|s| {
                    match graph_type {
                        GraphType::Axis => GraphTensors::from(game_to_axis_graph_raw(s)),
                        GraphType::Hex => GraphTensors::from(game_to_graph_raw(s)),
                    }
                }).collect()
            };
            client.evaluate(graphs)
        };

        let result = match gumbel_mcts(&game, mcts_config, rng, &mut eval) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("MCTS failed: {e}, dropping game");
                return None;
            }
        };

        let action = if move_count < exploration.load(Ordering::Relaxed) {
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
            graph: graph_output,
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

// ---------------------------------------------------------------------------
// Rotating file writer
// ---------------------------------------------------------------------------

/// Writes binary game records to sequentially numbered files, rotating
/// when the current file exceeds `max_bytes`. Files are named
/// `{stem}_{seq:04}.{ext}` (e.g. `games_0001.bin`).
struct RotatingWriter {
    dir: String,
    stem: String,
    ext: String,
    max_bytes: u64,
    seq: u32,
    writer: BufWriter<fs::File>,
    bytes_written: u64,
}

impl RotatingWriter {
    fn new(dir: &str, filename: &str, max_bytes: u64) -> Self {
        let (stem, ext) = match filename.rsplit_once('.') {
            Some((s, e)) => (s.to_string(), e.to_string()),
            None => (filename.to_string(), "bin".to_string()),
        };
        let mut rw = RotatingWriter {
            dir: dir.to_string(),
            stem,
            ext,
            max_bytes,
            seq: 0,
            writer: BufWriter::new(fs::File::create("/dev/null").unwrap()),
            bytes_written: 0,
        };
        rw.rotate();
        rw
    }

    fn current_path(&self) -> String {
        format!("{}/{}_{:04}.{}", self.dir, self.stem, self.seq, self.ext)
    }

    fn rotate(&mut self) {
        self.seq += 1;
        let path = self.current_path();
        self.writer = BufWriter::new(
            fs::File::create(&path).expect("Failed to create output file"),
        );
        self.bytes_written = 0;
        eprintln!("Writing to {}", path);
    }

    fn write_game(&mut self, result: &GameResult) -> std::io::Result<()> {
        if self.bytes_written >= self.max_bytes {
            self.rotate();
        }
        let pos_before = self.bytes_written;
        write_game_binary(&mut self.writer, result)?;
        // Estimate bytes written (flush ensures data hits OS)
        self.writer.flush()?;
        // Use the file position to track actual bytes
        self.bytes_written = self.writer.get_ref().metadata()
            .map(|m| m.len())
            .unwrap_or(pos_before + 1);
        Ok(())
    }
}

fn file_mtime(path: &str) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
    let mut exploration_fraction: Option<f64> = None;
    let mut output_path = "trajectories.json".to_string();
    let mut output_dir: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut n_threads: usize = 0;
    let mut omp_threads: usize = 0;
    let mut continuous = false;
    let mut graph_type_str = "hex".to_string();
    let mut output_filename = "games.bin".to_string();
    let mut max_batch: usize = 0; // 0 = auto
    let mut batch_timeout_ms: u64 = 0; // 0 = auto
    let mut max_file_mb: u64 = 2048;

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
            "--exploration-fraction" => { exploration_fraction = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--output-dir" => { output_dir = Some(args[i + 1].clone()); i += 2; }
            "--seed" => { seed = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--threads" => { n_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--omp" => { omp_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--continuous" => { continuous = true; i += 1; }
            "--graph-type" => { graph_type_str = args[i + 1].clone(); i += 2; }
            "--jsonl-filename" | "--output-filename" => { output_filename = args[i + 1].clone(); i += 2; }
            "--max-batch" => { max_batch = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-timeout-ms" => { batch_timeout_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--max-file-mb" => { max_file_mb = args[i + 1].parse().unwrap(); i += 2; }
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
        omp_threads = 1;
    }
    if n_threads == 0 {
        n_threads = (num_cpus / omp_threads).saturating_sub(1).max(1);
    }
    if max_batch == 0 {
        max_batch = n_threads * 2; // reasonable default
    }

    // SAFETY: called before spawning threads.
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", omp_threads.to_string());
        std::env::set_var("MKL_NUM_THREADS", omp_threads.to_string());
        std::env::set_var("OMP_PROC_BIND", "CLOSE");
    }
    tch::set_num_threads(omp_threads as i32);
    tch::set_num_interop_threads(1);

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

    // Load model once (shared via inference server)
    eprintln!("Loading model from {}...", model_path);
    let mut model = match TorchModel::load_with_graph_type(&model_path, tch_device, graph_type) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let batch_timeout = if batch_timeout_ms > 0 {
        Duration::from_millis(batch_timeout_ms)
    } else if tch_device == tch::Device::Cpu {
        Duration::from_millis(5)
    } else {
        Duration::from_millis(1)
    };

    let exploration_atomic = Arc::new(AtomicUsize::new(exploration));
    // EMA of game length stored as f64 bits in AtomicU64 (lock-free, all threads update)
    let ema_bits = Arc::new(AtomicU64::new(0u64)); // 0 bits = 0.0f64

    if !continuous {
        // --- Batch mode ---
        eprintln!(
            "Playing {} games (sims={}, m={}, explore={}, threads={}, omp={}, max_batch={})...",
            n_games, n_sims, m_actions, exploration, n_threads, omp_threads, max_batch,
        );
        let start = Instant::now();

        // Create inference channel
        let (request_tx, request_rx) = mpsc::channel::<EvalRequest>();
        let client = InferenceClient { request_tx };

        let all_games: Vec<GameResult> = std::thread::scope(|s| {
            // Spawn inference server
            let running_ref = &running;
            let model_ref = &mut model;
            let server_handle = s.spawn(move || {
                inference_server(
                    request_rx, model_ref, max_batch, batch_timeout,
                    running_ref, None, tch_device, graph_type,
                );
            });

            // Spawn game threads
            let games_per_thread = distribute(n_games, n_threads);
            let handles: Vec<_> = games_per_thread
                .into_iter()
                .enumerate()
                .map(|(ti, count)| {
                    let client = client.clone();
                    let mcts_config = &mcts_config;
                    let exploration_ref = &exploration_atomic;
                    s.spawn(move || {
                        let mut rng = make_rng(seed, ti as u64, 0);
                        let mut results = Vec::with_capacity(count);
                        for _ in 0..count {
                            if let Some(game) = play_one_game(
                                &client, game_config, mcts_config,
                                exploration_ref, graph_type, &mut rng,
                            ) {
                                results.push(game);
                            }
                        }
                        results
                    })
                })
                .collect();

            // Collect game results
            let results: Vec<GameResult> = handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect();

            // Drop the client so the inference server sees disconnect
            drop(client);
            server_handle.join().unwrap();

            results
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
        // --- Continuous mode ---
        let dir = output_dir.unwrap_or_else(|| "self_play_output".to_string());
        fs::create_dir_all(&dir).expect("Failed to create output directory");

        let rotating_writer = Arc::new(Mutex::new(
            RotatingWriter::new(&dir, &output_filename, max_file_mb * 1024 * 1024),
        ));

        if let Some(frac) = exploration_fraction {
            eprintln!(
                "Continuous self-play: threads={}, omp={}, max_batch={}, output={}/{}, exploration=adaptive({:.0}%)",
                n_threads, omp_threads, max_batch, dir, output_filename, frac * 100.0,
            );
        } else {
            eprintln!(
                "Continuous self-play: threads={}, omp={}, max_batch={}, output={}/{}, exploration={}",
                n_threads, omp_threads, max_batch, dir, output_filename, exploration,
            );
        }

        let game_counter = Arc::new(AtomicU64::new(0));

        // Create inference channel
        let (request_tx, request_rx) = mpsc::channel::<EvalRequest>();
        let client = InferenceClient { request_tx };

        std::thread::scope(|s| {
            // Spawn inference server (with model reloading)
            let running_ref = &running;
            let model_ref = &mut model;
            let model_path_ref = model_path.as_str();
            s.spawn(move || {
                inference_server(
                    request_rx, model_ref, max_batch, batch_timeout,
                    running_ref, Some(model_path_ref), tch_device, graph_type,
                );
            });

            // Spawn game threads
            for ti in 0..n_threads {
                let running = running.clone();
                let game_counter = game_counter.clone();
                let rotating_writer = rotating_writer.clone();
                let client = client.clone();
                let mcts_config = &mcts_config;
                let exploration_ref = &exploration_atomic;
                let ema_ref = &ema_bits;

                s.spawn(move || {
                    let mut rng = make_rng(seed, ti as u64, 0);
                    let mut last_logged_hundred: u64 = 0;
                    const EMA_ALPHA: f64 = 0.05;

                    while running.load(Ordering::Relaxed) {
                        if let Some(result) = play_one_game(
                            &client, game_config, mcts_config,
                            exploration_ref, graph_type, &mut rng,
                        ) {
                            let game_len = result.positions.len() as f64;

                            {
                                let mut rw = rotating_writer.lock().unwrap();
                                if let Err(e) = rw.write_game(&result) {
                                    eprintln!("Thread {ti}: failed to write game: {e}");
                                }
                            }

                            let count = game_counter.fetch_add(1, Ordering::Relaxed) + 1;

                            // All threads: update EMA via CAS loop
                            if exploration_fraction.is_some() {
                                loop {
                                    let old_bits = ema_ref.load(Ordering::Relaxed);
                                    let old_ema = f64::from_bits(old_bits);
                                    let updated = if old_ema == 0.0 {
                                        game_len
                                    } else {
                                        EMA_ALPHA * game_len + (1.0 - EMA_ALPHA) * old_ema
                                    };
                                    if ema_ref.compare_exchange_weak(
                                        old_bits, updated.to_bits(),
                                        Ordering::Relaxed, Ordering::Relaxed,
                                    ).is_ok() {
                                        break;
                                    }
                                }
                            }

                            // Thread 0: update exploration + progress logging
                            if ti == 0 {
                                if let Some(frac) = exploration_fraction {
                                    let ema = f64::from_bits(ema_ref.load(Ordering::Relaxed));
                                    let new_explore = (ema * frac).ceil() as usize;
                                    let new_explore = new_explore.max(2);
                                    let old_explore = exploration_ref.swap(new_explore, Ordering::Relaxed);
                                    if new_explore != old_explore {
                                        eprintln!(
                                            "exploration_moves: {} -> {} (ema_len={:.1}, frac={:.0}%)",
                                            old_explore, new_explore, ema, frac * 100.0,
                                        );
                                    }
                                }

                                let hundred = count / 100;
                                if hundred > last_logged_hundred {
                                    last_logged_hundred = hundred;
                                    eprintln!("{} games written", count);
                                }
                            }
                        }
                    }
                });
            }

            // Drop client so server shuts down when game threads finish
            drop(client);
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

    fn small_game_config() -> GameConfig {
        GameConfig { win_length: 4, placement_radius: 4, max_moves: 50 }
    }

    fn make_result(graph_type: GraphType, winner: &'static str) -> GameResult {
        let game = GameState::with_config(small_game_config());
        let (graph, _tensors) = build_position_graph(&game, graph_type);
        GameResult {
            positions: vec![
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph },
            ],
            winner,
        }
    }

    #[test]
    fn hex_binary_roundtrip() {
        let result = make_result(GraphType::Hex, "P1");
        let mut buf = Vec::new();
        write_game_binary(&mut buf, &result).unwrap();
        assert!(buf.len() > 12);
        // Check magic
        assert_eq!(&buf[..4], RECORD_MAGIC);
        // Read length prefix (after magic)
        let record_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        assert_eq!(record_len, buf.len() - 8); // total - magic - length
        // Read num_examples
        let num_examples = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(num_examples, 1);
    }

    #[test]
    fn axis_msgpack_has_edge_attr() {
        let result = make_result(GraphType::Axis, "draw");
        let record = result.to_record();
        assert!(record.examples[0].edge_attr.is_some());
    }

    #[test]
    fn value_targets_win() {
        let game = GameState::with_config(small_game_config());
        let (g1, _) = build_position_graph(&game, GraphType::Hex);
        let (g2, _) = build_position_graph(&game, GraphType::Hex);
        let result = GameResult {
            positions: vec![
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph: g1 },
                PositionData { policy: vec![0.3, 0.7], player: "P2", graph: g2 },
            ],
            winner: "P1",
        };
        let record = result.to_record();
        assert_eq!(record.examples[0].value, 1.0);  // P1 wins, P1's turn
        assert_eq!(record.examples[1].value, -1.0); // P1 wins, P2's turn
        assert_eq!(record.length, 2);
    }

    #[test]
    fn value_targets_draw() {
        let result = make_result(GraphType::Hex, "draw");
        let record = result.to_record();
        assert_eq!(record.examples[0].value, 0.0);
    }

    #[test]
    fn json_output_has_expected_keys() {
        let result = make_result(GraphType::Hex, "P1");
        let json = game_to_json(&result);
        let examples = json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 1);
        assert!(examples[0]["features"].is_array());
        assert!(examples[0]["policy"].is_array());
        assert_eq!(examples[0]["value"].as_f64().unwrap(), 1.0);
    }
}
