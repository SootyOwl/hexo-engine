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

use rustc_hash::FxHashMap as HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
// Request channel uses crossbeam (MPMC): the `shared` inference-dispatch mode
// clones one Receiver across N batcher workers so idle workers pull the next
// batch, avoiding the round-robin load bifurcation. Per-request response
// channels stay std mpsc (single-shot SPSC).
use crossbeam_channel::{
    Receiver as ReqReceiver, RecvTimeoutError as ReqRecvTimeoutError, Sender as ReqSender,
    unbounded as req_unbounded,
};
#[cfg(feature = "torch")]
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::{Duration, Instant, SystemTime};

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use hexo_rs::axis_graph::game_to_axis_graph_raw_opts;
use hexo_rs::graph::game_to_graph_raw_opts;
use hexo_rs::graph_tensors::{GraphTensors, GraphType};
#[cfg(feature = "torch")]
use hexo_rs::inference::TorchModel;
use hexo_rs::inference_subprocess::SubprocessModel;
use hexo_rs::mcts::MCTSConfig;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Per-move profiling for play_one_game (gated behind --profile-play / the
// PROFILE_PLAY_ENABLED static). Each game thread accumulates section-level
// timings in a thread-local and flushes one summary line at the end of its
// measurement allocation. Overhead per move when disabled = atomic load only.
// ---------------------------------------------------------------------------

static PROFILE_PLAY_ENABLED: AtomicBool = AtomicBool::new(false);

// Opening-move exploration distribution toggle (set once from
// `--exploration-distribution`). false (default) = sample ∝ improved_policy
// π' (σ-sharpened completed-Q, near one-hot at low sims); true = sample ∝
// root visit counts (paper-style, the broad Sequential-Halving staircase).
// Read once per explored move (atomic load only when disabled).
static EXPLORE_USE_VISIT_COUNTS: AtomicBool = AtomicBool::new(false);

/// Build the sampling weights for an opening-exploration move.
///
/// `use_visit_counts == false` returns the improved policy π' (the default,
/// matching prior HeXO behaviour). `true` returns the root visit counts cast
/// to f64 (the Gumbel-AZ-paper acting distribution). Both are unnormalised;
/// the caller normalises by the sum.
fn exploration_weights(
    improved_policy: &[f64],
    visit_counts: &[u32],
    use_visit_counts: bool,
) -> Vec<f64> {
    if use_visit_counts {
        visit_counts.iter().map(|&n| n as f64).collect()
    } else {
        improved_policy.to_vec()
    }
}

#[derive(Default)]
struct PlayPerfCounters {
    moves: u64,
    games: u64,
    graph_build_ns: u128,
    mcts_total_ns: u128,
    eval_wait_ns: u128,
    eval_graph_build_ns: u128,
    apply_move_ns: u128,
    iter_total_ns: u128,
}

thread_local! {
    static PLAY_PERF: std::cell::RefCell<PlayPerfCounters> =
        std::cell::RefCell::new(PlayPerfCounters::default());
}

fn play_perf_flush() {
    if !PROFILE_PLAY_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    PLAY_PERF.with(|cell| {
        let mut p = cell.borrow_mut();
        if p.moves == 0 {
            return;
        }
        let n = p.moves as f64;
        let to_us = |ns: u128| -> f64 { (ns as f64) / n / 1000.0 };
        // Pure MCTS CPU work = mcts_total − eval_wait (eval_wait is time
        // spent inside the eval closure, which gumbel_mcts called from
        // within its own loop). Pipe time inside eval_wait = eval_wait −
        // eval_graph_build.
        let mcts_cpu_us = to_us(p.mcts_total_ns.saturating_sub(p.eval_wait_ns));
        let pipe_wait_us = to_us(p.eval_wait_ns.saturating_sub(p.eval_graph_build_ns));
        eprintln!(
            "[play_perf] tid={:?} games={} moves={} per_move_us(total={:.0} \
             graph_build={:.0} mcts_total={:.0} mcts_cpu={:.0} \
             eval_wait={:.0} eval_graph_build={:.0} pipe_wait={:.0} apply={:.0})",
            std::thread::current().id(),
            p.games, p.moves,
            to_us(p.iter_total_ns),
            to_us(p.graph_build_ns),
            to_us(p.mcts_total_ns),
            mcts_cpu_us,
            to_us(p.eval_wait_ns),
            to_us(p.eval_graph_build_ns),
            pipe_wait_us,
            to_us(p.apply_move_ns),
        );
        *p = PlayPerfCounters::default();
    });
}

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

/// How eval requests are distributed across the N inference batcher workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchMode {
    /// Sender-side round-robin across N independent channels (one per worker).
    /// Balances request *count*, not load — with variable per-batch service
    /// time this is an unstable equilibrium: a transiently-slower worker
    /// accumulates a deeper backlog → larger batches → slower, and so on,
    /// bifurcating into one saturated worker + one starved tiny-batch worker.
    RoundRobin,
    /// One shared MPMC queue; every worker pulls the next available batch when
    /// it frees up. Self-balances batch sizes (no per-worker affinity) while
    /// preserving cross-worker latency hiding. Default for N>1.
    Shared,
}

/// Handle held by game threads to submit eval requests.
///
/// Holds one or more request channels. In [`DispatchMode::RoundRobin`] there is
/// one channel per worker and `evaluate()` round-robins across them via a
/// shared atomic counter. In [`DispatchMode::Shared`] there is a single channel
/// (`request_txs.len() == 1`) feeding a queue that all workers pull from, so
/// the counter trivially always selects index 0.
///
/// At N=1 both modes — and the historical single-Sender client — are
/// observationally identical: one channel, every request routed to it.
#[derive(Clone)]
struct InferenceClient {
    request_txs: Arc<Vec<ReqSender<EvalRequest>>>,
    counter: Arc<AtomicUsize>,
}

impl InferenceClient {
    /// Build a client over `request_txs.len()` worker channels. Panics if
    /// the vector is empty (callers always wire at least one worker).
    fn new(request_txs: Vec<ReqSender<EvalRequest>>) -> Self {
        assert!(
            !request_txs.is_empty(),
            "InferenceClient requires at least one worker channel"
        );
        Self {
            request_txs: Arc::new(request_txs),
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Submit graphs for evaluation and block until results arrive.
    /// Routes the request to one worker chosen by round-robin.
    fn evaluate(&self, graphs: Vec<GraphTensors>) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let (response_tx, response_rx) = mpsc::channel();
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.request_txs.len();
        self.request_txs[idx]
            .send(EvalRequest { graphs, response_tx })
            .expect("inference server gone");
        response_rx.recv().expect("inference server dropped response")
    }
}

/// Build the request channels for `n_workers` inference batchers under the
/// chosen [`DispatchMode`]. Returns `(client_txs, worker_rxs)` where
/// `worker_rxs.len() == n_workers` (one receiver per batcher thread):
///
/// - [`DispatchMode::RoundRobin`]: `n_workers` independent channels; sender `i`
///   goes to the client, receiver `i` to worker `i`.
/// - [`DispatchMode::Shared`]: one channel; its single sender goes to the
///   client and `n_workers` clones of the receiver go to the workers (crossbeam
///   delivers each request to exactly one receiver, so workers pull-balance).
fn build_inference_channels(
    n_workers: usize,
    dispatch: DispatchMode,
) -> (Vec<ReqSender<EvalRequest>>, Vec<ReqReceiver<EvalRequest>>) {
    match dispatch {
        DispatchMode::RoundRobin => {
            let mut txs = Vec::with_capacity(n_workers);
            let mut rxs = Vec::with_capacity(n_workers);
            for _ in 0..n_workers {
                let (tx, rx) = req_unbounded::<EvalRequest>();
                txs.push(tx);
                rxs.push(rx);
            }
            (txs, rxs)
        }
        DispatchMode::Shared => {
            let (tx, rx) = req_unbounded::<EvalRequest>();
            let rxs = (0..n_workers).map(|_| rx.clone()).collect();
            (vec![tx], rxs)
        }
    }
}

/// Run the inference server loop. Collects requests into batches and
/// runs forward passes. Returns when all senders are dropped or `running`
/// is set to false.
#[cfg(feature = "torch")]
fn inference_server(
    request_rx: ReqReceiver<EvalRequest>,
    model: &mut TorchModel,
    max_batch: usize,
    batch_timeout: Duration,
    running: &AtomicBool,
    // For continuous mode: poll for model updates
    model_path: Option<&str>,
    device: tch::Device,
    graph_type: GraphType,
    padded: bool,
) {
    let mut model_mtime = model_path.and_then(file_mtime);

    // Perf counters (mirrors the Python inference server's [perf] output)
    let mut perf_count: u64 = 0;
    let mut perf_total_us: u64 = 0;
    let mut perf_queue_us: u64 = 0;
    let mut perf_forward_us: u64 = 0;
    let mut perf_total_graphs: u64 = 0;
    let mut perf_total_nodes: u64 = 0;

    loop {
        // Block on first request (or check shutdown)
        let t0 = Instant::now();
        let first = match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => req,
            Err(ReqRecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // Check for model reload during idle
                if let Some(path) = model_path {
                    try_reload_model(model, path, &mut model_mtime, device, graph_type);
                }
                continue;
            }
            Err(ReqRecvTimeoutError::Disconnected) => break,
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
        let t_queue = Instant::now();

        // Flatten all graphs into one batch
        let mut all_graphs = Vec::with_capacity(total_graphs);
        let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        let mut batch_nodes: usize = 0;
        for req in &mut requests {
            let start = all_graphs.len();
            for g in req.graphs.drain(..) {
                batch_nodes += g.num_nodes;
                all_graphs.push(g);
            }
            slice_ranges.push((start, all_graphs.len()));
        }

        // Single forward pass
        let (all_logits, all_values) = if padded {
            model.forward_graphs_padded(all_graphs)
        } else {
            model.forward_graphs(all_graphs)
        };
        let t_done = Instant::now();

        // Scatter results back
        for (i, req) in requests.into_iter().enumerate() {
            let (start, end) = slice_ranges[i];
            let logits = all_logits[start..end].to_vec();
            let values = all_values[start..end].to_vec();
            let _ = req.response_tx.send((logits, values));
        }

        // Perf tracking
        perf_count += 1;
        perf_total_us += (t_done - t0).as_micros() as u64;
        perf_queue_us += (t_queue - t0).as_micros() as u64;
        perf_forward_us += (t_done - t_queue).as_micros() as u64;
        perf_total_graphs += total_graphs as u64;
        perf_total_nodes += batch_nodes as u64;
        if perf_count % 100 == 0 {
            let n = perf_count as f64;
            eprintln!(
                "[perf] batches={} avg_total={:.1}ms avg_queue_wait={:.1}ms \
                 avg_forward={:.1}ms avg_graphs={:.1} avg_nodes={:.0}",
                perf_count,
                perf_total_us as f64 / n / 1000.0,
                perf_queue_us as f64 / n / 1000.0,
                perf_forward_us as f64 / n / 1000.0,
                perf_total_graphs as f64 / n,
                perf_total_nodes as f64 / n,
            );
        }

        // Check for model reload after processing a batch
        if let Some(path) = model_path {
            try_reload_model(model, path, &mut model_mtime, device, graph_type);
        }
    }
}

/// Run the inference server loop using a Python subprocess model.
/// Same batching logic as `inference_server` but uses `SubprocessModel`.
fn inference_server_subprocess(
    request_rx: ReqReceiver<EvalRequest>,
    model: &mut SubprocessModel,
    max_batch: usize,
    batch_timeout: Duration,
    running: &AtomicBool,
    model_path: Option<&str>,
) {
    loop {
        // Block on first request (or check shutdown)
        let first = match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => req,
            Err(ReqRecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // Check for model reload during idle
                if let Some(path) = model_path {
                    model.try_reload(path);
                }
                continue;
            }
            Err(ReqRecvTimeoutError::Disconnected) => break,
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

        // Single forward pass via subprocess
        match model.forward_graphs(all_graphs) {
            Ok((all_logits, all_values)) => {
                // Scatter results back
                for (i, req) in requests.into_iter().enumerate() {
                    let (start, end) = slice_ranges[i];
                    let logits = all_logits[start..end].to_vec();
                    let values = all_values[start..end].to_vec();
                    let _ = req.response_tx.send((logits, values));
                }
            }
            Err(e) => {
                eprintln!("Subprocess forward error: {e}");
                // Send empty results so game threads don't hang
                for req in requests {
                    let n = req.graphs.len().max(1); // graphs already drained
                    let empty_logits: Vec<HashMap<Coord, f64>> =
                        (0..n).map(|_| HashMap::default()).collect();
                    let empty_values: Vec<f64> = vec![0.0; n];
                    let _ = req.response_tx.send((empty_logits, empty_values));
                }
            }
        }

        // Check for model reload after processing a batch
        if let Some(path) = model_path {
            model.try_reload(path);
        }
    }
}

/// Pool inference server for subprocess mode. Spawns its own SubprocessModel
/// and waits for the pool loader to signal which checkpoint to load via a
/// channel of file paths.
fn pool_inference_server_subprocess(
    request_rx: ReqReceiver<EvalRequest>,
    max_batch: usize,
    batch_timeout: Duration,
    running: &AtomicBool,
    pool_path_rx: mpsc::Receiver<String>,
    python_bin: &str,
    model_args: &[String],
) {
    // Wait for the first checkpoint path from the loader.
    let first_path = match pool_path_rx.recv() {
        Ok(p) => p,
        Err(_) => return, // loader exited before producing a path
    };

    let mut model = match SubprocessModel::spawn_labeled(python_bin, &first_path, model_args, "python pool") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Pool subprocess failed to spawn: {e}");
            return;
        }
    };

    loop {
        // Swap to any newly-staged checkpoint path before blocking on requests.
        if let Ok(new_path) = pool_path_rx.try_recv() {
            if !new_path.is_empty() {
                model.try_reload(&new_path);
            }
        }

        let first = match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => req,
            Err(ReqRecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) { break; }
                continue;
            }
            Err(ReqRecvTimeoutError::Disconnected) => break,
        };

        let mut requests = vec![first];
        let mut total_graphs: usize = requests[0].graphs.len();
        let deadline = Instant::now() + batch_timeout;

        while total_graphs < max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { break; }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => {
                    total_graphs += req.graphs.len();
                    requests.push(req);
                }
                Err(_) => break,
            }
        }

        let mut all_graphs = Vec::with_capacity(total_graphs);
        let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        for req in &mut requests {
            let start = all_graphs.len();
            all_graphs.extend(req.graphs.drain(..));
            slice_ranges.push((start, all_graphs.len()));
        }

        match model.forward_graphs(all_graphs) {
            Ok((all_logits, all_values)) => {
                for (i, req) in requests.into_iter().enumerate() {
                    let (start, end) = slice_ranges[i];
                    let logits = all_logits[start..end].to_vec();
                    let values = all_values[start..end].to_vec();
                    let _ = req.response_tx.send((logits, values));
                }
            }
            Err(e) => {
                eprintln!("Pool subprocess forward error: {e}");
                for req in requests {
                    let n = req.graphs.len().max(1);
                    let empty_logits: Vec<HashMap<Coord, f64>> =
                        (0..n).map(|_| HashMap::default()).collect();
                    let _ = req.response_tx.send((empty_logits, vec![0.0; n]));
                }
            }
        }
    }
}

/// Background pool loader for subprocess mode. Periodically picks a random
/// checkpoint from pool_dir and sends its path to the pool inference server
/// via a channel. The subprocess handles the actual reload.
fn pool_subprocess_loader(
    pool_dir: String,
    pool_path_tx: mpsc::SyncSender<String>,
    pool_disabled: Arc<AtomicBool>,
    pool_ready: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
) {
    const RELOAD_INTERVAL_SECS: u64 = 30;
    const WARMUP_INTERVAL_SECS: u64 = 5;

    eprintln!(
        "Pool loader started (dir={}, warmup_interval={}s, reload_interval={}s)",
        pool_dir, WARMUP_INTERVAL_SECS, RELOAD_INTERVAL_SECS
    );

    let mut rng = ChaCha8Rng::from_os_rng();
    let mut consecutive_failures: usize = 0;
    let mut last_loaded_path: Option<String> = None;
    let mut cached_snaps: Vec<std::path::PathBuf> = Vec::new();
    let mut cached_mtime: Option<SystemTime> = None;
    let mut first_iter = true;
    let mut ever_loaded = false;

    loop {
        if !first_iter {
            let interval = if ever_loaded { RELOAD_INTERVAL_SECS } else { WARMUP_INTERVAL_SECS };
            for _ in 0..interval {
                if !running.load(Ordering::Relaxed) {
                    eprintln!("Pool loader exited");
                    return;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        first_iter = false;

        if !running.load(Ordering::Relaxed) {
            eprintln!("Pool loader exited");
            return;
        }

        let dir_mtime = file_mtime(&pool_dir);
        if cached_snaps.is_empty() || dir_mtime != cached_mtime {
            cached_snaps = list_pool_snapshots(&pool_dir);
            cached_mtime = dir_mtime;
        }

        if cached_snaps.is_empty() {
            continue;
        }

        let idx = (rng.random::<u64>() as usize) % cached_snaps.len();
        let pick = &cached_snaps[idx];
        let path_str = pick.to_string_lossy().to_string();

        let same_as_last = last_loaded_path.as_deref() == Some(path_str.as_str());
        if !same_as_last {
            eprintln!("Pool snapshot selected -> {}", pick.display());
            last_loaded_path = Some(path_str.clone());
        }

        // Send path to pool inference server. If the channel is full
        // (server hasn't consumed the previous path yet), just skip —
        // the server will pick up this snapshot on the next reload cycle.
        match pool_path_tx.try_send(path_str) {
            Ok(_) => {
                consecutive_failures = 0;
                if !ever_loaded {
                    ever_loaded = true;
                    pool_ready.store(true, Ordering::Relaxed);
                    eprintln!("Pool ready: first snapshot path sent");
                }
            }
            Err(mpsc::TrySendError::Full(_)) => {
                eprintln!("Pool loader: channel full, skipping (server busy)");
            }
            Err(e) => {
                eprintln!("Pool loader: failed to send path: {e}");
                consecutive_failures += 1;
                if consecutive_failures >= 5 {
                    eprintln!("Pool disabled after 5 consecutive failures");
                    pool_disabled.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

/// Scan ``dir`` for ``*.pt`` files. Returns sorted list of paths.
fn list_pool_snapshots(dir: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("pt") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[cfg(feature = "torch")]
/// Drain a bounded channel of capacity 1 and then send ``value``. Used by
/// the pool loader to "always have the freshest model staged" without ever
/// blocking the inference thread on disk I/O: if a previously staged model
/// has not yet been consumed, it is dropped and replaced.
///
/// Returns ``Err`` only if the receiver has been dropped.
fn try_replace<T>(tx: &SyncSender<T>, value: T) -> Result<(), TrySendError<T>> {
    match tx.try_send(value) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(v)) => {
            // A stale model is already staged. The receiver is bounded at
            // capacity 1, so a single try_send after a full observation is
            // either consumed (producer wins) or still full (we win and
            // replace). Since we are the sole producer, the slot can only
            // transition full->empty (never empty->full) outside our control,
            // so the next try_send will always succeed.
            match tx.try_send(v) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(v)) => {
                    // The old staged value is still there because the
                    // consumer hasn't drained it. As the sole producer we
                    // cannot drain it ourselves without a receiver handle.
                    // Fall back to a blocking send — in practice this path
                    // is only hit when the consumer is overloaded, which
                    // is fine because our loader is on a background thread.
                    tx.send(v).map_err(|e| TrySendError::Disconnected(e.0))
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(feature = "torch")]
/// Pool inference server: processes eval requests using the currently-
/// loaded snapshot. Model reloads happen on a separate background loader
/// thread; this server just picks up newly-staged models non-blockingly
/// before each batch iteration so disk I/O never stalls workers.
///
/// The server blocks on `staged_model_rx.recv()` for its first model, so
/// callers can spawn it before any snapshot exists on disk. If the loader
/// gives up before staging anything (channel closed), the server exits
/// cleanly without ever entering the main batch loop.
fn pool_inference_server(
    request_rx: ReqReceiver<EvalRequest>,
    max_batch: usize,
    batch_timeout: Duration,
    running: &AtomicBool,
    staged_model_rx: mpsc::Receiver<TorchModel>,
    padded: bool,
) {
    // Block until the loader stages an initial model. This is the only
    // path by which the server obtains its first model now that the
    // startup-time `load_initial_pool_snapshot` has been removed.
    let mut model = match staged_model_rx.recv() {
        Ok(m) => m,
        Err(_) => {
            // Loader exited before producing any model (e.g. failure
            // threshold tripped or shutdown). Nothing to serve.
            return;
        }
    };
    loop {
        // Non-blocking swap to any freshly-staged model BEFORE we block on
        // the request channel. The loader thread is responsible for keeping
        // this channel topped up with the most recent snapshot.
        if let Ok(new_model) = staged_model_rx.try_recv() {
            model = new_model;
        }

        let first = match request_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => req,
            Err(ReqRecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
            Err(ReqRecvTimeoutError::Disconnected) => break,
        };

        let mut requests = vec![first];
        let mut total_graphs: usize = requests[0].graphs.len();
        let deadline = Instant::now() + batch_timeout;

        while total_graphs < max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { break; }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => {
                    total_graphs += req.graphs.len();
                    requests.push(req);
                }
                Err(_) => break,
            }
        }

        let mut all_graphs = Vec::with_capacity(total_graphs);
        let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        for req in &mut requests {
            let start = all_graphs.len();
            all_graphs.extend(req.graphs.drain(..));
            slice_ranges.push((start, all_graphs.len()));
        }

        let (all_logits, all_values) = if padded {
            model.forward_graphs_padded(all_graphs)
        } else {
            model.forward_graphs(all_graphs)
        };

        for (i, req) in requests.into_iter().enumerate() {
            let (start, end) = slice_ranges[i];
            let logits = all_logits[start..end].to_vec();
            let values = all_values[start..end].to_vec();
            let _ = req.response_tx.send((logits, values));
        }
    }
}

#[cfg(feature = "torch")]
/// Background pool model loader. Periodically picks a random ``.pt``
/// snapshot from ``pool_dir``, loads it off-thread, and stages it for the
/// pool inference server to atomically swap in. Never blocks the inference
/// thread: all disk I/O happens here. Exits cleanly when ``running`` is
/// cleared or after 5 consecutive distinct-file load failures (in which
/// case it also flips ``pool_disabled``).
fn pool_model_loader(
    pool_dir: String,
    device: tch::Device,
    graph_type: GraphType,
    staged_model_tx: SyncSender<TorchModel>,
    pool_disabled: Arc<AtomicBool>,
    pool_ready: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
) {
    /// Cadence once at least one snapshot has been loaded. The pool's job is
    /// a stationary opponent mix averaged over many games, not per-game
    /// freshness, so a coarse cadence is fine.
    const RELOAD_INTERVAL_SECS: u64 = 30;
    /// Cadence while we have *never* successfully loaded a snapshot. The
    /// pool dir is empty at curriculum start and only fills once the trainer
    /// exports its first checkpoint, so we poll quickly to keep startup
    /// latency low without burning CPU forever.
    const WARMUP_INTERVAL_SECS: u64 = 5;

    eprintln!(
        "Pool loader started (dir={}, warmup_interval={}s, reload_interval={}s)",
        pool_dir, WARMUP_INTERVAL_SECS, RELOAD_INTERVAL_SECS
    );

    let mut rng = ChaCha8Rng::from_os_rng();
    let mut consecutive_failures: usize = 0;
    let mut last_loaded_path: Option<String> = None;
    let mut cached_snaps: Vec<std::path::PathBuf> = Vec::new();
    let mut cached_mtime: Option<SystemTime> = None;
    let mut first_iter = true;
    let mut ever_loaded = false;

    loop {
        if !first_iter {
            // Responsive shutdown: split the sleep into 1s increments.
            // Use the short warmup interval until we have produced at least
            // one model, then switch to the normal cadence.
            let interval = if ever_loaded {
                RELOAD_INTERVAL_SECS
            } else {
                WARMUP_INTERVAL_SECS
            };
            for _ in 0..interval {
                if !running.load(Ordering::Relaxed) {
                    eprintln!("Pool loader exited");
                    return;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        first_iter = false;

        if !running.load(Ordering::Relaxed) {
            eprintln!("Pool loader exited");
            return;
        }

        // Refresh the cached file list when the dir mtime changes OR when
        // the cache is currently empty (so a freshly populated dir is picked
        // up on the very next warmup tick without waiting for an mtime
        // observation race).
        let dir_mtime = file_mtime(&pool_dir);
        if cached_snaps.is_empty() || dir_mtime != cached_mtime {
            cached_snaps = list_pool_snapshots(&pool_dir);
            cached_mtime = dir_mtime;
        }

        if cached_snaps.is_empty() {
            // Empty dir is *not* a load failure — it just means we are still
            // warming up before the trainer has exported its first checkpoint.
            // Don't increment the failure counter; just sleep and retry.
            continue;
        }

        let idx = (rng.random::<u64>() as usize) % cached_snaps.len();
        let pick = &cached_snaps[idx];
        let path_str = pick.to_string_lossy().to_string();
        match TorchModel::load_with_graph_type(&path_str, device, graph_type) {
            Ok(new_model) => {
                consecutive_failures = 0;
                let same_as_last = last_loaded_path.as_deref() == Some(path_str.as_str());
                if !same_as_last {
                    eprintln!("Pool snapshot loaded -> {}", pick.display());
                    last_loaded_path = Some(path_str);
                }
                if try_replace(&staged_model_tx, new_model).is_err() {
                    // Receiver gone: inference server has shut down.
                    eprintln!("Pool loader exited");
                    return;
                }
                if !ever_loaded {
                    ever_loaded = true;
                    // First successful load: flip the readiness flag so
                    // worker threads start routing games to the pool.
                    pool_ready.store(true, Ordering::Relaxed);
                    eprintln!("Pool ready: first snapshot staged");
                }
            }
            Err(e) => {
                eprintln!("Pool snapshot load failed: {}: {e}", pick.display());
                consecutive_failures += 1;
                if consecutive_failures >= 5 {
                    eprintln!("Pool disabled after 5 consecutive load failures");
                    pool_disabled.store(true, Ordering::Relaxed);
                    eprintln!("Pool loader exited");
                    return;
                }
            }
        }
    }
}

#[cfg(feature = "torch")]
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
    sample_weight: f32,
}

/// One game's worth of examples, written as a single msgpack record.
#[derive(serde::Serialize)]
struct GameRecord {
    length: usize,
    winner: &'static str,
    examples: Vec<ExampleRecord>,
}

/// Per-position data collected during self-play (internal, not serialized).
struct PositionData {
    policy: Vec<f64>,
    player: &'static str,
    graph: GraphOutput,
    /// Per-example loss weight in [0, 1]. 1.0 for full-cap MCTS searches;
    /// `1/playout_cap_divisor` for fast-cap searches under PCR. Reflects
    /// the relative trust in the improved-policy target as a function of
    /// the search budget that produced it. Written to the wire format
    /// (HX04) and applied multiplicatively in the trainer's loss.
    sample_weight: f32,
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
    /// Actual number of moves played in the game. May differ from
    /// `positions.len()` when playout cap randomization is active —
    /// fast-search positions advance `move_count` but are not recorded.
    move_count: u32,
}

impl GameResult {
    /// Convert to a serializable GameRecord, computing value targets.
    fn to_record(&self, draw_value: f32) -> GameRecord {
        let examples = self.positions.iter().map(|pos| {
            let value: f64 = if self.winner == "draw" {
                draw_value as f64
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
                    sample_weight: pos.sample_weight,
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
                    sample_weight: pos.sample_weight,
                },
            }
        }).collect();
        GameRecord {
            length: self.positions.len(),
            winner: self.winner,
            examples,
        }
    }
}

/// Magic bytes identifying the start of a game record.
///
/// Version history:
/// - HX01: original. No winner byte.
/// - HX02: added 1B winner.
/// - HX03: added 4B move_count.
/// - HX04: added 4B sample_weight per example (post-PCR-discard fix). With PCR
///   active, fast-cap positions are now written with weight = 1/divisor instead
///   of being discarded, since Gumbel-AZ's completed-Q targets are informative
///   even at low sims. Trainer applies weights multiplicatively in the loss.
const RECORD_MAGIC: &[u8; 4] = b"HX04";

/// Write a game as a length-prefixed binary record.
///
/// Format: `[4B magic "HX04"][4B LE record_size][4B LE num_examples][1B winner i8][4B LE move_count u32][example...]`
///
/// `move_count` is the total number of moves played in the game. With
/// playout-cap randomisation off (or with the HX04+weighted variant)
/// `move_count` equals `num_examples`. With legacy PCR-discard active
/// (HX03) `move_count` could exceed `num_examples`.
///
/// `winner`: 1 = P1, -1 = P2, 0 = draw.
///
/// Each example:
/// ```text
/// [2B num_nodes][2B num_edges][2B num_legal][1B has_edge_attr][4B value f32]
/// [4B sample_weight f32]                                         -- HX04+
/// [num_nodes*8*4B features f32]
/// [num_edges*8B edge_src i64][num_edges*8B edge_dst i64]
/// [num_edges*5*4B edge_attr f32]  -- only if has_edge_attr
/// [num_nodes*1B legal_mask u8][num_nodes*1B stone_mask u8]
/// [num_nodes*2*4B coords i32]
/// [num_legal*8B policy f64]
/// ```
fn write_game_binary<W: Write>(writer: &mut W, result: &GameResult, draw_value: f32) -> std::io::Result<()> {
    // First pass: compute total size
    let mut size = 4u32 + 1 + 4; // num_examples + winner byte + move_count
    for pos in &result.positions {
        let (n, e, nl, has_ea) = example_dims(pos);
        size += 2 + 2 + 2 + 1 + 4 + 4; // header (u16×3 + u8 + value f32 + sample_weight f32)
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
    let winner_byte: i8 = match result.winner {
        "P1" => 1,
        "P2" => -1,
        _ => 0,
    };
    writer.write_all(&[winner_byte as u8])?;
    writer.write_all(&result.move_count.to_le_bytes())?;

    for pos in &result.positions {
        let value: f64 = if result.winner == "draw" {
            draw_value as f64
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
    writer.write_all(&pos.sample_weight.to_le_bytes())?;

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
fn game_to_json(result: &GameResult, draw_value: f32) -> serde_json::Value {
    let record = result.to_record(draw_value);
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
    prune_empty_edges: bool,
    threat_features: bool,
) -> (GraphOutput, GraphTensors) {
    match graph_type {
        GraphType::Axis => {
            let g = game_to_axis_graph_raw_opts(game, prune_empty_edges, threat_features);
            let t = GraphTensors::from_axis(&g);
            (GraphOutput::Axis(g), t)
        }
        GraphType::Hex => {
            let g = game_to_graph_raw_opts(game, threat_features);
            let t = GraphTensors::from_hex(&g);
            (GraphOutput::Hex(g), t)
        }
    }
}

/// Decide which side (P1 or P2) the past-self pool plays in a pool game.
/// Pure: returns ``"P1"`` or ``"P2"`` with 50/50 probability. Exposed for
/// unit testing the bias property.
fn pick_pool_side(rng: &mut ChaCha8Rng) -> &'static str {
    if rng.random::<bool>() { "P1" } else { "P2" }
}

/// Play one complete game using batched inference via the client.
///
/// If ``pool_client`` is ``Some`` and ``pool_side`` is ``Some(side)`` then
/// MCTS searches whose root turn matches ``side`` will route their inference
/// requests through the pool client (the past-self snapshot). Otherwise the
/// live ``client`` is used for both sides (vanilla self-play).
/// Wu (2019) playout cap randomization decision: returns true when this
/// move should be played as a fast (low-sim) search whose training example
/// is dropped. Both `fraction > 0` and `divisor > 1` must hold for the
/// feature to ever activate; otherwise this is a no-op (always full search).
#[inline]
fn should_skip_example(roll: f64, fraction: f64, divisor: u32) -> bool {
    if fraction <= 0.0 || divisor <= 1 {
        return false;
    }
    roll < fraction
}

/// Build a reduced-budget MCTS config for the fast search arm of playout
/// cap randomization. The simulation count is divided by `divisor` and
/// clamped to a minimum of 1; all other fields are preserved.
fn reduced_sim_config(base: &MCTSConfig, divisor: u32) -> MCTSConfig {
    let div = divisor.max(1);
    MCTSConfig {
        n_simulations: (base.n_simulations / div).max(1),
        m_actions: base.m_actions,
        c_visit: base.c_visit,
        c_scale: base.c_scale,
        virtual_loss: base.virtual_loss,
        root_dirichlet_alpha: base.root_dirichlet_alpha,
        root_dirichlet_fraction: base.root_dirichlet_fraction,
        forced_candidate_capture_k: base.forced_candidate_capture_k,
    }
}

fn play_one_game(
    client: &InferenceClient,
    pool_client: Option<&InferenceClient>,
    pool_side: Option<&'static str>,
    game_config: GameConfig,
    mcts_config: &MCTSConfig,
    exploration: &AtomicUsize,
    graph_type: GraphType,
    prune_empty_edges: bool,
    threat_features: bool,
    rng: &mut ChaCha8Rng,
    playout_cap_fraction: f64,
    playout_cap_divisor: u32,
    running: &AtomicBool,
) -> Option<GameResult> {
    use hexo_rs::mcts::gumbel_mcts::gumbel_mcts;

    let mut game = GameState::with_config(game_config);
    let mut positions = Vec::new();
    let mut move_count = 0;
    let profile = PROFILE_PLAY_ENABLED.load(Ordering::Relaxed);

    while !game.is_terminal() {
        if !running.load(Ordering::Relaxed) {
            return None;
        }
        let iter_start = if profile { Some(Instant::now()) } else { None };
        let current_player = match game.current_player() {
            Some(hexo_engine::types::Player::P1) => "P1",
            Some(hexo_engine::types::Player::P2) => "P2",
            None => "?",
        };

        // Build graph ONCE for this position — reuse for both output and root eval
        let t = if profile { Some(Instant::now()) } else { None };
        let (graph_output, root_tensors) =
            build_position_graph(&game, graph_type, prune_empty_edges, threat_features);
        let graph_build_ns = t.map(|i| i.elapsed().as_nanos()).unwrap_or(0);

        // Choose which inference client services THIS root's MCTS search.
        // For pool games, the pool client serves searches whose side-to-move
        // matches the pre-assigned pool_side; the live client serves the rest.
        let active_client: &InferenceClient = match (pool_client, pool_side) {
            (Some(pc), Some(side)) if side == current_player => pc,
            _ => client,
        };

        // Eval closure: builds graphs on this thread, sends to inference server.
        // Root eval reuses the pre-built graph; leaf evals build fresh.
        let mut cached_root = Some(root_tensors);
        let mut eval_wait_ns: u128 = 0;
        let mut eval_graph_build_ns: u128 = 0;
        let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
            let eval_start = if profile { Some(Instant::now()) } else { None };
            let graphs = if let Some(gt) = cached_root.take() {
                debug_assert_eq!(states.len(), 1, "root eval should be single state");
                vec![gt]
            } else {
                let gb = if profile { Some(Instant::now()) } else { None };
                // Leaf evals: build graphs on this game thread (CPU work)
                let gs: Vec<_> = states.iter().map(|s| {
                    match graph_type {
                        GraphType::Axis => GraphTensors::from(game_to_axis_graph_raw_opts(s, prune_empty_edges, threat_features)),
                        GraphType::Hex => GraphTensors::from(game_to_graph_raw_opts(s, threat_features)),
                    }
                }).collect();
                if let Some(i) = gb { eval_graph_build_ns += i.elapsed().as_nanos(); }
                gs
            };
            let result = active_client.evaluate(graphs);
            if let Some(i) = eval_start { eval_wait_ns += i.elapsed().as_nanos(); }
            result
        };

        // Wu (2019) playout cap randomisation, adapted for Gumbel AZ (HX04):
        // with probability `playout_cap_fraction`, run a reduced-budget search.
        // Unlike vanilla AZ (which discards fast-cap targets as too noisy),
        // Gumbel AZ's improved_policy = softmax(logits + σ(completed_Q)) is
        // informative even at low sims, so we record every position. Fast-cap
        // examples carry a smaller `sample_weight` (= 1/divisor) so the
        // trainer down-weights them in the loss to reflect the smaller
        // search budget behind the target.
        let roll: f64 = rng.random::<f64>();
        let fast_search = should_skip_example(roll, playout_cap_fraction, playout_cap_divisor);
        let local_cfg;
        let active_cfg: &MCTSConfig = if fast_search {
            local_cfg = reduced_sim_config(mcts_config, playout_cap_divisor);
            &local_cfg
        } else {
            mcts_config
        };
        let sample_weight: f32 = if fast_search {
            // Reduced-search trust = reduced_sims / full_sims = 1 / divisor.
            // playout_cap_divisor==1 yields 1.0 (PCR fully off) and is safe.
            1.0 / (playout_cap_divisor.max(1) as f32)
        } else {
            1.0
        };

        let t = if profile { Some(Instant::now()) } else { None };
        let result = match gumbel_mcts(&game, active_cfg, rng, None, &mut eval) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("MCTS failed: {e}, dropping game");
                return None;
            }
        };
        let mcts_total_ns = t.map(|i| i.elapsed().as_nanos()).unwrap_or(0);

        let action = if move_count < exploration.load(Ordering::Relaxed) {
            let weights = exploration_weights(
                &result.improved_policy,
                &result.visit_counts,
                EXPLORE_USE_VISIT_COUNTS.load(Ordering::Relaxed),
            );
            let total: f64 = weights.iter().sum();
            if total > 0.0 {
                let r: f64 = rng.random::<f64>() * total;
                let mut cumsum = 0.0;
                let mut chosen = 0;
                for (i, &p) in weights.iter().enumerate() {
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
            sample_weight,
        });

        if !running.load(Ordering::Relaxed) {
            return None;
        }
        let t = if profile { Some(Instant::now()) } else { None };
        if let Err(e) = game.apply_move(action) {
            eprintln!("Invalid move from MCTS: {e:?}, dropping game");
            return None;
        }
        let apply_move_ns = t.map(|i| i.elapsed().as_nanos()).unwrap_or(0);
        move_count += 1;

        if profile {
            let iter_total_ns = iter_start.map(|i| i.elapsed().as_nanos()).unwrap_or(0);
            PLAY_PERF.with(|cell| {
                let mut p = cell.borrow_mut();
                p.moves += 1;
                p.graph_build_ns += graph_build_ns;
                p.mcts_total_ns += mcts_total_ns;
                p.eval_wait_ns += eval_wait_ns;
                p.eval_graph_build_ns += eval_graph_build_ns;
                p.apply_move_ns += apply_move_ns;
                p.iter_total_ns += iter_total_ns;
            });
        }
    }

    if profile {
        PLAY_PERF.with(|cell| cell.borrow_mut().games += 1);
    }

    let winner = match game.winner() {
        Some(hexo_engine::types::Player::P1) => "P1",
        Some(hexo_engine::types::Player::P2) => "P2",
        None => "draw",
    };

    Some(GameResult { positions, winner, move_count: move_count as u32 })
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
    draw_value: f32,
}

impl RotatingWriter {
    fn new(dir: &str, filename: &str, max_bytes: u64, draw_value: f32) -> Self {
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
            draw_value,
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
        write_game_binary(&mut self.writer, result, self.draw_value)?;
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
    let mut c_visit: u32 = 50;
    let mut c_scale: f64 = 1.0;
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
    let mut pool_dir: Option<String> = None;
    let mut pool_fraction: f64 = 0.0;
    let mut playout_cap_fraction: f64 = 0.0;
    let mut playout_cap_divisor: u32 = 1;
    #[allow(unused)] // used only with torch feature
    let mut padded_inference: bool = false;
    #[allow(unused)] // used only with torch feature
    let mut shape_hist: bool = false;
    let mut exploration_max: Option<f64> = None; // adaptive exploration ceiling
    let mut exploration_window: u32 = 200; // EMA effective window for p1 decided rate
    let mut exploration_exponent: f64 = 1.0; // deviation curve: 1.0=linear, 2.0=squared
    let mut exploration_distribution: String = "improved_policy".to_string(); // or "visit_counts"
    let mut draw_value: f32 = 0.0; // value target for drawn games (0.0=neutral, -0.1=penalty)
    let mut prune_empty_edges = false;
    let mut threat_features = false;

    // Python subprocess inference flags
    let mut python_inference = false;
    let mut python_bin = String::from("python");
    // Number of parallel Python inference subprocesses driving the main
    // self-play request stream. Default 1 = single subprocess (historical
    // behaviour, byte-identical CLI for existing configs). N>1 spawns N
    // independent inference workers with sender-side round-robin so multiple
    // GPU forward passes can overlap, hiding per-batch CUDA sync latency.
    // Only honored when --python-inference is set; the in-process torch
    // path is always single-worker.
    let mut python_inference_workers: usize = 1;
    // How requests reach the N inference batchers. Default Shared (MPMC pull
    // queue): idle workers pull the next batch, robust to any service-time
    // asymmetry. Validated neutral vs round-robin on this APU (both workers
    // ~67 graphs); strictly more general for future multi-GPU. Pass
    // `--inference-dispatch round-robin` for the legacy push behaviour.
    let mut inference_dispatch = DispatchMode::Shared;
    let mut checkpoint_path: Option<String> = None;
    let mut model_hidden_dim: usize = 256;
    let mut model_num_layers: usize = 3;
    let mut model_num_heads: usize = 8;
    let mut model_policy_hidden: usize = 128;
    let mut model_value_hidden: usize = 128;
    let mut model_conv_type = String::from("gine");
    let mut model_use_jk: bool = false;
    let mut model_jk_mode = String::from("sum");

    // Sequential Halving inner-loop fusion (virtual-loss leaf-parallel MCTS).
    // 0.0 = disabled (serial sims, current behaviour). 0.3–1.0 enables fusion.
    let mut virtual_loss: f64 = 0.0;

    // Root Dirichlet noise (AlphaZero-style root exploration applied before
    // Gumbel-Top-k candidate sampling). Both default to 0.0 → disabled,
    // bit-equivalent to existing behaviour.
    let mut root_dirichlet_alpha: f64 = 0.0;
    let mut root_dirichlet_fraction: f64 = 0.0;

    // Warmup games to play in batch mode BEFORE the timing-of-record begins.
    // Lets workers desync from the initial all-start-from-empty / all-at-
    // same-SH-phase lockstep, so the measured games/s reflects steady-state
    // batcher dynamics rather than the early synchronised regime. 0 disables.
    let mut warmup_games: usize = 0;

    // When true, each game worker accumulates per-section timing for moves
    // it plays during measurement and prints a summary at end of its run.
    let mut profile_play: bool = false;

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
            "--c-visit" => { c_visit = args[i + 1].parse().unwrap(); i += 2; }
            "--c-scale" => { c_scale = args[i + 1].parse().unwrap(); i += 2; }
            "--exploration" => { exploration = args[i + 1].parse().unwrap(); i += 2; }
            "--exploration-fraction" => { exploration_fraction = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--exploration-max" => { exploration_max = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--exploration-window" => { exploration_window = args[i + 1].parse().unwrap(); i += 2; }
            "--exploration-exponent" => { exploration_exponent = args[i + 1].parse().unwrap(); i += 2; }
            "--exploration-distribution" => { exploration_distribution = args[i + 1].clone(); i += 2; }
            "--draw-value" => { draw_value = args[i + 1].parse().unwrap(); i += 2; }
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
            "--pool-dir" => { pool_dir = Some(args[i + 1].clone()); i += 2; }
            "--pool-fraction" => {
                pool_fraction = args[i + 1].parse().unwrap();
                if !(0.0..=1.0).contains(&pool_fraction) {
                    eprintln!("--pool-fraction must be in [0.0, 1.0], got {}", pool_fraction);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--playout-cap-fraction" => {
                playout_cap_fraction = args[i + 1].parse().unwrap();
                if !(0.0..=1.0).contains(&playout_cap_fraction) {
                    eprintln!("--playout-cap-fraction must be in [0.0, 1.0], got {}", playout_cap_fraction);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--padded-inference" => { #[allow(unused)] { padded_inference = true; } i += 1; }
            "--prune-empty-edges" => { prune_empty_edges = true; i += 1; }
            "--threat-features" => { threat_features = true; i += 1; }
            "--shape-hist" => { #[allow(unused)] { shape_hist = true; } i += 1; }
            "--playout-cap-divisor" => {
                playout_cap_divisor = args[i + 1].parse().unwrap();
                if playout_cap_divisor < 1 {
                    eprintln!("--playout-cap-divisor must be >= 1, got {}", playout_cap_divisor);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--python-inference" => { python_inference = true; i += 1; }
            "--python-inference-workers" => {
                python_inference_workers = args[i + 1].parse().unwrap();
                if python_inference_workers == 0 {
                    eprintln!("--python-inference-workers must be >= 1, got 0");
                    std::process::exit(1);
                }
                i += 2;
            }
            "--inference-dispatch" => {
                inference_dispatch = match args[i + 1].as_str() {
                    "round-robin" => DispatchMode::RoundRobin,
                    "shared" => DispatchMode::Shared,
                    other => {
                        eprintln!(
                            "--inference-dispatch must be 'round-robin' or 'shared', got '{other}'"
                        );
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--python-bin" => { python_bin = args[i + 1].clone(); i += 2; }
            "--checkpoint" => { checkpoint_path = Some(args[i + 1].clone()); i += 2; }
            "--model-hidden-dim" => { model_hidden_dim = args[i + 1].parse().unwrap(); i += 2; }
            "--model-num-layers" => { model_num_layers = args[i + 1].parse().unwrap(); i += 2; }
            "--model-num-heads" => { model_num_heads = args[i + 1].parse().unwrap(); i += 2; }
            "--model-policy-hidden" => { model_policy_hidden = args[i + 1].parse().unwrap(); i += 2; }
            "--model-value-hidden" => { model_value_hidden = args[i + 1].parse().unwrap(); i += 2; }
            "--model-conv-type" => { model_conv_type = args[i + 1].clone(); i += 2; }
            "--model-use-jk" => { model_use_jk = true; i += 1; }
            "--model-jk-mode" => { model_jk_mode = args[i + 1].clone(); i += 2; }
            "--virtual-loss" => {
                virtual_loss = args[i + 1].parse().unwrap();
                if !virtual_loss.is_finite() || virtual_loss < 0.0 {
                    eprintln!("--virtual-loss must be a non-negative finite number, got {}", virtual_loss);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--root-dirichlet-alpha" => {
                root_dirichlet_alpha = args[i + 1].parse().unwrap();
                if !root_dirichlet_alpha.is_finite() || root_dirichlet_alpha < 0.0 {
                    eprintln!("--root-dirichlet-alpha must be a non-negative finite number, got {}", root_dirichlet_alpha);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--root-dirichlet-fraction" => {
                root_dirichlet_fraction = args[i + 1].parse().unwrap();
                if !root_dirichlet_fraction.is_finite() || !(0.0..=1.0).contains(&root_dirichlet_fraction) {
                    eprintln!("--root-dirichlet-fraction must be in [0, 1], got {}", root_dirichlet_fraction);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--warmup-games" => {
                warmup_games = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--profile-play" => { profile_play = true; i += 1; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    if model_path.is_empty() {
        eprintln!("Usage: self_play --model <path.pt> [--continuous] [options]");
        std::process::exit(1);
    }

    // python_inference_workers > 1 is only meaningful for the Python
    // subprocess inference path. The in-process torch path always shares a
    // single TorchModel, so multiple batchers would just serialise on it.
    if !python_inference && python_inference_workers > 1 {
        eprintln!(
            "Warning: --python-inference-workers={} requires --python-inference; \
             forcing to 1 (in-process torch is always single-worker).",
            python_inference_workers,
        );
        python_inference_workers = 1;
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
    #[cfg(feature = "torch")]
    if !python_inference {
        tch::set_num_threads(omp_threads as i32);
        tch::set_num_interop_threads(1);
    }

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

    PROFILE_PLAY_ENABLED.store(profile_play, Ordering::Relaxed);

    let explore_use_visit_counts = match exploration_distribution.as_str() {
        "visit_counts" => true,
        "improved_policy" => false,
        other => {
            eprintln!(
                "--exploration-distribution must be 'improved_policy' or 'visit_counts', got {other:?}"
            );
            std::process::exit(1);
        }
    };
    EXPLORE_USE_VISIT_COUNTS.store(explore_use_visit_counts, Ordering::Relaxed);

    let game_config = GameConfig { win_length, placement_radius: radius, max_moves };
    let mcts_config = Arc::new(MCTSConfig {
        n_simulations: n_sims,
        m_actions,
        c_visit,
        c_scale,
        virtual_loss,
        root_dirichlet_alpha,
        root_dirichlet_fraction,
        // E.4-inject capture: not exposed as a self_play CLI flag.
        // Self-play doesn't benefit from capture (the captured candidates
        // would need to flow into the *next* search, which self-play
        // doesn't track per-game-thread). Python eval harness uses it.
        forced_candidate_capture_k: 0,
    });

    let batch_timeout = if batch_timeout_ms > 0 {
        Duration::from_millis(batch_timeout_ms)
    } else {
        // Default: shorter timeout for GPU, longer for CPU
        if device_str == "cuda" {
            Duration::from_millis(1)
        } else {
            Duration::from_millis(5)
        }
    };

    let exploration_atomic = Arc::new(AtomicUsize::new(exploration));
    // EMA of game length stored as f64 bits in AtomicU64 (lock-free, all threads update)
    let ema_bits = Arc::new(AtomicU64::new(0u64)); // 0 bits = 0.0f64
    // EMA of p1 decided rate for adaptive exploration (0.5 = balanced)
    let p1_ema_bits = Arc::new(AtomicU64::new(0.5f64.to_bits())); // start balanced

    if !continuous {
        // --- Batch mode ---
        eprintln!(
            "Playing {} games (sims={}, m={}, c_visit={}, c_scale={}, explore={}, threads={}, omp={}, max_batch={}, warmup={})...",
            n_games, n_sims, m_actions, c_visit, c_scale, exploration, n_threads, omp_threads, max_batch, warmup_games,
        );

        // Create inference channel (batch mode is always single-worker, so
        // dispatch mode is irrelevant — one channel either way).
        let (request_tx, request_rx) = req_unbounded::<EvalRequest>();
        let client = InferenceClient::new(vec![request_tx]);

        // Each inference branch returns (all_games, measurement_elapsed).
        // measurement_elapsed is started AFTER warmup completes so the
        // reported games/s reflects steady-state worker distribution (warmup
        // breaks the initial all-workers-at-empty-board lockstep) and does
        // not include torch.compile or model load.
        let (all_games, elapsed): (Vec<GameResult>, Duration) = if python_inference {
            let ckpt = checkpoint_path.as_deref().unwrap_or(&model_path);
            let model_args = subprocess_model_args(
                model_hidden_dim, model_num_layers, model_num_heads,
                model_policy_hidden, model_value_hidden,
                &graph_type_str, &model_conv_type, device_str,
                padded_inference,
                model_use_jk, &model_jk_mode,
                threat_features,
            );
            eprintln!("Spawning Python inference subprocess...");
            let mut model = SubprocessModel::spawn(&python_bin, ckpt, &model_args)
                .unwrap_or_else(|e| { eprintln!("Failed to spawn Python: {e}"); std::process::exit(1); });

            std::thread::scope(|s| {
                let running_ref = &running;
                let model_ref = &mut model;
                let server_handle = s.spawn(move || {
                    inference_server_subprocess(
                        request_rx, model_ref, max_batch, batch_timeout,
                        running_ref, None,
                    );
                });

                let (results, measured) = run_batch_games_with_warmup(
                    s, client, warmup_games, n_games, n_threads, seed, game_config, &mcts_config,
                    &exploration_atomic, graph_type, prune_empty_edges, threat_features,
                    playout_cap_fraction, playout_cap_divisor,
                    &running,
                );

                server_handle.join().unwrap();
                (results, measured)
            })
        } else {
            #[cfg(not(feature = "torch"))]
            {
                eprintln!("Built without torch feature. Use --python-inference or rebuild with --features torch.");
                std::process::exit(1);
            }
            #[cfg(feature = "torch")]
            {
                let tch_device = match device_str {
                    "cuda" => tch::Device::Cuda(0),
                    _ => tch::Device::Cpu,
                };
                eprintln!("Loading model from {}...", model_path);
                let mut model = match TorchModel::load_with_graph_type(&model_path, tch_device, graph_type) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Failed to load model: {e}");
                        std::process::exit(1);
                    }
                };

                std::thread::scope(|s| {
                    let running_ref = &running;
                    let model_ref = &mut model;
                    let server_handle = s.spawn(move || {
                        inference_server(
                            request_rx, model_ref, max_batch, batch_timeout,
                            running_ref, None, tch_device, graph_type, padded_inference,
                        );
                    });

                    let (results, measured) = run_batch_games_with_warmup(
                        s, client, warmup_games, n_games, n_threads, seed, game_config, &mcts_config,
                        &exploration_atomic, graph_type, prune_empty_edges, threat_features,
                        playout_cap_fraction, playout_cap_divisor,
                        &running,
                    );

                    server_handle.join().unwrap();
                    (results, measured)
                })
            }
        };
        let total_moves: usize = all_games.iter().map(|g| g.move_count as usize).sum();

        let json: Vec<serde_json::Value> = all_games.iter()
            .map(|g| game_to_json(g, draw_value))
            .collect();
        let tmp_path = format!("{}.tmp", output_path);
        fs::write(&tmp_path, serde_json::to_string(&json).unwrap())
            .and_then(|_| fs::rename(&tmp_path, &output_path))
            .expect("Failed to write output");

        eprintln!(
            "Done: {} games, {} moves, {:.1}s ({:.2}s/game, {:.3} games/s)",
            n_games, total_moves, elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / n_games as f64,
            n_games as f64 / elapsed.as_secs_f64(),
        );
    } else {
        // --- Continuous mode ---
        #[cfg(feature = "torch")]
        if shape_hist && !python_inference {
            hexo_rs::inference::enable_shape_hist();
        }
        let dir = output_dir.unwrap_or_else(|| "self_play_output".to_string());
        fs::create_dir_all(&dir).expect("Failed to create output directory");

        let rotating_writer = Arc::new(Mutex::new(
            RotatingWriter::new(&dir, &output_filename, max_file_mb * 1024 * 1024, draw_value),
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

        // Create main inference channels — one batcher thread per
        // python_inference_workers. N=1 (default) is byte-for-byte equivalent
        // to the historical single-Sender path under either dispatch mode.
        // N>1 spawns N Python inference subprocesses; `inference_dispatch`
        // selects how requests reach their batcher threads:
        //   RoundRobin — N channels, sender-side `fetch_add%N` (legacy default)
        //   Shared     — one MPMC queue, workers pull-balance (see DispatchMode)
        let n_workers = python_inference_workers.max(1);
        let (main_request_txs, worker_rxs) =
            build_inference_channels(n_workers, inference_dispatch);
        // Wrap receivers in Option so the spawn loop / torch path can `.take()`
        // each exactly once (unchanged downstream).
        let mut main_request_rxs: Vec<Option<ReqReceiver<EvalRequest>>> =
            worker_rxs.into_iter().map(Some).collect();
        let client = InferenceClient::new(main_request_txs);

        // Past-self opponent pool (optional, torch-only). Always single-worker.
        let pool_disabled = Arc::new(AtomicBool::new(false));
        let pool_ready = Arc::new(AtomicBool::new(false));

        // Build the pool inference channel up front (when configured) so we
        // can clone the client into game threads inside the scope.
        let (pool_client_opt, pool_request_rx_opt) =
            if pool_fraction > 0.0 && pool_dir.is_some() {
                let (ptx, prx) = req_unbounded::<EvalRequest>();
                (Some(InferenceClient::new(vec![ptx])), Some(prx))
            } else {
                (None, None)
            };

        if python_inference {
            let ckpt = checkpoint_path.as_deref().unwrap_or(&model_path);
            let model_args = subprocess_model_args(
                model_hidden_dim, model_num_layers, model_num_heads,
                model_policy_hidden, model_value_hidden,
                &graph_type_str, &model_conv_type, device_str,
                padded_inference,
                model_use_jk, &model_jk_mode,
                threat_features,
            );
            if n_workers > 1 {
                let dispatch_label = match inference_dispatch {
                    DispatchMode::RoundRobin => "round-robin",
                    DispatchMode::Shared => "shared queue",
                };
                eprintln!(
                    "Spawning {} Python inference subprocesses ({})...",
                    n_workers, dispatch_label,
                );
            } else {
                eprintln!("Spawning Python inference subprocess...");
            }
            // Spawn one SubprocessModel per worker. We collect them into a
            // Vec so each batcher thread owns its own &mut SubprocessModel
            // inside the thread scope below.
            let mut models: Vec<SubprocessModel> = Vec::with_capacity(n_workers);
            for wid in 0..n_workers {
                // Tag each worker's stderr (incl. its `[perf]` line) `[python wN]`
                // so concurrent workers are distinguishable in the log. At N=1
                // this is still `[python w0]`; harmless and explicit.
                let label = format!("python w{wid}");
                match SubprocessModel::spawn_labeled(&python_bin, ckpt, &model_args, &label) {
                    Ok(m) => models.push(m),
                    Err(e) => {
                        eprintln!("Failed to spawn Python inference worker {wid}: {e}");
                        std::process::exit(1);
                    }
                }
            }

            std::thread::scope(|s| {
                let running_ref = &running;
                let model_path_ref = model_path.as_str();
                // Each batcher thread gets one (rx, &mut model) pair.
                for (rx_opt, model) in main_request_rxs.iter_mut().zip(models.iter_mut()) {
                    let rx = rx_opt.take().expect("each receiver consumed exactly once");
                    s.spawn(move || {
                        inference_server_subprocess(
                            rx, model, max_batch, batch_timeout,
                            running_ref, Some(model_path_ref),
                        );
                    });
                }

                // Spawn pool inference server + background loader (if configured).
                if let Some(prx) = pool_request_rx_opt {
                    let pdir = pool_dir.clone().expect("pool_dir set when pool configured");
                    let (pool_path_tx, pool_path_rx) = mpsc::sync_channel::<String>(1);

                    let python_bin_clone = python_bin.clone();
                    let model_args_clone = model_args.clone();
                    s.spawn(move || {
                        pool_inference_server_subprocess(
                            prx, max_batch, batch_timeout,
                            running_ref, pool_path_rx,
                            &python_bin_clone, &model_args_clone,
                        );
                    });

                    let pool_disabled_loader = pool_disabled.clone();
                    let pool_ready_loader = pool_ready.clone();
                    let running_loader = running.clone();
                    s.spawn(move || {
                        pool_subprocess_loader(
                            pdir, pool_path_tx,
                            pool_disabled_loader, pool_ready_loader,
                            running_loader,
                        );
                    });
                }

                run_continuous_game_threads(
                    s, client, pool_client_opt, n_threads, seed, game_config,
                    &mcts_config, &exploration_atomic, graph_type, prune_empty_edges,
                    threat_features,
                    playout_cap_fraction, playout_cap_divisor, pool_fraction,
                    &pool_disabled, &pool_ready, &running, &game_counter,
                    &rotating_writer, exploration_fraction, &ema_bits,
                    &p1_ema_bits, exploration_window, exploration_exponent,
                    exploration_max,
                );
            });
        } else {
            #[cfg(not(feature = "torch"))]
            {
                eprintln!("Built without torch feature. Use --python-inference or rebuild with --features torch.");
                std::process::exit(1);
            }
            #[cfg(feature = "torch")]
            {
                let tch_device = match device_str {
                    "cuda" => tch::Device::Cuda(0),
                    _ => tch::Device::Cpu,
                };
                eprintln!("Loading model from {}...", model_path);
                let mut model = match TorchModel::load_with_graph_type(&model_path, tch_device, graph_type) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Failed to load model: {e}");
                        std::process::exit(1);
                    }
                };

                // In-process torch inference is always single-worker; pull
                // the sole receiver out of the main_request_rxs vec.
                let request_rx = main_request_rxs[0]
                    .take()
                    .expect("torch path requires exactly one main receiver");

                std::thread::scope(|s| {
                    let running_ref = &running;
                    let model_ref = &mut model;
                    let model_path_ref = model_path.as_str();
                    s.spawn(move || {
                        inference_server(
                            request_rx, model_ref, max_batch, batch_timeout,
                            running_ref, Some(model_path_ref), tch_device, graph_type, padded_inference,
                        );
                    });

                    // Spawn pool inference server + background loader (if configured).
                    if let Some(prx) = pool_request_rx_opt {
                        let running_ref = &running;
                        let pdir = pool_dir.clone().expect("pool_dir set when pool configured");
                        let (staged_tx, staged_rx) = mpsc::sync_channel::<TorchModel>(1);

                        s.spawn(move || {
                            pool_inference_server(
                                prx, max_batch, batch_timeout,
                                running_ref, staged_rx, padded_inference,
                            );
                        });

                        let pool_disabled_loader = pool_disabled.clone();
                        let pool_ready_loader = pool_ready.clone();
                        let running_loader = running.clone();
                        s.spawn(move || {
                            pool_model_loader(
                                pdir, tch_device, graph_type,
                                staged_tx, pool_disabled_loader, pool_ready_loader,
                                running_loader,
                            );
                        });
                    }

                    run_continuous_game_threads(
                        s, client, pool_client_opt, n_threads, seed, game_config,
                        &mcts_config, &exploration_atomic, graph_type, prune_empty_edges,
                        threat_features,
                        playout_cap_fraction, playout_cap_divisor, pool_fraction,
                        &pool_disabled, &pool_ready, &running, &game_counter,
                        &rotating_writer, exploration_fraction, &ema_bits,
                        &p1_ema_bits, exploration_window, exploration_exponent,
                        exploration_max,
                    );
                });
            }
        }

        let total_games = game_counter.load(Ordering::Relaxed);
        eprintln!("Stopped after {} games.", total_games);
    }
}

/// Node-feature dim of threat-encoding graphs. Must match `fdim` in
/// hexo-mcts/src/graph.rs (`build_graph`): 8 base dims + 4 threat dims.
const THREAT_NODE_DIM: usize = 12;

/// Build the model args vector for `SubprocessModel::spawn`.
#[allow(clippy::too_many_arguments)]
fn subprocess_model_args(
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    policy_hidden: usize,
    value_hidden: usize,
    graph_type: &str,
    conv_type: &str,
    device: &str,
    padded_inference: bool,
    use_jk: bool,
    jk_mode: &str,
    threat_features: bool,
) -> Vec<String> {
    let mut v = vec![
        "--hidden-dim".into(), hidden_dim.to_string(),
        "--num-layers".into(), num_layers.to_string(),
        "--num-heads".into(), num_heads.to_string(),
        "--policy-hidden".into(), policy_hidden.to_string(),
        "--value-hidden".into(), value_hidden.to_string(),
        "--graph-type".into(), graph_type.to_string(),
        "--conv-type".into(), conv_type.to_string(),
        "--device".into(), device.to_string(),
    ];
    if padded_inference {
        v.push("--padded-inference".into());
    }
    // Only emit JK flags when JK is enabled — keeps the subprocess CLI
    // byte-identical for the (overwhelming majority of) no-JK runs.
    if use_jk {
        v.push("--use-jk".into());
        v.push("--jk-mode".into());
        v.push(jk_mode.to_string());
    }
    // 12-dim threat-feature graphs need the server to build the model with
    // node_features=12 and warm up with matching inputs. The server defaults
    // to 8, so only emit the flag for threat runs (keeps the CLI byte-identical
    // for plain 8-dim runs).
    if threat_features {
        v.push("--node-dim".into());
        v.push(THREAT_NODE_DIM.to_string());
    }
    v
}

/// Run batch-mode game threads inside a thread scope. Returns collected game results.
/// Takes `client` by value so it is dropped when all game threads finish,
/// signaling the inference server to shut down.
/// Run batch-mode games with an optional per-worker warmup phase.
///
/// Workers play `ceil(warmup_games / n_threads)` warmup games each (results
/// discarded), then wait at a barrier so all warmup completes before any
/// measurement starts, then play their measurement allocation. **Same OS
/// threads run both phases**, which is the entire point — without that, the
/// measurement phase would respawn workers that all start fresh from the
/// empty board and re-enter the same lockstep we were trying to escape.
///
/// Returns `(measurement_games, measurement_elapsed)`. The elapsed time is
/// measured from the moment the barrier releases (= last warmup-game done +
/// barrier sync latency) to when the last measurement game finishes.
#[allow(clippy::too_many_arguments)]
fn run_batch_games_with_warmup<'scope, 'env: 'scope>(
    s: &'scope std::thread::Scope<'scope, 'env>,
    client: InferenceClient,
    warmup_games: usize,
    n_games: usize,
    n_threads: usize,
    seed: Option<u64>,
    game_config: GameConfig,
    mcts_config: &'env MCTSConfig,
    exploration: &'env AtomicUsize,
    graph_type: GraphType,
    prune_empty_edges: bool,
    threat_features: bool,
    playout_cap_fraction: f64,
    playout_cap_divisor: u32,
    running: &'env Arc<AtomicBool>,
) -> (Vec<GameResult>, Duration) {
    let games_per_thread = distribute(n_games, n_threads);
    let warmup_per_thread = if warmup_games > 0 {
        distribute(warmup_games, n_threads)
    } else {
        vec![0usize; n_threads]
    };
    // n_threads workers + 1 caller, so the caller can take the measurement
    // start time at the moment workers all begin measurement games.
    let barrier = Arc::new(std::sync::Barrier::new(n_threads + 1));

    let handles: Vec<_> = warmup_per_thread
        .into_iter()
        .zip(games_per_thread)
        .enumerate()
        .map(|(ti, (warmup_count, count))| {
            let client = client.clone();
            let running_thread = running.clone();
            let barrier = barrier.clone();
            s.spawn(move || {
                let mut rng = make_rng(seed, ti as u64, 0);
                // Warmup: play games but discard the results. Same `rng`
                // carries forward so measurement games continue the RNG
                // stream — they don't replay the warmup positions.
                for _ in 0..warmup_count {
                    if !running_thread.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = play_one_game(
                        &client, None, None, game_config, mcts_config,
                        exploration, graph_type, prune_empty_edges, threat_features, &mut rng,
                        playout_cap_fraction, playout_cap_divisor,
                        &running_thread,
                    );
                }
                barrier.wait();
                let mut results = Vec::with_capacity(count);
                for _ in 0..count {
                    if let Some(game) = play_one_game(
                        &client, None, None, game_config, mcts_config,
                        exploration, graph_type, prune_empty_edges, threat_features, &mut rng,
                        playout_cap_fraction, playout_cap_divisor,
                        &running_thread,
                    ) {
                        results.push(game);
                    }
                }
                play_perf_flush();
                results
            })
        })
        .collect();

    // Drop the original client so server sees disconnect after threads finish
    drop(client);

    // Block here until every worker has finished warmup. Use the
    // post-barrier instant as the measurement start.
    barrier.wait();
    let measurement_start = Instant::now();

    let games: Vec<GameResult> = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap_or_default())
        .collect();

    (games, measurement_start.elapsed())
}

/// Spawn continuous-mode game threads inside a thread scope.
/// Takes `client` and `pool_client_opt` by value; the originals are dropped
/// after all game-thread clones have been created, so the inference servers
/// see a disconnect once every game thread finishes.
#[allow(clippy::too_many_arguments)]
fn run_continuous_game_threads<'scope, 'env: 'scope>(
    s: &'scope std::thread::Scope<'scope, 'env>,
    client: InferenceClient,
    pool_client_opt: Option<InferenceClient>,
    n_threads: usize,
    seed: Option<u64>,
    game_config: GameConfig,
    mcts_config: &'env MCTSConfig,
    exploration_atomic: &'env AtomicUsize,
    graph_type: GraphType,
    prune_empty_edges: bool,
    threat_features: bool,
    playout_cap_fraction: f64,
    playout_cap_divisor: u32,
    pool_fraction: f64,
    pool_disabled: &'env Arc<AtomicBool>,
    pool_ready: &'env Arc<AtomicBool>,
    running: &'env Arc<AtomicBool>,
    game_counter: &'env Arc<AtomicU64>,
    rotating_writer: &'env Arc<Mutex<RotatingWriter>>,
    exploration_fraction: Option<f64>,
    ema_bits: &'env AtomicU64,
    p1_ema_bits: &'env AtomicU64,
    exploration_window: u32,
    exploration_exponent: f64,
    exploration_max: Option<f64>,
) {
    let start_time = std::time::Instant::now();

    for ti in 0..n_threads {
        let running = running.clone();
        let game_counter = game_counter.clone();
        let rotating_writer = rotating_writer.clone();
        let client = client.clone();
        let pool_client = pool_client_opt.clone();
        let pool_disabled_ref = pool_disabled.clone();
        let pool_ready_ref = pool_ready.clone();

        s.spawn(move || {
            let mut rng = make_rng(seed, ti as u64, 0);
            let mut last_logged_hundred: u64 = 0;
            const EMA_ALPHA: f64 = 0.05;

            while running.load(Ordering::Relaxed) {
                // Per-game pool routing
                let (pool_client_ref, pool_side): (Option<&InferenceClient>, Option<&'static str>) =
                    if let Some(ref pc) = pool_client {
                        if !pool_disabled_ref.load(Ordering::Relaxed)
                            && pool_ready_ref.load(Ordering::Relaxed)
                            && rng.random::<f64>() < pool_fraction
                        {
                            (Some(pc), Some(pick_pool_side(&mut rng)))
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };

                if let Some(result) = play_one_game(
                    &client, pool_client_ref, pool_side,
                    game_config, mcts_config,
                    exploration_atomic, graph_type, prune_empty_edges, threat_features, &mut rng,
                    playout_cap_fraction, playout_cap_divisor,
                    &running,
                ) {
                    let game_len = result.move_count as f64;

                    {
                        let mut rw = rotating_writer.lock().unwrap();
                        if let Err(e) = rw.write_game(&result) {
                            eprintln!("Thread {ti}: failed to write game: {e}");
                        }
                    }

                    let count = game_counter.fetch_add(1, Ordering::Relaxed) + 1;

                    // All threads: update game-length EMA via CAS loop
                    if exploration_fraction.is_some() {
                        loop {
                            let old_bits = ema_bits.load(Ordering::Relaxed);
                            let old_ema = f64::from_bits(old_bits);
                            let updated = if old_ema == 0.0 {
                                game_len
                            } else {
                                EMA_ALPHA * game_len + (1.0 - EMA_ALPHA) * old_ema
                            };
                            if ema_bits.compare_exchange_weak(
                                old_bits, updated.to_bits(),
                                Ordering::Relaxed, Ordering::Relaxed,
                            ).is_ok() {
                                break;
                            }
                        }

                        // Update p1 decided rate EMA (skip draws)
                        let p1_sample = match result.winner {
                            "P1" => Some(1.0f64),
                            "P2" => Some(0.0f64),
                            _ => None,
                        };
                        let p1_ema_alpha: f64 = 2.0 / (exploration_window as f64 + 1.0);
                        if let Some(sample) = p1_sample {
                            loop {
                                let old_bits = p1_ema_bits.load(Ordering::Relaxed);
                                let old_ema = f64::from_bits(old_bits);
                                let updated = p1_ema_alpha * sample + (1.0 - p1_ema_alpha) * old_ema;
                                if p1_ema_bits.compare_exchange_weak(
                                    old_bits, updated.to_bits(),
                                    Ordering::Relaxed, Ordering::Relaxed,
                                ).is_ok() {
                                    break;
                                }
                            }
                        }
                    }

                    // Thread 0: update exploration + progress logging
                    if ti == 0 {
                        if let Some(base_frac) = exploration_fraction {
                            let ema = f64::from_bits(ema_bits.load(Ordering::Relaxed));

                            let frac = if let Some(max_frac) = exploration_max {
                                let p1_ema = f64::from_bits(p1_ema_bits.load(Ordering::Relaxed));
                                let deviation = (2.0 * (p1_ema - 0.5)).abs().min(1.0);
                                (base_frac + (max_frac - base_frac) * deviation.powf(exploration_exponent))
                                    .clamp(base_frac, max_frac)
                            } else {
                                base_frac
                            };

                            let new_explore = (ema * frac).ceil() as usize;
                            let new_explore = new_explore.max(2);
                            let old_explore = exploration_atomic.swap(new_explore, Ordering::Relaxed);
                            if new_explore != old_explore {
                                if exploration_max.is_some() {
                                    let p1_ema = f64::from_bits(p1_ema_bits.load(Ordering::Relaxed));
                                    eprintln!(
                                        "exploration_moves: {} -> {} (ema_len={:.1}, frac={:.0}%, p1_rate={:.2})",
                                        old_explore, new_explore, ema, frac * 100.0, p1_ema,
                                    );
                                } else {
                                    eprintln!(
                                        "exploration_moves: {} -> {} (ema_len={:.1}, frac={:.0}%)",
                                        old_explore, new_explore, ema, frac * 100.0,
                                    );
                                }
                            }
                        }

                        let hundred = count / 100;
                        if hundred > last_logged_hundred {
                            last_logged_hundred = hundred;
                            let elapsed = start_time.elapsed().as_secs_f64();
                            let gps = count as f64 / elapsed.max(0.001);
                            eprintln!("{} games written ({:.1} games/s)", count, gps);
                        }
                    }
                }
            }
        });
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
        let (graph, _tensors) = build_position_graph(&game, graph_type, false, false);
        GameResult {
            positions: vec![
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph, sample_weight: 1.0 },
            ],
            winner,
            move_count: 1,
        }
    }

    #[test]
    fn hex_binary_roundtrip() {
        let result = make_result(GraphType::Hex, "P1");
        let mut buf = Vec::new();
        write_game_binary(&mut buf, &result, 0.0).unwrap();
        assert!(buf.len() > 17);
        // Check magic
        assert_eq!(&buf[..4], RECORD_MAGIC);
        assert_eq!(&buf[..4], b"HX04");
        // Read length prefix (after magic)
        let record_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        assert_eq!(record_len, buf.len() - 8); // total - magic - length
        // Read num_examples
        let num_examples = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(num_examples, 1);
        // Winner byte follows num_examples
        assert_eq!(buf[12] as i8, 1);
        // move_count u32 follows winner byte
        let move_count = u32::from_le_bytes(buf[13..17].try_into().unwrap());
        assert_eq!(move_count, 1);
    }

    #[test]
    fn winner_byte_roundtrip() {
        for (w, expected) in [("P1", 1i8), ("P2", -1i8), ("draw", 0i8)] {
            let result = make_result(GraphType::Hex, w);
            let mut buf = Vec::new();
            write_game_binary(&mut buf, &result, 0.0).unwrap();
            assert_eq!(&buf[..4], b"HX04");
            assert_eq!(buf[12] as i8, expected, "winner {w}");
            // Length prefix should match buffer minus magic+size header
            let record_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
            assert_eq!(record_len, buf.len() - 8);
        }
    }

    #[test]
    fn move_count_roundtrip() {
        // Simulate playout cap: 10 actual moves but only 3 recorded examples.
        let game = GameState::with_config(small_game_config());
        let (g1, _) = build_position_graph(&game, GraphType::Hex, false, false);
        let (g2, _) = build_position_graph(&game, GraphType::Hex, false, false);
        let (g3, _) = build_position_graph(&game, GraphType::Hex, false, false);
        let result = GameResult {
            positions: vec![
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph: g1, sample_weight: 1.0 },
                PositionData { policy: vec![0.5, 0.5], player: "P2", graph: g2, sample_weight: 1.0 },
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph: g3, sample_weight: 1.0 },
            ],
            winner: "P1",
            move_count: 10,
        };
        let mut buf = Vec::new();
        write_game_binary(&mut buf, &result, 0.0).unwrap();
        let num_examples = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(num_examples, 3, "examples == positions.len()");
        let move_count = u32::from_le_bytes(buf[13..17].try_into().unwrap());
        assert_eq!(move_count, 10, "move_count distinct from examples");
    }

    #[test]
    fn axis_msgpack_has_edge_attr() {
        let result = make_result(GraphType::Axis, "draw");
        let record = result.to_record(0.0);
        assert!(record.examples[0].edge_attr.is_some());
    }

    #[test]
    fn exploration_weights_improved_policy_default() {
        // Default (use_visit_counts=false): returns π' verbatim.
        let pi = vec![0.7, 0.2, 0.1];
        let visits = vec![30u32, 14, 6];
        let w = exploration_weights(&pi, &visits, false);
        assert_eq!(w, pi);
    }

    #[test]
    fn exploration_weights_visit_counts() {
        // use_visit_counts=true: returns visit counts cast to f64, NOT π'.
        // This is the broad Sequential-Halving staircase the paper samples.
        let pi = vec![0.99, 0.005, 0.005];
        let visits = vec![30u32, 14, 6];
        let w = exploration_weights(&pi, &visits, true);
        assert_eq!(w, vec![30.0, 14.0, 6.0]);
        // The two distributions are genuinely different: π' is near one-hot
        // while the visit-count weights spread mass across the survivors.
        let pi_top = pi[0] / pi.iter().sum::<f64>();
        let vc_top = w[0] / w.iter().sum::<f64>();
        assert!(pi_top > 0.9, "π' should be peaky, got {pi_top}");
        assert!(vc_top < 0.7, "visit counts should be broad, got {vc_top}");
    }

    #[test]
    fn value_targets_win() {
        let game = GameState::with_config(small_game_config());
        let (g1, _) = build_position_graph(&game, GraphType::Hex, false, false);
        let (g2, _) = build_position_graph(&game, GraphType::Hex, false, false);
        let result = GameResult {
            positions: vec![
                PositionData { policy: vec![0.5, 0.5], player: "P1", graph: g1, sample_weight: 1.0 },
                PositionData { policy: vec![0.3, 0.7], player: "P2", graph: g2, sample_weight: 1.0 },
            ],
            winner: "P1",
            move_count: 2,
        };
        let record = result.to_record(0.0);
        assert_eq!(record.examples[0].value, 1.0);  // P1 wins, P1's turn
        assert_eq!(record.examples[1].value, -1.0); // P1 wins, P2's turn
        assert_eq!(record.length, 2);
    }

    #[test]
    fn value_targets_draw_neutral() {
        let result = make_result(GraphType::Hex, "draw");
        let record = result.to_record(0.0);
        assert_eq!(record.examples[0].value, 0.0);
    }

    #[test]
    fn value_targets_draw_penalty() {
        let result = make_result(GraphType::Hex, "draw");
        let record = result.to_record(-0.1);
        assert!((record.examples[0].value - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn pick_pool_side_is_unbiased() {
        // Both sides should appear with roughly 50/50 probability over many trials.
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let mut p1 = 0usize;
        let n = 20_000;
        for _ in 0..n {
            if pick_pool_side(&mut rng) == "P1" {
                p1 += 1;
            }
        }
        let frac = p1 as f64 / n as f64;
        assert!(
            (0.47..0.53).contains(&frac),
            "pick_pool_side bias: P1 fraction = {frac:.4}"
        );
    }

    #[test]
    fn pick_pool_side_returns_only_p1_or_p2() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..100 {
            let s = pick_pool_side(&mut rng);
            assert!(s == "P1" || s == "P2");
        }
    }

    #[cfg(feature = "torch")]
    #[test]
    fn list_pool_snapshots_filters_pt_extension() {
        let tmp = std::env::temp_dir().join(format!("hexo_pool_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("a.pt"), b"x").unwrap();
        fs::write(tmp.join("b.pt"), b"y").unwrap();
        fs::write(tmp.join("ignore.txt"), b"z").unwrap();
        fs::write(tmp.join("README"), b"q").unwrap();
        let snaps = list_pool_snapshots(tmp.to_str().unwrap());
        assert_eq!(snaps.len(), 2);
        for p in &snaps {
            assert_eq!(p.extension().and_then(|e| e.to_str()), Some("pt"));
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn try_replace_drains_and_sends_on_full_channel() {
        // Capacity-1 channel: once full, try_replace must evict the stale
        // value (via the consumer draining) and send the new one. We model
        // the "consumer eventually drains" case: send, drain, send-again.
        let (tx, rx) = mpsc::sync_channel::<i32>(1);

        // First send into empty channel succeeds.
        try_replace(&tx, 1).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1);

        // Stage a value, leave it buffered, then replace it. The first
        // try_send in try_replace will observe Full; a single drain by the
        // consumer between attempts allows the second try_send to land.
        try_replace(&tx, 2).unwrap();
        // Consumer drains so the replacement can proceed via try_replace's
        // retry path.
        let stale = rx.recv().unwrap();
        assert_eq!(stale, 2);

        // Rapid back-to-back sends without drains: last one wins via the
        // fallback blocking send after consumer drains. We spawn a consumer
        // that drains once with a small delay to mimic the real inference
        // thread picking up the stale model.
        try_replace(&tx, 3).unwrap();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let a = rx.recv().unwrap();
            let b = rx.recv().unwrap();
            (a, b)
        });
        try_replace(&tx, 4).unwrap();
        let (a, b) = handle.join().unwrap();
        // Order must be 3 then 4 (the replacement semantics preserve "most
        // recent wins" but may stage up to one older value in flight).
        assert_eq!(a, 3);
        assert_eq!(b, 4);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn try_replace_errors_when_receiver_dropped() {
        let (tx, rx) = mpsc::sync_channel::<i32>(1);
        drop(rx);
        assert!(try_replace(&tx, 42).is_err());
    }

    #[cfg(feature = "torch")]
    #[test]
    fn list_pool_snapshots_missing_dir_returns_empty() {
        let snaps = list_pool_snapshots("/nonexistent/path/that/does/not/exist/abc123");
        assert!(snaps.is_empty());
    }

    #[test]
    fn pool_ready_flag_default_false_then_settable() {
        // The pool_ready flag is the public contract between the loader
        // and the worker threads: it must start false (so workers do not
        // route games to a pool inference server that has no model yet)
        // and flip true exactly once the loader has staged a snapshot.
        let pool_ready = Arc::new(AtomicBool::new(false));
        assert!(!pool_ready.load(Ordering::Relaxed));
        // Loader simulates a successful first stage:
        pool_ready.store(true, Ordering::Relaxed);
        assert!(pool_ready.load(Ordering::Relaxed));
    }

    #[cfg(feature = "torch")]
    #[test]
    fn list_pool_snapshots_empty_then_populated() {
        // Models the "pool dir empty at curriculum start, files appear
        // later" flow at the unit level: the loader caches the file list
        // but must re-scan when the cache is empty so a freshly-populated
        // dir is picked up on the next iteration without waiting for an
        // mtime observation race.
        let tmp = std::env::temp_dir().join(format!("hexo_pool_warmup_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let dir_str = tmp.to_str().unwrap();

        // Initially empty.
        let snaps = list_pool_snapshots(dir_str);
        assert!(snaps.is_empty(), "fresh dir should have no snapshots");

        // Trainer drops a checkpoint.
        fs::write(tmp.join("model_step_100.pt"), b"fake").unwrap();
        let snaps = list_pool_snapshots(dir_str);
        assert_eq!(snaps.len(), 1, "newly-written .pt should be visible");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn playout_cap_disabled_never_skips() {
        // fraction=0.0, divisor=1: feature off; should never skip.
        for r_int in 0..1000 {
            let r = r_int as f64 / 1000.0;
            assert!(!should_skip_example(r, 0.0, 1));
            assert!(!should_skip_example(r, 0.0, 4));
            assert!(!should_skip_example(r, 0.5, 1));
        }
    }

    #[test]
    fn playout_cap_active_always_skips_at_fraction_one() {
        // fraction=1.0, divisor=4: every roll < 1.0 → skip.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..1000 {
            let r = rng.random::<f64>();
            assert!(should_skip_example(r, 1.0, 4));
        }
    }

    #[test]
    fn playout_cap_statistical_match() {
        // fraction=0.5, divisor=4: ~half the rolls should skip.
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
        let n = 10_000;
        let mut skips = 0usize;
        for _ in 0..n {
            let r = rng.random::<f64>();
            if should_skip_example(r, 0.5, 4) {
                skips += 1;
            }
        }
        let frac = skips as f64 / n as f64;
        assert!(
            (0.47..0.53).contains(&frac),
            "playout_cap fraction skew: {frac:.4}"
        );
    }

    #[test]
    fn reduced_sim_config_uses_divisor() {
        let base = MCTSConfig { n_simulations: 64, m_actions: 16, c_visit: 50, c_scale: 1.0, virtual_loss: 0.5, ..Default::default() };
        assert_eq!(reduced_sim_config(&base, 4).n_simulations, 16);
        assert_eq!(reduced_sim_config(&base, 1).n_simulations, 64);
        assert_eq!(reduced_sim_config(&base, 0).n_simulations, 64); // clamp divisor to 1
        let tiny = MCTSConfig { n_simulations: 1, m_actions: 16, c_visit: 50, c_scale: 1.0, virtual_loss: 0.0, ..Default::default() };
        assert_eq!(reduced_sim_config(&tiny, 4).n_simulations, 1); // clamp result to 1
        // Other fields preserved.
        let r = reduced_sim_config(&base, 4);
        assert_eq!(r.m_actions, 16);
        assert_eq!(r.c_visit, 50);
        assert_eq!(r.c_scale, 1.0);
        assert_eq!(r.virtual_loss, 0.5);
    }

    #[test]
    fn json_output_has_expected_keys() {
        let result = make_result(GraphType::Hex, "P1");
        let json = game_to_json(&result, 0.0);
        let examples = json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 1);
        assert!(examples[0]["features"].is_array());
        assert!(examples[0]["policy"].is_array());
        assert_eq!(examples[0]["value"].as_f64().unwrap(), 1.0);
    }

    // -----------------------------------------------------------------------
    // InferenceClient round-robin tests
    //
    // These tests exercise the sender-side round-robin distribution
    // implemented by `InferenceClient` without requiring a real Python
    // subprocess. A stub batcher thread per worker consumes EvalRequests
    // and echoes a per-worker tag back via the request's response_tx.
    // -----------------------------------------------------------------------

    use std::sync::atomic::AtomicUsize;

    /// Build a synthetic `EvalRequest` carrying a single dummy graph whose
    /// `num_nodes` field encodes a caller-supplied sentinel. The stub
    /// batcher echoes this sentinel into the response so the client can
    /// verify it received the response intended for *its* request.
    fn dummy_graph_with_sentinel(sentinel: usize) -> GraphTensors {
        GraphTensors {
            features: Vec::new(),
            edge_src: Vec::new(),
            edge_dst: Vec::new(),
            edge_attr: None,
            legal_mask: Vec::new(),
            stone_mask: Vec::new(),
            legal_coords: Vec::new(),
            num_nodes: sentinel,
            num_edges: 0,
        }
    }

    /// Stub batcher: pull EvalRequests from `rx`, record how many it saw,
    /// and reply with a value vector tagged with `worker_id` for every
    /// graph in the request (so the caller can detect crosstalk). Exits
    /// when `running` is cleared or when all senders are dropped.
    fn spawn_stub_batcher(
        worker_id: usize,
        rx: ReqReceiver<EvalRequest>,
        running: Arc<AtomicBool>,
        seen: Arc<AtomicUsize>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(req) => {
                        seen.fetch_add(1, Ordering::Relaxed);
                        // Tag every result slot with (worker_id, sentinel_from_graph).
                        // values[i] = worker_id as f64 lets the caller verify
                        // routing; logits[i] is empty (game code wouldn't read
                        // it in this stubbed environment).
                        let n = req.graphs.len();
                        let logits: Vec<HashMap<Coord, f64>> =
                            (0..n).map(|_| HashMap::default()).collect();
                        let values: Vec<f64> = req.graphs.iter()
                            .map(|g| {
                                // Pack worker_id in the low 32 bits and the
                                // sentinel (num_nodes) in the high 32 bits so
                                // the test can decode both from a single f64.
                                let packed: u64 =
                                    (worker_id as u64) | ((g.num_nodes as u64) << 32);
                                packed as f64
                            })
                            .collect();
                        let _ = req.response_tx.send((logits, values));
                    }
                    Err(ReqRecvTimeoutError::Timeout) => {
                        if !running.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                    Err(ReqRecvTimeoutError::Disconnected) => break,
                }
            }
        })
    }

    #[test]
    fn test_inference_client_single_worker_unchanged() {
        // N=1: behaviour must be observationally identical to the previous
        // single-Sender InferenceClient. One stub batcher, a handful of
        // evaluate() calls, every one routes to worker 0.
        let running = Arc::new(AtomicBool::new(true));
        let seen = Arc::new(AtomicUsize::new(0));

        let (tx, rx) = req_unbounded::<EvalRequest>();
        let stub = spawn_stub_batcher(0, rx, running.clone(), seen.clone());

        let client = InferenceClient::new(vec![tx]);
        for sentinel in 100..105 {
            let (_logits, values) = client.evaluate(vec![dummy_graph_with_sentinel(sentinel)]);
            assert_eq!(values.len(), 1);
            let packed = values[0] as u64;
            let worker = (packed & 0xFFFF_FFFF) as usize;
            let echoed = (packed >> 32) as usize;
            assert_eq!(worker, 0, "N=1 must always route to worker 0");
            assert_eq!(echoed, sentinel, "sentinel must round-trip");
        }
        assert_eq!(seen.load(Ordering::Relaxed), 5);

        // Shutdown.
        drop(client);
        running.store(false, Ordering::Relaxed);
        stub.join().expect("stub batcher must exit");
    }

    #[test]
    fn test_inference_client_round_robin_n3() {
        // N=3: 6 sequential evaluate() calls must hit workers in a strict
        // round-robin pattern starting from 0: [0,1,2,0,1,2].
        let running = Arc::new(AtomicBool::new(true));

        let mut txs: Vec<ReqSender<EvalRequest>> = Vec::new();
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
        let mut seens: Vec<Arc<AtomicUsize>> = Vec::new();
        for wid in 0..3 {
            let (tx, rx) = req_unbounded::<EvalRequest>();
            let seen = Arc::new(AtomicUsize::new(0));
            handles.push(spawn_stub_batcher(wid, rx, running.clone(), seen.clone()));
            txs.push(tx);
            seens.push(seen);
        }

        let client = InferenceClient::new(txs);
        let mut observed_workers: Vec<usize> = Vec::new();
        for sentinel in 0..6 {
            let (_logits, values) = client.evaluate(vec![dummy_graph_with_sentinel(sentinel)]);
            let packed = values[0] as u64;
            let worker = (packed & 0xFFFF_FFFF) as usize;
            let echoed = (packed >> 32) as usize;
            assert_eq!(echoed, sentinel, "sentinel must round-trip on call {sentinel}");
            observed_workers.push(worker);
        }

        assert_eq!(observed_workers, vec![0, 1, 2, 0, 1, 2]);
        for (wid, s) in seens.iter().enumerate() {
            assert_eq!(s.load(Ordering::Relaxed), 2, "worker {wid} should have seen 2 requests");
        }

        drop(client);
        running.store(false, Ordering::Relaxed);
        for h in handles { h.join().expect("stub batcher must exit"); }
    }

    #[test]
    fn test_inference_client_concurrent_n4_no_crosstalk() {
        // 4 stub batchers; 16 client threads × 50 calls each. Each call
        // submits a unique sentinel and verifies the response echoes that
        // exact sentinel back. Any crosstalk between concurrent callers
        // would surface as a mismatched sentinel.
        const N_WORKERS: usize = 4;
        const N_THREADS: usize = 16;
        const N_CALLS: usize = 50;

        let running = Arc::new(AtomicBool::new(true));
        let mut txs: Vec<ReqSender<EvalRequest>> = Vec::new();
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
        for wid in 0..N_WORKERS {
            let (tx, rx) = req_unbounded::<EvalRequest>();
            let seen = Arc::new(AtomicUsize::new(0));
            handles.push(spawn_stub_batcher(wid, rx, running.clone(), seen));
            txs.push(tx);
        }

        let client = InferenceClient::new(txs);

        let mut callers: Vec<std::thread::JoinHandle<usize>> = Vec::new();
        for tid in 0..N_THREADS {
            let client = client.clone();
            callers.push(std::thread::spawn(move || {
                let mut local_ok = 0usize;
                for i in 0..N_CALLS {
                    // Unique sentinel per (tid, i).
                    let sentinel = tid * N_CALLS + i + 1;
                    let (_logits, values) =
                        client.evaluate(vec![dummy_graph_with_sentinel(sentinel)]);
                    assert_eq!(values.len(), 1);
                    let packed = values[0] as u64;
                    let echoed = (packed >> 32) as usize;
                    // Per-call sentinel must round-trip: detects any crosstalk
                    // (response delivered to wrong caller's response_tx).
                    assert_eq!(echoed, sentinel,
                        "tid={tid} call={i}: expected sentinel {sentinel}, got {echoed}");
                    local_ok += 1;
                }
                local_ok
            }));
        }

        let total_ok: usize = callers.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total_ok, N_THREADS * N_CALLS);

        drop(client);
        running.store(false, Ordering::Relaxed);
        for h in handles { h.join().expect("stub batcher must exit"); }
    }

    #[test]
    fn test_inference_client_shutdown_clean() {
        // Drop the client *and* clear running; all stub batcher threads
        // must exit within 500ms. Catches "leaked threads on shutdown"
        // regressions.
        const N_WORKERS: usize = 3;
        let running = Arc::new(AtomicBool::new(true));
        let mut txs: Vec<ReqSender<EvalRequest>> = Vec::new();
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
        for wid in 0..N_WORKERS {
            let (tx, rx) = req_unbounded::<EvalRequest>();
            let seen = Arc::new(AtomicUsize::new(0));
            handles.push(spawn_stub_batcher(wid, rx, running.clone(), seen));
            txs.push(tx);
        }
        let client = InferenceClient::new(txs);

        // Hand out a few clones, do some traffic, then drop everything.
        let c2 = client.clone();
        let _ = c2.evaluate(vec![dummy_graph_with_sentinel(1)]);
        drop(c2);
        let _ = client.evaluate(vec![dummy_graph_with_sentinel(2)]);
        drop(client);

        running.store(false, Ordering::Relaxed);

        // All stub batchers must exit promptly. We give a generous budget
        // (500ms) — the stub's recv_timeout is 50ms.
        let deadline = Instant::now() + Duration::from_millis(500);
        for (i, h) in handles.into_iter().enumerate() {
            // join() blocks; rely on the stub timing out and seeing running=false
            // OR the channel being disconnected (since we dropped the senders).
            // Either way, the thread must exit before the deadline.
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(remaining > Duration::ZERO, "worker {i} exceeded shutdown deadline");
            h.join().expect("stub batcher must exit cleanly");
        }
    }

    #[test]
    fn test_build_inference_channels_shapes() {
        // RoundRobin: one channel per worker (N senders, N receivers).
        let (txs, rxs) = build_inference_channels(3, DispatchMode::RoundRobin);
        assert_eq!(txs.len(), 3, "round-robin: one sender per worker");
        assert_eq!(rxs.len(), 3, "round-robin: one receiver per worker");

        // Shared: a single sender feeding N cloned receivers off one queue.
        let (txs, rxs) = build_inference_channels(3, DispatchMode::Shared);
        assert_eq!(txs.len(), 1, "shared: single sender");
        assert_eq!(rxs.len(), 3, "shared: one receiver clone per worker");

        // N=1 is identical under both modes.
        for mode in [DispatchMode::RoundRobin, DispatchMode::Shared] {
            let (txs, rxs) = build_inference_channels(1, mode);
            assert_eq!(txs.len(), 1);
            assert_eq!(rxs.len(), 1);
        }
    }

    #[test]
    fn test_shared_dispatch_pull_balances_no_crosstalk() {
        // Shared mode: 3 stub batchers pull from ONE queue (the client holds a
        // single sender). Every request must be served exactly once with its
        // sentinel intact (no crosstalk, no drops), and the load must spread
        // across more than one worker — the property round-robin can't
        // guarantee under skewed service time but a pull queue gives for free.
        const N_WORKERS: usize = 3;
        const N_THREADS: usize = 8;
        const N_CALLS: usize = 100;

        let running = Arc::new(AtomicBool::new(true));
        let (txs, rxs) = build_inference_channels(N_WORKERS, DispatchMode::Shared);
        assert_eq!(txs.len(), 1);

        let mut handles = Vec::new();
        let mut seens: Vec<Arc<AtomicUsize>> = Vec::new();
        for (wid, rx) in rxs.into_iter().enumerate() {
            let seen = Arc::new(AtomicUsize::new(0));
            handles.push(spawn_stub_batcher(wid, rx, running.clone(), seen.clone()));
            seens.push(seen);
        }

        let client = InferenceClient::new(txs);

        let mut callers: Vec<std::thread::JoinHandle<usize>> = Vec::new();
        for tid in 0..N_THREADS {
            let client = client.clone();
            callers.push(std::thread::spawn(move || {
                let mut ok = 0usize;
                for i in 0..N_CALLS {
                    let sentinel = tid * N_CALLS + i + 1;
                    let (_logits, values) =
                        client.evaluate(vec![dummy_graph_with_sentinel(sentinel)]);
                    let echoed = (values[0] as u64 >> 32) as usize;
                    assert_eq!(echoed, sentinel, "shared dispatch crosstalk/drop");
                    ok += 1;
                }
                ok
            }));
        }
        let total_ok: usize = callers.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total_ok, N_THREADS * N_CALLS, "every request served once");

        drop(client);
        running.store(false, Ordering::Relaxed);
        for h in handles { h.join().expect("stub batcher must exit"); }

        let total_seen: usize = seens.iter().map(|s| s.load(Ordering::Relaxed)).sum();
        assert_eq!(total_seen, N_THREADS * N_CALLS, "no requests lost");
        let workers_used = seens.iter().filter(|s| s.load(Ordering::Relaxed) > 0).count();
        assert!(workers_used > 1,
            "shared queue should spread load across workers, used {workers_used}");
    }
}
