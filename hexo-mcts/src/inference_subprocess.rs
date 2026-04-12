//! `SubprocessModel` — spawns a Python inference subprocess and communicates
//! via a binary protocol over stdin/stdout.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write as _};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

use hexo_engine::types::Coord;

use crate::graph_tensors::GraphTensors;

const MAGIC: u32 = 0x48583034;
const VERSION: u8 = 1;
const MSG_FORWARD: u8 = 0x01;
const MSG_RELOAD: u8 = 0x02;
const MSG_SHUTDOWN: u8 = 0xFF;

pub struct SubprocessModel {
    child: Child,
    #[allow(dead_code)]
    model_args: Vec<String>,
    #[allow(dead_code)]
    python_bin: String,
    model_mtime: Option<SystemTime>,
    stderr_handle: Option<std::thread::JoinHandle<()>>,
}

impl SubprocessModel {
    /// Spawn the Python inference subprocess.
    ///
    /// Waits up to 600 seconds for the "READY" signal on stderr (torch.compile
    /// can take minutes on first run).
    pub fn spawn(python_bin: &str, model_path: &str, model_args: &[String]) -> Result<Self, String> {
        let mut child = Command::new(python_bin)
            .args(["-m", "hexo_a0.inference_server", "--checkpoint", model_path])
            .args(model_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn inference subprocess: {e}"))?;

        let stderr = child.stderr.take().expect("stderr was piped");

        // Wait for READY on stderr using thread + channel pattern.
        let (tx, rx) = mpsc::channel();
        let startup_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,  // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        eprintln!("[python] {trimmed}");
                        if trimmed == "READY" {
                            let _ = tx.send(Ok(reader));
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("stderr read error: {e}")));
                        return;
                    }
                }
            }
            let _ = tx.send(Err("subprocess exited before sending READY".into()));
        });

        let reader = match rx.recv_timeout(Duration::from_secs(600)) {
            Ok(Ok(reader)) => reader,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                // Timeout — kill the child and the thread
                let _ = child.kill();
                let _ = startup_thread.join();
                return Err("timed out waiting for READY from inference subprocess (600s)".into());
            }
        };

        // Spawn stderr drain thread
        let stderr_handle = std::thread::spawn(move || {
            let mut reader = reader;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => eprintln!("[python] {}", line.trim()),
                    Err(_) => break,
                }
            }
        });

        let model_mtime = std::fs::metadata(model_path)
            .and_then(|m| m.modified())
            .ok();

        Ok(SubprocessModel {
            child,
            model_args: model_args.to_vec(),
            python_bin: python_bin.to_string(),
            model_mtime,
            stderr_handle: Some(stderr_handle),
        })
    }

    /// Send a batch of graphs for inference and return (policy, value) results.
    ///
    /// Policy is returned as a map from `Coord` to logit for each graph.
    pub fn forward_graphs(
        &mut self,
        graphs: Vec<GraphTensors>,
    ) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
        if !self.is_alive() {
            return Err("inference subprocess is not running".into());
        }

        let num_graphs = graphs.len() as u32;
        let mut total_nodes: u32 = 0;
        let mut total_edges: u32 = 0;
        let has_edge_attr = graphs.first().map_or(false, |g| g.edge_attr.is_some());

        for g in &graphs {
            total_nodes += g.num_nodes as u32;
            total_edges += g.num_edges as u32;
        }

        // --- Serialize request to stdin ---
        let stdin = self.child.stdin.as_mut().expect("stdin was piped");
        let mut w = BufWriter::new(stdin);

        // Header
        w.write_all(&MAGIC.to_le_bytes()).map_err(|e| format!("write error: {e}"))?;
        w.write_all(&[VERSION, MSG_FORWARD]).map_err(|e| format!("write error: {e}"))?;
        w.write_all(&total_nodes.to_le_bytes()).map_err(|e| format!("write error: {e}"))?;
        w.write_all(&total_edges.to_le_bytes()).map_err(|e| format!("write error: {e}"))?;
        w.write_all(&num_graphs.to_le_bytes()).map_err(|e| format!("write error: {e}"))?;
        w.write_all(&[has_edge_attr as u8]).map_err(|e| format!("write error: {e}"))?;

        // Features: f32[total_nodes * 8]
        for g in &graphs {
            for &f in &g.features {
                w.write_all(&f.to_le_bytes()).map_err(|e| format!("write error: {e}"))?;
            }
        }

        // Edge indices with node offsets applied
        let mut node_offset: i64 = 0;
        for g in &graphs {
            for &src in &g.edge_src {
                w.write_all(&(src + node_offset).to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }
            node_offset += g.num_nodes as i64;
        }

        node_offset = 0;
        for g in &graphs {
            for &dst in &g.edge_dst {
                w.write_all(&(dst + node_offset).to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }
            node_offset += g.num_nodes as i64;
        }

        // Edge attr (if present)
        if has_edge_attr {
            for g in &graphs {
                if let Some(ref ea) = g.edge_attr {
                    for &f in ea {
                        w.write_all(&f.to_le_bytes())
                            .map_err(|e| format!("write error: {e}"))?;
                    }
                }
            }
        }

        // Legal mask
        for g in &graphs {
            for &m in &g.legal_mask {
                w.write_all(&[m as u8]).map_err(|e| format!("write error: {e}"))?;
            }
        }

        // Stone mask
        for g in &graphs {
            for &m in &g.stone_mask {
                w.write_all(&[m as u8]).map_err(|e| format!("write error: {e}"))?;
            }
        }

        // Batch indices
        for (batch_idx, g) in graphs.iter().enumerate() {
            let idx = batch_idx as i32;
            for _ in 0..g.num_nodes {
                w.write_all(&idx.to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }
        }

        w.flush().map_err(|e| format!("flush error: {e}"))?;

        // --- Read response from stdout ---
        let stdout = self.child.stdout.as_mut().expect("stdout was piped");

        let resp_magic = read_u32_le(stdout)?;
        if resp_magic != MAGIC {
            return Err(format!("bad magic in response: 0x{resp_magic:08X}"));
        }
        let resp_ver = read_u8(stdout)?;
        if resp_ver != VERSION {
            return Err(format!("bad version in response: {resp_ver}"));
        }
        let resp_type = read_u8(stdout)?;
        if resp_type != MSG_FORWARD {
            return Err(format!("unexpected response type: 0x{resp_type:02X}"));
        }

        let total_legal = read_u32_le(stdout)? as usize;
        let resp_num_graphs = read_u32_le(stdout)? as usize;
        if resp_num_graphs != graphs.len() {
            return Err(format!(
                "graph count mismatch: sent {}, got {resp_num_graphs}",
                graphs.len()
            ));
        }

        // Logits for all legal moves
        let mut logits = vec![0.0f32; total_legal];
        read_f32_slice(stdout, &mut logits)?;

        // Legal counts per graph
        let mut legal_counts = vec![0i32; resp_num_graphs];
        read_i32_slice(stdout, &mut legal_counts)?;

        // Values per graph
        let mut values = vec![0.0f32; resp_num_graphs];
        read_f32_slice(stdout, &mut values)?;

        // Map logits back to coordinates
        let mut policies = Vec::with_capacity(resp_num_graphs);
        let mut logit_offset = 0usize;
        for (i, g) in graphs.iter().enumerate() {
            let count = legal_counts[i] as usize;
            let mut policy = HashMap::with_capacity(count);
            for j in 0..count {
                let coord = g.legal_coords[j];
                policy.insert(coord, logits[logit_offset + j] as f64);
            }
            logit_offset += count;
            policies.push(policy);
        }

        let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();

        Ok((policies, values_f64))
    }

    /// Try to reload the model checkpoint if the file has been modified.
    /// Returns true if a reload was performed and acknowledged.
    pub fn try_reload(&mut self, path: &str) -> bool {
        let new_mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if self.model_mtime == Some(new_mtime) {
            return false;
        }

        if !self.is_alive() {
            return false;
        }

        // Send reload message
        let stdin = match self.child.stdin.as_mut() {
            Some(s) => s,
            None => return false,
        };
        let mut w = BufWriter::new(stdin);
        let path_bytes = path.as_bytes();
        let path_len = path_bytes.len() as u32;

        if w.write_all(&MAGIC.to_le_bytes()).is_err()
            || w.write_all(&[VERSION, MSG_RELOAD]).is_err()
            || w.write_all(&path_len.to_le_bytes()).is_err()
            || w.write_all(path_bytes).is_err()
            || w.flush().is_err()
        {
            return false;
        }

        // Read ACK
        let stdout = match self.child.stdout.as_mut() {
            Some(s) => s,
            None => return false,
        };

        let magic = match read_u32_le(stdout) {
            Ok(m) => m,
            Err(_) => return false,
        };
        if magic != MAGIC {
            return false;
        }
        let ver = match read_u8(stdout) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ver != VERSION {
            return false;
        }
        let msg_type = match read_u8(stdout) {
            Ok(t) => t,
            Err(_) => return false,
        };
        if msg_type != MSG_RELOAD {
            return false;
        }
        let success = match read_u8(stdout) {
            Ok(s) => s,
            Err(_) => return false,
        };

        if success != 0 {
            self.model_mtime = Some(new_mtime);
            true
        } else {
            false
        }
    }

    /// Check if the subprocess is still alive.
    fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false,  // exited
            Ok(None) => true,      // still running
            Err(_) => false,
        }
    }
}

impl Drop for SubprocessModel {
    fn drop(&mut self) {
        // Send shutdown message
        if let Some(stdin) = self.child.stdin.as_mut() {
            let mut w = BufWriter::new(stdin);
            let _ = w.write_all(&MAGIC.to_le_bytes());
            let _ = w.write_all(&[VERSION, MSG_SHUTDOWN]);
            let _ = w.flush();
        }
        // Drop stdin to signal EOF
        self.child.stdin.take();

        // Wait up to 5 seconds for graceful exit
        let start = std::time::Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(5) {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }

        // Join the stderr drain thread
        if let Some(handle) = self.stderr_handle.take() {
            let _ = handle.join();
        }
    }
}

// --- Wire-format reading helpers ---

fn read_exact(r: &mut impl std::io::Read, buf: &mut [u8]) -> Result<(), String> {
    r.read_exact(buf).map_err(|e| format!("read error: {e}"))
}

fn read_u8(r: &mut impl std::io::Read) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    read_exact(r, &mut buf)?;
    Ok(buf[0])
}

fn read_u32_le(r: &mut impl std::io::Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_slice(r: &mut impl std::io::Read, out: &mut [f32]) -> Result<(), String> {
    // Read as raw bytes, then convert
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn read_i32_slice(r: &mut impl std::io::Read, out: &mut [i32]) -> Result<(), String> {
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = i32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}
