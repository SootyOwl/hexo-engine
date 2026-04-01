//! TorchScript model inference for self-play.
//!
//! Loads a TorchScript-exported HeXO model and evaluates game states
//! entirely in Rust via tch-rs (libtorch bindings). No Python needed.

#![cfg(feature = "torch")]

use std::collections::HashMap;

use tch::{CModule, Device, Kind, Tensor};

use hexo_engine::types::{Coord, HEX_DIRS, Player};
use hexo_engine::hex::hex_distance;
use hexo_engine::GameState;

/// A loaded TorchScript model for inference.
pub struct TorchModel {
    model: CModule,
    device: Device,
    bf16: bool,
}

impl TorchModel {
    /// Load a TorchScript model from a file.
    ///
    /// Auto-detects bf16 models by checking the first parameter's dtype.
    pub fn load(path: &str, device: Device) -> Result<Self, tch::TchError> {
        let model = CModule::load_on_device(path, device)?;
        // Detect bf16 by checking the first named parameter's dtype
        let bf16 = model
            .named_parameters()
            .map(|params: Vec<(String, Tensor)>| {
                params.first().map(|(_, t)| t.kind() == Kind::BFloat16).unwrap_or(false)
            })
            .unwrap_or(false);
        if bf16 {
            eprintln!("Model loaded in bfloat16 mode");
        }
        Ok(TorchModel { model, device, bf16 })
    }

    /// Evaluate a batch of game states.
    ///
    /// Returns (logits_per_state, values) where:
    /// - logits_per_state[i] maps Coord → logit for state i
    /// - values[i] is the scalar value estimate
    pub fn evaluate(&self, states: &[GameState]) -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        let n = states.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        // Build batched graph tensors
        let graphs: Vec<GraphTensors> = states.iter().map(|s| build_graph_tensors(s)).collect();

        // Concatenate into batched tensors
        let mut total_nodes = 0usize;
        let mut total_edges = 0usize;
        for g in &graphs {
            total_nodes += g.num_nodes;
            total_edges += g.num_edges;
        }

        // Features: (total_nodes, 8)
        let mut all_features: Vec<f32> = Vec::with_capacity(total_nodes * 8);
        let mut all_edge_src: Vec<i64> = Vec::with_capacity(total_edges);
        let mut all_edge_dst: Vec<i64> = Vec::with_capacity(total_edges);
        let mut all_legal_mask: Vec<bool> = Vec::with_capacity(total_nodes);
        let mut all_stone_mask: Vec<bool> = Vec::with_capacity(total_nodes);
        let mut all_batch: Vec<i64> = Vec::with_capacity(total_nodes);

        let mut node_offset: i64 = 0;
        for (gi, g) in graphs.iter().enumerate() {
            all_features.extend_from_slice(&g.features);
            // Offset edge indices
            for &s in &g.edge_src {
                all_edge_src.push(s + node_offset);
            }
            for &d in &g.edge_dst {
                all_edge_dst.push(d + node_offset);
            }
            all_legal_mask.extend_from_slice(&g.legal_mask);
            all_stone_mask.extend_from_slice(&g.stone_mask);
            for _ in 0..g.num_nodes {
                all_batch.push(gi as i64);
            }
            node_offset += g.num_nodes as i64;
        }

        // Create tensors on device (convert to bf16 if model is bf16)
        let mut x = Tensor::from_slice(&all_features)
            .reshape([total_nodes as i64, 8])
            .to_device(self.device);
        if self.bf16 {
            x = x.to_kind(Kind::BFloat16);
        }
        let edge_index = Tensor::from_slice2(&[&all_edge_src, &all_edge_dst])
            .to_device(self.device);
        let legal_mask = Tensor::from_slice(
            &all_legal_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        ).to_kind(Kind::Bool).to_device(self.device);
        let stone_mask = Tensor::from_slice(
            &all_stone_mask.iter().map(|&b| b as i8).collect::<Vec<i8>>(),
        ).to_kind(Kind::Bool).to_device(self.device);
        let batch_tensor = Tensor::from_slice(&all_batch).to_device(self.device);
        let num_graphs = n as i64;

        // Run model via IValue interface (num_graphs is int, not Tensor)
        let _guard = tch::no_grad_guard();
        let result = match self.model.forward_is(&[
            tch::IValue::Tensor(x),
            tch::IValue::Tensor(edge_index),
            tch::IValue::Tensor(legal_mask),
            tch::IValue::Tensor(stone_mask),
            tch::IValue::Tensor(batch_tensor),
            tch::IValue::Int(num_graphs),
        ]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Model forward failed: {e}");
                let dummy_logits = states.iter().map(|_| HashMap::new()).collect();
                return (dummy_logits, vec![0.0; n]);
            }
        };

        // Parse tuple result — return dummy on unexpected format
        let (all_logits_tensor, legal_counts_tensor, values_tensor) = match result {
            tch::IValue::Tuple(ref parts) if parts.len() == 3 => {
                match (&parts[0], &parts[1], &parts[2]) {
                    (tch::IValue::Tensor(l), tch::IValue::Tensor(c), tch::IValue::Tensor(v)) => {
                        (l.shallow_clone(), c.shallow_clone(), v.shallow_clone())
                    }
                    _ => {
                        eprintln!("Unexpected tensor types in model output");
                        let dummy = states.iter().map(|_| HashMap::new()).collect();
                        return (dummy, vec![0.0; n]);
                    }
                }
            }
            _ => {
                eprintln!("Unexpected model output format");
                let dummy = states.iter().map(|_| HashMap::new()).collect();
                return (dummy, vec![0.0; n]);
            }
        };

        // Extract values
        let values_vec: Vec<f64> = match Vec::<f64>::try_from(values_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract values: {e}");
                return (states.iter().map(|_| HashMap::new()).collect(), vec![0.0; n]);
            }
        };

        // Extract logits per graph
        let counts_vec: Vec<i64> = match Vec::<i64>::try_from(legal_counts_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract counts: {e}");
                return (states.iter().map(|_| HashMap::new()).collect(), vec![0.0; n]);
            }
        };
        let all_logits_vec: Vec<f64> = match Vec::<f64>::try_from(all_logits_tensor.to_device(Device::Cpu)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to extract logits: {e}");
                return (states.iter().map(|_| HashMap::new()).collect(), vec![0.0; n]);
            }
        };

        let mut logits_maps = Vec::with_capacity(n);
        let mut offset = 0usize;
        for (gi, g) in graphs.iter().enumerate() {
            let count = counts_vec[gi] as usize;
            let logits_slice = &all_logits_vec[offset..offset + count];
            let legal_coords = &g.legal_coords;
            let map: HashMap<Coord, f64> = legal_coords
                .iter()
                .zip(logits_slice)
                .map(|(&c, &l)| (c, l))
                .collect();
            logits_maps.push(map);
            offset += count;
        }

        (logits_maps, values_vec)
    }
}

/// Raw graph tensors for one game state.
struct GraphTensors {
    features: Vec<f32>,    // N*8 flat
    edge_src: Vec<i64>,
    edge_dst: Vec<i64>,
    legal_mask: Vec<bool>,
    stone_mask: Vec<bool>,
    legal_coords: Vec<Coord>,
    num_nodes: usize,
    num_edges: usize,
}

/// Build graph tensors from a game state (mirrors graph.rs::build_graph).
fn build_graph_tensors(game: &GameState) -> GraphTensors {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(coord, _)| coord);
    let legal = game.legal_moves();
    let n_stones = stones.len();
    let n_legal = legal.len();
    let n = n_stones + n_legal;

    // Coords
    let mut coords: Vec<Coord> = Vec::with_capacity(n);
    for &(coord, _) in &stones {
        coords.push(coord);
    }
    for &c in &legal {
        coords.push(c);
    }

    // Global features
    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;

    // Centroid
    let (centroid_q, centroid_r, spread) = if n_stones > 0 {
        let sum_q: f64 = stones.iter().map(|&((q, _), _)| q as f64).sum();
        let sum_r: f64 = stones.iter().map(|&((_, r), _)| r as f64).sum();
        let cq = sum_q / n_stones as f64;
        let cr = sum_r / n_stones as f64;
        let max_dev = stones
            .iter()
            .map(|&((q, r), _)| (q as f64 - cq).abs().max((r as f64 - cr).abs()))
            .fold(0.0f64, f64::max);
        (cq, cr, max_dev.max(1.0))
    } else {
        (0.0, 0.0, 1.0)
    };

    let stone_coords: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();

    // Features
    let mut features = vec![0.0f32; n * 8];
    for (i, &(_, player)) in stones.iter().enumerate() {
        let base = i * 8;
        match player {
            Player::P1 => features[base] = 1.0,
            Player::P2 => features[base + 1] = 1.0,
        }
        features[base + 3] = player_feat;
        features[base + 4] = moves_feat;
        let q = coords[i].0 as f64;
        let r = coords[i].1 as f64;
        features[base + 5] = ((q - centroid_q) / spread) as f32;
        features[base + 6] = ((r - centroid_r) / spread) as f32;
    }
    for j in 0..n_legal {
        let idx = n_stones + j;
        let base = idx * 8;
        features[base + 2] = 1.0;
        features[base + 3] = player_feat;
        features[base + 4] = moves_feat;
        let q = coords[idx].0 as f64;
        let r = coords[idx].1 as f64;
        features[base + 5] = ((q - centroid_q) / spread) as f32;
        features[base + 6] = ((r - centroid_r) / spread) as f32;

        let coord = legal[j];
        let min_d = stone_coords
            .iter()
            .map(|&sc| hex_distance(coord, sc))
            .min()
            .unwrap_or(1);
        features[base + 7] = 1.0 / min_d.max(1) as f32;
    }

    // Edges
    let mut coord_to_idx: HashMap<Coord, usize> = HashMap::with_capacity(n);
    for (i, &c) in coords.iter().enumerate() {
        coord_to_idx.insert(c, i);
    }

    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    for (i, &c) in coords.iter().enumerate() {
        for &(dq, dr) in &HEX_DIRS {
            if let Some(&j) = coord_to_idx.get(&(c.0 + dq, c.1 + dr)) {
                edge_src.push(i as i64);
                edge_dst.push(j as i64);
            }
        }
    }
    let num_edges = edge_src.len();

    // Masks
    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for i in 0..n_stones {
        stone_mask[i] = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }

    GraphTensors {
        features,
        edge_src,
        edge_dst,
        legal_mask,
        stone_mask,
        legal_coords: legal,
        num_nodes: n,
        num_edges,
    }
}
