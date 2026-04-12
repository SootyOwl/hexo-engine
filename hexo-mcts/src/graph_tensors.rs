//! `GraphTensors` — raw graph data for one game state, without any `tch` dependency.
//!
//! This module is intentionally outside the `torch` feature gate so that
//! future consumers (e.g. `SubprocessModel`) can use it without linking libtorch.

use std::collections::HashMap;

use hexo_engine::types::{Coord, HEX_DIRS, Player};
use hexo_engine::hex::hex_distance;
use hexo_engine::GameState;

use crate::axis_graph::game_to_axis_graph_raw;

/// Graph type for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    Hex,
    Axis,
}

/// Raw graph tensors for one game state.
pub struct GraphTensors {
    pub features: Vec<f32>,    // N*8 flat
    pub edge_src: Vec<i64>,
    pub edge_dst: Vec<i64>,
    pub edge_attr: Option<Vec<f32>>,  // E*5 flat, only for axis graphs
    pub legal_mask: Vec<bool>,
    pub stone_mask: Vec<bool>,
    pub legal_coords: Vec<Coord>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl GraphTensors {
    /// Create from a hex graph reference (clones data).
    pub fn from_hex(g: &crate::graph::GraphData) -> Self {
        let legal_coords: Vec<Coord> = g.legal_mask.iter()
            .enumerate()
            .filter(|&(_, &is_legal)| is_legal)
            .map(|(i, _)| (g.coords[i * 2], g.coords[i * 2 + 1]))
            .collect();
        let num_edges = g.edge_src.len();
        GraphTensors {
            features: g.features.clone(),
            edge_src: g.edge_src.clone(),
            edge_dst: g.edge_dst.clone(),
            edge_attr: None,
            legal_mask: g.legal_mask.clone(),
            stone_mask: g.stone_mask.clone(),
            legal_coords,
            num_nodes: g.num_nodes,
            num_edges,
        }
    }

    /// Create from an axis graph reference (clones data).
    pub fn from_axis(g: &crate::axis_graph::AxisGraphData) -> Self {
        let legal_coords: Vec<Coord> = g.legal_mask.iter()
            .enumerate()
            .filter(|&(_, &is_legal)| is_legal)
            .map(|(i, _)| (g.coords[i * 2], g.coords[i * 2 + 1]))
            .collect();
        let num_edges = g.edge_src.len();
        GraphTensors {
            features: g.features.clone(),
            edge_src: g.edge_src.clone(),
            edge_dst: g.edge_dst.clone(),
            edge_attr: Some(g.edge_attr.clone()),
            legal_mask: g.legal_mask.clone(),
            stone_mask: g.stone_mask.clone(),
            legal_coords,
            num_nodes: g.num_nodes,
            num_edges,
        }
    }
}

impl From<crate::graph::GraphData> for GraphTensors {
    fn from(g: crate::graph::GraphData) -> Self {
        Self::from_hex(&g)
    }
}

impl From<crate::axis_graph::AxisGraphData> for GraphTensors {
    fn from(g: crate::axis_graph::AxisGraphData) -> Self {
        Self::from_axis(&g)
    }
}

/// Build graph tensors from a game state (mirrors graph.rs::build_graph).
pub fn build_graph_tensors(game: &GameState) -> GraphTensors {
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
        edge_attr: None,
        legal_mask,
        stone_mask,
        legal_coords: legal,
        num_nodes: n,
        num_edges,
    }
}

/// Build graph tensors from a game state using axis-window graph construction.
pub fn build_axis_graph_tensors(game: &GameState) -> GraphTensors {
    let axis_data = game_to_axis_graph_raw(game);

    // Extract legal_coords from coords + legal_mask
    let legal_coords: Vec<Coord> = axis_data.legal_mask
        .iter()
        .enumerate()
        .filter(|&(_, &is_legal)| is_legal)
        .map(|(i, _)| (axis_data.coords[i * 2], axis_data.coords[i * 2 + 1]))
        .collect();

    let num_edges = axis_data.edge_src.len();

    GraphTensors {
        features: axis_data.features,
        edge_src: axis_data.edge_src,
        edge_dst: axis_data.edge_dst,
        edge_attr: Some(axis_data.edge_attr),
        legal_mask: axis_data.legal_mask,
        stone_mask: axis_data.stone_mask,
        legal_coords,
        num_nodes: axis_data.num_nodes,
        num_edges,
    }
}
