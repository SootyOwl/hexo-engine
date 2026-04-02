//! Axis-window graph construction: connect nodes along the 3 hex axes within
//! a distance window, with edge features.
//!
//! Unlike `graph.rs` (hex-adjacency, distance 1 only), this builder creates
//! edges along each WIN_AXIS direction up to `win_length - 1` steps, plus a
//! global dummy node connected to all other nodes.

use std::collections::HashMap;

use hexo_engine::hex::hex_distance;
use hexo_engine::types::{Coord, Player, WIN_AXES};
use hexo_engine::GameState;

/// Raw axis-window graph data ready to be wrapped into PyG Data on the Python side.
pub struct AxisGraphData {
    /// Node features, flattened: (N+1)×8 row-major (includes dummy node).
    pub features: Vec<f32>,
    /// Edge source indices.
    pub edge_src: Vec<i64>,
    /// Edge destination indices.
    pub edge_dst: Vec<i64>,
    /// Edge attributes, flattened: E×5 row-major.
    pub edge_attr: Vec<f32>,
    /// Legal move mask (true for legal-move nodes).
    pub legal_mask: Vec<bool>,
    /// Stone mask (true for placed-stone nodes).
    pub stone_mask: Vec<bool>,
    /// Coordinates, flattened: (N+1)×2 row-major (q, r).
    pub coords: Vec<i32>,
    /// Number of nodes (N+1, includes dummy).
    pub num_nodes: usize,
}

/// Node classification for walk-stopping logic.
#[derive(Clone, Copy)]
enum NodeKind {
    Stone(Player),
    Empty,
}

/// Build axis-window graph arrays from pre-sorted stones and legal moves.
///
/// `stones` and `legal` must already be sorted by coord tuple (q, r).
fn build_axis_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
    win_length: u8,
) -> AxisGraphData {
    debug_assert!(win_length >= 2, "win_length must be >= 2, got {win_length}");
    let n_stones = stones.len();
    let n_legal = legal.len();
    let n_real = n_stones + n_legal; // nodes without dummy
    let n = n_real + 1; // total nodes including dummy
    let dummy_idx = n_real;
    let window = win_length as i32 - 1;

    // --- Coordinate arrays and lookup maps ---

    let mut coords = Vec::with_capacity(n * 2);
    let mut coord_to_idx: HashMap<Coord, usize> = HashMap::with_capacity(n_real);
    let mut node_kind: Vec<NodeKind> = Vec::with_capacity(n);

    for (i, &(coord, player)) in stones.iter().enumerate() {
        coords.push(coord.0);
        coords.push(coord.1);
        coord_to_idx.insert(coord, i);
        node_kind.push(NodeKind::Stone(player));
    }
    for (j, &coord) in legal.iter().enumerate() {
        let idx = n_stones + j;
        coords.push(coord.0);
        coords.push(coord.1);
        coord_to_idx.insert(coord, idx);
        node_kind.push(NodeKind::Empty);
    }
    // Dummy node coords
    coords.push(0);
    coords.push(0);
    // node_kind for dummy is not used directly; we handle dummy separately.

    // --- Stone centroid and spread ---

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

    // --- Node features (N+1)×8 ---

    let mut features = vec![0.0f32; n * 8];

    // Stone features
    for (i, &(_, player)) in stones.iter().enumerate() {
        let base = i * 8;
        match player {
            Player::P1 => features[base] = 1.0,
            Player::P2 => features[base + 1] = 1.0,
        }
        features[base + 3] = player_feat;
        features[base + 4] = moves_feat;
        let q = coords[i * 2] as f64;
        let r = coords[i * 2 + 1] as f64;
        features[base + 5] = ((q - centroid_q) / spread) as f32;
        features[base + 6] = ((r - centroid_r) / spread) as f32;
        // features[base + 7] = 0.0 (stones)
    }

    // Legal move features
    for j in 0..n_legal {
        let idx = n_stones + j;
        let base = idx * 8;
        features[base + 2] = 1.0; // empty one-hot
        features[base + 3] = player_feat;
        features[base + 4] = moves_feat;
        let q = coords[idx * 2] as f64;
        let r = coords[idx * 2 + 1] as f64;
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

    // Dummy node features: [0,0,0, player_feat, moves_feat, 0, 0, 0]
    let dummy_base = dummy_idx * 8;
    features[dummy_base + 3] = player_feat;
    features[dummy_base + 4] = moves_feat;

    // --- Masks ---

    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for i in 0..n_stones {
        stone_mask[i] = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }
    // Dummy: both false (default)

    // --- Axis-window edges ---

    let mut edge_src: Vec<i64> = Vec::new();
    let mut edge_dst: Vec<i64> = Vec::new();
    let mut edge_attr: Vec<f32> = Vec::new();

    // For each node i, for each of 3 axes, walk positive direction
    for i in 0..n_real {
        let iq = coords[i * 2];
        let ir = coords[i * 2 + 1];
        let i_kind = node_kind[i];

        for (axis_idx, &(dq, dr)) in WIN_AXES.iter().enumerate() {
            for d in 1..=window {
                let tq = iq + dq * d;
                let tr = ir + dr * d;

                let j = match coord_to_idx.get(&(tq, tr)) {
                    Some(&j) => j,
                    None => continue, // no node at this coord, keep walking
                };

                let j_kind = node_kind[j];

                // Compute same-color flag
                let same_color = match (i_kind, j_kind) {
                    (NodeKind::Stone(pi), NodeKind::Stone(pj)) => {
                        if pi == pj { 1.0 } else { -1.0 }
                    }
                    _ => 0.0,
                };

                // Add edge (i, j)
                edge_src.push(i as i64);
                edge_dst.push(j as i64);
                let mut attr_fwd = [0.0f32; 5];
                attr_fwd[axis_idx] = 1.0;
                attr_fwd[3] = d as f32;
                attr_fwd[4] = same_color;
                edge_attr.extend_from_slice(&attr_fwd);

                // Add edge (j, i)
                edge_src.push(j as i64);
                edge_dst.push(i as i64);
                let mut attr_rev = [0.0f32; 5];
                attr_rev[axis_idx] = 1.0;
                attr_rev[3] = -(d as f32);
                attr_rev[4] = same_color;
                edge_attr.extend_from_slice(&attr_rev);

                // Decide whether to stop walking
                let should_stop = match i_kind {
                    NodeKind::Stone(pi) => {
                        // Stop at opponent stone
                        matches!(j_kind, NodeKind::Stone(pj) if pj != pi)
                    }
                    NodeKind::Empty => {
                        // Stop at any stone
                        matches!(j_kind, NodeKind::Stone(_))
                    }
                };

                if should_stop {
                    break;
                }
            }
        }
    }

    // --- Dummy edges: bidirectional to all real nodes ---

    for i in 0..n_real {
        // dummy -> i
        edge_src.push(dummy_idx as i64);
        edge_dst.push(i as i64);
        edge_attr.extend_from_slice(&[0.0; 5]);

        // i -> dummy
        edge_src.push(i as i64);
        edge_dst.push(dummy_idx as i64);
        edge_attr.extend_from_slice(&[0.0; 5]);
    }

    AxisGraphData {
        features,
        edge_src,
        edge_dst,
        edge_attr,
        legal_mask,
        stone_mask,
        coords,
        num_nodes: n,
    }
}

/// Build axis-window graph arrays from a GameState.
pub fn game_to_axis_graph_raw(game: &GameState) -> AxisGraphData {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(coord, _)| coord);
    let legal = game.legal_moves(); // already sorted by engine

    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;
    let win_length = game.config().win_length;

    build_axis_graph(&stones, &legal, player_feat, moves_feat, win_length)
}

/// Build axis-window graph arrays for a batch of game states in parallel.
pub fn game_to_axis_graph_batch(games: &[GameState]) -> Vec<AxisGraphData> {
    use rayon::prelude::*;
    games.par_iter().map(game_to_axis_graph_raw).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{GameConfig, GameState};
    use std::collections::{HashMap, HashSet};

    fn small_game() -> GameState {
        GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        })
    }

    /// Apply moves to create a mid-game state with stones from both players.
    /// P1 starts at (0,0). P2 gets 2 moves, then alternating 2 each.
    fn mid_game() -> GameState {
        let mut game = small_game();
        // P2 turn (2 moves)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((-1, 0)).unwrap();
        // P1 turn (2 moves)
        game.apply_move((0, 1)).unwrap();
        game.apply_move((0, -1)).unwrap();
        // P2 turn (2 moves)
        game.apply_move((1, -1)).unwrap();
        game.apply_move((-1, 1)).unwrap();
        // P1 turn (2 moves)
        game.apply_move((2, 0)).unwrap();
        game.apply_move((-2, 0)).unwrap();
        // P2 turn (2 moves)
        game.apply_move((0, 2)).unwrap();
        game.apply_move((0, -2)).unwrap();
        game
    }

    // AC-1a: num_nodes == n_stones + n_legal + 1
    #[test]
    fn ac1a_initial_position_node_count() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        let n_stones = game.placed_stones().len();
        let n_legal = game.legal_moves().len();
        assert_eq!(g.num_nodes, n_stones + n_legal + 1);
        assert_eq!(g.features.len(), g.num_nodes * 8);
        assert_eq!(g.coords.len(), g.num_nodes * 2);
    }

    // AC-1b: All axis edges have |signed_distance| <= win_length - 1
    #[test]
    fn ac1b_signed_distance_within_window() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let wl = game.config().win_length as f32;
        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let dist = g.edge_attr[e * 5 + 3];
            assert!(
                dist.abs() <= wl - 1.0,
                "edge {e}: |signed_distance| = {} > win_length - 1 = {}",
                dist.abs(),
                wl - 1.0
            );
        }
    }

    // AC-1c: edge_attr is 5-dim per edge
    #[test]
    fn ac1c_edge_attr_length() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        assert_eq!(g.edge_attr.len(), g.edge_src.len() * 5);
    }

    // AC-1d: Axis one-hot sums to 0.0 (dummy) or 1.0 (axis edges)
    #[test]
    fn ac1d_axis_one_hot_valid() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;
        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let oh_sum: f32 =
                g.edge_attr[e * 5] + g.edge_attr[e * 5 + 1] + g.edge_attr[e * 5 + 2];
            let is_dummy_edge =
                g.edge_src[e] == dummy || g.edge_dst[e] == dummy;
            if is_dummy_edge {
                assert_eq!(
                    oh_sum, 0.0,
                    "dummy edge {e} should have axis one-hot sum 0.0, got {oh_sum}"
                );
            } else {
                assert_eq!(
                    oh_sum, 1.0,
                    "axis edge {e} should have axis one-hot sum 1.0, got {oh_sum}"
                );
            }
        }
    }

    // AC-1e: Edges are symmetric — every (s,d) has a corresponding (d,s)
    #[test]
    fn ac1e_edges_symmetric() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();
        for (s, d) in g.edge_src.iter().zip(&g.edge_dst) {
            assert!(
                edges.contains(&(*d, *s)),
                "missing reverse edge ({d}, {s}) for ({s}, {d})"
            );
        }
    }

    // AC-1f: Dummy node connects to every non-dummy node (bidirectional)
    #[test]
    fn ac1f_dummy_connects_all() {
        let game = small_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;
        let n_real = g.num_nodes - 1;

        // Collect nodes connected to dummy as source
        let from_dummy: HashSet<i64> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(s, _)| **s == dummy)
            .map(|(_, d)| *d)
            .collect();
        // Collect nodes connected to dummy as destination
        let to_dummy: HashSet<i64> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(_, d)| **d == dummy)
            .map(|(s, _)| *s)
            .collect();

        for i in 0..n_real {
            assert!(
                from_dummy.contains(&(i as i64)),
                "dummy should connect to node {i}"
            );
            assert!(
                to_dummy.contains(&(i as i64)),
                "node {i} should connect to dummy"
            );
        }
    }

    // AC-1g: All hex-neighbor edges (distance 1) present — axis edges subsume hex adjacency
    #[test]
    fn ac1g_subsumes_hex_neighbors() {
        use crate::graph::game_to_graph_raw;

        let game = mid_game();
        let hex_g = game_to_graph_raw(&game);
        let axis_g = game_to_axis_graph_raw(&game);

        // Build coord->index maps for both graphs
        // Hex graph has no dummy, axis graph does. Use coords to match.
        let hex_edges: HashSet<(i32, i32, i32, i32)> = hex_g
            .edge_src
            .iter()
            .zip(&hex_g.edge_dst)
            .map(|(s, d)| {
                let si = *s as usize;
                let di = *d as usize;
                (hex_g.coords[si * 2], hex_g.coords[si * 2 + 1],
                 hex_g.coords[di * 2], hex_g.coords[di * 2 + 1])
            })
            .collect();

        let axis_edges: HashSet<(i32, i32, i32, i32)> = axis_g
            .edge_src
            .iter()
            .zip(&axis_g.edge_dst)
            .filter(|(s, d)| {
                **s != (axis_g.num_nodes - 1) as i64
                    && **d != (axis_g.num_nodes - 1) as i64
            })
            .map(|(s, d)| {
                let si = *s as usize;
                let di = *d as usize;
                (axis_g.coords[si * 2], axis_g.coords[si * 2 + 1],
                 axis_g.coords[di * 2], axis_g.coords[di * 2 + 1])
            })
            .collect();

        for edge in &hex_edges {
            assert!(
                axis_edges.contains(edge),
                "hex neighbor edge ({},{}) -> ({},{}) missing from axis graph",
                edge.0,
                edge.1,
                edge.2,
                edge.3,
            );
        }
    }

    // AC-1h: Mid-game same_color features correctly encode +1.0, -1.0, 0.0
    #[test]
    fn ac1h_same_color_features() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;

        let n_edges = g.edge_src.len();
        for e in 0..n_edges {
            let s = g.edge_src[e] as usize;
            let d = g.edge_dst[e] as usize;
            let same_color = g.edge_attr[e * 5 + 4];

            let s_is_dummy = s as i64 == dummy;
            let d_is_dummy = d as i64 == dummy;
            let s_is_stone = !s_is_dummy && g.stone_mask[s];
            let d_is_stone = !d_is_dummy && g.stone_mask[d];
            let s_is_empty = !s_is_dummy && g.legal_mask[s];
            let d_is_empty = !d_is_dummy && g.legal_mask[d];

            if s_is_dummy || d_is_dummy || s_is_empty || d_is_empty {
                assert_eq!(
                    same_color, 0.0,
                    "edge {e} ({s}->{d}): expected same_color 0.0 (has empty/dummy endpoint)"
                );
            } else if s_is_stone && d_is_stone {
                assert!(
                    same_color == 1.0 || same_color == -1.0,
                    "edge {e} ({s}->{d}): stone-stone should be +1 or -1, got {same_color}"
                );
            }
        }

        // Verify specific sign correctness: find a same-player stone-stone edge
        // mid_game has P1 at (0,0) and P2 stones. Stones are sorted by coord.
        let stones = game.placed_stones();
        let mut sorted_stones = stones.clone();
        sorted_stones.sort_by_key(|&(coord, _)| coord);

        // Build coord -> (index, player) map
        let stone_info: HashMap<Coord, (usize, Player)> = sorted_stones
            .iter()
            .enumerate()
            .map(|(i, &(c, p))| (c, (i, p)))
            .collect();

        let mut found_same = false;
        let mut found_diff = false;
        let mut found_legal = false;

        for e in 0..n_edges {
            let s = g.edge_src[e] as usize;
            let d = g.edge_dst[e] as usize;
            if s as i64 == dummy || d as i64 == dummy {
                continue;
            }
            let same_color = g.edge_attr[e * 5 + 4];
            let s_stone = g.stone_mask[s];
            let d_stone = g.stone_mask[d];

            if s_stone && d_stone {
                // Look up players from coords
                let sq = g.coords[s * 2];
                let sr = g.coords[s * 2 + 1];
                let dq = g.coords[d * 2];
                let dr = g.coords[d * 2 + 1];
                if let (Some(&(_, sp)), Some(&(_, dp))) =
                    (stone_info.get(&(sq, sr)), stone_info.get(&(dq, dr)))
                {
                    if sp == dp {
                        assert_eq!(same_color, 1.0,
                            "edge {e} ({s}->{d}): same-player stone-stone should be +1.0, got {same_color}");
                        found_same = true;
                    } else {
                        assert_eq!(same_color, -1.0,
                            "edge {e} ({s}->{d}): different-player stone-stone should be -1.0, got {same_color}");
                        found_diff = true;
                    }
                }
            } else if g.legal_mask[s] || g.legal_mask[d] {
                assert_eq!(same_color, 0.0,
                    "edge {e} ({s}->{d}): edge involving legal move should be 0.0, got {same_color}");
                found_legal = true;
            }
        }
        assert!(found_same, "should find at least one same-player stone-stone edge");
        assert!(found_diff, "should find at least one different-player stone-stone edge");
        assert!(found_legal, "should find at least one edge involving a legal move");
    }

    // AC-1i: Known 3-stone collinear configuration produces exact expected edge count
    #[test]
    fn ac1i_three_collinear_stones() {
        // Create a game and place 3 stones in a line along axis (1,0)
        // P1 at (0,0) by default. P2 places at (1,0) and (2,0).
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 3,
            max_moves: 80,
        });
        // P2 turn: place at (1,0) and (2,0)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((2, 0)).unwrap();

        let g = game_to_axis_graph_raw(&game);
        let dummy = (g.num_nodes - 1) as i64;

        // Count non-dummy edges
        let axis_edge_count = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .filter(|(s, d)| **s != dummy && **d != dummy)
            .count();

        // Pin the exact non-dummy axis edge count as a snapshot test.
        // This count covers all edges (stone-stone, stone-legal, legal-stone,
        // legal-legal) discovered by the walk logic, excluding dummy edges.
        eprintln!("AC-1i: non-dummy axis edge count = {axis_edge_count}");
        assert!(axis_edge_count > 0, "should have non-dummy edges");
        // Snapshot: pin exact count so any walk-logic change is caught.
        assert_eq!(
            axis_edge_count, 572,
            "pinned non-dummy axis edge count"
        );

        // Count edges involving only the 3 stones (indices 0, 1, 2)
        let stone_stone_edges: Vec<_> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .enumerate()
            .filter(|(_, (s, d))| {
                (**s as usize) < 3 && (**d as usize) < 3
            })
            .collect();

        // From the walk analysis above: discovered pairs among stones are
        // (0,0)-(1,0) and (1,0)-(2,0), giving 2*2=4 directed edges from axis (1,0).
        // Other axes: check collinearity on (0,1) and (1,-1).
        // (0,0),(1,0),(2,0) on axis (0,1): step (0,+1). None of these differ only in r.
        // So no pairs on (0,1) axis among these 3.
        // (1,-1) axis: (0,0)+1*(1,-1)=(1,-1) - not a stone. No pairs.
        // So exactly 4 stone-stone edges.
        assert_eq!(
            stone_stone_edges.len(),
            4,
            "3 collinear stones should produce 4 directed stone-stone edges, got {}",
            stone_stone_edges.len()
        );
    }

    // AC-1k: No duplicate edges
    #[test]
    fn ac1k_no_duplicate_edges() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edges: Vec<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();
        let unique: HashSet<(i64, i64)> = edges.iter().copied().collect();
        assert_eq!(
            edges.len(),
            unique.len(),
            "found {} duplicate edges",
            edges.len() - unique.len()
        );
    }

    // AC-1l: For every edge pair (i,j) and (j,i), signed distances sum to zero
    #[test]
    fn ac1l_signed_distance_antisymmetric() {
        let game = mid_game();
        let g = game_to_axis_graph_raw(&game);
        let edge_map: HashMap<(i64, i64), f32> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .enumerate()
            .map(|(e, (s, d))| ((*s, *d), g.edge_attr[e * 5 + 3]))
            .collect();

        for (&(s, d), &dist) in &edge_map {
            let rev_dist = edge_map
                .get(&(d, s))
                .expect(&format!("missing reverse edge ({d},{s})"));
            assert!(
                (dist + rev_dist).abs() < 1e-6,
                "signed distances for ({s},{d}) and ({d},{s}) should sum to 0: {dist} + {rev_dist}"
            );
        }
    }

    // AC-1m: Edge count for 50-stone game is in a reasonable range; hex/axis ratio < 15.0
    #[test]
    fn ac1m_edge_count_50_stones() {
        use crate::graph::game_to_graph_raw;

        let mut game = GameState::with_config(GameConfig {
            win_length: 6,
            placement_radius: 6,
            max_moves: 200,
        });
        // Place 49 more stones (P1 already at origin)
        let mut placed = 0;
        while placed < 49 && !game.is_terminal() {
            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }
            game.apply_move(moves[0]).unwrap();
            placed += 1;
        }

        let g = game_to_axis_graph_raw(&game);
        let hex_g = game_to_graph_raw(&game);
        let n_edges = g.edge_src.len();
        let n_real = g.num_nodes - 1;
        let dummy_edges = 2 * n_real;
        let axis_edges = n_edges - dummy_edges;
        let hex_edges = hex_g.edge_src.len();

        assert!(
            n_edges >= dummy_edges,
            "should have at least {dummy_edges} edges (dummy), got {n_edges}"
        );
        assert!(
            axis_edges > 0,
            "should have axis edges beyond just dummy edges"
        );

        let ratio = axis_edges as f64 / hex_edges.max(1) as f64;
        eprintln!(
            "AC-1m: 50-stone game: {} nodes, {} axis edges, {} hex edges, ratio = {:.2}",
            g.num_nodes, axis_edges, hex_edges, ratio
        );
        assert!(
            ratio < 15.0,
            "axis/hex edge ratio {ratio:.2} exceeds design threshold 15.0"
        );
    }

    // AC-1n: Two stones at distance win_length-1 connected; at distance win_length not connected
    #[test]
    fn ac1n_boundary_distance() {
        // win_length=4, so window=3
        // P1 at (0,0). Place P2 at (3,0) => distance 3 along axis (1,0) => should connect
        // Place P2 at (0,4) => distance 4 along axis (0,1) => should NOT connect
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 5,
            max_moves: 80,
        });
        // P2 places at (3,0) and (0,4)
        game.apply_move((3, 0)).unwrap();
        game.apply_move((0, 4)).unwrap();

        let g = game_to_axis_graph_raw(&game);

        // Build coord -> index map, excluding dummy node
        let mut coord_idx: HashMap<Coord, usize> = HashMap::new();
        for i in 0..g.num_nodes - 1 {
            let q = g.coords[i * 2];
            let r = g.coords[i * 2 + 1];
            coord_idx.insert((q, r), i);
        }

        let idx_origin = *coord_idx.get(&(0, 0)).unwrap();
        let idx_3_0 = *coord_idx.get(&(3, 0)).unwrap();

        // Check (0,0) and (3,0) are connected (distance 3 = win_length - 1)
        // Walking from (0,0) P1 along (1,0):
        // step 1: (1,0) — empty node, continue
        // step 2: (2,0) — empty node, continue
        // step 3: (3,0) — P2 stone, add edge and STOP
        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();

        assert!(
            edges.contains(&(idx_origin as i64, idx_3_0 as i64)),
            "stones at distance win_length-1 should be connected"
        );

        // (0,0) and (0,4): distance 4 = win_length, should NOT be directly connected
        let idx_0_4 = *coord_idx.get(&(0, 4)).unwrap();
        assert!(
            !edges.contains(&(idx_origin as i64, idx_0_4 as i64)),
            "stones at distance win_length should NOT be connected"
        );
    }

    // AC-1j: game_to_axis_graph_batch matches individual game_to_axis_graph_raw
    #[test]
    fn ac1j_batch_matches_individual() {
        let games = vec![small_game(), mid_game()];
        let batch = game_to_axis_graph_batch(&games);
        assert_eq!(batch.len(), games.len());

        for (i, game) in games.iter().enumerate() {
            let individual = game_to_axis_graph_raw(game);
            assert_eq!(batch[i].num_nodes, individual.num_nodes);
            assert_eq!(batch[i].features, individual.features);
            assert_eq!(batch[i].edge_src.len(), individual.edge_src.len());

            // Build (src, dst) -> [f32; 5] maps for both and compare
            let batch_attr_map: HashMap<(i64, i64), Vec<f32>> = batch[i]
                .edge_src
                .iter()
                .zip(&batch[i].edge_dst)
                .enumerate()
                .map(|(e, (s, d))| {
                    let attr = batch[i].edge_attr[e * 5..(e + 1) * 5].to_vec();
                    ((*s, *d), attr)
                })
                .collect();
            let ind_attr_map: HashMap<(i64, i64), Vec<f32>> = individual
                .edge_src
                .iter()
                .zip(&individual.edge_dst)
                .enumerate()
                .map(|(e, (s, d))| {
                    let attr = individual.edge_attr[e * 5..(e + 1) * 5].to_vec();
                    ((*s, *d), attr)
                })
                .collect();

            assert_eq!(
                batch_attr_map.len(),
                ind_attr_map.len(),
                "game {i}: edge count mismatch"
            );
            for (key, batch_attr) in &batch_attr_map {
                let ind_attr = ind_attr_map
                    .get(key)
                    .unwrap_or_else(|| panic!("game {i}: missing edge {:?} in individual", key));
                assert_eq!(
                    batch_attr, ind_attr,
                    "game {i}: edge_attr mismatch for edge {:?}",
                    key
                );
            }
        }
    }

    // AC: Verify actual node feature values for specific nodes.
    #[test]
    fn ac_node_features() {
        let game = small_game(); // P1 at origin, P2 to move
        let g = game_to_axis_graph_raw(&game);

        // Node 0 is P1 stone (stones sorted by coord, (0,0) is first/only stone)
        let base0 = 0 * 8;
        // P1 one-hot: [1, 0, 0]
        assert_eq!(g.features[base0], 1.0, "node 0: P1 one-hot[0]");
        assert_eq!(g.features[base0 + 1], 0.0, "node 0: P1 one-hot[1]");
        assert_eq!(g.features[base0 + 2], 0.0, "node 0: P1 one-hot[2]");
        // player_feat: P2 to move => -1.0
        assert_eq!(g.features[base0 + 3], -1.0, "node 0: player_feat");
        // moves_feat: moves_remaining / 2.0 = 2/2.0 = 1.0
        assert_eq!(g.features[base0 + 4], 1.0, "node 0: moves_feat");

        // Node 1 is first legal move (empty)
        let n_stones = game.placed_stones().len();
        assert_eq!(n_stones, 1);
        let base1 = 1 * 8; // first legal move node
        // Empty one-hot: [0, 0, 1]
        assert_eq!(g.features[base1 + 2], 1.0, "node 1: empty one-hot");
        // inv_distance > 0 for legal moves near a stone
        assert!(g.features[base1 + 7] > 0.0, "node 1: inv_distance > 0");

        // Dummy node (last)
        let dummy_idx = g.num_nodes - 1;
        let dummy_base = dummy_idx * 8;
        // Dummy has no type one-hot
        assert_eq!(g.features[dummy_base], 0.0, "dummy: one-hot[0]");
        assert_eq!(g.features[dummy_base + 1], 0.0, "dummy: one-hot[1]");
        assert_eq!(g.features[dummy_base + 2], 0.0, "dummy: one-hot[2]");
        // Dummy still has player_feat and moves_feat
        assert_eq!(g.features[dummy_base + 3], -1.0, "dummy: player_feat");
        assert_eq!(g.features[dummy_base + 4], 1.0, "dummy: moves_feat");
    }

    // AC: Verify walk-stopping logic in isolation.
    #[test]
    fn ac_walk_stopping() {
        // P1 at (0,0), P2 at (1,0) and (2,0). win_length=4 => window=3.
        let mut game = GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 3,
            max_moves: 80,
        });
        // P2 places at (1,0) and (2,0)
        game.apply_move((1, 0)).unwrap();
        game.apply_move((2, 0)).unwrap();

        let g = game_to_axis_graph_raw(&game);

        // Build coord -> index map
        let mut coord_idx: HashMap<Coord, usize> = HashMap::new();
        for i in 0..g.num_nodes - 1 {
            coord_idx.insert((g.coords[i * 2], g.coords[i * 2 + 1]), i);
        }

        let idx_00 = *coord_idx.get(&(0, 0)).expect("(0,0) should exist");
        let idx_10 = *coord_idx.get(&(1, 0)).expect("(1,0) should exist");
        let idx_20 = *coord_idx.get(&(2, 0)).expect("(2,0) should exist");

        let edges: HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(s, d)| (*s, *d))
            .collect();

        // (0,0) P1 walking +axis (1,0): hits P2 at (1,0) => stops.
        // So (0,0)->(1,0) exists.
        assert!(
            edges.contains(&(idx_00 as i64, idx_10 as i64)),
            "(0,0)->(1,0) should exist"
        );

        // (1,0) P2 walking +axis (1,0): hits P2 at (2,0) => same color, continue.
        // So (1,0)->(2,0) exists.
        assert!(
            edges.contains(&(idx_10 as i64, idx_20 as i64)),
            "(1,0)->(2,0) should exist"
        );

        // (0,0)->(2,0) should NOT exist: walk from (0,0) along +axis (1,0)
        // stopped at (1,0) because it hit opponent P2.
        assert!(
            !edges.contains(&(idx_00 as i64, idx_20 as i64)),
            "(0,0)->(2,0) should NOT exist (walk stopped at opponent)"
        );
    }
}
