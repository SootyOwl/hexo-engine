//! Graph construction: convert a GameState to raw arrays for PyG Data objects.
//!
//! Mirrors the Python `game_to_graph` but returns flat Vecs instead of tensors.
//! Python wraps these into `torch.tensor` + `torch_geometric.data.Data`.

use std::collections::HashMap;

use hexo_engine::hex::hex_distance;
use hexo_engine::symmetry::D6_TRANSFORMS;
use hexo_engine::types::{Coord, HEX_DIRS, Player};
use hexo_engine::GameState;

/// Raw graph data ready to be wrapped into PyG Data on the Python side.
pub struct GraphData {
    /// Node features, flattened: N×8 row-major.
    pub features: Vec<f32>,
    /// Edge source indices.
    pub edge_src: Vec<i64>,
    /// Edge destination indices.
    pub edge_dst: Vec<i64>,
    /// Legal move mask (true for legal-move nodes).
    pub legal_mask: Vec<bool>,
    /// Stone mask (true for placed-stone nodes).
    pub stone_mask: Vec<bool>,
    /// Neighbor indices per node: N×6, flattened row-major.
    /// For each node, 6 entries giving the index of each hex neighbor.
    /// -1 (as i64) for missing neighbors (boundary nodes).
    /// Neighbor order matches HEX_DIRS: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    pub neighbor_index: Vec<i64>,
    /// Coordinates, flattened: N×2 row-major (q, r).
    pub coords: Vec<i32>,
    /// Number of nodes.
    pub num_nodes: usize,
}

/// Build graph arrays from pre-sorted stones and legal moves plus global features.
///
/// `stones` and `legal` must already be sorted by coord tuple (q, r).
///
/// Node ordering:
///   - Placed stones first, sorted by (q, r)
///   - Legal moves next, sorted by (q, r)
///
/// Node features (8-dim):
///   [0:3] Stone-type one-hot: [1,0,0]=P1, [0,1,0]=P2, [0,0,1]=empty
///   [3]   Current player: +1.0 if P1, -1.0 if P2
///   [4]   Moves remaining this turn / 2.0
///   [5]   Normalised q (relative to stone centroid)
///   [6]   Normalised r (relative to stone centroid)
///   [7]   Inverse distance to nearest stone (0 for stones themselves)
fn build_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
) -> GraphData {
    let n_stones = stones.len();
    let n_legal = legal.len();
    let n = n_stones + n_legal;

    // Coords (flattened N×2)
    let mut coords = Vec::with_capacity(n * 2);
    for &(coord, _) in stones {
        coords.push(coord.0);
        coords.push(coord.1);
    }
    for &(q, r) in legal {
        coords.push(q);
        coords.push(r);
    }

    // Stone centroid and spread
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

    // Stone coords as a vec for fast nearest-stone lookup
    let stone_coords: Vec<Coord> = stones.iter().map(|&(c, _)| c).collect();

    // Features (flattened N×8)
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
        // features[base + 7] = 0.0 (stones get 0 for inverse distance)
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

        // Inverse distance to nearest stone
        let coord = legal[j];
        let min_d = stone_coords
            .iter()
            .map(|&sc| hex_distance(coord, sc))
            .min()
            .unwrap_or(1);
        features[base + 7] = 1.0 / min_d.max(1) as f32;
    }

    // Edges (hex adjacency)
    let mut coord_to_idx: HashMap<Coord, usize> = HashMap::with_capacity(n);
    for i in 0..n_stones {
        coord_to_idx.insert((coords[i * 2], coords[i * 2 + 1]), i);
    }
    for j in 0..n_legal {
        let idx = n_stones + j;
        coord_to_idx.insert((coords[idx * 2], coords[idx * 2 + 1]), idx);
    }

    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    for i in 0..n {
        let q = coords[i * 2];
        let r = coords[i * 2 + 1];
        for &(dq, dr) in &HEX_DIRS {
            if let Some(&j) = coord_to_idx.get(&(q + dq, r + dr)) {
                edge_src.push(i as i64);
                edge_dst.push(j as i64);
            }
        }
    }

    // Neighbor index (N×6, -1 for missing)
    let mut neighbor_index = Vec::with_capacity(n * 6);
    for i in 0..n {
        let q = coords[i * 2];
        let r = coords[i * 2 + 1];
        for &(dq, dr) in &HEX_DIRS {
            match coord_to_idx.get(&(q + dq, r + dr)) {
                Some(&j) => neighbor_index.push(j as i64),
                None => neighbor_index.push(-1),
            }
        }
    }

    // Masks
    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for i in 0..n_stones {
        stone_mask[i] = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }

    GraphData {
        features,
        edge_src,
        edge_dst,
        legal_mask,
        stone_mask,
        neighbor_index,
        coords,
        num_nodes: n,
    }
}

/// Build graph arrays from a GameState.
///
/// Extracts stones, legal moves, and global features from the game state,
/// then delegates to `build_graph`.
pub fn game_to_graph_raw(game: &GameState) -> GraphData {
    let mut stones = game.placed_stones();
    stones.sort_by_key(|&(coord, _)| coord);
    let legal = game.legal_moves(); // already sorted by engine

    let player_feat: f32 = match game.current_player() {
        Some(Player::P1) => 1.0,
        _ => -1.0,
    };
    let moves_feat: f32 = game.moves_remaining_this_turn() as f32 / 2.0;

    build_graph(&stones, &legal, player_feat, moves_feat)
}

/// Apply all 11 non-identity D6 transforms to produce augmented graphs.
///
/// Returns Vec of (GraphData, permutation) where permutation[i] gives the
/// index in the ORIGINAL legal moves array that maps to position i in the
/// augmented (re-sorted) legal moves.
pub fn augment_graph(
    stones: &[(Coord, Player)],
    legal: &[Coord],
    player_feat: f32,
    moves_feat: f32,
) -> Vec<(GraphData, Vec<usize>)> {
    // Skip transform 0 (identity)
    D6_TRANSFORMS[1..]
        .iter()
        .map(|transform| {
            // Transform and sort stones
            let mut t_stones: Vec<(Coord, Player)> = stones
                .iter()
                .map(|&(c, p)| (transform(c), p))
                .collect();
            t_stones.sort_by_key(|&(c, _)| c);

            // Transform legal moves, keeping track of original indices
            let mut indexed_legal: Vec<(Coord, usize)> = legal
                .iter()
                .enumerate()
                .map(|(i, &c)| (transform(c), i))
                .collect();
            indexed_legal.sort_by_key(|&(c, _)| c);

            let t_legal: Vec<Coord> = indexed_legal.iter().map(|&(c, _)| c).collect();
            let permutation: Vec<usize> = indexed_legal.iter().map(|&(_, i)| i).collect();

            let graph = build_graph(&t_stones, &t_legal, player_feat, moves_feat);
            (graph, permutation)
        })
        .collect()
}

/// Build graph arrays for a batch of game states in parallel using rayon.
pub fn game_to_graph_batch(games: &[GameState]) -> Vec<GraphData> {
    use rayon::prelude::*;
    games.par_iter().map(game_to_graph_raw).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::GameConfig;

    fn small_game() -> GameState {
        GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 80,
        })
    }

    #[test]
    fn graph_has_correct_node_count() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // 1 stone (P1 at origin) + legal_moves
        let expected = 1 + game.legal_moves().len();
        assert_eq!(g.num_nodes, expected);
        assert_eq!(g.features.len(), expected * 8);
        assert_eq!(g.coords.len(), expected * 2);
    }

    #[test]
    fn features_are_8_dim() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.features.len() % 8, 0);
    }

    #[test]
    fn stone_one_hot_is_correct() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // First node is P1 stone at (0,0)
        assert_eq!(g.features[0], 1.0); // P1
        assert_eq!(g.features[1], 0.0); // not P2
        assert_eq!(g.features[2], 0.0); // not empty
    }

    #[test]
    fn legal_node_has_empty_one_hot() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Second node (first legal move) should be empty
        let base = 1 * 8; // node index 1
        assert_eq!(g.features[base], 0.0);     // not P1
        assert_eq!(g.features[base + 1], 0.0); // not P2
        assert_eq!(g.features[base + 2], 1.0); // empty
    }

    #[test]
    fn masks_are_correct() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert!(g.stone_mask[0]);   // first node is a stone
        assert!(!g.legal_mask[0]);  // first node is not legal
        assert!(!g.stone_mask[1]);  // second node is not a stone
        assert!(g.legal_mask[1]);   // second node is legal
    }

    #[test]
    fn edges_are_symmetric() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Every (src, dst) should have a corresponding (dst, src)
        let edges: std::collections::HashSet<(i64, i64)> = g
            .edge_src
            .iter()
            .zip(&g.edge_dst)
            .map(|(&s, &d)| (s, d))
            .collect();
        for (&s, &d) in g.edge_src.iter().zip(&g.edge_dst) {
            assert!(edges.contains(&(d, s)), "missing reverse edge ({d}, {s})");
        }
    }

    #[test]
    fn current_player_feature() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Game starts with P2 to move → player_feat = -1.0
        assert_eq!(g.features[3], -1.0);
    }

    #[test]
    fn inverse_distance_is_positive_for_legal() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        let n_stones = g.stone_mask.iter().filter(|&&b| b).count();
        for j in 0..g.legal_mask.iter().filter(|&&b| b).count() {
            let base = (n_stones + j) * 8;
            assert!(g.features[base + 7] > 0.0, "legal node {j} should have positive inv distance");
        }
    }

    #[test]
    fn stone_inverse_distance_is_zero() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.features[7], 0.0); // stone node's inv distance
    }

    #[test]
    fn neighbor_index_correct_for_origin() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        // Node 0 is the stone at (0,0). Check its 6 neighbors are present
        // (all should be legal moves adjacent to origin)
        let ni = &g.neighbor_index[0..6];
        // At least some neighbors should be valid (>= 0)
        assert!(ni.iter().any(|&idx| idx >= 0));
        // None should be out of bounds
        for &idx in ni {
            assert!(idx == -1 || (idx >= 0 && (idx as usize) < g.num_nodes));
        }
    }

    #[test]
    fn neighbor_index_has_correct_length() {
        let game = small_game();
        let g = game_to_graph_raw(&game);
        assert_eq!(g.neighbor_index.len(), g.num_nodes * 6);
    }

    #[test]
    fn augment_produces_11_graphs() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let results = augment_graph(&stones, &legal, -1.0, 1.0);
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn augmented_graphs_have_same_node_count() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let original = game_to_graph_raw(&game);
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (g, _perm) in &augmented {
            assert_eq!(g.num_nodes, original.num_nodes);
            assert_eq!(g.features.len(), original.features.len());
        }
    }

    #[test]
    fn permutation_has_correct_length() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (_g, perm) in &augmented {
            assert_eq!(perm.len(), legal.len());
        }
    }

    #[test]
    fn permutation_is_a_valid_bijection() {
        let game = small_game();
        let mut stones = game.placed_stones();
        stones.sort_by_key(|&(c, _)| c);
        let legal = game.legal_moves();
        let augmented = augment_graph(&stones, &legal, -1.0, 1.0);
        for (_g, perm) in &augmented {
            let mut sorted = perm.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), legal.len());
        }
    }
}
