//! Batch tensor construction: collate multiple GraphData into flat tensors
//! ready for direct model consumption, bypassing PyG Batch.from_data_list().

use crate::graph::GraphData;

pub struct BatchTensors {
    pub features: Vec<f32>,       // total_nodes * 8, row-major
    pub edge_index_src: Vec<i64>, // total_edges (with node offsets applied)
    pub edge_index_dst: Vec<i64>,
    pub legal_mask: Vec<bool>,    // total_nodes
    pub stone_mask: Vec<bool>,    // total_nodes
    pub batch: Vec<i64>,          // total_nodes, each = graph index
    pub num_graphs: usize,
    pub legal_counts: Vec<usize>, // per-graph count of legal nodes
}

pub fn collate_graphs(graphs: &[GraphData]) -> BatchTensors {
    let num_graphs = graphs.len();
    let total_nodes: usize = graphs.iter().map(|g| g.num_nodes).sum();
    let total_edges: usize = graphs.iter().map(|g| g.edge_src.len()).sum();

    let mut features = Vec::with_capacity(total_nodes * 8);
    let mut edge_index_src = Vec::with_capacity(total_edges);
    let mut edge_index_dst = Vec::with_capacity(total_edges);
    let mut legal_mask = Vec::with_capacity(total_nodes);
    let mut stone_mask = Vec::with_capacity(total_nodes);
    let mut batch = Vec::with_capacity(total_nodes);
    let mut legal_counts = Vec::with_capacity(num_graphs);

    let mut node_offset: i64 = 0;

    for (graph_idx, g) in graphs.iter().enumerate() {
        features.extend_from_slice(&g.features);
        legal_mask.extend_from_slice(&g.legal_mask);
        stone_mask.extend_from_slice(&g.stone_mask);

        for &src in &g.edge_src {
            edge_index_src.push(src + node_offset);
        }
        for &dst in &g.edge_dst {
            edge_index_dst.push(dst + node_offset);
        }

        for _ in 0..g.num_nodes {
            batch.push(graph_idx as i64);
        }

        legal_counts.push(g.legal_mask.iter().filter(|&&m| m).count());
        node_offset += g.num_nodes as i64;
    }

    BatchTensors {
        features,
        edge_index_src,
        edge_index_dst,
        legal_mask,
        stone_mask,
        batch,
        num_graphs,
        legal_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::game_to_graph_raw;
    use hexo_engine::GameState;

    #[test]
    fn collate_two_graphs() {
        let g1 = game_to_graph_raw(&GameState::new());
        let g2 = game_to_graph_raw(&GameState::new());
        let n1 = g1.num_nodes;
        let n2 = g2.num_nodes;

        let bt = collate_graphs(&[g1, g2]);

        assert_eq!(bt.num_graphs, 2);
        assert_eq!(bt.batch.len(), n1 + n2);
        assert_eq!(bt.features.len(), (n1 + n2) * 8);
        assert_eq!(bt.legal_counts.len(), 2);
        assert!(bt.batch[..n1].iter().all(|&b| b == 0));
        assert!(bt.batch[n1..].iter().all(|&b| b == 1));
        assert!(bt.edge_index_src.iter().any(|&s| s >= n1 as i64));
    }
}
