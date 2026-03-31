pub mod backup;
pub mod batched;
pub mod gumbel_mcts;
pub mod halving;
pub mod node;
pub mod scoring;
pub mod select;
pub mod simulate;

/// MCTS configuration parameters.
#[derive(Clone)]
pub struct MCTSConfig {
    pub n_simulations: u32,
    pub m_actions: usize,
    pub c_visit: u32,
    pub c_scale: f64,
}
