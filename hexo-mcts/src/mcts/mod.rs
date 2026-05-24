pub mod batcher;
pub mod backup;
pub mod batched;
pub mod gumbel_mcts;
pub mod halving;
pub mod node;
pub mod scoring;
pub mod select;
pub mod simulate;

#[cfg(feature = "dedup_count")]
pub mod dedup_count;

/// MCTS configuration parameters.
///
/// `Default` is derived for ergonomic test construction via
/// `MCTSConfig { n_simulations: 64, m_actions: 16, ..Default::default() }`,
/// but the default values for `n_simulations` and `m_actions` are 0 and not
/// usable at runtime — always set these explicitly in real configurations.
#[derive(Clone, Default)]
pub struct MCTSConfig {
    pub n_simulations: u32,
    pub m_actions: usize,
    pub c_visit: u32,
    pub c_scale: f64,
    /// Virtual-loss magnitude used during SH inner-loop fusion. 0.0 disables
    /// fusion (serial sims, current behaviour); >0.0 enables leaf-parallel
    /// MCTS within each SH phase. Typical range 0.3–1.0.
    pub virtual_loss: f64,
    /// Dirichlet noise concentration α applied to root priors before
    /// Gumbel-Top-k candidate sampling. 0.0 disables (bit-equivalent to
    /// pre-noise behaviour). For HeXO with ~200 legal moves at full board,
    /// α ≈ 0.05 is a reasonable starting point (AlphaZero heuristic
    /// α ≈ 10 / branching_factor). Lower α → more concentrated noise
    /// (sparser exploration boost). Higher α → flatter noise (uniform-ish).
    pub root_dirichlet_alpha: f64,
    /// Mix fraction ε for Dirichlet noise at the root.
    /// `mixed_prior = (1 - ε) * network_prior + ε * Dirichlet(α)`.
    /// 0.0 disables (no noise applied regardless of α). Standard AlphaZero
    /// uses 0.25. Ignored when `root_dirichlet_alpha == 0.0`.
    pub root_dirichlet_fraction: f64,
    /// Weight α on a lookahead value bonus applied at start-of-turn root SH
    /// elimination. For each Gumbel-Top-K p₁ candidate, the network's value
    /// head is queried on the post-p₁ state and the score is biased by α·v.
    /// 0.0 (default) disables the bonus path entirely (bit-equivalent to
    /// current behaviour). Only active when `game.moves_remaining_this_turn() == 2`.
    pub lookahead_value_bonus: f64,
}
