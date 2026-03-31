use hexo_engine::Coord;
use rand::Rng;
use rand::distr::{Distribution, Uniform};

use super::node::MCTSNode;
use super::scoring::{QContext, sigma};

/// Sample Gumbel(0,1) noise and return top-min(m, n) action indices.
///
/// Returns `(candidate_indices, gumbel_samples)` where:
/// - `candidate_indices` are indices into the actions array, sorted by descending
///   score (gumbel + logit).
/// - `gumbel_samples` are the full Gumbel noise values for all actions.
pub fn gumbel_top_k<R: Rng>(
    logits: &[f64],
    m: usize,
    rng: &mut R,
) -> (Vec<usize>, Vec<f64>) {
    let n = logits.len();
    let k = m.min(n);

    let uniform = Uniform::new(1e-20, 1.0 - 1e-20).unwrap();
    let gumbels: Vec<f64> = (0..n)
        .map(|_| {
            let u: f64 = uniform.sample(rng);
            -(-u.ln()).ln()
        })
        .collect();

    let mut scores: Vec<(usize, f64)> = gumbels
        .iter()
        .zip(logits.iter())
        .enumerate()
        .map(|(i, (&g, &l))| (i, g + l))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let candidates: Vec<usize> = scores.iter().take(k).map(|&(i, _)| i).collect();

    (candidates, gumbels)
}

/// Compute the improved policy distribution from a completed search.
///
/// π_improved(a) = softmax(logits(a) + σ(normalized_completedQ(a)))
///
/// Q-values are read from the root's perspective. Unvisited actions get v_mix.
pub fn compute_improved_policy(
    logits: &[f64],
    coords: &[Coord],
    root: &MCTSNode,
    c_visit: u32,
    c_scale: f64,
) -> Vec<f64> {
    let qctx = QContext::new(root, coords);

    let log_scores: Vec<f64> = coords
        .iter()
        .zip(logits.iter())
        .map(|(coord, logit)| logit + sigma(qctx.norm_q[coord], qctx.max_child_visits, c_visit, c_scale))
        .collect();

    // Step 4: softmax
    super::scoring::softmax(&log_scores)
}

use super::MCTSConfig;

/// Allocate simulations across candidates using Sequential Halving.
///
/// Each phase distributes simulations to surviving candidates. After each phase,
/// the bottom half (by Gumbel score) is eliminated.
///
/// The evaluation callback `eval_fn` receives a list of game states that need
/// network evaluation and returns `(priors_per_state, values)`.
///
/// Returns the list of surviving candidate indices.
pub fn sequential_halving<F>(
    root: &mut MCTSNode,
    candidate_indices: &[usize],
    candidate_gumbels: &[f64],
    coords: &[Coord],
    logits: &[f64],
    config: &MCTSConfig,
    eval_fn: &mut F,
) -> Vec<usize>
where
    F: FnMut(&mut MCTSNode, &[Coord], u32, f64) -> (),
{
    let n_simulations = config.n_simulations;
    let c_visit = config.c_visit;
    let c_scale = config.c_scale;

    if n_simulations == 0 {
        return candidate_indices.to_vec();
    }
    if candidate_indices.len() <= 1 {
        return candidate_indices.to_vec();
    }

    let num_phases = (candidate_indices.len() as f64).log2().ceil() as u32;
    let mut remaining: Vec<usize> = candidate_indices.to_vec();
    let mut total_sims_used: u32 = 0;

    for _ in 0..num_phases {
        if total_sims_used >= n_simulations || remaining.len() <= 1 {
            break;
        }

        let sims_per_action =
            (n_simulations / (num_phases * remaining.len() as u32)).max(1);

        for _ in 0..sims_per_action {
            if total_sims_used >= n_simulations {
                break;
            }

            let mut batch_actions: Vec<Coord> = Vec::new();
            for &idx in &remaining {
                if total_sims_used >= n_simulations {
                    break;
                }
                batch_actions.push(coords[idx]);
                total_sims_used += 1;
            }

            eval_fn(root, &batch_actions, c_visit, c_scale);
        }

        // Eliminate bottom half by Gumbel score
        let qctx = QContext::new(root, coords);

        remaining.sort_by(|&a, &b| {
            let score_a = gumbel_score(
                a, coords, candidate_gumbels, logits, &qctx, c_visit, c_scale,
            );
            let score_b = gumbel_score(
                b, coords, candidate_gumbels, logits, &qctx, c_visit, c_scale,
            );
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let keep = (remaining.len() + 1) / 2; // ceil(len/2)
        remaining.truncate(keep);
    }

    remaining
}

fn gumbel_score(
    idx: usize,
    coords: &[Coord],
    gumbels: &[f64],
    logits: &[f64],
    qctx: &QContext,
    c_visit: u32,
    c_scale: f64,
) -> f64 {
    let action = coords[idx];
    let norm_q = qctx.norm_q[&action];
    gumbels[idx] + logits[idx] + sigma(norm_q, qctx.max_child_visits, c_visit, c_scale)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use hexo_engine::Player;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn gumbel_top_k_returns_min_m_n() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0];
        let (candidates, gumbels) = gumbel_top_k(&logits, 5, &mut rng);
        assert_eq!(candidates.len(), 3); // min(5, 3)
        assert_eq!(gumbels.len(), 3);
    }

    #[test]
    fn gumbel_top_k_deterministic() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let (c1, g1) = gumbel_top_k(&logits, 2, &mut rng1);
        let (c2, g2) = gumbel_top_k(&logits, 2, &mut rng2);
        assert_eq!(c1, c2);
        assert_eq!(g1, g2);
    }

    #[test]
    fn gumbel_top_k_sorted_descending() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (candidates, gumbels) = gumbel_top_k(&logits, 3, &mut rng);

        // Verify candidates are sorted by descending gumbel+logit score
        for i in 0..candidates.len() - 1 {
            let score_a = gumbels[candidates[i]] + logits[candidates[i]];
            let score_b = gumbels[candidates[i + 1]] + logits[candidates[i + 1]];
            assert!(score_a >= score_b);
        }
    }

    #[test]
    fn compute_improved_policy_sums_to_one() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0)];
        let logits = vec![1.0, 2.0, 0.5];
        let priors: HashMap<Coord, f64> =
            coords.iter().map(|&c| (c, 1.0 / 3.0)).collect();
        root.expand(priors, 0.5);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_improved_policy_length_matches_coords() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1)];
        let logits = vec![1.0, 2.0];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.5)).collect();
        root.expand(priors, 0.0);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        assert_eq!(policy.len(), coords.len());
    }

    #[test]
    fn sequential_halving_zero_budget() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let config = MCTSConfig {
            n_simulations: 0,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let candidates = vec![0, 1, 2, 3];
        let gumbels = vec![1.0, 2.0, 3.0, 4.0];
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let logits = vec![0.0; 4];

        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert_eq!(result, candidates);
    }

    #[test]
    fn sequential_halving_single_candidate() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let config = MCTSConfig {
            n_simulations: 16,
            m_actions: 1,
            c_visit: 50,
            c_scale: 1.0,
        };
        let candidates = vec![0];
        let gumbels = vec![1.0];
        let coords = vec![(1, 0)];
        let logits = vec![0.0];

        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn sequential_halving_eliminates() {
        // With 4 candidates and non-zero budget, should end with fewer
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.25)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 16,
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let candidates = vec![0, 1, 2, 3];
        // Different gumbels so elimination has a preference
        let gumbels = vec![4.0, 3.0, 2.0, 1.0];
        let logits = vec![0.0; 4];

        // Dummy eval_fn that doesn't actually evaluate (just for testing structure)
        let result = sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, _, _, _| {},
        );
        assert!(result.len() < candidates.len());
        assert!(!result.is_empty());
    }

    #[test]
    fn sequential_halving_respects_budget() {
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1), (-1, 0), (0, -1)];
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.25)).collect();
        root.expand(priors, 0.0);

        let config = MCTSConfig {
            n_simulations: 4, // very small budget
            m_actions: 4,
            c_visit: 50,
            c_scale: 1.0,
        };
        let candidates = vec![0, 1, 2, 3];
        let gumbels = vec![1.0; 4];
        let logits = vec![0.0; 4];

        let mut total_eval_calls = 0u32;
        sequential_halving(
            &mut root,
            &candidates,
            &gumbels,
            &coords,
            &logits,
            &config,
            &mut |_, actions, _, _| {
                total_eval_calls += actions.len() as u32;
            },
        );
        assert!(total_eval_calls <= 4);
    }

    #[test]
    fn compute_improved_policy_favors_high_q() {
        // Root with two children: one visited with Q=1.0, one with Q=-1.0.
        // The improved policy should put more weight on the high-Q action.
        let mut root = MCTSNode::new(1.0, None, Player::P1);
        let coords = vec![(1, 0), (0, 1)];
        let logits = vec![0.0, 0.0]; // equal logits so only Q matters
        let priors: HashMap<Coord, f64> = coords.iter().map(|&c| (c, 0.5)).collect();
        root.expand(priors, 0.0);

        // Child (1,0): same player, Q=1.0
        let mut c_good = MCTSNode::new(0.5, Some((1, 0)), Player::P1);
        c_good.visit_count = 10;
        c_good.value_sum = 10.0; // Q = 1.0

        // Child (0,1): same player, Q=-1.0
        let mut c_bad = MCTSNode::new(0.5, Some((0, 1)), Player::P1);
        c_bad.visit_count = 10;
        c_bad.value_sum = -10.0; // Q = -1.0

        root.children.insert((1, 0), c_good);
        root.children.insert((0, 1), c_bad);

        let policy = compute_improved_policy(&logits, &coords, &root, 50, 1.0);
        // coords[0] = (1,0) has Q=1.0, coords[1] = (0,1) has Q=-1.0
        // With equal logits, the high-Q action should dominate the policy.
        assert!(
            policy[0] > policy[1],
            "Policy for Q=1.0 action ({}) should exceed Q=-1.0 action ({})",
            policy[0],
            policy[1],
        );
        // The difference should be substantial, not just a rounding artifact
        assert!(policy[0] > 0.9, "High-Q action should dominate: {}", policy[0]);
    }
}
