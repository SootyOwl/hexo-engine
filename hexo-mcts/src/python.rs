use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use hexo_engine::game::{GameConfig, GameState, MoveError};
use hexo_engine::types::{Coord, Player};

use crate::mcts::gumbel_mcts;
use crate::mcts::MCTSConfig;

fn player_str(p: Player) -> &'static str {
    match p {
        Player::P1 => "P1",
        Player::P2 => "P2",
    }
}

#[pyclass(name = "GameConfig", skip_from_py_object)]
#[derive(Clone)]
struct PyGameConfig {
    inner: GameConfig,
}

#[pymethods]
impl PyGameConfig {
    #[new]
    fn new(win_length: u8, placement_radius: i32, max_moves: u32) -> PyResult<Self> {
        if win_length < 2 {
            return Err(PyValueError::new_err("win_length must be >= 2"));
        }
        if placement_radius < 1 {
            return Err(PyValueError::new_err("placement_radius must be >= 1"));
        }
        if max_moves < 1 {
            return Err(PyValueError::new_err("max_moves must be >= 1"));
        }
        Ok(PyGameConfig {
            inner: GameConfig { win_length, placement_radius, max_moves },
        })
    }

    #[staticmethod]
    fn full_hexo() -> Self {
        PyGameConfig { inner: GameConfig::FULL_HEXO }
    }

    #[getter]
    fn win_length(&self) -> u8 {
        self.inner.win_length
    }

    #[getter]
    fn placement_radius(&self) -> i32 {
        self.inner.placement_radius
    }

    #[getter]
    fn max_moves(&self) -> u32 {
        self.inner.max_moves
    }

    fn __repr__(&self) -> String {
        format!(
            "GameConfig(win_length={}, placement_radius={}, max_moves={})",
            self.inner.win_length, self.inner.placement_radius, self.inner.max_moves,
        )
    }
}

#[pyclass(name = "GameState")]
struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyGameConfig>) -> Self {
        let inner = match config {
            Some(cfg) => GameState::with_config(cfg.inner),
            None => GameState::new(),
        };
        PyGameState { inner }
    }

    fn apply_move(&mut self, q: i32, r: i32) -> PyResult<()> {
        self.inner.apply_move((q, r)).map_err(|e| match e {
            MoveError::GameOver => PyValueError::new_err("game is over"),
            MoveError::CellOccupied => PyValueError::new_err("cell is occupied"),
            MoveError::OutOfRange => PyValueError::new_err("move is out of range"),
        })
    }

    fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner.legal_moves()
    }

    fn legal_move_count(&self) -> usize {
        self.inner.legal_move_count()
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn winner(&self) -> Option<&'static str> {
        self.inner.winner().map(player_str)
    }

    fn current_player(&self) -> Option<&'static str> {
        self.inner.current_player().map(player_str)
    }

    fn moves_remaining_this_turn(&self) -> u8 {
        self.inner.moves_remaining_this_turn()
    }

    fn placed_stones(&self) -> Vec<((i32, i32), &'static str)> {
        self.inner
            .placed_stones()
            .into_iter()
            .map(|(coord, player)| (coord, player_str(player)))
            .collect()
    }

    fn move_count(&self) -> u32 {
        self.inner.move_count()
    }

    fn config(&self) -> PyGameConfig {
        PyGameConfig { inner: *self.inner.config() }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn clone(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __copy__(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __repr__(&self) -> String {
        let status = if self.inner.is_terminal() {
            match self.inner.winner() {
                Some(p) => format!("{} wins", player_str(p)),
                None => "draw".into(),
            }
        } else {
            let p = player_str(self.inner.current_player().unwrap());
            let rem = self.inner.moves_remaining_this_turn();
            format!("{p} to move, {rem} left")
        };
        format!(
            "GameState({} stones, {})",
            self.inner.placed_stones().len(),
            status,
        )
    }
}

#[pyclass(name = "MCTSConfig", skip_from_py_object)]
#[derive(Clone)]
struct PyMCTSConfig {
    inner: MCTSConfig,
}

#[pymethods]
impl PyMCTSConfig {
    #[new]
    fn new(n_simulations: u32, m_actions: usize, c_visit: u32, c_scale: f64) -> PyResult<Self> {
        if m_actions < 1 {
            return Err(PyValueError::new_err("m_actions must be >= 1"));
        }
        if !c_scale.is_finite() || c_scale <= 0.0 {
            return Err(PyValueError::new_err("c_scale must be a positive finite number"));
        }
        Ok(PyMCTSConfig {
            inner: MCTSConfig {
                n_simulations,
                m_actions,
                c_visit,
                c_scale,
            },
        })
    }
}

/// Run Gumbel MCTS from a game state.
///
/// Args:
///     game: hexo_rs.GameState (non-terminal)
///     eval_fn: callable(list[GameState]) -> tuple[list[dict[(int,int), float]], list[float]]
///     config: hexo_rs.MCTSConfig
///     seed: optional int for deterministic RNG
///
/// Returns:
///     tuple[tuple[int,int], list[float]] — (action, improved_policy)
#[pyfunction(name = "gumbel_mcts")]
#[pyo3(signature = (game, eval_fn, config, seed=None))]
fn py_gumbel_mcts(
    py: Python<'_>,
    game: &PyGameState,
    eval_fn: Py<PyAny>,
    config: &PyMCTSConfig,
    seed: Option<u64>,
) -> PyResult<((i32, i32), Vec<f64>)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };

    let mut eval_error: Option<PyErr> = None;

    let mut eval = |states: &[GameState]| -> (Vec<HashMap<Coord, f64>>, Vec<f64>) {
        // Build list of PyGameState objects for the callback
        let py_states: Vec<PyGameState> = states
            .iter()
            .map(|s| PyGameState { inner: s.clone() })
            .collect();

        // Call the Python eval_fn
        let result = match eval_fn.call1(py, (py_states,)) {
            Ok(r) => r,
            Err(e) => {
                eval_error = Some(e);
                let dummy = states.iter().map(|_| HashMap::new()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        // Parse result: (list[list[float]], list[float]) — flat logit arrays
        let parsed: Result<(Vec<Vec<f64>>, Vec<f64>), _> = result.extract(py);
        let (logits_per_state, values_raw) = match parsed {
            Ok(v) => v,
            Err(e) => {
                eval_error = Some(e.into());
                let dummy = states.iter().map(|_| HashMap::new()).collect();
                return (dummy, vec![0.0; states.len()]);
            }
        };

        // Map flat logit arrays to coord→logit HashMaps using legal_moves()
        let logits_maps: Vec<HashMap<Coord, f64>> = states
            .iter()
            .zip(logits_per_state)
            .map(|(state, logits)| {
                let moves = state.legal_moves();
                moves.into_iter().zip(logits).collect()
            })
            .collect();

        (logits_maps, values_raw)
    };

    let result = gumbel_mcts::gumbel_mcts(&game.inner, &config.inner, &mut rng, &mut eval)
        .map_err(PyValueError::new_err)?;

    // Check if the eval callback encountered a Python error
    if let Some(e) = eval_error {
        return Err(e);
    }

    Ok((result.action, result.improved_policy))
}

/// Build graph arrays from a game state (Rust-accelerated game_to_graph).
///
/// Returns a dict with keys: features, edge_src, edge_dst, legal_mask, stone_mask,
/// coords, num_nodes — all as flat Python lists ready for torch.tensor().
#[pyfunction(name = "game_to_graph_raw")]
fn py_game_to_graph_raw(py: Python<'_>, game: &PyGameState) -> PyResult<Py<PyAny>> {
    if game.inner.is_terminal() {
        return Err(PyValueError::new_err(
            "Cannot construct graph for a terminal game state.",
        ));
    }

    let g = crate::graph::game_to_graph_raw(&game.inner);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("features", g.features)?;
    dict.set_item("edge_src", g.edge_src)?;
    dict.set_item("edge_dst", g.edge_dst)?;
    dict.set_item("legal_mask", g.legal_mask)?;
    dict.set_item("stone_mask", g.stone_mask)?;
    dict.set_item("neighbor_index", g.neighbor_index)?;
    dict.set_item("coords", g.coords)?;
    dict.set_item("num_nodes", g.num_nodes)?;
    Ok(dict.into())
}

/// Build graph arrays for a batch of game states in parallel.
///
/// Returns a list of dicts, one per state. Uses rayon for parallel construction.
#[pyfunction(name = "game_to_graph_batch")]
fn py_game_to_graph_batch(py: Python<'_>, games: Vec<Py<PyGameState>>) -> PyResult<Vec<Py<PyAny>>> {
    // Extract inner GameStates while we hold the GIL
    let states: Vec<GameState> = games
        .iter()
        .map(|g| g.borrow(py).inner.clone())
        .collect();

    // Parallel graph construction (no GIL needed — pure Rust)
    let graphs = crate::graph::game_to_graph_batch(&states);

    graphs
        .into_iter()
        .map(|g| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("features", g.features)?;
            dict.set_item("edge_src", g.edge_src)?;
            dict.set_item("edge_dst", g.edge_dst)?;
            dict.set_item("legal_mask", g.legal_mask)?;
            dict.set_item("stone_mask", g.stone_mask)?;
            dict.set_item("neighbor_index", g.neighbor_index)?;
            dict.set_item("coords", g.coords)?;
            dict.set_item("num_nodes", g.num_nodes)?;
            Ok(dict.into())
        })
        .collect()
}

#[pymodule]
fn hexo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameConfig>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyMCTSConfig>()?;
    m.add_function(wrap_pyfunction!(py_gumbel_mcts, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_graph_raw, m)?)?;
    m.add_function(wrap_pyfunction!(py_game_to_graph_batch, m)?)?;
    Ok(())
}
