use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use hexo_engine::game::{GameConfig, GameState, MoveError};
use hexo_engine::types::Player;

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

#[pymodule]
fn hexo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameConfig>()?;
    m.add_class::<PyGameState>()?;
    Ok(())
}
