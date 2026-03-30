"""Tests for the hexo_rs Python bindings (PyO3)."""

import copy

import pytest

import hexo_rs


# ------------------------------------------------------------------
# GameConfig
# ------------------------------------------------------------------

class TestGameConfig:
    def test_full_hexo_defaults(self):
        cfg = hexo_rs.GameConfig.full_hexo()
        assert cfg.win_length == 6
        assert cfg.placement_radius == 8
        assert cfg.max_moves == 200

    def test_custom_config(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
        assert cfg.win_length == 4
        assert cfg.placement_radius == 4
        assert cfg.max_moves == 50

    def test_repr(self):
        cfg = hexo_rs.GameConfig.full_hexo()
        assert repr(cfg) == "GameConfig(win_length=6, placement_radius=8, max_moves=200)"

    def test_invalid_win_length_zero(self):
        with pytest.raises(ValueError, match="win_length"):
            hexo_rs.GameConfig(win_length=0, placement_radius=8, max_moves=200)

    def test_invalid_win_length_one(self):
        with pytest.raises(ValueError, match="win_length"):
            hexo_rs.GameConfig(win_length=1, placement_radius=8, max_moves=200)

    def test_invalid_placement_radius(self):
        with pytest.raises(ValueError, match="placement_radius"):
            hexo_rs.GameConfig(win_length=4, placement_radius=0, max_moves=200)

    def test_invalid_placement_radius_negative(self):
        with pytest.raises(ValueError, match="placement_radius"):
            hexo_rs.GameConfig(win_length=4, placement_radius=-1, max_moves=200)

    def test_invalid_max_moves_zero(self):
        with pytest.raises(ValueError, match="max_moves"):
            hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=0)


# ------------------------------------------------------------------
# GameState — initial state
# ------------------------------------------------------------------

class TestInitialState:
    def test_default_new(self):
        game = hexo_rs.GameState()
        assert not game.is_terminal()
        assert game.winner() is None

    def test_from_config(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=100)
        game = hexo_rs.GameState(cfg)
        assert not game.is_terminal()

    def test_current_player_is_p2(self):
        game = hexo_rs.GameState()
        assert game.current_player() == "P2"

    def test_moves_remaining_is_2(self):
        game = hexo_rs.GameState()
        assert game.moves_remaining_this_turn() == 2

    def test_p1_stone_at_origin(self):
        game = hexo_rs.GameState()
        stones = game.placed_stones()
        assert ((0, 0), "P1") in stones

    def test_move_count_starts_at_zero(self):
        game = hexo_rs.GameState()
        assert game.move_count() == 0

    def test_legal_moves_nonempty(self):
        game = hexo_rs.GameState()
        moves = game.legal_moves()
        assert len(moves) > 0
        for m in moves:
            assert isinstance(m, tuple)
            assert len(m) == 2

    def test_legal_move_count(self):
        game = hexo_rs.GameState()
        assert game.legal_move_count() == len(game.legal_moves())

    def test_origin_not_in_legal_moves(self):
        game = hexo_rs.GameState()
        assert (0, 0) not in game.legal_moves()


# ------------------------------------------------------------------
# GameState — apply_move
# ------------------------------------------------------------------

class TestApplyMove:
    def test_valid_move_places_stone(self):
        game = hexo_rs.GameState()
        game.apply_move(1, 0)
        assert game.move_count() == 1
        stones = game.placed_stones()
        assert ((1, 0), "P2") in stones

    def test_two_moves_switches_player(self):
        game = hexo_rs.GameState()
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        assert game.current_player() == "P1"

    def test_occupied_raises(self):
        game = hexo_rs.GameState()
        with pytest.raises(ValueError, match="occupied"):
            game.apply_move(0, 0)

    def test_out_of_range_raises(self):
        game = hexo_rs.GameState()
        with pytest.raises(ValueError, match="range"):
            game.apply_move(100, 100)

    def test_game_over_raises(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        assert game.is_terminal()
        with pytest.raises(ValueError, match="over"):
            game.apply_move(5, 0)

    def test_placed_stones_grows(self):
        game = hexo_rs.GameState()
        assert len(game.placed_stones()) == 1
        game.apply_move(1, 0)
        assert len(game.placed_stones()) == 2


# ------------------------------------------------------------------
# GameState — win detection
# ------------------------------------------------------------------

class TestWinDetection:
    def test_p2_wins_4_in_a_row(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        assert game.is_terminal()
        assert game.winner() == "P2"

    def test_no_winner_during_play(self):
        game = hexo_rs.GameState()
        game.apply_move(1, 0)
        assert game.winner() is None


# ------------------------------------------------------------------
# GameState — draw
# ------------------------------------------------------------------

class TestDraw:
    def test_draw_by_move_limit(self):
        cfg = hexo_rs.GameConfig(win_length=6, placement_radius=8, max_moves=4)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(0, 1)
        game.apply_move(-1, 0)
        game.apply_move(0, -1)
        assert game.is_terminal()
        assert game.winner() is None


# ------------------------------------------------------------------
# GameState — terminal state accessors
# ------------------------------------------------------------------

class TestTerminalState:
    def test_current_player_none_when_terminal(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        assert game.is_terminal()
        assert game.current_player() is None

    def test_moves_remaining_zero_when_terminal(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        assert game.moves_remaining_this_turn() == 0

    def test_legal_moves_empty_when_terminal(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        assert game.legal_moves() == []
        assert game.legal_move_count() == 0


# ------------------------------------------------------------------
# GameState — clone / copy
# ------------------------------------------------------------------

class TestClone:
    def test_clone_method(self):
        game = hexo_rs.GameState()
        cloned = game.clone()
        game.apply_move(1, 0)
        assert game.move_count() == 1
        assert cloned.move_count() == 0

    def test_copy_copy(self):
        game = hexo_rs.GameState()
        cloned = copy.copy(game)
        game.apply_move(1, 0)
        assert cloned.move_count() == 0

    def test_copy_deepcopy(self):
        game = hexo_rs.GameState()
        cloned = copy.deepcopy(game)
        game.apply_move(1, 0)
        assert cloned.move_count() == 0

    def test_clone_independence_both_directions(self):
        game = hexo_rs.GameState()
        game.apply_move(1, 0)
        cloned = game.clone()
        cloned.apply_move(2, 0)
        # Original unaffected by clone mutation
        assert game.move_count() == 1
        assert cloned.move_count() == 2


# ------------------------------------------------------------------
# GameState — config accessor
# ------------------------------------------------------------------

class TestConfigAccessor:
    def test_config_returns_gameconfig(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=5, max_moves=100)
        game = hexo_rs.GameState(cfg)
        got = game.config()
        assert got.win_length == 4
        assert got.placement_radius == 5
        assert got.max_moves == 100


# ------------------------------------------------------------------
# GameState — repr
# ------------------------------------------------------------------

class TestRepr:
    def test_repr_initial(self):
        game = hexo_rs.GameState()
        r = repr(game)
        assert "P2 to move" in r
        assert "1 stones" in r

    def test_repr_terminal(self):
        cfg = hexo_rs.GameConfig(win_length=4, placement_radius=8, max_moves=200)
        game = hexo_rs.GameState(cfg)
        game.apply_move(1, 0)
        game.apply_move(2, 0)
        game.apply_move(0, 3)
        game.apply_move(0, -3)
        game.apply_move(3, 0)
        game.apply_move(4, 0)
        r = repr(game)
        assert "P2 wins" in r


# ------------------------------------------------------------------
# Fuzz: random games with invariant checks
# ------------------------------------------------------------------

class TestFuzz:
    def test_random_games_with_invariants(self):
        """Play 200 random games, verify post-game invariants."""
        import random
        rng = random.Random(42)

        for _ in range(200):
            cfg = hexo_rs.GameConfig(win_length=4, placement_radius=4, max_moves=50)
            game = hexo_rs.GameState(cfg)
            while not game.is_terminal():
                moves = game.legal_moves()
                assert len(moves) > 0, "legal_moves() empty on non-terminal state"
                q, r = rng.choice(moves)
                game.apply_move(q, r)

            # Post-game invariants
            assert game.is_terminal()
            assert game.legal_moves() == []
            assert game.legal_move_count() == 0
            assert game.current_player() is None
            assert game.moves_remaining_this_turn() == 0
            assert game.move_count() <= cfg.max_moves
            w = game.winner()
            assert w in (None, "P1", "P2")
