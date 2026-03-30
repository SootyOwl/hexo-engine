# Phase 1: GNN Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PyTorch Geometric GNN that takes a HeXO `GameState` (via the `hexo-rs` PyO3 bindings) as a graph and outputs a policy distribution over legal moves + a scalar value estimate.

**Architecture:** Graph construction converts `GameState` into a PyG `Data` object: one node per placed stone + one node per empty legal cell, 7-dim feature vectors, undirected hex-adjacency edges. The GNN uses GATv2Conv message-passing layers with residual connections for the representation, a per-node MLP policy head (masked to empty/legal nodes), and a global-pooling MLP value head producing a scalar in [-1, 1]. Variable-size graphs are batched via PyG's `Batch.from_data_list()`.

**Tech Stack:** Python 3.13, PyTorch 2.6+, PyTorch Geometric 2.7+, `hexo-rs` (PyO3 bindings), `uv` for packaging, `pytest`. ROCm 7.2 via toolbox for GPU (CPU-first development).

**Key decision:** Neither MiniZero nor LightZero are viable bases — both assume fixed-size 2D grids with CNN architectures throughout. We build a custom lean stack, borrowing algorithmic reference (Gumbel AZ sequential halving, MCTS structure, KL-divergence policy loss) from their source code.

---

## Requirements

### Graph Construction

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-1 | Convert `GameState` to a PyG `Data` object with nodes for all placed stones and all legal empty cells | AC-1a: Node count equals `len(placed_stones()) + len(legal_moves())`. AC-1b: Function accepts any non-terminal `GameState` from `hexo-rs`. AC-1c: Function raises `ValueError` on terminal states (callers use game outcome directly) | T3 |
| REQ-2 | 7-dim node feature vector: stone type (3-hot: P1/P2/empty), current player (+1/-1), moves remaining (0.5/1.0), normalised axial q, normalised axial r | AC-2a: P1 stone nodes have `[1,0,0,...]`. AC-2b: P2 stone nodes have `[0,1,0,...]`. AC-2c: Empty cell nodes have `[0,0,1,...]`. AC-2d: Current player feature is +1.0 when P1 to move, -1.0 when P2. AC-2e: Moves remaining is 1.0 for 2-left, 0.5 for 1-left. AC-2f: Axial coords are centred on the centroid of all nodes and normalised by the maximum spread | T3 |
| REQ-3 | Undirected edges between all hex-adjacent node pairs (hex distance == 1) | AC-3a: Every edge (u, v) has a reverse edge (v, u). AC-3b: No self-loops. AC-3c: Edge count matches expected hex adjacency for test positions | T3 |
| REQ-4 | `Data` object carries a `legal_mask` boolean tensor (True for empty/legal nodes) and a `coords` int tensor of axial coordinates per node | AC-4a: `legal_mask` is False for all stone nodes, True for all empty nodes. AC-4b: `coords` shape is `(N, 2)` with correct axial coordinates | T3 |
| REQ-5 | Deterministic node ordering: placed stones first (sorted by coord), then empty legal cells (sorted by coord) | AC-5: Calling `game_to_graph` twice on the same `GameState` produces identical `Data` objects | T3 |

### GNN Model

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-6 | GATv2Conv representation network with configurable depth, width, and attention heads, plus residual connections | AC-6a: Output shape is `(N, hidden_dim)` for any input graph size. AC-6b: Gradients flow through all layers | T4 |
| REQ-7 | Policy head: per-node MLP producing logits for legal (empty) nodes only | AC-7a: Output length equals number of legal moves. AC-7b: `softmax(logits)` sums to ~1.0 | T5 |
| REQ-8 | Value head: global mean pooling + MLP with tanh, producing scalar in [-1, 1] | AC-8a: Output is a scalar. AC-8b: Output is in range [-1, 1]. AC-8c: Gradients flow through pooling | T5 |
| REQ-9 | Combined `HeXONet` model with `forward_batch()` as the primary interface and `forward()` as a convenience wrapper | AC-9a: `forward()` returns `(policy_logits, value)` tuple. AC-9b: `forward_batch()` returns `(list[policy_logits], values_tensor)` with correct shapes per graph. AC-9c: `forward(graph)` produces identical results to `forward_batch([graph])` unpacked | T5, T7 |

### Loss Function

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-10 | AlphaZero loss: cross-entropy (policy) + MSE (value). Policy cross-entropy is `-(target * log_softmax(logits)).sum()` where `target` is a probability distribution from MCTS, NOT `F.cross_entropy` (which expects class indices) | AC-10a: Loss is a positive finite scalar. AC-10b: Gradients flow to both policy logits and value. AC-10c: Near-perfect predictions produce near-zero loss | T6 |

### Batching

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-11 | Variable-size game graphs can be collated into a single PyG `Batch` via `Batch.from_data_list()` | AC-11a: Total node count equals sum of individual graph node counts. AC-11b: `legal_mask` is preserved across batching. AC-11c: `num_graphs` matches input list length | T7 |
| REQ-12 | Batched model forward produces correct per-graph policy logits and values | AC-12a: `policy_logits_list[i]` length matches `legal_mask.sum()` for graph `i`. AC-12b: `values` tensor shape is `(batch_size,)` | T7 |

### Configuration

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-13 | All model hyperparameters in a single `ModelConfig` dataclass with sensible defaults | AC-13a: Default config instantiates without arguments. AC-13b: Fields include `node_features=7`, `hidden_dim=128`, `num_layers=6`, `num_heads=4`, `policy_hidden=64`, `value_hidden=64`, `dropout=0.0`. AC-13c: Custom values can be passed at construction | T2 |

### Device Compatibility

| ID | Requirement | Acceptance Criteria | Task |
|----|-------------|---------------------|------|
| REQ-14 | Model works on CPU and ROCm GPU | AC-14a: Forward + backward pass completes on CPU. AC-14b: Forward + backward pass completes on ROCm GPU (skipped if unavailable). AC-14c: Model can overfit a single position (loss decreases over 50 optimiser steps) | T8 |

---

## File Structure

```
hexo-a0/
├── pyproject.toml              # uv project, deps: torch, torch-geometric, hexo-rs, pytest
├── src/
│   └── hexo_a0/
│       ├── __init__.py         # Package root, public exports
│       ├── config.py           # ModelConfig dataclass
│       ├── graph.py            # GameState → PyG Data conversion
│       ├── model.py            # HeXONet: GATv2Conv repr + policy/value heads
│       └── loss.py             # AlphaZero loss function
└── tests/
    ├── conftest.py             # Shared fixtures (game states at various stages)
    ├── test_graph.py           # Graph construction + batching tests
    ├── test_model.py           # Model config + forward pass tests (repr, heads, combined)
    ├── test_loss.py            # Loss computation tests
    └── test_integration.py     # End-to-end pipeline + GPU smoke tests
```

**Implementation notes:**
- `placed_stones()` returns HashMap-order data from Rust. The Python side **must sort** by coordinate for determinism (REQ-5).
- `current_player` and `moves_remaining` are global game-state properties broadcast identically to all nodes (stones and empties alike).
- Centroid-based coordinate normalisation provides translation invariance for the infinite board. Tradeoff: adding a stone shifts all nodes' coordinate features. This is a deliberate choice — static normalisation would break when games drift far from origin.
- Edge construction uses a coord→index dict lookup: for each node, check its 6 hex neighbours in the dict. This is O(6N), not O(N^2).
- Policy-logit-to-coordinate mapping: callers recover legal-move coordinates via `data.coords[data.legal_mask]`, which is in the same order as the policy logits due to REQ-5.
- Edge features and axis-aligned skip edges are documented as future extensions for Phase 2+ tuning, not Phase 1 scope.
- Inference latency profiling belongs in Phase 2 when integrated with MCTS. Phase 1 focuses on correctness.

---

## Task 1: Project Scaffolding

**Files:** Create `hexo-a0/pyproject.toml`, `hexo-a0/src/hexo_a0/__init__.py`

- [ ] Run `uv init` in `hexo-a0/`, configure `pyproject.toml` with dependencies: `torch>=2.6`, `torch-geometric>=2.7`, dev dependency `pytest>=8.0`
- [ ] Install `hexo-rs` into the venv: run `maturin develop` inside `../hexo-rs/` (this builds and installs the PyO3 extension into the active venv). Do NOT also run `uv pip install -e` — `maturin develop` is the canonical method
- [ ] Create `src/hexo_a0/__init__.py` with a docstring
- [ ] Verify imports: `import hexo_a0; import hexo_rs; import torch; import torch_geometric` all succeed
- [ ] Commit

---

## Task 2: Model Config

**Files:** Create `src/hexo_a0/config.py`, create `tests/test_model.py` (started here, extended in T4/T5)

- [ ] Write failing tests for `ModelConfig` in `test_model.py`: default values match REQ-13 AC-13b, custom values override correctly
- [ ] Run tests — verify they fail
- [ ] Implement `ModelConfig` as a `dataclass` with the fields from REQ-13
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 3: Graph Construction

**Files:** Create `src/hexo_a0/graph.py`, create `tests/conftest.py`, create `tests/test_graph.py`

**Node ordering convention:** Placed stones first (sorted by coord), then empty legal cells (sorted by coord). This makes it easy to separate stone nodes from legal-move nodes via the `legal_mask`.

**Coordinate normalisation:** Compute centroid of all node coordinates (stones + legal cells). Compute spread as `max(max(|q - centroid_q|), max(|r - centroid_r|), 1)`. Normalised coords: `(q - centroid_q) / spread`, `(r - centroid_r) / spread`.

- [ ] Create `conftest.py` with fixtures: `initial_game` (default `GameState()`), `small_game` (4-in-a-row config with ~5 stones played, P2 to move)
- [ ] Write failing tests for `game_to_graph()` covering:
  - Node count matches `len(placed_stones()) + len(legal_moves())` (AC-1a)
  - Feature tensor shape is `(N, 7)`, dtype `float32` (AC-2a–f)
  - P1 stone features: `[1,0,0,...]` (AC-2a)
  - Empty cell features: `[0,0,1,...]` (AC-2c)
  - Current player feature correct for P2-to-move and P1-to-move states (AC-2d)
  - Moves remaining feature correct for 2-left vs 1-left (AC-2e)
  - Normalised coords are approximately centred (mean near 0) (AC-2f)
  - Edges are undirected: every (u,v) has a reverse (v,u) (AC-3a)
  - `legal_mask` is False for stone nodes, True for empty nodes (AC-4a)
  - `coords` tensor has correct shape and values (AC-4b)
  - Occupied cell (0,0) is not in legal moves (AC-1a)
  - Determinism: two calls on same state produce identical results (AC-5)
  - Terminal state raises `ValueError` (AC-1c)
  - Policy-logit-to-coordinate roundtrip: `data.coords[data.legal_mask]` returns the same coordinates as `game.legal_moves()` in the same order (critical invariant for Phase 2 MCTS integration)
- [ ] Run tests — verify they fail
- [ ] Implement `game_to_graph(game) -> Data` using the node ordering convention and feature spec above. Use `HEX_DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]` for edge construction
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 4: GNN Representation Network

**Files:** Create `src/hexo_a0/model.py`, create `tests/test_model.py`

The representation network projects 7-dim input to `hidden_dim`, then applies `num_layers` GATv2Conv layers with residual connections and LayerNorm. Each GATv2Conv uses `num_heads` attention heads where `head_dim = hidden_dim // num_heads`. Validate that `hidden_dim % num_heads == 0` at construction.

- [ ] Write failing tests for `RepresentationNetwork`:
  - Output shape is `(N, hidden_dim)` for various graph sizes (AC-6a)
  - Output dtype is `float32`
  - Gradients flow back to input (AC-6b)
- [ ] Run tests — verify they fail
- [ ] Implement `RepresentationNetwork(config)` with: `Linear` input projection, `ModuleList` of `GATv2Conv` layers with `LayerNorm` and residual connections, optional dropout
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 5: Policy Head, Value Head, and HeXONet

**Files:** Modify `src/hexo_a0/model.py`, modify `tests/test_model.py`

- [ ] Write failing tests for `PolicyHead`:
  - Output length equals number of True entries in `legal_mask` (AC-7a)
  - `softmax(output)` sums to ~1.0 (AC-7b)
- [ ] Write failing tests for `ValueHead`:
  - Output is a scalar (AC-8a)
  - Output is in [-1, 1] (AC-8b)
  - Gradients flow through pooling (AC-8c)
- [ ] Write failing tests for `HeXONet`:
  - `forward()` returns `(policy_logits, value)` with correct shapes (AC-9a)
  - Gradients flow to all parameters
- [ ] Run tests — verify they fail
- [ ] Implement:
  - `PolicyHead(hidden_dim, policy_hidden)`: MLP producing per-node logits, indexed by `legal_mask`
  - `ValueHead(hidden_dim, value_hidden)`: global mean pooling → MLP → tanh → scalar
  - `HeXONet(config)`: composes `RepresentationNetwork`, `PolicyHead`, `ValueHead`. `forward_batch(batch)` is the primary path; `forward(x, edge_index, legal_mask)` is a convenience wrapper that creates a single-element batch and unpacks the result
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 6: Loss Function

**Files:** Create `src/hexo_a0/loss.py`, create `tests/test_loss.py`

Loss = `alpha * policy_loss + beta * value_loss`. Policy loss is cross-entropy: `-(target * log_softmax(logits, dim=0)).sum()` where `target` is a probability distribution from MCTS (NOT `F.cross_entropy` which expects class indices). Value loss is MSE. L2 regularisation is handled by the optimiser (`weight_decay`), not the loss function. KL-divergence mode for Gumbel AZ is deferred to Phase 2.

- [ ] Write failing tests for `alphazero_loss()`:
  - Returns scalar loss + components dict with `policy_loss` and `value_loss` keys (AC-10a)
  - Loss is positive and finite (AC-10a)
  - Near-perfect predictions produce near-zero loss (AC-10c)
  - Gradients flow to policy logits and value (AC-10b)
- [ ] Run tests — verify they fail
- [ ] Implement `alphazero_loss(policy_logits, policy_target, value, value_target, *, alpha=1.0, beta=1.0)` using `F.log_softmax` and `F.mse_loss`
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 7: Batching Support

**Files:** Modify `src/hexo_a0/graph.py`, modify `src/hexo_a0/model.py`, modify `tests/test_graph.py`, modify `tests/test_model.py`

PyG's `Batch.from_data_list()` handles variable-size graph batching by concatenating nodes and offsetting edge indices. Custom tensors (`legal_mask`, `coords`) are concatenated along dim 0 automatically.

- [ ] Write failing tests for `Batch.from_data_list()` on game graphs:
  - Returns a PyG `Batch` with correct `num_graphs` (AC-11c)
  - Total node count is sum of individual graphs (AC-11a)
  - `legal_mask` total is preserved (AC-11b)
- [ ] Write failing tests for `HeXONet.forward_batch()`:
  - Returns `(list[policy_logits], values)` with correct per-graph shapes (AC-12a, AC-12b)
  - `forward(graph)` produces identical results to `forward_batch` with a single graph (AC-9c)
- [ ] Run tests — verify they fail
- [ ] Batching uses `Batch.from_data_list()` directly — no custom wrapper needed
- [ ] Implement `HeXONet.forward_batch(batch)`: run representation on concatenated graph, then extract per-graph policy logits and values using `batch.batch` assignments
- [ ] Run tests — all pass
- [ ] Commit

---

## Task 8: Integration Tests and GPU Smoke Test

**Files:** Create `tests/test_integration.py`, update `src/hexo_a0/__init__.py` with public exports

- [ ] Write end-to-end test: create a `GameState`, convert to graph, run `HeXONet` forward, compute `alphazero_loss` with fake MCTS targets, verify loss is finite and gradients reach all parameters (AC-14a)
- [ ] Write batched end-to-end test: multiple game states at different stages, collate, batched forward, per-graph loss, backward (AC-14a)
- [ ] Write overfit test: train on a single position for 50 steps, verify loss decreases (AC-14c)
- [ ] Write GPU smoke tests (skip if `torch.cuda.is_available()` is False): forward + backward on ROCm device, verify output tensors are on GPU (AC-14b)
- [ ] Update `__init__.py` to export `ModelConfig`, `game_to_graph`, `HeXONet`, `alphazero_loss`
- [ ] Run all tests — all pass on CPU
- [ ] Run GPU tests inside ROCm toolbox: `toolbox run -c llama-rocm-7.2 ...` (AC-14b)
- [ ] Commit

---

## Summary

| Task | What | Requirements |
|------|------|-------------|
| 1 | Project scaffolding (uv, deps) | — |
| 2 | ModelConfig dataclass | REQ-13 |
| 3 | Graph construction: GameState → PyG Data | REQ-1–5 |
| 4 | GATv2Conv representation network | REQ-6 |
| 5 | Policy head, value head, HeXONet | REQ-7–9 |
| 6 | AlphaZero loss function | REQ-10 |
| 7 | Batching (PyG Batch + forward_batch) | REQ-11–12 |
| 8 | Integration tests + GPU smoke test | REQ-14 |

Every REQ has labeled ACs; every AC maps to at least one test.
