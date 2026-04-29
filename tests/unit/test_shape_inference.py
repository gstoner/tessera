"""
test_shape_inference.py — ShapeInferenceEngine tests (Phase 6)

API notes (from diagnostics.py):
  - infer_matmul(lhs, rhs) reports error via reporter on mismatch, returns None
  - infer_elementwise(*shapes) reports error via reporter on mismatch, returns None
  - infer_flash_attn(Q, K, V) reports error via reporter on mismatch, returns None
  - infer_graph(ops) — ops use "inputs" (list[str]) and "output" (str) keys
  - get_shape(name) — returns None on miss (uses dict.get()), not KeyError
  - To surface shape errors, call reporter.raise_if_errors() after inference
"""
from __future__ import annotations

import pytest
from tessera.diagnostics import (
    ErrorReporter,
    ShapeInferenceEngine,
    TesseraShapeError,
)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_basic_2d(self, engine):
        shape = engine.infer_matmul((4, 8), (8, 16))
        assert shape == (4, 16)

    def test_square(self, engine):
        assert engine.infer_matmul((64, 64), (64, 64)) == (64, 64)

    def test_k_mismatch_records_error(self, engine):
        """K mismatch adds error to reporter; infer_matmul returns None."""
        result = engine.infer_matmul((4, 8), (9, 16))
        assert result is None
        assert engine.reporter.has_errors()

    def test_k_mismatch_raises_after_check(self, engine):
        """Calling raise_if_errors() after mismatch raises TesseraShapeError."""
        engine.infer_matmul((4, 8), (9, 16))
        with pytest.raises(TesseraShapeError):
            engine.reporter.raise_if_errors()

    def test_batched_3d(self, engine):
        shape = engine.infer_matmul((2, 4, 8), (2, 8, 16))
        assert shape == (2, 4, 16)

    def test_batched_4d(self, engine):
        shape = engine.infer_matmul((3, 2, 4, 8), (3, 2, 8, 16))
        assert shape == (3, 2, 4, 16)

    def test_1d_lhs_records_error(self, engine):
        """Rank < 2 is invalid; reporter gets an error, returns None."""
        result = engine.infer_matmul((8,), (8, 4))
        assert result is None
        assert engine.reporter.has_errors()

    def test_output_not_same_as_input(self, engine):
        shape = engine.infer_matmul((3, 5), (5, 7))
        assert shape[0] == 3 and shape[1] == 7

    def test_m_1_vector_matmul(self, engine):
        shape = engine.infer_matmul((1, 512), (512, 256))
        assert shape == (1, 256)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

class TestElementwise:
    def test_same_shape_passes(self, engine):
        shape = engine.infer_elementwise((4, 4), (4, 4))
        assert shape == (4, 4)

    def test_single_operand(self, engine):
        shape = engine.infer_elementwise((2, 3, 4))
        assert shape == (2, 3, 4)

    def test_three_operands(self, engine):
        shape = engine.infer_elementwise((8,), (8,), (8,))
        assert shape == (8,)

    def test_shape_mismatch_records_error(self, engine):
        result = engine.infer_elementwise((4, 4), (4, 5))
        assert result is None
        assert engine.reporter.has_errors()

    def test_broadcast_passes(self, engine):
        shape = engine.infer_elementwise((8, 1, 128), (8, 1024, 128))
        assert shape == (8, 1024, 128)

    def test_broadcast_mismatch_records_error(self, engine):
        result = engine.infer_elementwise((3,), (4, 4))
        assert result is None
        assert engine.reporter.has_errors()

    def test_shape_mismatch_raises_after_check(self, engine):
        engine.infer_elementwise((4, 4), (4, 5))
        with pytest.raises(TesseraShapeError):
            engine.reporter.raise_if_errors()


# ---------------------------------------------------------------------------
# Flash attention
# ---------------------------------------------------------------------------

class TestFlashAttention:
    def test_basic_shapes(self, engine):
        out = engine.infer_flash_attn((2, 8, 512, 64),
                                      (2, 8, 512, 64),
                                      (2, 8, 512, 64))
        assert out == (2, 8, 512, 64)

    def test_output_shape_equals_q(self, engine):
        Q = (1, 16, 1024, 128)
        K = (1, 16, 1024, 128)
        V = (1, 16, 1024, 128)
        out = engine.infer_flash_attn(Q, K, V)
        assert out == Q

    def test_kv_seq_mismatch_records_error(self, engine):
        result = engine.infer_flash_attn((1, 8, 512, 64),
                                          (1, 8, 256, 64),   # K seq mismatch
                                          (1, 8, 512, 64))
        assert result is None
        assert engine.reporter.has_errors()

    def test_head_dim_mismatch_records_error(self, engine):
        result = engine.infer_flash_attn((1, 8, 512, 64),
                                          (1, 8, 512, 128),  # head_dim mismatch
                                          (1, 8, 512, 64))
        assert result is None
        assert engine.reporter.has_errors()

    def test_wrong_rank_records_error(self, engine):
        result = engine.infer_flash_attn((8, 512, 64),  # rank 3
                                          (8, 512, 64),
                                          (8, 512, 64))
        assert result is None
        assert engine.reporter.has_errors()

    def test_kv_seq_mismatch_raises_after_check(self, engine):
        engine.infer_flash_attn((1, 8, 512, 64),
                                 (1, 8, 256, 64),
                                 (1, 8, 512, 64))
        with pytest.raises(TesseraShapeError):
            engine.reporter.raise_if_errors()

    def test_cross_attention_kv_seq_different(self, engine):
        """Cross-attention: Q seq != KV seq is VALID (K and V seqs must match)."""
        out = engine.infer_flash_attn((2, 8, 128, 64),
                                       (2, 8, 512, 64),
                                       (2, 8, 512, 64))
        assert out == (2, 8, 128, 64)  # output seq = Q seq


# ---------------------------------------------------------------------------
# Graph inference
# ---------------------------------------------------------------------------

class TestInferGraph:
    def test_matmul_chain(self, engine):
        # infer_graph expects "inputs" (list[str]) and "output" (str)
        engine.set_shape("x",  (4, 512))
        engine.set_shape("w1", (512, 256))
        engine.set_shape("w2", (256, 128))
        ops = [
            {"op": "matmul", "inputs": ["x", "w1"],  "output": "h1"},
            {"op": "matmul", "inputs": ["h1", "w2"], "output": "h2"},
        ]
        results = engine.infer_graph(ops)
        assert results["h1"] == (4, 256)
        assert results["h2"] == (4, 128)

    def test_elementwise_after_matmul(self, engine):
        engine.set_shape("a",    (8, 64))
        engine.set_shape("b",    (64, 32))
        engine.set_shape("bias", (8, 32))
        ops = [
            {"op": "matmul",     "inputs": ["a", "b"],        "output": "c"},
            {"op": "elementwise", "inputs": ["c", "bias"],     "output": "d"},
        ]
        results = engine.infer_graph(ops)
        assert results["c"] == (8, 32)
        assert results["d"] == (8, 32)

    def test_unknown_op_handled_gracefully(self, engine):
        engine.set_shape("x", (4, 4))
        ops = [{"op": "custom_exotic_op", "inputs": ["x"], "output": "y"}]
        results = engine.infer_graph(ops)
        assert results.get("y") is None

    def test_set_and_get_shape(self, engine):
        engine.set_shape("tensor", (3, 4, 5))
        assert engine.get_shape("tensor") == (3, 4, 5)

    def test_get_unknown_shape_returns_none(self, engine):
        """get_shape() uses dict.get() — returns None for unknown keys."""
        result = engine.get_shape("nonexistent_tensor")
        assert result is None

    def test_graph_propagates_shapes_forward(self, engine):
        """After infer_graph, intermediate shapes are queryable."""
        engine.set_shape("x", (2, 4))
        engine.set_shape("w", (4, 8))
        ops = [{"op": "matmul", "inputs": ["x", "w"], "output": "y"}]
        engine.infer_graph(ops)
        assert engine.get_shape("y") == (2, 8)
