"""Tests for tessera_gemma.utils.shapes — no torch required."""
import pytest
from tessera_gemma.utils.shapes import ShapeSpec, check_shape, assert_divisible, TesseraShapeError


class TestShapeSpec:
    def test_string_dims(self):
        s = ShapeSpec(["B", "T", "H"])
        assert s.dims == ["B", "T", "H"]

    def test_int_dims_coerced(self):
        s = ShapeSpec([2, 64, 256])
        assert s.dims == ["2", "64", "256"]

    def test_rank(self):
        assert ShapeSpec(["B", "T"]).rank == 2


class TestCheckShape:
    def test_exact_match(self):
        syms = {}
        check_shape("x", (2, 16, 512), ShapeSpec(["B", "T", "C"]), syms)
        assert syms == {"B": 2, "T": 16, "C": 512}

    def test_literal_pass(self):
        syms = {}
        check_shape("w", (256, 512), ShapeSpec(["256", "512"]), syms)
        assert syms == {}

    def test_literal_fail(self):
        with pytest.raises(TesseraShapeError, match="dim\\[0\\]"):
            check_shape("w", (128, 512), ShapeSpec(["256", "512"]), {})

    def test_wildcard(self):
        syms = {}
        check_shape("x", (3, 7, 99), ShapeSpec(["?", "?", "?"]), syms)
        assert syms == {}

    def test_symbol_binding(self):
        syms = {}
        check_shape("q", (2, 16, 8, 64), ShapeSpec(["B", "T", "H", "D"]), syms)
        assert syms["H"] == 8

    def test_symbol_conflict(self):
        syms = {"B": 2}
        with pytest.raises(TesseraShapeError, match="conflict"):
            check_shape("x", (3, 16), ShapeSpec(["B", "T"]), syms)

    def test_rank_mismatch(self):
        with pytest.raises(TesseraShapeError, match="rank"):
            check_shape("x", (2, 16), ShapeSpec(["B", "T", "C"]), {})

    def test_chaining(self):
        syms = {}
        check_shape("x", (2, 16, 512), ShapeSpec(["B", "T", "C"]), syms)
        check_shape("y", (2, 16, 512), ShapeSpec(["B", "T", "C"]), syms)
        assert syms == {"B": 2, "T": 16, "C": 512}

    def test_returns_symbols(self):
        syms = {}
        ret = check_shape("x", (2, 8), ShapeSpec(["B", "T"]), syms)
        assert ret is syms


class TestAssertDivisible:
    def test_passes(self):
        assert_divisible(512, 8)

    def test_fails(self):
        with pytest.raises(TesseraShapeError, match="not divisible"):
            assert_divisible(513, 8)

    def test_zero_denom(self):
        with pytest.raises(TesseraShapeError, match="positive"):
            assert_divisible(512, 0)

    def test_message_included(self):
        with pytest.raises(TesseraShapeError, match="head_dim"):
            assert_divisible(513, 16, "head_dim must divide hidden_size")
