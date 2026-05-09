"""Dynamic-shape support audit (Phase A2).

This file is the executable counterpart of
`docs/spec/SHAPE_SYSTEM.md` §10 "Dynamic Shape Support Matrix". Each test
documents one cell of the matrix — a backend × symbolic-dim case — so
regressions are caught and the matrix stays grounded in real behavior.

If you add a new executing backend, add a cell here.
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

import tessera as ts


_IS_DARWIN = platform.system() == "Darwin"


# ─────────────────────────────────────────────────────────────────────────────
# CPU reference path (no target= argument)
# ─────────────────────────────────────────────────────────────────────────────


class TestCPUReferenceDynamicShapes:
    def test_two_concrete_shapes_one_decorator(self):
        """Same @jit accepts different concrete batch / seq shapes per call."""
        @ts.jit
        def add_fn(
            x: ts.Tensor["B", "S", "D"],
            y: ts.Tensor["B", "S", "D"],
        ) -> ts.Tensor["B", "S", "D"]:
            return ts.ops.add(x, y)

        a1 = np.ones((2, 4, 8), dtype=np.float32)
        b1 = np.ones((2, 4, 8), dtype=np.float32)
        assert add_fn(a1, b1).shape == (2, 4, 8)

        a2 = np.ones((3, 5, 8), dtype=np.float32)
        b2 = np.ones((3, 5, 8), dtype=np.float32)
        assert add_fn(a2, b2).shape == (3, 5, 8)

    def test_symbolic_K_flows_through_gemm(self):
        @ts.jit
        def matmul(
            A: ts.Tensor["M", "K"],
            B: ts.Tensor["K", "N"],
        ) -> ts.Tensor["M", "N"]:
            return ts.ops.gemm(A, B)

        # K differs between calls; result shape adapts
        out_k8 = matmul(np.random.randn(4, 8).astype(np.float32),
                        np.random.randn(8, 16).astype(np.float32))
        out_k16 = matmul(np.random.randn(4, 16).astype(np.float32),
                         np.random.randn(16, 16).astype(np.float32))
        assert out_k8.shape == (4, 16)
        assert out_k16.shape == (4, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Apple targets (require Darwin)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _IS_DARWIN, reason="Apple targets require Darwin")
class TestAppleCPUDynamicShapes:
    def test_gemm_accepts_dynamic_K(self):
        @ts.jit(target="apple_cpu")
        def matmul(
            A: ts.Tensor["M", "K"],
            B: ts.Tensor["K", "N"],
        ) -> ts.Tensor["M", "N"]:
            return ts.ops.gemm(A, B)

        out_k8 = matmul(np.random.randn(4, 8).astype(np.float32),
                        np.random.randn(8, 16).astype(np.float32))
        out_k16 = matmul(np.random.randn(4, 16).astype(np.float32),
                         np.random.randn(16, 16).astype(np.float32))
        assert out_k8.shape == (4, 16)
        assert out_k16.shape == (4, 16)


@pytest.mark.skipif(not _IS_DARWIN, reason="Apple targets require Darwin")
class TestAppleGPUDynamicShapes:
    def test_gemm_accepts_dynamic_K(self):
        @ts.jit(target="apple_gpu")
        def matmul(
            A: ts.Tensor["M", "K"],
            B: ts.Tensor["K", "N"],
        ) -> ts.Tensor["M", "N"]:
            return ts.ops.gemm(A, B)

        out = matmul(np.random.randn(4, 8).astype(np.float32),
                     np.random.randn(8, 16).astype(np.float32))
        assert out.shape == (4, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint enforcement — decoration-time vs call-time
# ─────────────────────────────────────────────────────────────────────────────


class TestConstraintEnforcement:
    def test_decoration_time_check_with_bindings(self):
        """`@jit(bindings={K:7})` + Divisible(K,8) raises at decoration time."""
        with pytest.raises(ts.TesseraConstraintError, match="not divisible"):
            @ts.jit(bindings={"K": 7})
            def bad(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
                ts.require(ts.constraint.Divisible("K", 8))
                return ts.ops.gemm(A, B)

    def test_decoration_time_check_with_valid_bindings(self):
        """`@jit(bindings={K:8})` + Divisible(K,8) decorates successfully."""
        @ts.jit(bindings={"K": 8})
        def good(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Divisible("K", 8))
            return ts.ops.gemm(A, B)

        out = good(np.random.randn(4, 8).astype(np.float32),
                   np.random.randn(8, 16).astype(np.float32))
        assert out.shape == (4, 16)

    def test_call_time_check_divisible(self):
        """Phase A2-followup: constraint violations raise at first call."""
        @ts.jit
        def aligned(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Divisible("K", 8))
            return ts.ops.gemm(A, B)

        # Aligned: passes
        out = aligned(
            np.random.randn(4, 8).astype(np.float32),
            np.random.randn(8, 16).astype(np.float32),
        )
        assert out.shape == (4, 16)

        # K=7 violates Divisible(K, 8) → raises at call time.
        with pytest.raises(ts.TesseraConstraintError, match="not divisible"):
            aligned(
                np.random.randn(4, 7).astype(np.float32),
                np.random.randn(7, 16).astype(np.float32),
            )

    def test_call_time_inconsistent_dim_binding_raises(self):
        """If the same symbolic dim resolves to two different concrete values
        across args, raise with a clear message."""
        @ts.jit
        def matmul(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
            return ts.ops.gemm(A, B)

        with pytest.raises(ts.TesseraConstraintError, match="Inconsistent binding"):
            matmul(
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(7, 16).astype(np.float32),  # K=7 vs K=8
            )

    def test_call_time_check_range_constraint(self):
        """`Range(dim, lo, hi)` at call time."""
        @ts.jit
        def bounded(x: ts.Tensor["B", "S"]) -> ts.Tensor["B", "S"]:
            ts.require(ts.constraint.Range("S", 4, 32))
            return ts.ops.add(x, x)

        # In range
        assert bounded(np.zeros((2, 16), dtype=np.float32)).shape == (2, 16)
        # Below
        with pytest.raises(ts.TesseraConstraintError, match=r"S.*below|out of range|lo"):
            bounded(np.zeros((2, 2), dtype=np.float32))
        # Above
        with pytest.raises(ts.TesseraConstraintError, match=r"S.*above|out of range|hi"):
            bounded(np.zeros((2, 64), dtype=np.float32))

    def test_call_time_check_equal_constraint(self):
        """`Equal(dim_a, dim_b)` at call time enforces cross-arg dim equality."""
        @ts.jit
        def square(
            A: ts.Tensor["M", "K"],
            B: ts.Tensor["K", "N"],
        ) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Equal("M", "N"))
            return ts.ops.gemm(A, B)

        # M == N  passes
        assert square(np.random.randn(8, 4).astype(np.float32),
                      np.random.randn(4, 8).astype(np.float32)).shape == (8, 8)
        # M != N  fails
        with pytest.raises(ts.TesseraConstraintError, match=r"M.*N|equal|Equal"):
            square(np.random.randn(8, 4).astype(np.float32),
                   np.random.randn(4, 16).astype(np.float32))

    def test_call_time_check_multiple_constraints(self):
        """Multiple `require()`s — first violation raises with that constraint's message."""
        @ts.jit
        def doubly_constrained(
            A: ts.Tensor["M", "K"],
            B: ts.Tensor["K", "N"],
        ) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Divisible("K", 8))
            ts.require(ts.constraint.Range("M", 1, 64))
            return ts.ops.gemm(A, B)

        # Violates Divisible(K, 8)
        with pytest.raises(ts.TesseraConstraintError, match="not divisible"):
            doubly_constrained(np.random.randn(4, 7).astype(np.float32),
                                np.random.randn(7, 8).astype(np.float32))
        # Violates Range(M, 1, 64)
        with pytest.raises(ts.TesseraConstraintError, match=r"M.*range|out of range|hi"):
            doubly_constrained(np.random.randn(128, 8).astype(np.float32),
                                np.random.randn(8, 8).astype(np.float32))

    def test_call_time_check_caches_per_shape(self):
        """Repeated calls with the same shape should re-check at most once."""
        @ts.jit
        def aligned(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Divisible("K", 8))
            return ts.ops.gemm(A, B)

        # Call twice with same shape — second call should hit the cache (no error,
        # and `_constraint_cache` should have exactly one entry).
        for _ in range(3):
            aligned(
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(8, 16).astype(np.float32),
            )
        cache = getattr(aligned, "_constraint_cache", None)
        assert cache is not None
        assert len(cache) == 1
        # Different shape → second cache entry
        aligned(
            np.random.randn(4, 16).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32),
        )
        assert len(cache) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: shape mismatches at numpy layer are still caught (loud failure)
# ─────────────────────────────────────────────────────────────────────────────


class TestShapeMismatchDiagnostics:
    def test_inconsistent_dim_caught_before_numpy(self):
        """When two args share a symbolic dim that resolves to different
        concrete values, the call-time constraint check (Phase A2-followup)
        raises ``TesseraConstraintError`` *before* numpy gets a chance to
        raise its own broadcast error — strictly better diagnostic.
        """
        @ts.jit
        def add_fn(
            x: ts.Tensor["B", "S", "D"],
            y: ts.Tensor["B", "S", "D"],
        ) -> ts.Tensor["B", "S", "D"]:
            return ts.ops.add(x, y)

        a = np.ones((2, 4, 8), dtype=np.float32)
        b = np.ones((3, 4, 8), dtype=np.float32)  # B differs
        with pytest.raises(ts.TesseraConstraintError, match="Inconsistent binding"):
            add_fn(a, b)
