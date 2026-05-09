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

    @pytest.mark.xfail(
        reason=(
            "Documented gap (Phase A2 in execution_roadmap.md): call-time "
            "constraint enforcement is not yet implemented. CANONICAL_API.md "
            "§Constraint API explicitly notes this. When this is fixed, this "
            "test should pass and the xfail mark removed."
        ),
        strict=True,
    )
    def test_call_time_check_xfail(self):
        @ts.jit
        def aligned(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]) -> ts.Tensor["M", "N"]:
            ts.require(ts.constraint.Divisible("K", 8))
            return ts.ops.gemm(A, B)

        # K=7 violates Divisible(K, 8). This SHOULD raise but currently doesn't.
        with pytest.raises(ts.TesseraConstraintError):
            aligned(np.random.randn(4, 7).astype(np.float32),
                    np.random.randn(7, 16).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: shape mismatches at numpy layer are still caught (loud failure)
# ─────────────────────────────────────────────────────────────────────────────


class TestNumpyShapeMismatchSanity:
    def test_broadcast_mismatch_raises(self):
        @ts.jit
        def add_fn(
            x: ts.Tensor["B", "S", "D"],
            y: ts.Tensor["B", "S", "D"],
        ) -> ts.Tensor["B", "S", "D"]:
            return ts.ops.add(x, y)

        a = np.ones((2, 4, 8), dtype=np.float32)
        b = np.ones((3, 4, 8), dtype=np.float32)  # B differs
        with pytest.raises(ValueError, match="broadcast"):
            add_fn(a, b)
