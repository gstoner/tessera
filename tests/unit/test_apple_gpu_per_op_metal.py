"""General residency gate — `_apple_gpu_chain_kind` "per_op_metal" (2026-06-17).

A multi-op apple_gpu program where every op has a GPU lane now runs per-op on
Metal instead of demoting the whole program to artifact_only just because the
chain isn't a named fusion. This keeps mixed programs that interleave compute
with structural ops (matmul -> transpose -> gelu) on metal_runtime end-to-end.
Specific fused chains still win (checked first); a program with any non-GPU op
stays conservative (None -> artifact_only).

NB: the functions below are real ``def``s, not lambdas — @jit source-inspects the
body, which is unreliable for lambdas (they'd fall back to eager).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def _gelu(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608028654 * (v + 0.044715 * v**3)))


def _matmul_transpose_gelu(a, b):
    return ts.ops.gelu(ts.ops.transpose(ts.ops.matmul(a, b)))


def _matmul_softmax(a, b):
    return ts.ops.softmax(ts.ops.matmul(a, b))


def _compiled(fn, *arrays):
    f = ts.jit(target="apple_gpu")(fn)
    f(*arrays)  # force compile/run
    return f


# ── recognizer-level: structure, not execution ───────────────────────────────
def test_recognizer_accepts_all_gpu_capable_chain():
    from tessera.compiler.driver import _apple_gpu_chain_kind

    a = np.ones((4, 8), np.float32)
    b = np.ones((8, 16), np.float32)
    f = _compiled(_matmul_transpose_gelu, a, b)
    plan = getattr(f, "cpu_plan", None)
    assert plan is not None, "expected a compiled cpu_plan (def, not lambda)"
    assert _apple_gpu_chain_kind(plan) == "per_op_metal"


def test_specific_fusion_still_wins_over_per_op_metal():
    """matmul -> softmax must stay the named fusion, not per_op_metal."""
    from tessera.compiler.driver import _apple_gpu_chain_kind

    a = np.ones((4, 8), np.float32)
    b = np.ones((8, 16), np.float32)
    f = _compiled(_matmul_softmax, a, b)
    plan = getattr(f, "cpu_plan", None)
    assert plan is not None
    assert _apple_gpu_chain_kind(plan) == "matmul_softmax"


def test_non_gpu_op_keeps_chain_conservative():
    """An op with NO GPU lane must not be claimed per-op metal — the recognizer
    sees lane_for == None and returns None (stays artifact_only)."""
    from tessera.compiler.apple_gpu_envelope import lane_for

    # einsum has no apple_gpu lane today (numpy-only, hard_kernel disposition).
    assert lane_for("tessera.einsum") is None


# ── execution: mixed program stays GPU-resident + correct ─────────────────────
def test_matmul_transpose_gelu_stays_gpu_resident():
    a = _RNG.standard_normal((4, 8)).astype(np.float32)
    b = _RNG.standard_normal((8, 16)).astype(np.float32)
    f = _compiled(_matmul_transpose_gelu, a, b)
    out = np.asarray(f(a, b))
    np.testing.assert_allclose(out, _gelu((a @ b).T), rtol=1e-3, atol=1e-3)
    if DARWIN:
        assert f.execution_kind == "native_gpu"


@pytest.mark.hardware_apple_gpu
def test_mixed_program_no_fallback_on_metal():
    from tessera.compiler import apple_gpu_coverage as cov
    from tests._support.apple import assert_native_apple_jit

    a = _RNG.standard_normal((4, 8)).astype(np.float32)
    b = _RNG.standard_normal((8, 16)).astype(np.float32)
    f = _compiled(_matmul_transpose_gelu, a, b)
    assert cov.fallback_histogram(lambda: f(a, b)) == {}
    assert_native_apple_jit(f)
