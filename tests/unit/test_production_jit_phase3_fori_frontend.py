"""Phase-G G-C — `@jit`-style bounded-loop front-end
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`jit_fori_loop` / `build_fori_loop` trace a natural bounded loop body into the
`tessera.control_for` Graph-IR op and execute it: on apple_gpu through the
Target-IR path (control_for -> tessera-opt -> tessera_apple.gpu.control_loop ->
run_graph_loop_f32); on cpu the scf.for is compiled natively. This is the
front-end half of "@jit -> tessera.control_for".

Apple GPU cases need the runtime + a built tessera-opt; CPU cases need only the
LLVM JIT lane. Each skips when its prerequisites are missing.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import (
    TesseraJitError,
    _find_tessera_opt,
    build_fori_loop,
    jit_fori_loop,
)

_GPU = agb.is_available() and jb.is_available() and _find_tessera_opt() is not None
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / tessera-opt unavailable")
cpu = pytest.mark.skipif(not jb.is_available(), reason="libtessera_jit unavailable")


def _np_silu(z):
    return z / (1.0 + np.exp(-z))


# --- IR shape (no GPU/runtime needed) --------------------------------------- #
def test_frontend_builds_control_for_op():
    """The front-end produces a graph carrying tessera.control_for with the
    executable body payload."""
    d, trip = 8, 4
    g = build_fori_loop(
        trip, lambda g, c, w: g.matmul(c, w),
        init_shape=(1, d), const_shapes=[(d, d)], target="apple_gpu")
    mlir = g._emit_control_for_mlir()
    assert "tessera.control_for" in mlir
    assert "stop = 4 : i64" in mlir
    assert "carry_arg_index = 0 : i64" in mlir
    assert "body_opcodes = array<i32: 0>" in mlir  # one matmul


# --- apple_gpu execution via the Target-IR control_loop path ---------------- #
@gpu
@pytest.mark.parametrize("d,trip", [(8, 4), (16, 5)])
def test_frontend_linear_recurrence_matches_numpy(d, trip):
    rng = np.random.default_rng(d * 7 + trip)
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = jit_fori_loop(trip, lambda g, c, w_: g.matmul(c, w_), init=c0, consts=[w])
    ref = c0.copy()
    for _ in range(trip):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_frontend_silu_matmul_body_matches_numpy():
    rng = np.random.default_rng(3)
    d, trip = 16, 6
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = jit_fori_loop(
        trip, lambda g, c, w_: g.silu(g.matmul(c, w_)), init=c0, consts=[w])
    ref = c0.copy()
    for _ in range(trip):
        ref = _np_silu(ref @ w)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_frontend_prenorm_residual_block_matches_numpy():
    """A pre-norm residual transformer-ish block iterated as a GPU loop."""
    rng = np.random.default_rng(2026)
    d, trip = 32, 5
    c0 = (rng.standard_normal((1, d)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def body(g, c, w_):
        return g.add(c, g.matmul(g.rmsnorm(c), w_))

    out = jit_fori_loop(trip, body, init=c0, consts=[w])

    def np_rms(x, eps=1e-5):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

    ref = c0.copy()
    for _ in range(trip):
        ref = ref + np_rms(ref) @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_frontend_two_invariant_consts():
    """Multiple loop-invariant operands thread through the front-end + the
    control_loop arg ABI."""
    rng = np.random.default_rng(11)
    d, trip = 16, 4
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w1 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    w2 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = jit_fori_loop(
        trip, lambda g, c, a, b: g.matmul(g.silu(g.matmul(c, a)), b),
        init=c0, consts=[w1, w2])
    ref = c0.copy()
    for _ in range(trip):
        ref = _np_silu(ref @ w1) @ w2
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@gpu
def test_frontend_drives_control_loop_op():
    """The apple_gpu front-end runs through the lowered control_loop op."""
    d, trip = 8, 3
    g = build_fori_loop(
        trip, lambda g, c, w: g.matmul(c, w),
        init_shape=(1, d), const_shapes=[(d, d)], target="apple_gpu")
    c0 = np.ones((1, d), np.float32) / d
    w = np.eye(d, dtype=np.float32)
    g.run_via_target_ir(c0, w)
    assert g.last_dispatch() == ["control_loop"]


@gpu
def test_frontend_via_target_ir_matches_direct_dispatch():
    """IR-driven path is bit-identical to forcing the direct in-memory run."""
    rng = np.random.default_rng(99)
    d, trip = 16, 5
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    body = lambda g, c, w_: g.silu(g.matmul(c, w_))  # noqa: E731
    via = jit_fori_loop(trip, body, init=c0, consts=[w], via_target_ir=True)
    direct = jit_fori_loop(trip, body, init=c0, consts=[w], via_target_ir=False)
    np.testing.assert_allclose(via, direct, rtol=1e-6, atol=1e-6)


# --- cpu lane: same front-end, scf.for compiled natively -------------------- #
@cpu
def test_frontend_cpu_lane_matches_numpy():
    rng = np.random.default_rng(7)
    d, trip = 8, 5
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = jit_fori_loop(
        trip, lambda g, c, w_: g.matmul(c, w_), init=c0, consts=[w], target="cpu")
    ref = c0.copy()
    for _ in range(trip):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
