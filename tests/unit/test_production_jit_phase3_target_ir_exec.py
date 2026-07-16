"""Phase-G G-B.2 — MLIR-driven execution of the lowered control-flow Target IR
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`GraphFn.run_via_target_ir()` proves the *lowered IR executes*, not just
lit-checks: it emits `tessera.control_for` (with the serialized body op-list
payload), lowers it through `tessera-opt --tessera-control-for-to-apple_gpu` to
`tessera_apple.gpu.control_loop`, then dispatches off the lowered op's recorded
runtime `symbol` (`tessera_apple_gpu_run_graph_loop_f32`). The result must match
`run()`'s direct in-memory dispatch AND numpy.

Needs the Apple GPU runtime + a built `tessera-opt`; skips otherwise.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError, _find_tessera_opt

pytestmark = [pytest.mark.hardware_apple_gpu, pytest.mark.compiler_tool,
              pytest.mark.usefixtures("apple_gpu_jit_runtime")]


def _np_silu(z):
    return z / (1.0 + np.exp(-z))


@pytest.mark.parametrize("d,trip", [(8, 4), (16, 6)])
def test_target_ir_exec_linear_matches_run_and_numpy(d, trip):
    rng = np.random.default_rng(d + trip)
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(trip, init=ci, body=lambda c: g.matmul(c, wi)))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gd = GraphFn(target="apple_gpu")
    build(gd)
    via = gg.run_via_target_ir(c0, w)   # lowered MLIR → execute
    direct = gd.run(c0, w)              # direct in-memory dispatch
    # Same runtime path + identical op-list → bit-for-bit equal.
    np.testing.assert_allclose(via, direct, rtol=1e-6, atol=1e-6)
    assert gg.last_dispatch() == ["control_loop"]  # the Target-IR op drove it
    ref = c0.copy()
    for _ in range(trip):
        ref = ref @ w
    np.testing.assert_allclose(via, ref, rtol=1e-4, atol=1e-4)


def test_target_ir_exec_silu_matmul_body():
    rng = np.random.default_rng(1)
    d, trip = 16, 5
    c0 = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(trip, init=ci, body=lambda c: g.silu(g.matmul(c, wi))))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gd = GraphFn(target="apple_gpu")
    build(gd)
    via = gg.run_via_target_ir(c0, w)
    np.testing.assert_allclose(via, gd.run(c0, w), rtol=1e-6, atol=1e-6)
    ref = c0.copy()
    for _ in range(trip):
        ref = _np_silu(ref @ w)
    np.testing.assert_allclose(via, ref, rtol=1e-4, atol=1e-4)


def test_target_ir_exec_prenorm_residual_body():
    rng = np.random.default_rng(2026)
    d, trip = 32, 5
    c0 = (rng.standard_normal((1, d)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)

    def build(g):
        ci, wi = g.arg((1, d)), g.arg((d, d))
        g.ret(g.for_loop(trip, init=ci,
                         body=lambda c: g.add(c, g.matmul(g.rmsnorm(c), wi))))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    gd = GraphFn(target="apple_gpu")
    build(gd)
    np.testing.assert_allclose(gg.run_via_target_ir(c0, w), gd.run(c0, w),
                               rtol=1e-4, atol=1e-4)


def test_target_ir_exec_emits_control_for_with_payload():
    """The emitted Graph IR carries the executable op-list payload that the
    lowering threads through to the Target op."""
    d, trip = 8, 3
    g = GraphFn(target="apple_gpu")
    ci, wi = g.arg((1, d)), g.arg((d, d))
    g.ret(g.for_loop(trip, init=ci, body=lambda c: g.matmul(c, wi)))
    mlir = g._emit_control_for_mlir()
    assert 'tessera.control_for' in mlir
    assert 'body_opcodes = array<i32: 0>' in mlir       # one matmul
    assert 'carry_arg_index = 0 : i64' in mlir
    assert 'stop = 3 : i64' in mlir


def test_target_ir_exec_rejects_non_loop_graph():
    g = GraphFn(target="apple_gpu")
    a, b = g.arg((4, 8)), g.arg((8, 4))
    g.ret(g.matmul(a, b))  # straight-line, no loop
    with pytest.raises(TesseraJitError, match="for_loop graph"):
        g.run_via_target_ir(np.ones((4, 8), np.float32), np.ones((8, 4), np.float32))
