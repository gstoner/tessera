"""Phase-G close-out C — MLIR-driven execution of the lowered control_if op.

`GraphFn.run_cond_via_target_ir()` proves the lowered Target IR executes: emit
`tessera.control_if` (with the then/else op-list payload + out_shape), lower it
through `tessera-opt --tessera-control-if-to-apple_gpu` to
`tessera_apple.gpu.control_if`, then dispatch off the lowered op's recorded
runtime `symbol` (`tessera_apple_gpu_run_graph_cond_f32`). Mirrors the control_for
exec test.

Needs the Apple GPU runtime + a built `tessera-opt`; skips otherwise.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError, _find_tessera_opt

pytestmark = [pytest.mark.hardware_apple_gpu, pytest.mark.compiler_tool,
              pytest.mark.usefixtures("apple_gpu_jit_runtime")]


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _build(d=8):
    g = GraphFn(target="apple_gpu")
    flag, x, w = g.arg((1,)), g.arg((1, d)), g.arg((d, d))
    g.ret(g.cond(flag,
                 then_fn=lambda: g.silu(g.matmul(x, w)),
                 else_fn=lambda: g.relu(g.matmul(x, w))))
    return g


@pytest.mark.parametrize("flagv", [1.0, -1.0])
def test_control_if_exec_selects_branch_matches_numpy(flagv):
    rng = np.random.default_rng(int(abs(flagv)) + 5)
    d = 8
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    g = _build(d)
    out = g.run_cond_via_target_ir(np.array([flagv], np.float32), x, w)
    assert g.last_dispatch() == ["control_if"]
    z = x @ w
    ref = _silu(z) if flagv > 0 else np.maximum(z, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_control_if_exec_matches_direct_dispatch():
    rng = np.random.default_rng(99)
    d = 16
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    flag = np.array([1.0], np.float32)
    via = _build(d).run_cond_via_target_ir(flag, x, w)
    direct = _build(d).run(flag, x, w)  # direct in-memory cond dispatch
    np.testing.assert_allclose(via, direct, rtol=1e-6, atol=1e-6)


def test_control_if_exec_emits_payload():
    g = _build(8)
    mlir = g._emit_control_if_mlir()
    assert "tessera.control_if" in mlir
    assert "flag_arg_index = 0 : i64" in mlir
    assert "then_opcodes = array<i32:" in mlir
    assert "else_opcodes = array<i32:" in mlir
    assert "out_shape = array<i64: 1, 8>" in mlir


def test_control_if_exec_rejects_non_cond_graph():
    g = GraphFn(target="apple_gpu")
    a, b = g.arg((4, 8)), g.arg((8, 4))
    g.ret(g.matmul(a, b))
    with pytest.raises(TesseraJitError, match="cond graph"):
        g.run_cond_via_target_ir(np.ones((4, 8), np.float32),
                                 np.ones((8, 4), np.float32))
