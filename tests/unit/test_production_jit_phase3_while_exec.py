"""Phase-G close-out D — MLIR-driven execution of the lowered control_while op.

`GraphFn.run_while_via_target_ir()` proves the lowered Target IR executes: emit
`tessera.control_while` (with the body+cond op-list payload), lower it through
`tessera-opt --tessera-control-while-to-apple_gpu` to
`tessera_apple.gpu.control_while`, then dispatch off the lowered op's recorded
runtime `symbol` (`tessera_apple_gpu_run_graph_while_f32`). The while lowers to an
MPSGraph forLoop + select-masking (native `while` is unstable).

Needs the Apple GPU runtime + a built `tessera-opt`; skips otherwise.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError, _find_tessera_opt

pytestmark = [pytest.mark.hardware_apple_gpu, pytest.mark.compiler_tool,
              pytest.mark.usefixtures("apple_gpu_jit_runtime")]


def _build_always_run(d=4, max_iters=3):
    """Predicate is a positive constant arg → runs all max_iters; the carry
    iterates carry = carry @ w. (Keeps the oracle a plain matmul power.)"""
    g = GraphFn(target="apple_gpu")
    x, w, thr = g.arg((1, d)), g.arg((d, d)), g.arg((1,))
    g.ret(g.while_loop(max_iters, cond=lambda c: thr,
                       body=lambda c: g.matmul(c, w), init=x))
    return g


@pytest.mark.parametrize("max_iters", [2, 4])
def test_control_while_exec_matches_numpy(max_iters):
    rng = np.random.default_rng(max_iters)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([1.0], np.float32)  # pred > 0 → all iterations run
    g = _build_always_run(d, max_iters)
    out = g.run_while_via_target_ir(x, w, thr)
    assert g.last_dispatch() == ["control_while"]
    ref = x.copy()
    for _ in range(max_iters):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_control_while_exec_matches_direct_dispatch():
    rng = np.random.default_rng(7)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([1.0], np.float32)
    via = _build_always_run(d, 3).run_while_via_target_ir(x, w, thr)
    direct = _build_always_run(d, 3).run(x, w, thr)
    np.testing.assert_allclose(via, direct, rtol=1e-6, atol=1e-6)


def test_control_while_exec_early_stop_freezes_carry():
    """A predicate that starts non-positive runs zero iterations → carry == init
    (the select-masking freezes the carry once the predicate is false)."""
    rng = np.random.default_rng(3)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([-1.0], np.float32)  # pred <= 0 from the start → no updates
    out = _build_always_run(d, 3).run_while_via_target_ir(x, w, thr)
    np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-4)


def test_control_while_exec_emits_payload():
    g = _build_always_run(4, 3)
    mlir = g._emit_control_while_mlir()
    assert "tessera.control_while" in mlir
    assert "carry_arg_index = 0 : i64" in mlir
    assert "max_iters = 3 : i64" in mlir
    assert "body_opcodes = array<i32:" in mlir
    assert "cond_opcodes = array<i32>" in mlir  # empty cond op-list (pred is an arg)


def test_control_while_exec_rejects_non_while_graph():
    g = GraphFn(target="apple_gpu")
    a, b = g.arg((4, 8)), g.arg((8, 4))
    g.ret(g.matmul(a, b))
    with pytest.raises(TesseraJitError, match="while graph"):
        g.run_while_via_target_ir(np.ones((4, 8), np.float32),
                                  np.ones((8, 4), np.float32))
