"""Phase-G close-out E — `jit_while_loop` front-end.

`jit_while_loop(max_iters, cond, body, init=, consts=)` traces a bounded while
into `tessera.control_while` and executes it on Apple GPU (one MPSGraph forLoop +
select-masking; once the predicate goes false the carry freezes). The
bounded-while user surface is this explicit front-end rather than an AST bridge,
because a Python `while` carries no max_iters.

Needs the Apple GPU runtime + a built `tessera-opt`; skips otherwise.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import _find_tessera_opt, build_while_loop, jit_while_loop

pytestmark = [pytest.mark.hardware_apple_gpu, pytest.mark.compiler_tool,
              pytest.mark.usefixtures("apple_gpu_jit_runtime")]


@pytest.mark.parametrize("max_iters", [2, 4])
def test_while_frontend_runs_to_max_when_pred_positive(max_iters):
    rng = np.random.default_rng(max_iters)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([1.0], np.float32)  # pred > 0 → all iterations run
    out = jit_while_loop(
        max_iters,
        cond=lambda g, c, w_, t: t,
        body=lambda g, c, w_, t: g.matmul(c, w_),
        init=x, consts=[w, thr])
    ref = x.copy()
    for _ in range(max_iters):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_while_frontend_early_stop_freezes_carry():
    rng = np.random.default_rng(3)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([-1.0], np.float32)  # pred <= 0 from the start → carry == init
    out = jit_while_loop(
        3,
        cond=lambda g, c, w_, t: t,
        body=lambda g, c, w_, t: g.matmul(c, w_),
        init=x, consts=[w, thr])
    np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-4)


def test_while_frontend_via_ir_matches_direct():
    rng = np.random.default_rng(9)
    d = 4
    x = (rng.standard_normal((1, d)) / d).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    thr = np.array([1.0], np.float32)
    cond = lambda g, c, w_, t: t           # noqa: E731
    body = lambda g, c, w_, t: g.matmul(c, w_)  # noqa: E731
    via = jit_while_loop(3, cond, body, init=x, consts=[w, thr], via_target_ir=True)
    direct = jit_while_loop(3, cond, body, init=x, consts=[w, thr],
                            via_target_ir=False)
    np.testing.assert_allclose(via, direct, rtol=1e-6, atol=1e-6)


def test_build_while_loop_emits_control_while():
    g = build_while_loop(
        3,
        cond=lambda g, c, w_, t: t,
        body=lambda g, c, w_, t: g.matmul(c, w_),
        init_shape=(1, 4), const_shapes=[(4, 4), (1,)])
    mlir = g._emit_control_while_mlir()
    assert "tessera.control_while" in mlir
    assert "max_iters = 3 : i64" in mlir
