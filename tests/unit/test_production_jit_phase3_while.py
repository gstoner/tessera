"""Phase 3 G-A.3 — GraphFn bounded while on the Apple GPU
(docs/spec/PRODUCTION_COMPILER_PLAN.md, docs/audit/backend/apple/archive/apple_gpu_control_flow_lowering.md).

`GraphFn(target="apple_gpu").while_loop(max_iters, cond, body, init)` authors a
max-iter-capped while as ONE MPSGraph `forLoop` with select-masking — once the
predicate `cond(carry) > 0` goes false, the carry freezes. MPSGraph's native
`while` is unstable (SIGSEGV under churn), so this is the safe lowering.

There is no CPU `scf.while` lane, so the oracle is a numpy bounded-while-masking
reference (the exact semantics the GPU lowering implements).

v1: apple_gpu only, f32, init is a function arg. Skips off-Darwin / when the
runtime is unavailable.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera._jit_boundary import GraphFn, TesseraJitError

pytestmark = [
    pytest.mark.hardware_apple_gpu,
    pytest.mark.usefixtures("apple_gpu_jit_runtime"),
]


def _np_while(c0, max_iters, cond_np, body_np):
    """The masking-while the GPU lowering implements: keep iterating to
    max_iters, freezing the carry once cond <= 0."""
    carry = c0.copy()
    for _ in range(max_iters):
        pred = cond_np(carry) > 0
        nxt = body_np(carry)
        carry = np.where(pred, nxt, carry)
    return carry


# ── decay-while: halve the carry until its row-sum drops to the threshold ────


@pytest.mark.parametrize("d,mx", [(8, 10), (16, 12), (4, 6)])
def test_while_decay_stops(d, mx):
    c0 = np.ones((1, d), np.float32)
    half = np.full((1, d), 0.5, np.float32)
    ones = np.ones((d, 1), np.float32)
    thr = np.array([[1.0]], np.float32)

    def build(g):
        c, h, o, t = g.arg((1, d)), g.arg((1, d)), g.arg((d, 1)), g.arg((1, 1))
        g.ret(g.while_loop(
            mx,
            cond=lambda cr: g.sub(g.matmul(cr, o), t),  # rowsum - thr > 0?
            body=lambda cr: g.mul(cr, h),               # halve
            init=c))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    out = gg.run(c0, half, ones, thr)
    ref = _np_while(c0, mx, lambda cr: cr @ ones - thr, lambda cr: cr * half)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    assert gg.last_dispatch() == ["while"]


# ── runs all max_iters (predicate always true) ───────────────────────────────


def test_while_runs_all_iters():
    d, mx = 8, 5
    rng = np.random.default_rng(0)
    c0 = (rng.standard_normal((1, d)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    ones = np.ones((d, 1), np.float32)
    bias = np.array([[1e3]], np.float32)  # cond = rowsum + 1e3 > 0 always

    def build(g):
        c, wi, o, b = g.arg((1, d)), g.arg((d, d)), g.arg((d, 1)), g.arg((1, 1))
        g.ret(g.while_loop(
            mx,
            cond=lambda cr: g.add(g.matmul(cr, o), b),  # always > 0
            body=lambda cr: g.matmul(cr, wi),
            init=c))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    out = gg.run(c0, w, ones, bias)
    ref = c0.copy()
    for _ in range(mx):
        ref = ref @ w
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── stops immediately (predicate false at iter 0) → returns init ─────────────


def test_while_stops_immediately_returns_init():
    d, mx = 8, 6
    c0 = (np.ones((1, d), np.float32) * 0.01)  # rowsum 0.08 < thr
    half = np.full((1, d), 0.5, np.float32)
    ones = np.ones((d, 1), np.float32)
    thr = np.array([[1.0]], np.float32)

    def build(g):
        c, h, o, t = g.arg((1, d)), g.arg((1, d)), g.arg((d, 1)), g.arg((1, 1))
        g.ret(g.while_loop(
            mx, cond=lambda cr: g.sub(g.matmul(cr, o), t),
            body=lambda cr: g.mul(cr, h), init=c))

    gg = GraphFn(target="apple_gpu")
    build(gg)
    out = gg.run(c0, half, ones, thr)
    np.testing.assert_allclose(out, c0, rtol=1e-6, atol=1e-6)  # never updated


# ── envelope: v1 restrictions rejected ───────────────────────────────────────


def test_while_runs_on_a_cpu_target():
    """The v1 `apple_gpu-only` restriction is gone; this pins that it stays gone.

    This test previously asserted `pytest.raises(..., match="apple_gpu-only")`.
    No such rejection exists for `while_loop` -- the only `apple_gpu-only`
    raise in `_jit_boundary` belongs to `scan` -- and the assertion was never
    reached, because the predicate-shape check fired first with a message that
    does not match. Two defects masked each other: a stale expectation behind
    an unsatisfiable contract.

    `while_loop`'s docstring is the accurate account: the canonical lane emits
    a bounded `scf.while`, and Apple's MPSGraph `forLoop` is one physical
    choice under the same Graph/SCF contract. Adding the guard to make the old
    assertion pass would have broken a working, numerically correct CPU lane to
    satisfy an obsolete restriction.
    """
    g = GraphFn(target="cpu")
    c, h, o, t = g.arg((1, 8)), g.arg((1, 8)), g.arg((8, 1)), g.arg((1, 1))
    g.ret(g.while_loop(6, cond=lambda cr: g.sub(g.matmul(cr, o), t),
                       body=lambda cr: g.mul(cr, h), init=c))

    c0 = np.ones((1, 8), np.float32)
    half = np.full((1, 8), 0.5, np.float32)
    ones = np.ones((8, 1), np.float32)
    thr = np.array([[1.0]], np.float32)

    expected = c0.copy()
    for _ in range(6):
        if (expected @ ones - thr).item() <= 0:
            break
        expected = expected * half

    out = np.asarray(g.run(c0, half, ones, thr))
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_while_bf16_now_supported_via_host_upcast():
    """Phase-G close-out D: a bf16 while no longer raises 'f32-only' — it runs the
    f32 executor via host upcast and returns bf16."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    g = GraphFn(target="apple_gpu", elem="bf16")
    c, h, o, t = g.arg((1, 8)), g.arg((1, 8)), g.arg((8, 1)), g.arg((1, 1))
    g.ret(g.while_loop(4, cond=lambda cr: g.sub(g.matmul(cr, o), t),
                       body=lambda cr: g.mul(cr, h), init=c))
    spec = g._serialize_while_spec()  # former f32-only gate must accept bf16
    assert spec[1] == 4  # max_iters
    out = g.run(np.ones((1, 8), bf16), np.ones((1, 8), bf16),
                np.ones((8, 1), bf16), np.ones((1, 1), bf16))
    assert out.dtype == bf16


def test_while_rejects_init_not_an_arg():
    g = GraphFn(target="apple_gpu")
    x, w, o, t = g.arg((1, 8)), g.arg((8, 8)), g.arg((8, 1)), g.arg((1, 1))
    init = g.matmul(x, w)  # computed, not an arg
    g.ret(g.while_loop(4, cond=lambda cr: g.sub(g.matmul(cr, o), t),
                       body=lambda cr: g.matmul(cr, w), init=init))
    with pytest.raises(TesseraJitError, match="must be a function arg"):
        g.run(np.ones((1, 8), np.float32), np.ones((8, 8), np.float32),
              np.ones((8, 1), np.float32), np.ones((1, 1), np.float32))


# ── the predicate-shape contract ─────────────────────────────────────────────
#
# `while_loop`'s predicate is COMPUTED FROM THE CARRY inside the loop, so it is
# limited to what the op surface can produce -- and that surface has no
# reshape, flatten, squeeze or reduce. From a (1, d) carry the only scalar
# reduction available is `matmul((1,d), (d,1))`, which yields (1, 1). The old
# `pred.shape != (1,)` rule therefore stated a contract no caller could meet
# for a rank-2 carry, and every test above writes the matmul form.
#
# The rule guarded something real, though: the emitter hardcoded a single-index
# `tensor.extract` against a rank-1 type, so a (1, 1) predicate would have
# lowered to malformed IR. The fix is one zero index per axis, not a relaxed
# assertion.


def test_a_rank_two_single_element_predicate_is_accepted():
    """(1, 1) is the natural output of a matmul-based scalar reduction."""
    g = GraphFn(target="apple_gpu")
    c, o, t = g.arg((1, 4)), g.arg((4, 1)), g.arg((1, 1))
    g.ret(g.while_loop(3, cond=lambda cr: g.sub(g.matmul(cr, o), t),
                       body=lambda cr: cr, init=c))
    assert g.build()


def test_a_rank_one_single_element_predicate_is_still_accepted():
    """Generalising must not drop the shape the rule used to name."""
    g = GraphFn(target="apple_gpu")
    c, t = g.arg((1,)), g.arg((1,))
    g.ret(g.while_loop(3, cond=lambda cr: g.sub(cr, t),
                       body=lambda cr: cr, init=c))
    assert g.build()


def test_a_multi_element_predicate_is_still_rejected():
    """The rule was widened to single-element, not removed. A (1, 4) predicate
    has no single truth value and must not silently use element 0."""
    g = GraphFn(target="apple_gpu")
    c, t = g.arg((1, 4)), g.arg((1, 4))
    with pytest.raises(TesseraJitError, match="single-element"):
        g.while_loop(3, cond=lambda cr: g.sub(cr, t), body=lambda cr: cr,
                     init=c)


def test_the_emitted_extract_has_one_index_per_axis():
    """The actual defect the old rule was standing in for.

    A single index against a rank-2 tensor is malformed IR, so widening the
    acceptance without fixing the emission would have traded a false rejection
    for a broken lowering.
    """
    g = GraphFn(target="apple_gpu")
    c, o, t = g.arg((1, 4)), g.arg((4, 1)), g.arg((1, 1))
    g.ret(g.while_loop(3, cond=lambda cr: g.sub(g.matmul(cr, o), t),
                       body=lambda cr: cr, init=c))
    extract = next(line for line in g.build().splitlines()
                   if "tensor.extract" in line and "wpscalar" in line)
    subscript = extract[extract.index("[") + 1:extract.index("]")]
    assert subscript.count(",") == 1, (
        f"a (1, 1) predicate needs two indices, got: {extract.strip()}")
    assert "tensor<1x1x" in extract, (
        f"the extract must be typed at the predicate's real rank: {extract.strip()}")
