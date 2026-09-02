"""Reverse-over-reverse autodiff must fail closed, not report zeros.

Before this guard, ``grad(grad(f))`` returned an all-zero gradient for a
function with a nonzero second derivative, and ``jacrev(grad(f))`` returned an
all-zero Jacobian by a second path. Nothing raised.

The mechanism: ``_ACTIVE_TAPE`` is a single contextvar, so an inner ``tape()``
shadows the outer one completely — every ``ops.*`` call in that stretch is
recorded on the inner tape and the outer records nothing. ``backward`` then
walks a tape the differentiated input never reached and reports no cotangent,
which the zero branch in ``grad`` read as "this function is constant in that
argument".

Every pre-existing guard checked the OUTPUT side (``tape.backward``'s "target
is not a tape-recorded output", ``jacrev``'s refusal to claim constancy for an
untraced tail). Nothing checked the input side, and an input that never reached
the tape is indistinguishable from an unused one without the extra fact this
suite pins: ``Tape.shadowed_buffer_ids`` -- the set of buffer ids an
inner tape consumed, so each parameter is judged on its own evidence
rather than any nested tape poisoning the whole pass.

The forward-mode twin of this failure was already handled — ``jacfwd(grad(f))``
raises via ``active_jvp_trace()``. These tests hold the two modes symmetric.

Scope: the missing CAPABILITY (composing the modes) is AD-WEIL-1 and is not in
question here. These tests are about refusing to answer, not about answering.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import grad, hvp, jacfwd, jacrev
from tessera.autodiff.rematerialize import rematerialize
from tessera.autodiff.tape import TesseraAutodiffError, tape

X = np.array([0.3, 0.7, 1.1])


def f(x):
    """f(x) = Σ sin(x)·x   →   f'' = 2cos(x) − x·sin(x), nowhere identically 0."""
    return ops.sum(ops.mul(ops.sin(x), x))


def analytic_second_derivative(x: np.ndarray) -> np.ndarray:
    return -np.sin(x) * x + 2 * np.cos(x)


# ── the defect: these must raise rather than return a wrong number ───────────


def test_grad_of_grad_raises_instead_of_returning_zeros():
    def g(x):
        return ops.sum(grad(f)(x))

    with pytest.raises(TesseraAutodiffError, match="inner tape"):
        grad(g)(X)


def test_jacrev_of_grad_raises_instead_of_returning_zeros():
    with pytest.raises(TesseraAutodiffError, match="inner tape"):
        jacrev(grad(f))(X)


def test_the_refused_answer_would_have_been_wrong():
    """The point of the guard: zero is not merely unproven here, it is false."""
    truth = analytic_second_derivative(X)
    assert np.all(np.abs(truth) > 1e-3), "test function must have a real 2nd derivative"

    step = 1e-5
    fd = np.array([
        (
            float(np.sum(np.asarray(grad(f)(np.where(np.arange(X.size) == i, X + step, X)))))
            - float(np.sum(np.asarray(grad(f)(np.where(np.arange(X.size) == i, X - step, X)))))
        ) / (2 * step)
        for i in range(X.size)
    ])
    np.testing.assert_allclose(fd, truth, rtol=1e-6, atol=1e-6)


def test_diagnostic_names_the_cause_and_the_alternative():
    def g(x):
        return ops.sum(grad(f)(x))

    with pytest.raises(TesseraAutodiffError) as excinfo:
        grad(g)(X)
    msg = str(excinfo.value)
    assert "inner tape" in msg
    assert "AD-WEIL-1" in msg, "must route to the plan item that owns the capability"
    assert "hvp" in msg, "must name what works today"
    assert "constant" in msg, "must say why zero would be a false claim, not just unproven"


def test_forward_and_reverse_nesting_both_fail_closed():
    """The two orders are symmetric; neither may return a number."""
    with pytest.raises(TesseraAutodiffError):
        jacfwd(grad(f))(X)
    with pytest.raises(TesseraAutodiffError):
        jacrev(grad(f))(X)


# ── no false positives: everything legitimate is untouched ───────────────────


def test_first_order_grad_is_unchanged():
    np.testing.assert_allclose(grad(f)(X), np.cos(X) * X + np.sin(X), rtol=1e-9)


def test_hvp_still_matches_the_analytic_second_derivative():
    got = hvp(f, X, np.ones_like(X))
    np.testing.assert_allclose(got, analytic_second_derivative(X), rtol=1e-5, atol=1e-6)


def test_genuinely_unused_argument_still_returns_zeros():
    """A real constant-in-argument must NOT be upgraded to an error."""
    def h(a, b):
        return ops.sum(ops.mul(a, a))

    np.testing.assert_array_equal(grad(h, argnums=1)(X, X), np.zeros_like(X))


def test_jacrev_identity_path_still_resolves_structurally():
    np.testing.assert_allclose(jacrev(lambda z: z)(X), np.eye(X.size), atol=0)


def test_rematerialize_nested_tape_is_not_flagged():
    """`rematerialize` opens a nested tape from inside a VJP — i.e. during the
    outer tape's BACKWARD, when the forward is already over. That is the
    legitimate pattern the `_forward_closed` gate exists to preserve."""
    remat_f = rematerialize(lambda z: ops.mul(ops.sin(z), z))

    with tape() as t:
        y = ops.sum(remat_f(X))
        t.backward(y)

    assert t.shadowed_buffer_ids == set()
    # and the gradient is still the real one
    np.testing.assert_allclose(
        t.cotangent[id(X)], np.cos(X) * X + np.sin(X), rtol=1e-9
    )


# ── the flag itself ──────────────────────────────────────────────────────────


def test_flag_is_set_on_the_tape_that_loses_the_ops():
    with tape() as outer:
        with tape() as inner:
            ops.sum(ops.sin(X))
        assert outer.shadowed_buffer_ids, "outer lost the ops"
        assert inner.shadowed_buffer_ids == set(), "inner gained them"


def test_flag_is_not_set_for_sequential_sibling_tapes():
    with tape() as first:
        ops.sum(ops.sin(X))
    with tape() as second:
        ops.sum(ops.sin(X))
    assert first.shadowed_buffer_ids == set()
    assert second.shadowed_buffer_ids == set()


# ─────────────────────────────────────────────────────────────────────────────
# Review on #678: the flag was set on context-manager ENTRY, so ANY nested tape
# poisoned the whole outer pass -- including one that swallowed nothing.
# ─────────────────────────────────────────────────────────────────────────────


def test_an_unused_argument_still_gets_its_honest_zero():
    """A nested tape must not turn a legitimate zero into a refusal.

    `b` is genuinely unused and the inner tape computes an unrelated
    diagnostic. Before the fix this raised for `b`, reporting "swallowed" about
    an argument no tape ever touched.
    """
    def fn(a, b):
        with tape():
            _ = ops.mul(a, a)
        return ops.sum(ops.mul(a, a))

    a = np.array([2.0])
    b = np.array([5.0])
    da, db = grad(fn, argnums=(0, 1))(a, b)
    np.testing.assert_allclose(da, [4.0])
    np.testing.assert_allclose(db, [0.0])


def test_a_returned_argument_is_an_identity_jacobian_regardless_of_tapes():
    """Provable structure outranks shadow evidence.

    `fn` returns one of its own arguments, which is an identity Jacobian by
    construction -- no tape needs consulting. Checking the shadow flag first
    rejected it merely because an unrelated nested tape had opened.
    """
    def ident(a, b):
        with tape():
            _ = ops.mul(a, a)
        return a

    jacrev(ident, argnums=(0, 1))(np.array([2.0]), np.array([5.0]))


def test_an_inner_tape_that_swallows_nothing_records_nothing():
    """An empty inner tape consumed no buffer, so it shadows no path."""
    with tape() as outer:
        with tape():
            pass
        _ = ops.mul(np.array([1.0]), np.array([2.0]))
    assert outer.shadowed_buffer_ids == set()


def test_the_guard_still_fires_for_a_genuinely_swallowed_path():
    """The whole point of the flag, preserved.

    `jacrev(grad(f))` is the case this suite exists for: the inner tape
    swallows the very value the outer pass is differentiating, and the honest
    answer is a refusal rather than zeros. Narrowing the flag from "any nested
    tape opened" to "this buffer was consumed" must not weaken it.

    (`grad(grad(f))` is refused earlier, by the scalar-output check, so it does
    not exercise this branch.)
    """
    with pytest.raises(TesseraAutodiffError, match="inner tape"):
        jacrev(grad(f))(X)


# ── the shadow can sever the chain anywhere, not just at the parameter ───────
#
# Second review round (#679). The first guard asked "was THIS parameter's buffer
# consumed by an inner tape?" — which is only the narrowest instance. A nested
# tape can swallow an INTERMEDIATE, leaving the differentiated input untouched
# and the reverse sweep severed upstream of it. Both cases below returned a
# silent zero with the parameter-only test in place.


def test_nested_tape_swallowing_an_intermediate_raises():
    """`shadowed_buffer_ids` holds y and z but NOT x; the sweep dies at z."""
    def f_mid(a):
        y = ops.mul(a, a)          # outer tape
        with tape():
            z = ops.sin(y)         # inner tape swallows y -> z
        return ops.sum(z)          # outer tape again

    truth = np.cos(X * X) * 2 * X
    assert np.all(np.abs(truth) > 1e-3), "the refused answer must not be zero"
    with pytest.raises(TesseraAutodiffError, match="inner tape"):
        grad(f_mid)(X)


def test_jacrev_raises_when_an_outer_op_records_the_output_after_a_nest():
    """`out` IS tape-produced here, so the not-tape-produced branch is skipped
    entirely — the zero row has to be refused on the normal sweep path."""
    def f_j(a):
        with tape():
            y = ops.sin(a)
        return ops.sum(y)

    with pytest.raises(TesseraAutodiffError, match="inner tape"):
        jacrev(f_j)(X)


def test_truncation_is_recorded_on_the_tape_that_lost_the_edge():
    def f_mid(a):
        y = ops.mul(a, a)
        with tape():
            z = ops.sin(y)
        return ops.sum(z)

    with tape() as t:
        out = f_mid(X)
        t.backward(out)
    assert t.truncated_at_shadow, "the severed edge must be recorded"
    assert t.truncated_at_shadow <= t.shadowed_buffer_ids


def test_a_severed_sweep_does_not_condemn_an_untouched_argument():
    """The severed edge makes upstream values unprovable — it must NOT make a
    never-recorded argument unprovable too, or the guard is back to the coarse
    bool this design replaced."""
    def f(a, b):
        y = ops.mul(a, a)
        with tape():
            z = ops.sin(y)
        return ops.sum(z)          # `b` never appears anywhere

    with pytest.raises(TesseraAutodiffError):
        grad(f, argnums=0)(X, np.ones_like(X))
    np.testing.assert_array_equal(grad(f, argnums=1)(X, np.ones_like(X)),
                                  np.zeros_like(X))


def test_vector_valued_jacrev_is_unaffected():
    np.testing.assert_allclose(jacrev(lambda z: ops.mul(z, z))(X),
                               np.diag(2 * X), atol=1e-9)
