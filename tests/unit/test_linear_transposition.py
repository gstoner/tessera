"""C1 — automatic linear transposition.

Proves the review's claim that the linear-primitive JVPs are a mechanical
transposition of the forwards (so the two hand-maintained registries are one),
and that the declared ``transpose_rule`` axis now has a consumer (Decision #29).

Part 1 cross-checks the *derived* JVP (from linearity, ``make_linear_jvp``)
against the hand-written JVP in ``jvp.py`` for every op in the overlap. If they
agree, the hand-written JVP is redundant with the forward — the review's point.

Part 2 exercises the ``transpose_rule`` consumer: a linear ``@custom_primitive``
that declares only a forward and a transpose gets a working VJP (from the
transpose) and a working JVP (derived from linearity), and both match finite
differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.linear import (
    MULTILINEAR_PRIMITIVES,
    make_linear_jvp,
    _resolve_forward,
)


# ── Part 1: derived JVP == hand-written JVP over the overlap ──────────────────

_CASES = {
    "transpose": (lambda: (np.arange(6.0).reshape(2, 3),), {"axes": (1, 0)}),
    "reshape": (lambda: (np.arange(6.0),), {"shape": (2, 3)}),
    "matmul": (lambda: (np.random.default_rng(0).standard_normal((3, 4)),
                        np.random.default_rng(1).standard_normal((4, 2))), {}),
    "gemm": (lambda: (np.random.default_rng(2).standard_normal((3, 4)),
                      np.random.default_rng(3).standard_normal((4, 2))), {}),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_derived_jvp_matches_handwritten(name):
    hand = get_jvp(name)
    if hand is None:
        pytest.skip(f"{name} has no hand-written JVP to compare against")
    forward = _resolve_forward(name)
    assert forward is not None
    derived = make_linear_jvp(forward, MULTILINEAR_PRIMITIVES[name])

    make_primals, kwargs = _CASES[name]
    primals = make_primals()
    rng = np.random.default_rng(99)
    tangents = tuple(rng.standard_normal(np.asarray(p).shape) for p in primals)

    p_hand, t_hand = hand(primals, tangents, **kwargs)
    p_der, t_der = derived(primals, tangents, **kwargs)

    np.testing.assert_allclose(np.asarray(p_der), np.asarray(p_hand), atol=1e-10)
    np.testing.assert_allclose(np.asarray(t_der), np.asarray(t_hand), atol=1e-10)


def test_register_derived_linear_jvps_is_additive():
    # The gap-filler must never override a hand-written JVP (Decision #31: the
    # surviving path must already carry what a duplicate would). Every declared
    # multilinear built-in currently has a hand-written JVP, so the filler adds
    # nothing today — and it must stay that way (no silent clobber).
    from tessera.autodiff.linear import register_derived_linear_jvps
    from tessera.autodiff.jvp import get_jvp as _g

    before = {n: _g(n) for n in MULTILINEAR_PRIMITIVES}
    filled = register_derived_linear_jvps()
    after = {n: _g(n) for n in MULTILINEAR_PRIMITIVES}
    # No op that already had a JVP had it replaced.
    for n, fn in before.items():
        if fn is not None:
            assert after[n] is fn, f"{n} JVP was overridden"
    # Anything reported as filled must have had no JVP before.
    for n in filled:
        assert before[n] is None


def test_derived_jvp_matches_finite_difference():
    # Sanity: the linearity identity really is the derivative, vs central diff.
    forward = _resolve_forward("matmul")
    derived = make_linear_jvp(forward, (0, 1))
    rng = np.random.default_rng(7)
    A = rng.standard_normal((2, 3))
    B = rng.standard_normal((3, 2))
    dA = rng.standard_normal((2, 3))
    dB = rng.standard_normal((3, 2))
    _, tan = derived((A, B), (dA, dB))
    eps = 1e-6
    fd = (forward(A + eps * dA, B + eps * dB) - forward(A - eps * dA, B - eps * dB)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-6)


# ── Part 2: transpose_rule is a real consumer (Decision #29) ─────────────────

def test_linear_custom_primitive_transpose_drives_grad():
    from tessera.custom import custom_primitive
    from tessera.autodiff.vjp import get_vjp
    from tessera.autodiff.jvp import get_jvp as _get_jvp

    W = np.array([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]])  # fixed linear operator

    # y = W @ x, linear in x (arg 0). Declared linear with a transpose rule; no
    # VJP or JVP written by hand.
    @custom_primitive("c1_linear_apply", linear=True, linear_args=(0,))
    def c1_linear_apply(x):
        return W @ x

    @c1_linear_apply.def_transpose
    def _transpose(dout, x, **_):
        # adjoint of x -> W@x is u -> Wᵀ@u
        return (W.T @ np.asarray(dout),)

    # Consumer wired the transpose as the VJP and derived a JVP from linearity.
    assert get_vjp("c1_linear_apply") is not None
    assert _get_jvp("c1_linear_apply") is not None

    x = np.array([1.0, -2.0, 0.5])
    # VJP (reverse): dout -> Wᵀ dout
    dout = np.array([1.0, 1.0])
    (gx,) = get_vjp("c1_linear_apply")(dout, x)
    np.testing.assert_allclose(gx, W.T @ dout)

    # JVP (forward, derived): v -> W v
    v = np.array([0.3, 0.1, -0.2])
    _, tan = _get_jvp("c1_linear_apply")((x,), (v,))
    np.testing.assert_allclose(np.asarray(tan), W @ v)
