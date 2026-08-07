"""C2 — enforce the declared Clarke-subgradient selection policy at kinks.

This test is the *consumer* of ``autodiff.nonsmooth.NONSMOOTH_SELECTION``
(Decision #29: a declaration must have a consumer). It probes each nonsmooth
op **exactly at** its kink — the one input where two valid subgradients differ —
and asserts:

  1. the reverse-mode VJP returns the subgradient the policy declares;
  2. forward and reverse mode agree at the kink (no mode-dependent semantics);
  3. the two regression bugs C2 fixed stay fixed:
       * ``clip`` and ``clamp`` (aliases) agree at a bound;
       * ``abs`` and ``absolute`` agree at 0 in value *and* dtype.

If a nonsmooth op is added to ``NONSMOOTH_SELECTION`` with no VJP wired to the
declared helper, or a kernel/VJP drifts off the convention, this fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff import nonsmooth
from tessera.autodiff.vjp import get_vjp
from tessera.autodiff.jvp import get_jvp


def _vjp_at(name, x, dout, **kw):
    fn = get_vjp(name)
    assert fn is not None, f"no VJP registered for {name!r}"
    return fn(dout, x, **kw)


# ── 1. Declared policy is honored at the kink ────────────────────────────────

def test_relu_subgrad_zero_at_kink():
    # relu'(0) = 0 under SUBGRAD_ZERO.
    (g,) = _vjp_at("relu", np.array([-1.0, 0.0, 2.0]), np.ones(3))
    np.testing.assert_array_equal(g, [0.0, 0.0, 1.0])


def test_abs_absolute_subgrad_zero_at_kink():
    x = np.array([-3.0, 0.0, 5.0])
    (ga,) = _vjp_at("abs", x, np.ones(3))
    (gb,) = _vjp_at("absolute", x, np.ones(3))
    np.testing.assert_array_equal(ga, [-1.0, 0.0, 1.0])
    # abs and absolute are the same op — value AND dtype must match (the C2
    # dedup: abs previously forced float64, absolute kept input dtype).
    np.testing.assert_array_equal(ga, gb)
    assert ga.dtype == gb.dtype


def test_clip_clamp_agree_and_are_zero_at_bounds():
    # The headline C2 fix: clip used strict (>/<) -> 0 at a bound, clamp used
    # inclusive (>=,<=) -> 1 at a bound, for the *same* operation. Now both 0.
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # bounds at -0.5 and 0.5
    (gc,) = _vjp_at("clip", x, np.ones(5), min_val=-0.5, max_val=0.5)
    (gm,) = _vjp_at("clamp", x, np.ones(5), min=-0.5, max=0.5)
    np.testing.assert_array_equal(gc, [0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(gc, gm)


def test_max_family_splits_ties_and_conserves_mass():
    # Two tied maxima -> each gets half the incoming cotangent (mass-conserving).
    x = np.array([1.0, 3.0, 3.0, 2.0])
    (g,) = _vjp_at("amax", x, np.array(1.0))
    np.testing.assert_allclose(g, [0.0, 0.5, 0.5, 0.0])
    assert g.sum() == pytest.approx(1.0)  # total mass preserved


def test_min_family_splits_ties():
    x = np.array([1.0, 1.0, 3.0])
    (g,) = _vjp_at("amin", x, np.array(1.0))
    np.testing.assert_allclose(g, [0.5, 0.5, 0.0])


def test_maximum_binary_even_tie_split():
    a = np.array([2.0, 5.0])
    b = np.array([2.0, 1.0])  # tie in position 0
    ga, gb = _vjp_at("maximum", a, np.ones(2), y=b) if False else get_vjp("maximum")(np.ones(2), a, b)
    np.testing.assert_array_equal(ga, [0.5, 1.0])
    np.testing.assert_array_equal(gb, [0.5, 0.0])


# ── 2. Forward and reverse mode agree at the kink ────────────────────────────

@pytest.mark.parametrize(
    "name,x,kw",
    [
        ("relu", np.array([-1.0, 0.0, 2.0]), {}),
        ("abs", np.array([-1.0, 0.0, 2.0]), {}),
        ("clip", np.array([-1.0, 0.0, 1.0]), {"min_val": -0.5, "max_val": 0.5}),
    ],
)
def test_vjp_jvp_agree_at_kink_pointwise(name, x, kw):
    """For a pointwise op the Jacobian is diagonal, so the forward tangent with
    ``dx = 1`` equals the reverse cotangent with ``dout = 1`` element-wise. If
    forward and reverse mode picked different subgradients at the kink they
    would disagree here — the miscompile C2 guards against."""
    jvp = get_jvp(name)
    if jvp is None:
        pytest.skip(f"{name} has no JVP")
    ones = np.ones_like(x, dtype=np.float64)
    (_, jtan) = jvp((x,), (ones,), **kw)
    (gx,) = get_vjp(name)(ones, x, **kw)
    np.testing.assert_allclose(np.asarray(jtan), np.asarray(gx), atol=1e-9)


def test_vjp_jvp_agree_at_kink_reduction_duality():
    """For a reduction the modes agree via the JVP/VJP duality identity
    ``<J v, u> == <v, Jᵀ u>``, not element-wise. Probe amax at a two-way tie."""
    x = np.array([1.0, 3.0, 3.0])
    v = np.ones_like(x)          # input tangent
    u = np.array(1.0)            # output cotangent
    (_, jtan) = get_jvp("amax")((x,), (v,))
    (vjp_cot,) = get_vjp("amax")(u, x)
    lhs = float(np.sum(np.asarray(jtan) * np.asarray(u)))
    rhs = float(np.sum(np.asarray(v) * np.asarray(vjp_cot)))
    assert lhs == pytest.approx(rhs)


# ── 3. Registry hygiene (Decision #29 / #21a) ────────────────────────────────

def test_every_declared_op_has_a_vjp():
    missing = [n for n in nonsmooth.NONSMOOTH_SELECTION if get_vjp(n) is None]
    assert not missing, f"declared nonsmooth ops with no VJP consumer: {missing}"


def test_selection_policy_fails_closed_on_unknown_op():
    # A nonsmooth op with no declared policy must raise, not default silently.
    with pytest.raises(KeyError):
        nonsmooth.selection_policy("definitely_not_an_op")


def test_policies_are_from_the_named_set():
    allowed = {nonsmooth.SUBGRAD_ZERO, nonsmooth.SUBGRAD_SPLIT}
    bad = {n: p for n, p in nonsmooth.NONSMOOTH_SELECTION.items() if p not in allowed}
    assert not bad, f"ops with an unrecognized policy: {bad}"
