"""AD-DATUM-POLYGAMMA — lgamma/digamma join the datum; the tower is tested.

The last two datum holdouts needed the polygamma family (ψ, ψ′, ψ″, …) as
their slope tower — `algebra._polygamma` + `_jet_from_tower` + the auxiliary
recurrence lookup (`recurrence_for`) provide it. Claims:

* the tower is numerically right: exact anchors (ψ″(1) = −2ζ(3),
  ψ‴(1) = π⁴/15, ψ⁗(1) = −24ζ(5), half-integer values), finite-difference
  cross-order consistency including the reflection branch, poles → nan;
* the n = 0/1 rungs ARE the displaced hand VJPs' helpers (bit-for-bit —
  §8: the survivor carries the deleted path's numerics at the orders it
  actually had);
* the datum values mirror the canonical forwards bit-for-bit (the PR #600
  primal-replacement lesson);
* production switched through the same fill-or-displace protocol as the
  other 19 ops, and the auxiliary tower rungs never leak into the
  production registries (#29 — their consumers are the nested-dual
  reference and the jet lane only);
* an unknown recurrence name still fails closed.

The differential proof vs the displaced oracles, the k=1..4 ODE-table
proof, and the law sweep run in test_retired_pointwise.py /
test_autodiff_laws.py, whose parametrization picked these two ops up
automatically when the entries were declared.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.autodiff.algebra import (
    SCALAR_RECURRENCES,
    TruncatedJet,
    _digamma_value,
    _lgamma_value,
    _polygamma,
    recurrence_for,
)

ZETA3 = 1.2020569031595942854
ZETA5 = 1.0369277551433699263


def test_polygamma_exact_anchors():
    # n = 1 is the displaced hand helper VERBATIM (deliberately — see the
    # bitwise test below), so it is held to that helper's own ~5e-12
    # precision; the NEW n ≥ 2 core is held to machine precision.
    anchors = [
        (1, 1.0, math.pi ** 2 / 6.0, 1e-11),
        (1, 0.5, math.pi ** 2 / 2.0, 1e-11),
        (2, 1.0, -2.0 * ZETA3, 5e-15),
        (2, 0.5, -14.0 * ZETA3, 5e-15),
        (3, 1.0, math.pi ** 4 / 15.0, 5e-15),
        (3, 0.5, math.pi ** 4, 5e-15),
        (4, 1.0, -24.0 * ZETA5, 5e-15),
    ]
    for n, x, ref, rtol in anchors:
        got = float(_polygamma(n, np.array([x]))[0])
        np.testing.assert_allclose(got, ref, rtol=rtol,
                                   err_msg=f"psi^({n})({x})")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_polygamma_fd_cross_order_consistency(n):
    """ψ⁽ⁿ⁺¹⁾ == d/dx ψ⁽ⁿ⁾ by central differences — including NEGATIVE x,
    which exercises the reflection branch (cot-derivative polynomials)."""
    h = 1e-6
    for xv in (0.3, 0.7, 3.3, 11.2, -0.4, -1.7, -2.6):
        fd = (_polygamma(n, np.array([xv + h]))
              - _polygamma(n, np.array([xv - h]))) / (2.0 * h)
        an = _polygamma(n + 1, np.array([xv]))
        np.testing.assert_allclose(fd, an, rtol=1e-6,
                                   err_msg=f"n={n} at x={xv}")


def test_polygamma_low_rungs_are_the_displaced_helpers_bitwise():
    from tessera.autodiff.vjp import _digamma_positive, _trigamma_positive
    x = np.concatenate([np.linspace(0.05, 12.0, 200),
                        -np.linspace(0.13, 5.87, 60)])
    np.testing.assert_array_equal(_polygamma(0, x), _digamma_positive(x))
    np.testing.assert_array_equal(_polygamma(1, x), _trigamma_positive(x))


def test_polygamma_poles_are_nan():
    for n in (0, 1, 2, 3):
        out = _polygamma(n, np.array([0.0, -1.0, -3.0]))
        assert np.all(np.isnan(out)), n


def test_datum_values_mirror_canonical_forwards_bitwise():
    """The rule's primal replaces canonical execution under AD (#600), so
    the datum value must be the canonical forward bit-for-bit — over the
    whole real line for digamma (reflection and pole conventions too)."""
    from tessera import ops
    rng = np.random.default_rng(11)
    x = np.concatenate([np.abs(rng.standard_normal(4000)) * 8 + 1e-4,
                        -(np.abs(rng.standard_normal(1000)) * 4 + 0.11),
                        np.array([0.5, 1.0, 1.5, 7.9999, 8.0, -0.5, -2.5])])
    np.testing.assert_array_equal(_digamma_value(x), ops.digamma(x))
    np.testing.assert_array_equal(
        _digamma_value(np.array([0.0, -1.0, -2.0])),
        ops.digamma(np.array([0.0, -1.0, -2.0])))
    xp = np.abs(rng.standard_normal(4000)) * 8 + 1e-4
    np.testing.assert_array_equal(_lgamma_value(xp), ops.lgamma(xp))


def test_production_switched_with_the_wave_protocol():
    from tessera.autodiff.derivative_contract import RETIRED_HAND_RULES
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS
    for name in ("lgamma", "digamma"):
        assert name in SCALAR_RECURRENCES
        assert getattr(_JVPS[name], "_derived_from_datum", None) == name
        assert getattr(_VJPS[name], "_derived_from_datum", None) == name
        old_jvp, old_vjp = RETIRED_HAND_RULES[name]
        assert getattr(old_jvp, "_derived_from_datum", None) is None
        assert getattr(old_vjp, "_derived_from_datum", None) is None


def test_aux_tower_never_leaks_into_production():
    """#29 discipline: the tower rungs' consumers are the nested-dual
    reference and the jet lane — not the op registries, not the law
    sweep's parametrization source."""
    from tessera.autodiff.jvp import _JVPS
    from tessera.autodiff.vjp import _VJPS
    rec = recurrence_for("trigamma")
    assert recurrence_for("trigamma") is rec          # cached
    assert recurrence_for("polygamma4") is recurrence_for("polygamma4")
    for registry in (SCALAR_RECURRENCES, _JVPS, _VJPS):
        assert "trigamma" not in registry
        assert not any(k.startswith("polygamma") for k in registry)


def test_unknown_recurrence_name_fails_closed():
    with pytest.raises(KeyError):
        recurrence_for("not_a_primitive")
    with pytest.raises(KeyError):
        recurrence_for("polygamma1")   # spelled "trigamma"; alias rejected
    with pytest.raises(KeyError):
        recurrence_for("polygammaX")


def test_aux_jets_match_finite_differences():
    """The tower rungs' own jets (used when the nested reference descends
    past digamma) agree with FD on their k=1 coefficient and with the
    tower on higher ones."""
    W = TruncatedJet(3, coefficient_scaling="derivative")
    for name, n in (("trigamma", 1), ("polygamma2", 2)):
        rec = recurrence_for(name)
        x0 = 1.7
        w = rec.jet(W, W.lift(np.asarray(x0), np.asarray(1.0)))
        for k in range(4):
            ref = float(_polygamma(n + k, np.array([x0]))[0])
            got = float(np.asarray(W.extract(w, k)))
            np.testing.assert_allclose(got, ref, rtol=1e-11,
                                       err_msg=f"{name} k={k}")


def test_polygamma_reflection_stable_at_high_order_deep_negative():
    """PR #604 review (P2): cot(πx) via `1/tan(πx)` leaves an
    argument-reduction residual that the cotangent-derivative polynomial
    and π^{n+1} amplify — the naive form was ~1.4% wrong for ψ⁽⁸⁾(−9.5)
    and sign-wrong for ψ⁽¹⁰⁾. The reduced-phase evaluation makes the
    reflection EXACT at half-integers: for even n the cot term vanishes
    identically (ψ⁽ⁿ⁾(−9.5) = ψ⁽ⁿ⁾(10.5) to the last bit), and for odd n
    the constant p_n(0) = cot⁽ⁿ⁾(π/2) survives exactly."""
    for n in (2, 4, 8, 10):
        v = float(_polygamma(n, np.array([-9.5]))[0])
        ref = float(_polygamma(n, np.array([10.5]))[0])
        np.testing.assert_array_equal(v, ref)
    # Odd order: ψ⁽³⁾(−2.5) = −ψ⁽³⁾(3.5) + 2π⁴  (p₃(0) = cot‴(π/2) = −2)
    v3 = float(_polygamma(3, np.array([-2.5]))[0])
    ref3 = -float(_polygamma(3, np.array([3.5]))[0]) + 2.0 * math.pi ** 4
    np.testing.assert_allclose(v3, ref3, rtol=1e-14)
    # Off the half-integers, high order stays FD-consistent.
    h = 1e-6
    for n in (7, 9):
        for xv in (-9.4, -9.6):
            fd = (_polygamma(n, np.array([xv + h]))
                  - _polygamma(n, np.array([xv - h]))) / (2.0 * h)
            an = _polygamma(n + 1, np.array([xv]))
            np.testing.assert_allclose(fd, an, rtol=1e-6,
                                       err_msg=f"n={n} at x={xv}")
