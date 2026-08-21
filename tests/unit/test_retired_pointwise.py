"""AD-RETIRE-1 — the ODE-family hand rules are retired behind the datum.

The first evidence-backed retirement (AUTODIFF_NEXTGEN_PLAN §2.2's
protocol): production JVP/VJP for the 13 holonomic-ODE primitives are now
DERIVED from `ScalarRecurrence.pointwise`; the displaced hand rules are
the declared oracles (#31) in `derivative_contract.RETIRED_HAND_RULES`.

Four claims:
* production actually switched — every ODE-family registration carries
  the datum marker, and the switch is idempotent;
* the derived pair is bit-identical to the displaced oracles on the law
  inputs (in-domain), with a mutation control proving the comparison can
  fail;
* at the domain boundary the ONE carried guard fixes the displaced
  pair's measured inconsistency (jvp_sqrt clamped √x while vjp_sqrt
  clamped x; jvp_log had no guard at all) — J and Jᵀ of one function now
  agree at the same point, and the forward-mode boundary change is
  pinned deliberately, not discovered;
* the full law sweep (adjoint + chain-vs-canonical-forward + kink) stays
  green over the derived rules — enforced by `test_autodiff_laws.py`,
  which now runs against the switched registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.algebra import SCALAR_RECURRENCES
from tessera.autodiff.derivative_contract import (
    RETIRED_HAND_RULES,
    _make_derived_pair,
    register_datum_derived_rules,
)
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.law_inputs import LAW_INPUT_SPECS
from tessera.autodiff.laws import op_rng
from tessera.autodiff.vjp import _VJPS

ODE_FAMILY = sorted(SCALAR_RECURRENCES)


def test_production_switched_and_oracles_held():
    # The ledger also carries the structured retirees (AD-RETIRE-2, see
    # test_retired_structured.py); this file owns the ODE family's rows.
    assert set(ODE_FAMILY) <= set(RETIRED_HAND_RULES)
    for name in ODE_FAMILY:
        assert getattr(_JVPS[name], "_derived_from_datum", None) == name
        assert getattr(_VJPS[name], "_derived_from_datum", None) == name
        old_jvp, old_vjp = RETIRED_HAND_RULES[name]
        assert getattr(old_jvp, "_derived_from_datum", None) is None
        assert getattr(old_vjp, "_derived_from_datum", None) is None


def test_switch_is_idempotent():
    before = dict(RETIRED_HAND_RULES)
    switched = register_datum_derived_rules()
    assert sorted(switched) == ODE_FAMILY
    assert RETIRED_HAND_RULES == before, (
        "re-entry must not displace the derived rules into the oracle slot")


@pytest.mark.parametrize("name", ODE_FAMILY)
def test_derived_pair_matches_displaced_oracle_in_domain(name):
    """Bit-identical on the law inputs — the differential proof that the
    switch changed the AUTHORITY, not the numbers."""
    spec = LAW_INPUT_SPECS[name]
    rng = op_rng(name, "retire-differential")
    (x,), kwargs = spec.make(rng)
    dx = rng.standard_normal(np.shape(x))
    dout = rng.standard_normal(np.shape(x))
    old_jvp, old_vjp = RETIRED_HAND_RULES[name]

    y_new, t_new = _JVPS[name]((x,), (dx,), **kwargs)
    y_old, t_old = old_jvp((x,), (dx,), **kwargs)
    np.testing.assert_array_equal(np.asarray(y_new, dtype=np.float64),
                                  np.asarray(y_old, dtype=np.float64))
    np.testing.assert_allclose(t_new, t_old, rtol=1e-15, atol=0)

    (g_new,) = _VJPS[name](dout, x, **kwargs)
    (g_old,) = old_vjp(dout, x, **kwargs)
    np.testing.assert_allclose(g_new, g_old, rtol=1e-15, atol=0)


def test_differential_has_teeth():
    """Mutation control: a corrupted datum produces a derived rule the
    oracle comparison rejects."""
    import dataclasses

    rec = SCALAR_RECURRENCES["tanh"]
    corrupted = dataclasses.replace(
        rec, derivative_expr=lambda o, x: o.mul(
            rec.derivative_expr(o, x),
            np.float64(1.01)),  # planted 1% error
    )
    bad_jvp, _ = _make_derived_pair("tanh", corrupted)
    x = np.array([0.3, -0.8])
    dx = np.array([1.0, 1.0])
    _, t_bad = bad_jvp((x,), (dx,))
    _, t_ref = RETIRED_HAND_RULES["tanh"][0]((x,), (dx,))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(t_bad, t_ref, rtol=1e-15, atol=0)


def test_boundary_guard_is_one_convention_for_both_modes():
    """The fixed inconsistency, pinned. At x below the domain eps the
    displaced pair disagreed between modes; the derived pair applies ONE
    guard (the VJP's convention) to both, so ⟨J v, u⟩ = ⟨v, Jᵀ u⟩ holds
    AT the boundary too."""
    for name, x in (("sqrt", np.array([0.0, 1e-30, 4.0])),
                    ("log", np.array([1e-30, 2.0]))):
        dx = np.ones_like(x)
        dout = np.full_like(x, 0.5)
        _, t = _JVPS[name]((x,), (dx,))
        (g,) = _VJPS[name](dout, x)
        # One slope for both modes: t = f'(x̂)·dx and g = f'(x̂)·dout with
        # the SAME guarded x̂.
        np.testing.assert_allclose(t * 0.5, g, rtol=1e-15, atol=0)

    # The VJP convention survived: derived VJP == displaced VJP everywhere,
    # including at the boundary.
    for name, x in (("sqrt", np.array([0.0, 1e-30, 4.0])),
                    ("log", np.array([1e-30, 2.0]))):
        dout = np.full_like(x, 0.5)
        (g_new,) = _VJPS[name](dout, x)
        (g_old,) = RETIRED_HAND_RULES[name][1](dout, x)
        np.testing.assert_allclose(g_new, g_old, rtol=1e-15, atol=0)

    # The forward-mode boundary CHANGE is deliberate and visible: the old
    # jvp_log had no guard (slope 1/x → 1e30 here); the derived rule caps
    # at the declared 1e-12 domain eps (slope 1e12).
    x = np.array([1e-30])
    _, t_new = _JVPS["log"]((x,), (np.ones(1),))
    _, t_old = RETIRED_HAND_RULES["log"][0]((x,), (np.ones(1),))
    np.testing.assert_allclose(t_new, [1e12], rtol=1e-12)
    np.testing.assert_allclose(t_old, [1e30], rtol=1e-12)


def test_derived_rules_preserve_the_canonical_dtype():
    """PR #600 review (P1): forward-mode dispatch returns the RULE's
    primal instead of re-executing the canonical op, so a promoting rule
    silently changes a function's result dtype the moment AD is enabled.
    Every datum-derived rule keeps float32 primals/tangents/cotangents for
    float32 inputs — bit-compatible with the dtype-preserving displaced
    hand rules (tanh/sin/sigmoid), and a deliberate, pinned FIX of the
    displaced factory's float64 promotion (exp/log/…), which is asserted
    here as the old behavior so the fix stays visible."""
    x32 = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    d32 = np.ones(5, dtype=np.float32)
    positive = np.abs(x32) + np.float32(0.5)
    for name in ODE_FAMILY:
        xin = positive if name in ("log", "sqrt", "reciprocal",
                                   "log1p", "rsqrt") else x32
        y, t = _JVPS[name]((xin,), (d32,))
        (g,) = _VJPS[name](d32, xin)
        assert np.asarray(y).dtype == np.float32, name
        assert np.asarray(t).dtype == np.float32, name
        assert np.asarray(g).dtype == np.float32, name

    # Bit-compat with the displaced rules that already preserved dtype.
    for name in ("tanh", "sin", "sigmoid"):
        y, t = _JVPS[name]((x32,), (d32,))
        y_old, t_old = RETIRED_HAND_RULES[name][0]((x32,), (d32,))
        np.testing.assert_array_equal(np.asarray(t), np.asarray(t_old))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(y_old))

    # The documented fix: the displaced factory PROMOTED (old behavior,
    # asserted so the change stays a decision, not an accident).
    y_old_exp, _ = RETIRED_HAND_RULES["exp"][0]((x32,), (d32,))
    assert np.asarray(y_old_exp).dtype == np.float64
