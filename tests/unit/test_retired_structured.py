"""AD-RETIRE-2 — the structured family retires behind the jet route.

softmax / logsumexp / rmsnorm-core production JVP/VJP pairs are now
first-order specializations of the structured jets
(`jet.register_jet_derived_structured_rules`); the displaced hand rules
are the declared oracles (#31) in the shared retirement ledger.

Envelope audit outcome (the §8 bar): softmax's pair speaks `axis`;
logsumexp's speaks `axis` (incl. None) + `keepdims`; rmsnorm's is the
gamma-less last-axis core with `eps` inside the sqrt — every combination
is differential-tested against the displaced oracle below. VJPs derive by
the stated transpose structure: softmax and the rmsnorm kernel are
symmetric (pullback = pushforward on the cotangent); logsumexp's
linearization is the softmax row-functional, so its transpose is the
broadcast multiply.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.autodiff.derivative_contract import RETIRED_HAND_RULES
from tessera.autodiff.jet import (
    STRUCTURED_RETIREES,
    register_jet_derived_structured_rules,
)
from tessera.autodiff.jvp import _JVPS
from tessera.autodiff.laws import op_rng
from tessera.autodiff.vjp import _VJPS

ENVELOPES = {
    "softmax": [{"axis": -1}, {"axis": 0}],
    "logsumexp": [{"axis": -1, "keepdims": False},
                  {"axis": 0, "keepdims": True},
                  {"axis": None, "keepdims": False}],
    "rmsnorm": [{"eps": 1e-5}, {"eps": 1e-3}],
}


def test_production_switched_and_oracles_held():
    for name in STRUCTURED_RETIREES:
        assert getattr(_JVPS[name], "_derived_from_jet", None) == name
        assert getattr(_VJPS[name], "_derived_from_jet", None) == name
        assert name in RETIRED_HAND_RULES
        old_jvp, old_vjp = RETIRED_HAND_RULES[name]
        assert getattr(old_jvp, "_derived_from_jet", None) is None
        assert getattr(old_vjp, "_derived_from_jet", None) is None


def test_switch_is_idempotent():
    before = dict(RETIRED_HAND_RULES)
    switched = register_jet_derived_structured_rules()
    assert sorted(switched) == sorted(STRUCTURED_RETIREES)
    assert RETIRED_HAND_RULES == before


@pytest.mark.parametrize("name", STRUCTURED_RETIREES)
def test_derived_pair_matches_displaced_oracle_across_the_envelope(name):
    """The whole audited kwarg envelope, per op. The expressions differ in
    association from the hand rules (jet pieces vs fused formulas), so the
    bar is float64 rounding — 1e-14 relative — not bit equality; a wrong
    derivation shows as O(1), not O(1e-16)."""
    rng = op_rng(name, "retire-structured")
    x = rng.standard_normal((3, 5))
    dx = rng.standard_normal((3, 5))
    old_jvp, old_vjp = RETIRED_HAND_RULES[name]
    for kwargs in ENVELOPES[name]:
        y_new, t_new = _JVPS[name]((x,), (dx,), **kwargs)
        y_old, t_old = old_jvp((x,), (dx,), **kwargs)
        np.testing.assert_allclose(np.asarray(y_new), np.asarray(y_old),
                                   rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(np.asarray(t_new), np.asarray(t_old),
                                   rtol=1e-14, atol=1e-15)
        u = rng.standard_normal(np.shape(np.asarray(t_new)))
        (g_new,) = _VJPS[name](u, x, **kwargs)
        (g_old,) = old_vjp(u, x, **kwargs)
        np.testing.assert_allclose(g_new, g_old, rtol=1e-14, atol=1e-15)


def test_derived_rules_preserve_the_canonical_dtype():
    """The PR #600 dtype lesson, applied from the start: float32 traces
    stay float32 through primal, tangent, and cotangent for all three."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    dx = rng.standard_normal((3, 5)).astype(np.float32)
    for name in STRUCTURED_RETIREES:
        y, t = _JVPS[name]((x,), (dx,))
        assert np.asarray(y).dtype == np.float32, name
        assert np.asarray(t).dtype == np.float32, name
        u = np.asarray(t, dtype=np.float32)
        (g,) = _VJPS[name](u if name != "logsumexp" else u, x)
        assert np.asarray(g).dtype == np.float32, name


def test_differential_has_teeth():
    """A corrupted derived tangent fails the oracle comparison."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 5))
    dx = rng.standard_normal((3, 5))
    _, t = _JVPS["softmax"]((x,), (dx,))
    _, t_ref = RETIRED_HAND_RULES["softmax"][0]((x,), (dx,))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(t) * 1.01, t_ref,
                                   rtol=1e-14, atol=1e-15)


def test_symmetric_transpose_delegation_is_a_true_adjoint():
    """The stated derivation, checked as the law: for softmax and the
    rmsnorm kernel, ⟨Jv, u⟩ = ⟨v, Jᵀu⟩ where Jᵀu is the SAME derived rule
    applied to u — the symmetry is measured, not assumed."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal((3, 5))
    for name in ("softmax", "rmsnorm"):
        v = rng.standard_normal((3, 5))
        u = rng.standard_normal((3, 5))
        _, jv = _JVPS[name]((x,), (v,))
        (jtu,) = _VJPS[name](u, x)
        lhs = float(np.sum(np.asarray(jv) * u))
        rhs = float(np.sum(v * np.asarray(jtu)))
        assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-12, name
