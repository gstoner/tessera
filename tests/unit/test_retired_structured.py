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


# ── rmsnorm γ envelope (AD-DATUM-POLYGAMMA wave) ────────────────────────────
# The hand pair was x-only while the canonical forward takes an optional
# broadcast γ, so tape-reverse through `ops.rmsnorm(x, gamma)` was broken
# BEFORE retirement (the hand VJP swallowed γ and returned one cotangent
# for two operands). The derived pair now carries γ:
#   JVP  dy = J_core(dx)·γ + core·dγ ;  VJP  dx = J_core(γ⊙dout),
#   dγ = Σ_broadcast dout⊙core reduced to γ's shape.
# There is no displaced oracle for this half — its proof is adjoint +
# finite differences + tape end-to-end, below.


def test_rmsnorm_gamma_adjoint_identity():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((3, 4, 8))
    g = rng.standard_normal(8)
    dx = rng.standard_normal(x.shape)
    dg = rng.standard_normal(g.shape)
    u = rng.standard_normal(x.shape)
    _, t = _JVPS["rmsnorm"]((x, g), (dx, dg), eps=1e-5)
    gx, gg = _VJPS["rmsnorm"](u, x, g, eps=1e-5)
    lhs = float(np.sum(t * u))
    rhs = float(np.sum(dx * gx) + np.sum(dg * gg))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-12


def test_rmsnorm_gamma_matches_finite_differences():
    from tessera import ops
    rng = np.random.default_rng(32)
    x = rng.standard_normal((2, 6))
    g = rng.standard_normal(6)
    eps = 1e-5

    def loss(xv, gv):
        return float(np.sum(np.sin(ops.rmsnorm(xv, gv, eps=eps))))

    dout = np.cos(ops.rmsnorm(x, g, eps=eps))
    gx, gg = _VJPS["rmsnorm"](dout, x, g, eps=eps)
    h = 1e-6
    for idx in ((0, 1), (1, 4)):
        e = np.zeros_like(x)
        e[idx] = h
        fd = (loss(x + e, g) - loss(x - e, g)) / (2.0 * h)
        np.testing.assert_allclose(fd, gx[idx], rtol=1e-6)
    for j in (0, 5):
        e = np.zeros_like(g)
        e[j] = h
        fd = (loss(x, g + e) - loss(x, g - e)) / (2.0 * h)
        np.testing.assert_allclose(fd, gg[j], rtol=1e-6)


def test_rmsnorm_gamma_tape_reverse_end_to_end():
    """The original failing repro: before this wave the derived (and hand)
    VJP raised/returned wrong arity for two operands."""
    from tessera import ops
    from tessera.autodiff.tape import tape
    rng = np.random.default_rng(33)
    x = rng.standard_normal((2, 5))
    g = rng.standard_normal(5)

    with tape() as t:
        y = ops.rmsnorm(x, g)
        target = ops.sum(y)
    t.backward(target)
    gx = t.cotangent[id(x)]
    gg = t.cotangent[id(g)]
    h = 1e-6

    def loss(xv, gv):
        return float(np.sum(np.asarray(ops.rmsnorm(xv, gv))))

    e = np.zeros_like(g)
    e[2] = h
    fd = (loss(x, g + e) - loss(x, g - e)) / (2.0 * h)
    np.testing.assert_allclose(fd, np.asarray(gg)[2], rtol=1e-6)
    ex = np.zeros_like(x)
    ex[1, 3] = h
    fd = (loss(x + ex, g) - loss(x - ex, g)) / (2.0 * h)
    np.testing.assert_allclose(fd, np.asarray(gx)[1, 3], rtol=1e-6)


def test_rmsnorm_gamma_broadcast_shapes_and_dtype():
    rng = np.random.default_rng(34)
    x = rng.standard_normal((3, 4, 8)).astype(np.float32)
    u = rng.standard_normal(x.shape).astype(np.float32)
    for gshape in [(8,), (1, 1, 8), ()]:
        g = np.asarray(rng.standard_normal(gshape), dtype=np.float32)
        gx, gg = _VJPS["rmsnorm"](u, x, g)
        assert np.shape(gg) == gshape
        assert gx.dtype == np.float32 and np.asarray(gg).dtype == np.float32
    # γ-less arity is unchanged — one operand in, one cotangent out.
    (gx_only,) = _VJPS["rmsnorm"](u, x)
    assert gx_only.shape == x.shape


def test_rmsnorm_gamma_keyword_spelling_routes_as_operand():
    """PR #604 review (P1): `ops.rmsnorm(x, gamma=g)` left γ in kwargs, so
    the record never saw it — reverse mode rejected the arity ("2
    cotangents, expected 1") and forward mode ran the rule's γ-less
    branch, making the PRIMAL silently wrong. `promote_operand_kwargs`
    now routes a keyword-spelled operand into the positional record; both
    spellings must be identical end to end."""
    import tessera.autodiff as ad
    from tessera import ops
    from tessera.autodiff.tape import tape
    rng = np.random.default_rng(35)
    x = rng.standard_normal((2, 5))
    g = rng.standard_normal(5)
    dx = rng.standard_normal(x.shape)

    with tape() as t_kw:
        target = ops.sum(ops.rmsnorm(x, gamma=g))
    t_kw.backward(target)
    with tape() as t_pos:
        target = ops.sum(ops.rmsnorm(x, g))
    t_pos.backward(target)
    for operand in (x, g):
        np.testing.assert_array_equal(t_kw.cotangent[id(operand)],
                                      t_pos.cotangent[id(operand)])

    y_kw, dy_kw = ad.jvp(lambda v: ops.rmsnorm(v, gamma=g), (x,), (dx,))
    y_pos, dy_pos = ad.jvp(lambda v: ops.rmsnorm(v, g), (x,), (dx,))
    np.testing.assert_array_equal(np.asarray(y_kw), np.asarray(y_pos))
    np.testing.assert_array_equal(np.asarray(dy_kw), np.asarray(dy_pos))
