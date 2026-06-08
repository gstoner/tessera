"""Autodiff + surface for the GA differential-form ops (the last deferred tail).

Closes autodiff for hodge_star + the three field operators (ext_deriv, vec_deriv,
codiff). All are linear, so the JVP re-applies the op to the tangent and the VJP
is the exact discrete adjoint:

  * hodge_star — constant 8×8 map; VJP = its transpose.
  * ext_deriv / vec_deriv — Σ_axis (blade matrix) · (np.gradient along axis);
    VJP = Σ_axis (gradient operatorᵀ) ∘ (blade matrixᵀ).
  * codiff = hodge ∘ ext_deriv ∘ hodge; VJP = the adjoint composition.

Correctness is locked by the adjoint identity <op(x), g> == <x, adj(g)> (exact
for a linear op) plus a JVP-vs-finite-difference check.

clifford_integral is intentionally not on the flat tape (it reduces over a
Manifold / accepts a callable integrand) — its flat-lane autodiff is
not_applicable, asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import get_jvp, get_vjp

O = ts.ops
_FIELD_OPS = ["clifford_ext_deriv", "clifford_vec_deriv", "clifford_codiff"]
_SHAPE = (3, 3, 3)
_SPACING = (0.5, 0.7, 0.9)


def _f(seed):
    return np.random.default_rng(seed).standard_normal((*_SHAPE, 8))


# ── surface + registration ───────────────────────────────────────────────────
def test_ops_on_namespace_and_registered():
    for n in ["clifford_hodge_star", *_FIELD_OPS]:
        assert hasattr(O, n), n
        assert get_vjp(n) is not None and get_jvp(n) is not None, n


def test_surface_matches_ga_lane():
    from tessera.ga import calculus as cal
    from tessera.ga.multivector import Multivector
    from tessera.ga.signature import Cl
    cl = Cl(3, 0, 0)
    a = _f(0).astype(np.float32)
    fld = cal.MultivectorField(a, cl, spacing=_SPACING)
    np.testing.assert_allclose(O.clifford_ext_deriv(a, spacing=_SPACING), cal.ext_deriv(fld).values, atol=1e-5)
    np.testing.assert_allclose(O.clifford_codiff(a, spacing=_SPACING), cal.codiff(fld).values, atol=1e-5)


# ── hodge_star (flat, constant linear map) ────────────────────────────────────
def test_hodge_star_jvp_and_adjoint():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(8); da = rng.standard_normal(8); g = rng.standard_normal(8)
    _, tan = get_jvp("clifford_hodge_star")((a,), (da,))
    eps = 1e-6
    fd = (np.asarray(O.clifford_hodge_star(a + eps * da)) - np.asarray(O.clifford_hodge_star(a - eps * da))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-6)
    grad = get_vjp("clifford_hodge_star")(g, a)[0]
    np.testing.assert_allclose(np.dot(np.asarray(tan), g), np.dot(da, grad), atol=1e-6)


# ── field ops: JVP vs FD + exact adjoint identity ────────────────────────────
@pytest.mark.parametrize("name", _FIELD_OPS)
def test_field_op_jvp_matches_fd(name):
    op = getattr(O, name)
    jvp = get_jvp(name)
    x, dx = _f(hash(name) % 99), _f(hash(name) % 99 + 1)
    _, tan = jvp((x,), (dx,), spacing=_SPACING)
    eps = 1e-6
    fd = (np.asarray(op(x + eps * dx, spacing=_SPACING)) - np.asarray(op(x - eps * dx, spacing=_SPACING))) / (2 * eps)
    np.testing.assert_allclose(np.asarray(tan), fd, atol=1e-5)


@pytest.mark.parametrize("name", _FIELD_OPS)
def test_field_op_vjp_is_exact_adjoint(name):
    op = getattr(O, name)
    vjp = get_vjp(name)
    rng = np.random.default_rng(hash(name) % 97)
    x = rng.standard_normal((*_SHAPE, 8))
    g = rng.standard_normal((*_SHAPE, 8))
    grad = vjp(g, x, spacing=_SPACING)[0]
    lhs = np.sum(np.asarray(op(x, spacing=_SPACING)) * g)
    rhs = np.sum(x * grad)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)


def test_field_op_default_spacing_unit():
    # spacing=None ⇒ unit grid step (no crash, finite output).
    x = _f(5)
    out = np.asarray(O.clifford_vec_deriv(x))
    assert out.shape == x.shape and np.all(np.isfinite(out))


# ── integral is not a flat-tape op ───────────────────────────────────────────
def test_integral_marked_not_applicable():
    from tessera.compiler import primitive_coverage as pc
    entry = pc.all_primitive_coverages()["clifford_integral"]
    assert entry.contract_status.get("vjp") == "not_applicable"
    assert entry.contract_status.get("jvp") == "not_applicable"


# ── whole GA autodiff surface is closed ──────────────────────────────────────
def test_ga_autodiff_surface_complete():
    from tessera.compiler import primitive_coverage as pc
    ga = {n: e for n, e in pc.all_primitive_coverages().items() if e.category == "geometric_algebra"}
    planned = [n for n in ga if ga[n].contract_status.get("vjp") == "planned"]
    assert planned == [], f"unexpected planned-vjp GA ops: {planned}"
