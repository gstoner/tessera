"""GA6 — Forward-mode JVPs for multivector operations.

Each JVP function ``jvp_<op>(tangents, primals, **kwargs) -> Multivector``
returns the forward pushforward at the given primal arguments. Conventions:

  - ``tangents`` is a tuple matching the differentiable subset of
    ``primals``; entries that are non-multivector or non-differentiable
    are ``None``.
  - The return value is the tangent at the op's output.

For linear ops, JVP applies the op itself to the input tangent. For
bilinear ops like ``geometric_product``, JVP applies the product rule:
``d(a · b) = da · b + a · db``.
"""

from __future__ import annotations

import numpy as np

from tessera.autodiff.geometric.registry import register_jvp_geo
from tessera.ga.multivector import Multivector
from tessera.ga.ops import (
    conjugate,
    geometric_product,
    grade_involution,
    grade_projection,
    inner,
    left_contraction,
    reverse,
    wedge,
)


# ---------------------------------------------------------------------------
# Linear ops — JVP applies the op to the tangent
# ---------------------------------------------------------------------------

@register_jvp_geo("add")
def jvp_add(tangents, primals):
    da, db = tangents
    return da + db


@register_jvp_geo("sub")
def jvp_sub(tangents, primals):
    da, db = tangents
    return da - db


@register_jvp_geo("neg")
def jvp_neg(tangents, primals):
    (da,) = tangents
    return -da


@register_jvp_geo("scalar_mul")
def jvp_scalar_mul(tangents, primals):
    da, dscalar = tangents
    a, scalar = primals
    return float(scalar) * da + (float(dscalar) if dscalar is not None else 0.0) * a


@register_jvp_geo("grade_projection")
def jvp_grade_projection(tangents, primals, k=None):
    (da, _) = tangents
    _, k_arg = primals
    target = k if k is not None else k_arg
    return grade_projection(da, target)


@register_jvp_geo("reverse")
def jvp_reverse(tangents, primals):
    (da,) = tangents
    return reverse(da)


@register_jvp_geo("grade_involution")
def jvp_grade_involution(tangents, primals):
    (da,) = tangents
    return grade_involution(da)


@register_jvp_geo("conjugate")
def jvp_conjugate(tangents, primals):
    (da,) = tangents
    return conjugate(da)


@register_jvp_geo("hodge_star")
def jvp_hodge_star(tangents, primals):
    """⋆ is linear in its argument — apply it to the tangent."""
    from tessera.ga.calculus import hodge_star
    (da,) = tangents
    return hodge_star(da)


# ---------------------------------------------------------------------------
# Bilinear ops — product rule
# ---------------------------------------------------------------------------

@register_jvp_geo("geometric_product")
def jvp_geometric_product(tangents, primals):
    """d(a · b) = da · b + a · db."""
    da, db = tangents
    a, b = primals
    return geometric_product(da, b) + geometric_product(a, db)


@register_jvp_geo("wedge")
def jvp_wedge(tangents, primals):
    """d(a ∧ b) = da ∧ b + a ∧ db."""
    da, db = tangents
    a, b = primals
    return wedge(da, b) + wedge(a, db)


@register_jvp_geo("left_contraction")
def jvp_left_contraction(tangents, primals):
    """d(a ⌋ b) = da ⌋ b + a ⌋ db."""
    da, db = tangents
    a, b = primals
    return left_contraction(da, b) + left_contraction(a, db)


# ---------------------------------------------------------------------------
# Scalar-valued ops — JVP returns a scalar (np.ndarray)
# ---------------------------------------------------------------------------

@register_jvp_geo("inner")
def jvp_inner(tangents, primals):
    """d<a, b> = <da, b> + <a, db>."""
    da, db = tangents
    a, b = primals
    return inner(da, b) + inner(a, db)


@register_jvp_geo("norm_squared")
def jvp_norm_squared(tangents, primals):
    """d|a|² = 2 <a, da>."""
    (da,) = tangents
    (a,) = primals
    return 2.0 * inner(a, da)


@register_jvp_geo("norm")
def jvp_norm(tangents, primals):
    """d|a| = <a, da> / |a|."""
    (da,) = tangents
    (a,) = primals
    n = float(np.sqrt(np.sum(a.coefficients ** 2)))
    if n < 1e-30:
        return np.array(0.0)
    return inner(a, da) / n


@register_jvp_geo("rotor_sandwich")
def jvp_rotor_sandwich(tangents, primals):
    """d(R · x · R†) = dR · x · R† + R · dx · R† + R · x · dR†."""
    dR, dx = tangents
    R, x = primals
    R_dag = reverse(R)
    dR_dag = reverse(dR)
    return (
        geometric_product(geometric_product(dR, x), R_dag)
        + geometric_product(geometric_product(R, dx), R_dag)
        + geometric_product(geometric_product(R, x), dR_dag)
    )
