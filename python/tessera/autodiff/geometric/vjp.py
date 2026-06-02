"""GA6 — Reverse-mode VJPs for multivector operations.

Each VJP function ``vjp_<op>(dout, *primals, **kwargs) -> tuple_of_grads``
takes the upstream cotangent ``dout`` (a ``Multivector`` with the same
algebra and shape as the op's output) and the op's primal arguments,
and returns the gradient w.r.t. each primal — ``None`` for
non-differentiable args like scalars.

All VJPs use the **direct Cayley-table-adjoint formula** under the
Frobenius inner product on coefficient vectors. This formulation works
for Cl(p, 0) signatures cleanly; on Cl(p, q) with q > 0 the user must
interpret cotangents via the Frobenius product (not the Hestenes
``<x, reverse(y)>`` form). See ``docs/audit/domain/DOMAIN_AUDIT.md`` § Q3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tessera.autodiff.geometric.registry import register_vjp_geo
from tessera.ga.multivector import Multivector
from tessera.ga.ops import (
    grade_projection,
    reverse,
    grade_involution,
    conjugate,
)


# ---------------------------------------------------------------------------
# Linear ops (self-adjoint under Frobenius)
# ---------------------------------------------------------------------------

@register_vjp_geo("add")
def vjp_add(dout: Multivector, a: Multivector, b: Multivector):
    """c = a + b — gradient flows unchanged to both inputs."""
    return dout, dout


@register_vjp_geo("sub")
def vjp_sub(dout: Multivector, a: Multivector, b: Multivector):
    """c = a - b — gradient flows positive to a, negated to b."""
    return dout, -dout


@register_vjp_geo("neg")
def vjp_neg(dout: Multivector, a: Multivector):
    """c = -a."""
    return (-dout,)


@register_vjp_geo("scalar_mul")
def vjp_scalar_mul(dout: Multivector, a: Multivector, scalar: float):
    """c = scalar · a — grad wrt a is scalar · dout; grad wrt scalar
    is the Frobenius inner product <dout, a>."""
    grad_a = float(scalar) * dout
    grad_scalar = float(
        np.sum(dout.coefficients * a.coefficients)
    )
    return grad_a, grad_scalar


@register_vjp_geo("grade_projection")
def vjp_grade_projection(dout: Multivector, a: Multivector, k):
    """grade_projection extracts grade(s) k; gradient flows only through
    the selected grades."""
    return grade_projection(dout, k), None


@register_vjp_geo("reverse")
def vjp_reverse(dout: Multivector, a: Multivector):
    """reverse is a diagonal sign matrix → self-adjoint."""
    return (reverse(dout),)


@register_vjp_geo("grade_involution")
def vjp_grade_involution(dout: Multivector, a: Multivector):
    """Self-adjoint."""
    return (grade_involution(dout),)


@register_vjp_geo("conjugate")
def vjp_conjugate(dout: Multivector, a: Multivector):
    """conjugate = reverse ∘ grade_involution; both self-adjoint."""
    return (conjugate(dout),)


@register_vjp_geo("hodge_star")
def vjp_hodge_star(dout: Multivector, a: Multivector):
    """Hodge star ``⋆ω = reverse(ω)·I`` is linear. Its Frobenius adjoint
    is computed via the direct Cayley-table contraction.

    Concretely: ``(⋆ω)_k = Σ_i Hodge_{k,i} ω_i`` where
    ``Hodge_{k,i} = T(i, I_mask).sign · rev_sign(i)`` if ``T(i, I_mask).mask == k``
    else 0. The transpose-apply ``(Hodge^T)_{i,k} = Hodge_{k,i}`` gives
    the same formula with ``i`` indexing the output.
    """
    algebra = a.algebra
    table = algebra.product_table()
    I_mask = algebra.pseudoscalar.mask
    dim = algebra.dim
    out_coeffs = np.zeros_like(a.coefficients, dtype=np.result_type(a.dtype, dout.dtype))
    for i in range(dim):
        k_mask, sign = table[i][I_mask]
        if sign == 0:
            continue
        grade_i = i.bit_count()
        rev_sign = (-1) ** ((grade_i * (grade_i - 1)) // 2)
        coeff = sign * rev_sign
        if coeff == 1:
            out_coeffs[..., i] = out_coeffs[..., i] + dout.coefficients[..., k_mask]
        else:
            out_coeffs[..., i] = out_coeffs[..., i] - dout.coefficients[..., k_mask]
    return (Multivector(out_coeffs, algebra),)


# ---------------------------------------------------------------------------
# Bilinear ops — direct Cayley-table adjoint
# ---------------------------------------------------------------------------

def _table_adjoint_product(
    dout: Multivector,
    a: Multivector,
    b: Multivector,
    *,
    disjoint: bool = False,
    grade_filter=None,
):
    """Compute (grad_a, grad_b) for c = bilinear(a, b) using the direct
    Cayley-table adjoint. ``disjoint=True`` enforces the wedge gate
    (shared index ⇒ zero), ``grade_filter`` (callable taking grade_i,
    grade_j, grade_k and returning bool) further filters terms.
    """
    algebra = a.algebra
    table = algebra.product_table()
    dim = algebra.dim
    leading_a = np.broadcast_shapes(dout.coefficients.shape[:-1], b.coefficients.shape[:-1])
    leading_b = np.broadcast_shapes(dout.coefficients.shape[:-1], a.coefficients.shape[:-1])
    dtype = np.result_type(dout.dtype, a.dtype, b.dtype)
    grad_a = np.zeros((*leading_a, dim), dtype=dtype)
    grad_b = np.zeros((*leading_b, dim), dtype=dtype)
    for i in range(dim):
        gi = i.bit_count()
        for j in range(dim):
            if disjoint and (i & j) != 0:
                continue
            k_mask, sign = table[i][j]
            if sign == 0:
                continue
            if grade_filter is not None:
                gj = j.bit_count()
                gk = k_mask.bit_count()
                if not grade_filter(gi, gj, gk):
                    continue
            dout_k = dout.coefficients[..., k_mask]
            b_j = b.coefficients[..., j]
            a_i = a.coefficients[..., i]
            if sign == 1:
                grad_a[..., i] = grad_a[..., i] + dout_k * b_j
                grad_b[..., j] = grad_b[..., j] + dout_k * a_i
            else:
                grad_a[..., i] = grad_a[..., i] - dout_k * b_j
                grad_b[..., j] = grad_b[..., j] - dout_k * a_i
    return Multivector(grad_a, algebra), Multivector(grad_b, algebra)


@register_vjp_geo("geometric_product")
def vjp_geometric_product(dout: Multivector, a: Multivector, b: Multivector):
    """c = a · b (Clifford product). Direct table contraction."""
    return _table_adjoint_product(dout, a, b)


@register_vjp_geo("wedge")
def vjp_wedge(dout: Multivector, a: Multivector, b: Multivector):
    """c = a ∧ b — same as geometric product but with the disjoint-index gate."""
    return _table_adjoint_product(dout, a, b, disjoint=True)


@register_vjp_geo("left_contraction")
def vjp_left_contraction(dout: Multivector, a: Multivector, b: Multivector):
    """c = a ⌋ b — grade-difference filter ``grade(k) == grade(j) - grade(i)``
    (and non-negative)."""
    def _filter(gi, gj, gk):
        target = gj - gi
        return target >= 0 and gk == target
    return _table_adjoint_product(dout, a, b, grade_filter=_filter)


# ---------------------------------------------------------------------------
# Scalar-valued ops — inner / norm
# ---------------------------------------------------------------------------

@register_vjp_geo("inner")
def vjp_inner(dout, a: Multivector, b: Multivector):
    """L = <a, b>_F (Frobenius) on Cl(p, 0); ``dout`` is a scalar.

    For Cl(p, 0) signatures, ``inner(a, b)`` reduces to ``Σ_i a_i b_i``
    after the reverse-and-multiply algebra cancellation, so the
    gradients are simply ``dout · b`` and ``dout · a``. For other
    signatures the relation involves grade-dependent signs; users on
    Cl(p, q, r) with q > 0 should call the direct geometric_product VJP
    chain.
    """
    scale = float(np.asarray(dout))
    return scale * b, scale * a


@register_vjp_geo("norm_squared")
def vjp_norm_squared(dout, a: Multivector):
    """L = <a, a>_F. ∂L/∂a = 2 · dout · a."""
    scale = 2.0 * float(np.asarray(dout))
    return (scale * a,)


@register_vjp_geo("norm")
def vjp_norm(dout, a: Multivector, *, norm_value: Optional[float] = None):
    """L = |a| = sqrt(<a, a>). ∂L/∂a = (dout / |a|) · a."""
    if norm_value is None:
        n = float(np.sqrt(np.sum(a.coefficients ** 2)))
    else:
        n = float(norm_value)
    if n < 1e-30:
        # Subgradient at zero — return zero (cheap convention).
        return (Multivector.zeros(a.algebra, shape=a.shape, dtype=a.dtype),)
    scale = float(np.asarray(dout)) / n
    return (scale * a,)


# ---------------------------------------------------------------------------
# Rotor sandwich — chains 2 geometric products + reverse
# ---------------------------------------------------------------------------

@register_vjp_geo("rotor_sandwich")
def vjp_rotor_sandwich(dout: Multivector, R: Multivector, x: Multivector):
    """y = R · x · R† chained through the registered geometric_product VJPs.

    Let ``y = R · t`` where ``t = x · R†``. Then
        ∂L/∂R          = first arg of vjp_gp(dout, R, t)
        ∂L/∂t          = second arg of vjp_gp(dout, R, t)
        ∂L/∂x          = first arg of vjp_gp(∂L/∂t, x, R†)
        ∂L/∂R†         = second arg of vjp_gp(∂L/∂t, x, R†)
        ∂L/∂R         += vjp_reverse(∂L/∂R†, R)

    Because R is even-grade for a rotor, the accumulated gradient
    likewise lies in the even-grade subspace — verified by
    `test_rotor_sandwich_gradient_is_even_grade` in
    `tests/unit/test_ga_autodiff.py`.
    """
    from tessera.ga.ops import geometric_product

    R_dagger = reverse(R)
    t = geometric_product(x, R_dagger)
    # First product VJP: y = R · t.
    grad_R_1, grad_t = vjp_geometric_product(dout, R, t)
    # Second product VJP: t = x · R†.
    grad_x, grad_R_dagger = vjp_geometric_product(grad_t, x, R_dagger)
    # Reverse VJP on R†.
    (grad_R_2,) = vjp_reverse(grad_R_dagger, R)
    return grad_R_1 + grad_R_2, grad_x
