"""M6 Step 3 (start) — closed-form VJP table for the energy whitelist.

The :mod:`tessera.compiler.energy_jit` lowering surface defines a
14-op whitelist of energy primitives.  Step 3 of M6 generates a
fused energy + ``grad_y`` kernel per primitive; this module ships
the **symbolic gradients** (∂E/∂y per op) that the future Apple GPU
MSL codegen will consume.

Each entry maps a canonical IR op name (the value in
:data:`energy_jit._ENERGY_ATTR_TO_OP_NAME`) to a Python callable
``(*operands, out_grad) -> tuple[grad-per-input, ...]`` that
returns the VJP w.r.t. **every input** the op consumes.  Inputs
the caller marks as constant (``W``, ``coefs``, ``b``, ...) get
their gradients computed too — the M6 Step 4 codegen is free to
discard ones the user pinned.

The VJP table is validated by a ``check_grad`` harness (finite
differences) so any incorrect rule is caught by tests, not by
silent silent NaNs at training time.

Why this is the right shape:

- The VJPs are pure numpy today — they run on the host until M6
  Step 4 lowers them to MSL.  M2's frontend unification can read
  this table to determine whether an energy function admits a
  ``grad_y`` rule.
- The same symbolic-grad core also feeds M7's Cauchy-Riemann
  verifier (which is the gating dependency in the M7 plan).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# VJP type
#
# A VJP function takes the original positional inputs to the op plus
# the upstream cotangent (``out_grad``) and returns a tuple of
# cotangents — one per input, in the same positional order.
# Constant arguments still receive a cotangent so the caller can
# decide whether to drop it.
# ─────────────────────────────────────────────────────────────────────────────

EnergyVJP = Callable[..., tuple[np.ndarray, ...]]


def _as(x: Any) -> np.ndarray:
    return np.asarray(x)


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear / quadratic
# ─────────────────────────────────────────────────────────────────────────────

def _vjp_quadratic(y, W, out_grad):
    """``E = y^T W y``  ⇒  ``∂E/∂y = (W + W^T) y``, ``∂E/∂W = y ⊗ y``.

    The cotangent ``out_grad`` is scalar (or batched scalar); it
    scales every output gradient.
    """
    y = _as(y); W = _as(W); g = _as(out_grad)
    g = np.asarray(g).reshape(*g.shape, 1)
    grad_y = g * np.einsum("ij,...j->...i", (W + W.T), y)
    grad_W = np.einsum("...,...i,...j->ij", _as(out_grad), y, y)
    return (grad_y, grad_W)


def _vjp_bilinear(y, x, W, out_grad):
    """``E = y^T W x``  ⇒  ``∂E/∂y = W x``, ``∂E/∂x = W^T y``,
    ``∂E/∂W = y ⊗ x``."""
    y = _as(y); x = _as(x); W = _as(W); g = _as(out_grad)
    g_ = g.reshape(*g.shape, 1)
    grad_y = g_ * np.einsum("ij,...j->...i", W, x)
    grad_x = g_ * np.einsum("ij,...i->...j", W, y)
    grad_W = np.einsum("...,...i,...j->ij", g, y, x)
    return (grad_y, grad_x, grad_W)


def _vjp_inner(y, x, out_grad):
    """``E = y · x``  ⇒  ``∂E/∂y = x``, ``∂E/∂x = y``."""
    y = _as(y); x = _as(x); g = _as(out_grad)
    g_ = g.reshape(*g.shape, 1)
    return (g_ * x, g_ * y)


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial / norms
# ─────────────────────────────────────────────────────────────────────────────

def _vjp_polynomial(y, coefs, out_grad):
    """``E = Σ_k coefs[k] · y^k``  (elementwise).
    ``∂E/∂y_i = Σ_{k ≥ 1} k · coefs[k] · y_i^(k-1)``.
    ``∂E/∂coefs[k] = y^k`` (elementwise; the caller usually drops it)."""
    y = _as(y); g = _as(out_grad)
    deriv = np.zeros_like(y, dtype=np.float64)
    yk = np.ones_like(y, dtype=np.float64)
    for k in range(1, len(coefs)):
        # d/dy of coefs[k] * y^k = k * coefs[k] * y^(k-1)
        deriv = deriv + k * float(coefs[k]) * yk
        yk = yk * y
    grad_y = (g * deriv).astype(y.dtype, copy=False)
    grad_coefs = tuple(
        float((g * (y ** k)).sum()) for k in range(len(coefs))
    )
    return (grad_y, grad_coefs)


def _vjp_norm(y, out_grad):
    """``E = ‖y‖₂``  ⇒  ``∂E/∂y = y / ‖y‖``.  Subgradient is 0 at the
    origin to keep the VJP well-defined (matches PyTorch / JAX)."""
    y = _as(y); g = _as(out_grad)
    n = np.linalg.norm(y, axis=-1, keepdims=True)
    safe = np.where(n > 0, n, 1.0)
    g_ = np.asarray(out_grad).reshape(*g.shape, 1)
    return (g_ * (y / safe) * (n > 0),)


def _vjp_norm_sq(y, out_grad):
    """``E = ‖y‖₂²``  ⇒  ``∂E/∂y = 2 y``."""
    y = _as(y); g = _as(out_grad)
    g_ = np.asarray(out_grad).reshape(*g.shape, 1)
    return (g_ * 2.0 * y,)


# ─────────────────────────────────────────────────────────────────────────────
# Activations — elementwise; cotangent is same shape as input.
# ─────────────────────────────────────────────────────────────────────────────

def _vjp_relu(x, out_grad):
    return (_as(out_grad) * (_as(x) > 0).astype(_as(x).dtype),)


def _vjp_tanh(x, out_grad):
    t = np.tanh(_as(x))
    return (_as(out_grad) * (1.0 - t * t),)


def _vjp_sigmoid(x, out_grad):
    s = 1.0 / (1.0 + np.exp(-_as(x)))
    return (_as(out_grad) * s * (1.0 - s),)


def _vjp_gelu(x, out_grad):
    """Approximate-GELU VJP (tanh form), matching the reference
    impl + the MSL kernel.

    Let ``u = sqrt(2/π) (x + 0.044715 x³)``.  Then GELU(x) =
    0.5 x (1 + tanh(u)) and
    ``∂GELU/∂x = 0.5 (1 + tanh(u)) + 0.5 x sech²(u) · du/dx``,
    with ``du/dx = sqrt(2/π) (1 + 3·0.044715·x²)``.
    """
    x = _as(x).astype(np.float64, copy=False)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    u = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
    du_dx = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x ** 2)
    t = np.tanh(u)
    sech2 = 1.0 - t * t
    deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx
    g = _as(out_grad).astype(np.float64, copy=False) * deriv
    return (g.astype(_as(x).dtype, copy=False),)


def _vjp_softplus(x, out_grad):
    """``softplus(x) = log(1 + exp(x))`` ⇒ ``∂/∂x = sigmoid(x)``."""
    s = 1.0 / (1.0 + np.exp(-_as(x)))
    return (_as(out_grad) * s,)


# ─────────────────────────────────────────────────────────────────────────────
# Small dense heads
# ─────────────────────────────────────────────────────────────────────────────

def _vjp_linear(y, W, b, out_grad):
    """``out = y @ W + b``.
    ``∂out/∂y = out_grad @ W^T``,
    ``∂out/∂W = y^T @ out_grad``,
    ``∂out/∂b = sum_batch(out_grad)``.
    """
    y = _as(y); W = _as(W); g = _as(out_grad)
    grad_y = g @ W.T
    if y.ndim == 1:
        grad_W = np.outer(y, g)
    else:
        grad_W = y.T @ g
    # Sum every batch axis except the trailing feature axis.
    grad_b = g.sum(axis=tuple(range(g.ndim - 1))) if g.ndim > 1 else g
    return (grad_y, grad_W, grad_b)


def _vjp_mlp_head(y, W1, b1, W2, b2, out_grad):
    """``mlp_head(y) = linear(relu(linear(y, W1, b1)), W2, b2)``.

    The VJP composes the linear+relu rules above so the M6 Step 3
    codegen can either lower the chain as one fused kernel or as
    three primitive ops — both produce identical gradients.
    """
    y = _as(y); W1 = _as(W1); b1 = _as(b1); W2 = _as(W2); b2 = _as(b2)
    g = _as(out_grad)
    h_pre = y @ W1 + b1
    h = np.maximum(h_pre, 0)
    # Backprop through the trailing linear(., W2, b2):
    grad_h, grad_W2, grad_b2 = _vjp_linear(h, W2, b2, g)
    # Through the relu mask:
    grad_h_pre = grad_h * (h_pre > 0).astype(h_pre.dtype)
    # Through the leading linear(y, W1, b1):
    grad_y, grad_W1, grad_b1 = _vjp_linear(y, W1, b1, grad_h_pre)
    return (grad_y, grad_W1, grad_b1, grad_W2, grad_b2)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _vjp_reduce_sum(y, out_grad):
    """``∂(Σ y_i)/∂y = ones_like(y)`` ⇒ broadcasts ``out_grad`` back."""
    y = _as(y)
    return (np.full_like(y, float(_as(out_grad))),)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — keyed on canonical IR op name (the manifest name).
# ─────────────────────────────────────────────────────────────────────────────

ENERGY_VJPS: Mapping[str, EnergyVJP] = {
    "energy_quadratic":   _vjp_quadratic,
    "energy_bilinear":    _vjp_bilinear,
    "energy_inner":       _vjp_inner,
    "energy_polynomial":  _vjp_polynomial,
    "energy_norm":        _vjp_norm,
    "energy_norm_sq":     _vjp_norm_sq,
    "energy_relu":        _vjp_relu,
    "energy_tanh":        _vjp_tanh,
    "energy_sigmoid":     _vjp_sigmoid,
    "energy_gelu":        _vjp_gelu,
    "energy_softplus":    _vjp_softplus,
    "energy_linear":      _vjp_linear,
    "energy_mlp_head":    _vjp_mlp_head,
    "energy_reduce_sum":  _vjp_reduce_sum,
}


def vjp_for(op_name: str) -> EnergyVJP:
    """Lookup the closed-form VJP for a canonical energy IR op.

    Raises :class:`KeyError` for unknown ops — the caller is
    expected to have validated against the energy whitelist before
    requesting a VJP.
    """
    try:
        return ENERGY_VJPS[op_name]
    except KeyError as exc:
        raise KeyError(
            f"no closed-form VJP for {op_name!r}; valid ops: "
            f"{sorted(ENERGY_VJPS)}"
        ) from exc


def has_vjp(op_name: str) -> bool:
    """Predicate version of :func:`vjp_for` — useful for the
    `claim_lint`-style audits that want to know which ops are
    M6 Step 3-ready."""
    return op_name in ENERGY_VJPS


__all__ = [
    "EnergyVJP",
    "ENERGY_VJPS",
    "vjp_for",
    "has_vjp",
]
