"""``tessera.energy`` — restricted energy primitive namespace.

M6 Step 2 ships the **lowering surface**: every name here must be
in ``energy_jit._ENERGY_ATTR_TO_OP_NAME`` so the AST lowerer
recognizes it.  The Python implementations below are numpy reference
paths used until M6 Steps 3 + 4 land fused MSL kernels for the same
ops.

The reference implementations are intentionally trivial — their
job is to make ``E(y, *params)`` callable by users who haven't yet
opted into the JIT path.  Correctness vs performance is the same
story as ``tessera.ga.*``: the public API works on numpy arrays,
and ``@energy_jit`` is what eventually swaps in a fused kernel.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "quadratic", "bilinear", "inner",
    "polynomial", "norm", "norm_sq",
    "relu", "tanh", "sigmoid", "gelu", "softplus",
    "linear", "mlp_head",
    "reduce_sum",
]


# ─── Bilinear / quadratic ────────────────────────────────────────────────────

def quadratic(y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``y^T W y`` — scalar (or batched scalar) quadratic form."""
    y = np.asarray(y)
    W = np.asarray(W)
    return np.einsum("...i,ij,...j->...", y, W, y)


def bilinear(y: np.ndarray, x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``y^T W x``."""
    return np.einsum("...i,ij,...j->...",
                     np.asarray(y), np.asarray(W), np.asarray(x))


def inner(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``y · x``."""
    return np.einsum("...i,...i->...", np.asarray(y), np.asarray(x))


# ─── Polynomial / norms ──────────────────────────────────────────────────────

def polynomial(y: np.ndarray, coefs: Sequence[float]) -> np.ndarray:
    """``Σ_k coefs[k] · y^k`` — small-degree element-wise polynomial."""
    y = np.asarray(y)
    out = np.zeros_like(y, dtype=np.float64) + float(coefs[0])
    yk = np.ones_like(y, dtype=np.float64)
    for c in coefs[1:]:
        yk = yk * y
        out = out + float(c) * yk
    return out.astype(y.dtype, copy=False)


def norm(y: np.ndarray) -> np.ndarray:
    """``‖y‖₂``."""
    return np.linalg.norm(np.asarray(y), axis=-1)


def norm_sq(y: np.ndarray) -> np.ndarray:
    """``‖y‖₂²``."""
    y = np.asarray(y)
    return np.einsum("...i,...i->...", y, y)


# ─── Activations ─────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x), 0)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """Approximate GELU (tanh form), matching the MSL kernel used
    by Tessera's Apple GPU runtime."""
    x = np.asarray(x).astype(np.float64, copy=False)
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))).astype(np.asarray(x).dtype, copy=False)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(np.asarray(x), 0)


# ─── Small dense heads ───────────────────────────────────────────────────────

def linear(y: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``y @ W + b``."""
    return np.asarray(y) @ np.asarray(W) + np.asarray(b)


def mlp_head(
    y: np.ndarray, W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
) -> np.ndarray:
    """linear → relu → linear — the canonical 2-layer MLP energy head."""
    return linear(relu(linear(y, W1, b1)), W2, b2)


# ─── Aggregation ─────────────────────────────────────────────────────────────

def reduce_sum(y: np.ndarray) -> np.ndarray:
    """Scalar reduction for the final energy value."""
    return np.sum(np.asarray(y))
