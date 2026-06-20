"""SmoothQuant activation-scale migration pass (Workstream D).

The audit's #4 lesson, re-scoped per the P1 feedback: Tessera's backend *already*
direct-consumes compact quantized operands (the packed-int4 Apple-GPU
``quantized_matmul`` lane, ``ops.dequant_matmul``). What was missing is the
**producer** — a compiler pass that migrates per-channel activation scale into the
adjacent weights and emits calibrated W8A8 operands, so the difficulty of
quantizing activations is moved into the (easier-to-quantize) weights.

SmoothQuant (Xiao et al. 2022): for ``Y = X @ W`` with per-input-channel
activation outliers, pick a per-channel smoothing factor ``s`` and rewrite

    Y = (X / s) @ (diag(s) @ W) = X̂ @ Ŵ       # mathematically identical in fp

with ``s_j = max|X_j|^α / max|W_j|^(1-α)``. ``X̂`` has its outliers tamed, so it
quantizes to int8 cleanly; ``Ŵ`` absorbs the scale and quantizes per-output-channel.

The win only counts if the backend consumes the compact operands **directly** —
int8 × int8 → int32 accumulation + rescale, never dequantize-to-fp-then-GEMM.
:func:`smoothquant_matmul` runs exactly that W8A8 path; the oracle
(:func:`verify_w8a8`) checks parity vs fp16 *and* asserts the operands stayed int8
(the anti-fallback invariant).

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SmoothQuantConfig:
    alpha: float = 0.5        # migration strength: 0 = none, 1 = all-to-weights
    eps: float = 1e-5         # floor for the smoothing denominators


@dataclass
class MigratedLinear:
    """A linear layer after activation→weight scale migration.

    Carries the per-channel smoothing factor, the smoothed+quantized weight
    operand the backend consumes directly, and the metadata to dequantize it.
    """

    smoothing: np.ndarray     # s, shape (C_in,)
    w_smoothed: np.ndarray    # diag(s) @ W (fp reference for the exactness check)
    w_q: np.ndarray           # int8 weights, shape (C_in, C_out)
    w_scale: np.ndarray       # per-output-channel scale, shape (C_out,)

    def numeric_policy(self) -> Any:
        """The W8A8 contract this layer declares to the backend."""
        from .primitive_coverage import NumericPolicy
        return NumericPolicy(storage="int8", accum="int32", scale="per_channel",
                             quant_axis=1)

    def apply_smoothing(self, X: np.ndarray) -> np.ndarray:
        """Return the smoothed activation X̂ = X / s."""
        return np.asarray(X, np.float32) / self.smoothing[None, :]

    def dequantized_weight(self) -> np.ndarray:
        """W̃ ≈ Ŵ, reconstructed from the int8 operand (for inspection/tests)."""
        return self.w_q.astype(np.float32) * self.w_scale[None, :]


def compute_smoothing_factor(
    act_absmax: np.ndarray, weight_absmax: np.ndarray, *, alpha: float, eps: float,
) -> np.ndarray:
    """Per-channel ``s = max|X|^α / max|W|^(1-α)`` (SmoothQuant Eq. 4)."""
    a = np.maximum(np.asarray(act_absmax, np.float64), eps)
    w = np.maximum(np.asarray(weight_absmax, np.float64), eps)
    s = (a ** alpha) / (w ** (1.0 - alpha))
    return np.maximum(s, eps).astype(np.float32)


def _quantize_int8_per_channel(M: np.ndarray, axis: int):
    """Symmetric int8 quantize along the kept axis (scale per other-axis index)."""
    absmax = np.maximum(np.abs(M).max(axis=axis, keepdims=True), 1e-12)
    scale = absmax / 127.0
    q = np.round(M / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def migrate_activation_scale(
    X_calib: np.ndarray, W: np.ndarray, *, config: SmoothQuantConfig | None = None,
) -> MigratedLinear:
    """Migrate per-channel activation scale into ``W`` and quantize the result.

    ``X_calib`` is calibration activations ``(N, C_in)``; ``W`` is ``(C_in, C_out)``.
    Returns a :class:`MigratedLinear` whose ``w_q`` is the int8 operand the backend
    consumes directly.
    """
    cfg = config or SmoothQuantConfig()
    X_calib = np.asarray(X_calib, np.float32)
    W = np.asarray(W, np.float32)
    if X_calib.ndim != 2 or W.ndim != 2 or X_calib.shape[1] != W.shape[0]:
        raise ValueError(
            f"shape mismatch: X_calib {X_calib.shape} @ W {W.shape}")

    act_absmax = np.abs(X_calib).max(axis=0)          # (C_in,)
    weight_absmax = np.abs(W).max(axis=1)             # (C_in,) — across outputs
    s = compute_smoothing_factor(act_absmax, weight_absmax,
                                 alpha=cfg.alpha, eps=cfg.eps)

    w_smoothed = s[:, None] * W                       # diag(s) @ W
    # Quantize per output channel (axis 0 = C_in reduced, scale per C_out).
    w_q, w_scale_col = _quantize_int8_per_channel(w_smoothed, axis=0)
    return MigratedLinear(smoothing=s, w_smoothed=w_smoothed,
                          w_q=w_q, w_scale=w_scale_col.reshape(-1))


def smoothquant_matmul(X: np.ndarray, migrated: MigratedLinear) -> np.ndarray:
    """The W8A8 direct-consume path: int8 × int8 → int32 accum → rescale.

    No dequantize-then-GEMM: weights stay int8, activations are quantized int8,
    the contraction is integer, and only the final accumulator is rescaled to fp.
    """
    X_hat = migrated.apply_smoothing(X)               # (N, C_in)
    x_q, a_scale = _quantize_int8_per_channel(X_hat, axis=1)  # per-token
    acc = x_q.astype(np.int32) @ migrated.w_q.astype(np.int32)  # (N, C_out) int32
    return acc.astype(np.float32) * a_scale * migrated.w_scale[None, :]


@dataclass(frozen=True)
class W8A8Verdict:
    relation: str            # "equivalent" | "divergent"
    rel_err: float
    operands_int8: bool      # the anti-fallback invariant
    detail: str = ""
    exact_residual: float = 0.0   # relative error of the *pre-quant* rewrite (must be ~0)

    @property
    def is_equivalent(self) -> bool:
        return self.relation == "equivalent" and self.operands_int8


def verify_w8a8(
    X: np.ndarray, W: np.ndarray, migrated: MigratedLinear, *, rtol: float = 0.06,
    exact_rtol: float = 1e-4,
) -> W8A8Verdict:
    """Oracle: smoothed W8A8 matmul ≈ fp ``X @ W`` AND operands stayed int8.

    Three checks, not one — because the ``rtol`` parity bound has to be loose
    (~6%) to admit genuine int8×int8 quant error, which would also mask a real
    regression in the *migration*. So we additionally pin the part that must be
    tight:

      1. ``rel_err ≤ rtol`` — the end-to-end W8A8 result tracks fp ``X @ W`` up to
         int8 quant error.
      2. ``operands_int8`` — the anti-fallback invariant; a path that silently
         dequantized to fp would still be "correct" but would not be the win.
      3. ``exact_residual ≤ exact_rtol`` — the smoothing factorization
         ``X̂ @ Ŵ`` is an *exact* fp rewrite of ``X @ W``; all admissible error must
         come from quantization, not from the migration. This catches a buggy
         smoothing factor that the loose ``rtol`` would otherwise wave through.
    """
    X = np.asarray(X, np.float32)
    W = np.asarray(W, np.float32)
    y_ref = X @ W
    y_sq = smoothquant_matmul(X, migrated)
    scale = float(np.max(np.abs(y_ref))) or 1.0
    rel_err = float(np.max(np.abs(y_sq - y_ref)) / scale)
    exact_residual = exact_smoothing_residual(X, W, migrated) / scale
    operands_int8 = (migrated.w_q.dtype == np.int8)
    smoothing_exact = exact_residual <= exact_rtol
    rel = "equivalent" if (rel_err <= rtol and smoothing_exact) else "divergent"
    if rel == "equivalent":
        detail = (f"W8A8 rel_err={rel_err:.3e} (≤ {rtol}); smoothing exact "
                  f"(residual={exact_residual:.2e}); operands int8={operands_int8}")
    elif not smoothing_exact:
        detail = (f"smoothing factorization not exact: residual={exact_residual:.3e} "
                  f"exceeds {exact_rtol} — the migration itself is wrong")
    else:
        detail = f"W8A8 rel_err={rel_err:.3e} exceeds {rtol}"
    return W8A8Verdict(rel, rel_err, operands_int8, detail, exact_residual)


def exact_smoothing_residual(X: np.ndarray, W: np.ndarray,
                             migrated: MigratedLinear) -> float:
    """Max abs error of the *pre-quant* rewrite ``X̂ @ Ŵ`` vs ``X @ W``.

    Must be ~0 — the smoothing itself is an exact fp factorization; all error is
    introduced only by the subsequent int8 quantization.
    """
    X = np.asarray(X, np.float32)
    lhs = migrated.apply_smoothing(X) @ migrated.w_smoothed
    return float(np.max(np.abs(lhs - (X @ np.asarray(W, np.float32)))))


__all__ = [
    "SmoothQuantConfig",
    "MigratedLinear",
    "compute_smoothing_factor",
    "migrate_activation_scale",
    "smoothquant_matmul",
    "verify_w8a8",
    "exact_smoothing_residual",
    "W8A8Verdict",
]
