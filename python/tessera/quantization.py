"""Reference numerics and quantization helpers for standalone Tessera tests.

These functions intentionally start as numpy-reference semantics. They give the
compiler a stable API vocabulary for S9 while backend-specific packing,
observer fusion, and QAT transform rules mature in later sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _symmetric_scale(x: np.ndarray, qmax: int) -> np.float32:
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs == 0.0:
        return np.float32(1.0)
    return np.float32(max_abs / qmax)


def quantize_int8(x: Any, scale: float | None = None, zero_point: int = 0, *, symmetric: bool = True):
    """Quantize to int8 values and return ``(q, scale, zero_point)``."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if symmetric:
        # Local `s: np.float32` so mypy doesn't have to reconcile the
        # parameter's `float | None` type with the np.float32 widening.
        s = _symmetric_scale(x_arr, 127) if scale is None else np.float32(scale)
        q = np.round(x_arr / s).clip(-127, 127).astype(np.int8)
        return q, np.float32(s), 0
    s = np.float32(scale if scale is not None else ((x_arr.max() - x_arr.min()) / 255.0 if x_arr.size else 1.0))
    if float(s) == 0.0:
        s = np.float32(1.0)
    q = np.round(x_arr / s + zero_point).clip(-128, 127).astype(np.int8)
    return q, s, int(zero_point)


def dequantize_int8(q: Any, scale: float, zero_point: int = 0) -> np.ndarray:
    return (_asarray(q).astype(np.float32) - float(zero_point)) * np.float32(scale)


def quantize_int4(x: Any, scale: float | None = None, zero_point: int = 0, *, symmetric: bool = True):
    """Quantize to signed int4 values stored in int8 containers."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if symmetric:
        s = _symmetric_scale(x_arr, 7) if scale is None else np.float32(scale)
        q = np.round(x_arr / s).clip(-7, 7).astype(np.int8)
        return q, np.float32(s), 0
    s = np.float32(scale if scale is not None else ((x_arr.max() - x_arr.min()) / 15.0 if x_arr.size else 1.0))
    if float(s) == 0.0:
        s = np.float32(1.0)
    q = np.round(x_arr / s + zero_point).clip(-8, 7).astype(np.int8)
    return q, s, int(zero_point)


def dequantize_int4(q: Any, scale: float, zero_point: int = 0) -> np.ndarray:
    q_arr = _asarray(q).astype(np.float32)
    if np.any(q_arr < -8) or np.any(q_arr > 7):
        raise ValueError("int4 containers must hold values in [-8, 7]")
    return (q_arr - float(zero_point)) * np.float32(scale)


def fake_quantize(x: Any, num_bits: int = 8, scale: float | None = None, zero_point: int = 0, *, symmetric: bool = True) -> np.ndarray:
    """Quantize then dequantize, preserving the original floating output dtype."""
    if num_bits == 8:
        q, s, zp = quantize_int8(x, scale=scale, zero_point=zero_point, symmetric=symmetric)
        return dequantize_int8(q, s, zp).astype(_asarray(x).dtype, copy=False)
    if num_bits == 4:
        q, s, zp = quantize_int4(x, scale=scale, zero_point=zero_point, symmetric=symmetric)
        return dequantize_int4(q, s, zp).astype(_asarray(x).dtype, copy=False)
    raise ValueError("fake_quantize supports num_bits 4 or 8")


@dataclass
class CalibrationObserver:
    """Min/max calibration observer for reference quantization flows."""

    min_val: float | None = None
    max_val: float | None = None

    def observe(self, x: Any) -> "CalibrationObserver":
        x_arr = _asarray(x).astype(np.float32, copy=False)
        if x_arr.size == 0:
            return self
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        self.min_val = x_min if self.min_val is None else min(self.min_val, x_min)
        self.max_val = x_max if self.max_val is None else max(self.max_val, x_max)
        return self

    def calculate_qparams(self, num_bits: int = 8, *, symmetric: bool = True) -> tuple[np.float32, int]:
        if self.min_val is None or self.max_val is None:
            return np.float32(1.0), 0
        if symmetric:
            qmax = (2 ** (num_bits - 1)) - 1
            max_abs = max(abs(self.min_val), abs(self.max_val))
            return np.float32(max_abs / qmax if max_abs else 1.0), 0
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        scale = (self.max_val - self.min_val) / float(qmax - qmin)
        if scale == 0.0:
            return np.float32(1.0), 0
        zp = int(np.clip(round(qmin - self.min_val / scale), qmin, qmax))
        return np.float32(scale), zp


def calibration_observer() -> CalibrationObserver:
    return CalibrationObserver()


def grad_scaler_step(
    grads: Any,
    scale: float,
    *,
    found_inf: bool = False,
    growth_tracker: int = 0,
    growth_interval: int = 2000,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
):
    """Unscale gradients and update a loss-scaling value.

    Returns ``(unscaled_grads, new_scale, new_growth_tracker, should_step)``.
    Nested lists/tuples/dicts are handled recursively.
    """
    scale_f = float(scale)

    def unscale(g):
        if isinstance(g, dict):
            return {k: unscale(v) for k, v in g.items()}
        if isinstance(g, tuple):
            return tuple(unscale(v) for v in g)
        if isinstance(g, list):
            return [unscale(v) for v in g]
        return _asarray(g).astype(np.float32, copy=False) / scale_f

    if found_inf:
        return unscale(grads), np.float32(scale_f * backoff_factor), 0, False
    tracker = int(growth_tracker) + 1
    if tracker >= int(growth_interval):
        return unscale(grads), np.float32(scale_f * growth_factor), 0, True
    return unscale(grads), np.float32(scale_f), tracker, True


def quantize_int4_packed(w: Any, group_size: int = 64):
    """Affine per-group int4 packing of a 2-D weight ``W[N, K]`` for the Apple GPU
    packed quantized-matmul lane (P3).

    Each group of ``group_size`` columns gets its own affine scale + bias; codes
    are 4-bit (``[0, 15]``) and packed 2 nibbles per byte (low nibble = even k,
    high nibble = odd k) so weight storage is ``0.5`` bytes/element — ~8× less
    than full-width f32 codes. Dequant is ``w ≈ scale·code + bias`` (MLX
    convention).

    Returns ``(packed, scales, biases)`` where ``packed`` is ``uint8
    [N, ceil(K/2)]`` and ``scales``/``biases`` are ``f32 [N, NG]`` with
    ``NG = ceil(K/group_size)``.
    """
    w = _asarray(w).astype(np.float32, copy=False)
    if w.ndim != 2:
        raise ValueError("quantize_int4_packed expects a 2-D weight [N, K]")
    n, k = w.shape
    gs = int(group_size) if group_size and group_size > 0 else k
    ng = (k + gs - 1) // gs
    scales = np.empty((n, ng), np.float32)
    biases = np.empty((n, ng), np.float32)
    codes = np.empty((n, k), np.uint8)
    for g in range(ng):
        k0 = g * gs
        k1 = min(k0 + gs, k)
        sl = w[:, k0:k1]
        wmin = sl.min(axis=1)
        wmax = sl.max(axis=1)
        scale = (wmax - wmin) / 15.0
        scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
        biases[:, g] = wmin.astype(np.float32)
        scales[:, g] = scale
        q = np.rint((sl - wmin[:, None]) / scale[:, None])
        codes[:, k0:k1] = np.clip(q, 0, 15).astype(np.uint8)
    evens = codes[:, 0:k:2]                  # length ceil(K/2) = pitch
    odds = codes[:, 1:k:2]                    # length floor(K/2)
    packed = evens.astype(np.uint8, copy=True)
    packed[:, : odds.shape[1]] |= (odds.astype(np.uint8) << 4)
    return packed, scales, biases


#: OCP FP4 e2m1 positive magnitudes, indexed by the low 3 code bits; bit 3 = sign.
_FP4_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def quantize_fp4_packed(w: Any, group_size: int = 32, scale_mode: str = "mx"):
    """FP4 e2m1 packing — the MXFP4 / NVFP4 packed layout for the Apple GPU
    quantized-matmul lane (P3 follow-up).

    Per-group *symmetric* scale + 4-bit e2m1 codes packed 2 nibbles/byte (low =
    even k, high = odd k; bit 3 of each code is the sign, bits 0-2 index
    :data:`_FP4_E2M1_LUT`). ``scale_mode="mx"`` rounds the group scale up to a
    power of two (MXFP4 shared exponent, group 32); ``"nv"`` keeps an fp32 scale
    (NVFP4 uses group 16 + an fp8 scale — pass ``group_size=16``). Dequant is
    ``w ≈ scale_g · sign · e2m1_lut[code]`` (no bias — FP4 is symmetric).

    Returns ``(packed uint8 [N, ceil(K/2)], scales f32 [N, NG])`` with
    ``NG = ceil(K/group_size)``.
    """
    w = _asarray(w).astype(np.float32, copy=False)
    if w.ndim != 2:
        raise ValueError("quantize_fp4_packed expects a 2-D weight [N, K]")
    n, k = w.shape
    gs = int(group_size) if group_size and group_size > 0 else k
    ng = (k + gs - 1) // gs
    lut = _FP4_E2M1_LUT
    scales = np.empty((n, ng), np.float32)
    codes = np.zeros((n, k), np.uint8)
    for g in range(ng):
        k0 = g * gs
        k1 = min(k0 + gs, k)
        sl = w[:, k0:k1]
        amax = np.max(np.abs(sl), axis=1)
        scale = amax / 6.0  # largest |w| maps to the max FP4 magnitude (6)
        scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
        if scale_mode == "mx":
            scale = np.exp2(np.ceil(np.log2(scale))).astype(np.float32)
        scales[:, g] = scale
        t = sl / scale[:, None]
        idx = np.argmin(
            np.abs(np.abs(t)[..., None] - lut), axis=-1).astype(np.uint8)
        sign = (t < 0.0).astype(np.uint8) * 8
        codes[:, k0:k1] = idx | sign
    evens = codes[:, 0:k:2]
    odds = codes[:, 1:k:2]
    packed = evens.astype(np.uint8, copy=True)
    packed[:, : odds.shape[1]] |= (odds.astype(np.uint8) << 4)
    return packed, scales


def dequantize_fp4_packed(
    packed: Any, scales: Any, k: int, group_size: int = 32
) -> np.ndarray:
    """Inverse of :func:`quantize_fp4_packed` — reconstruct ``W[N,K]`` f32 from
    packed FP4 e2m1 codes + per-group scale."""
    packed = _asarray(packed).astype(np.uint8, copy=False)
    scales = _asarray(scales).astype(np.float32, copy=False)
    n = packed.shape[0]
    k = int(k)
    gs = int(group_size) if group_size and group_size > 0 else k
    ng = (k + gs - 1) // gs
    codes = np.empty((n, k), np.uint8)
    low = packed & 0x0F
    high = packed >> 4
    codes[:, 0:k:2] = low[:, : len(range(0, k, 2))]
    codes[:, 1:k:2] = high[:, : len(range(1, k, 2))]
    mag = _FP4_E2M1_LUT[codes & 7]
    sign = np.where((codes & 8) > 0, -1.0, 1.0).astype(np.float32)
    w = np.empty((n, k), np.float32)
    for g in range(ng):
        k0 = g * gs
        k1 = min(k0 + gs, k)
        w[:, k0:k1] = scales[:, g : g + 1] * sign[:, k0:k1] * mag[:, k0:k1]
    return w


def dequantize_int4_packed(
    packed: Any, scales: Any, biases: Any, k: int, group_size: int = 64
) -> np.ndarray:
    """Inverse of :func:`quantize_int4_packed` — reconstruct ``W[N, K]`` f32 from
    packed int4 codes + per-group affine scale/bias."""
    packed = _asarray(packed).astype(np.uint8, copy=False)
    scales = _asarray(scales).astype(np.float32, copy=False)
    biases = _asarray(biases).astype(np.float32, copy=False)
    n = packed.shape[0]
    k = int(k)
    gs = int(group_size) if group_size and group_size > 0 else k
    ng = (k + gs - 1) // gs
    codes = np.empty((n, k), np.uint8)
    low = packed & 0x0F
    high = packed >> 4
    codes[:, 0:k:2] = low[:, : len(range(0, k, 2))]
    codes[:, 1:k:2] = high[:, : len(range(1, k, 2))]
    w = np.empty((n, k), np.float32)
    for g in range(ng):
        k0 = g * gs
        k1 = min(k0 + gs, k)
        w[:, k0:k1] = (
            scales[:, g : g + 1] * codes[:, k0:k1].astype(np.float32)
            + biases[:, g : g + 1]
        )
    return w


__all__ = [
    "CalibrationObserver",
    "calibration_observer",
    "dequantize_fp4_packed",
    "dequantize_int4",
    "dequantize_int4_packed",
    "dequantize_int8",
    "fake_quantize",
    "grad_scaler_step",
    "quantize_fp4_packed",
    "quantize_int4",
    "quantize_int4_packed",
    "quantize_int8",
]
