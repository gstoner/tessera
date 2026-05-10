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
        scale = _symmetric_scale(x_arr, 127) if scale is None else np.float32(scale)
        q = np.round(x_arr / scale).clip(-127, 127).astype(np.int8)
        return q, np.float32(scale), 0
    scale = np.float32(scale if scale is not None else ((x_arr.max() - x_arr.min()) / 255.0 if x_arr.size else 1.0))
    if float(scale) == 0.0:
        scale = np.float32(1.0)
    q = np.round(x_arr / scale + zero_point).clip(-128, 127).astype(np.int8)
    return q, scale, int(zero_point)


def dequantize_int8(q: Any, scale: float, zero_point: int = 0) -> np.ndarray:
    return (_asarray(q).astype(np.float32) - float(zero_point)) * np.float32(scale)


def quantize_int4(x: Any, scale: float | None = None, zero_point: int = 0, *, symmetric: bool = True):
    """Quantize to signed int4 values stored in int8 containers."""
    x_arr = _asarray(x).astype(np.float32, copy=False)
    if symmetric:
        scale = _symmetric_scale(x_arr, 7) if scale is None else np.float32(scale)
        q = np.round(x_arr / scale).clip(-7, 7).astype(np.int8)
        return q, np.float32(scale), 0
    scale = np.float32(scale if scale is not None else ((x_arr.max() - x_arr.min()) / 15.0 if x_arr.size else 1.0))
    if float(scale) == 0.0:
        scale = np.float32(1.0)
    q = np.round(x_arr / scale + zero_point).clip(-8, 7).astype(np.int8)
    return q, scale, int(zero_point)


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


__all__ = [
    "CalibrationObserver",
    "calibration_observer",
    "dequantize_int4",
    "dequantize_int8",
    "fake_quantize",
    "grad_scaler_step",
    "quantize_int4",
    "quantize_int8",
]
