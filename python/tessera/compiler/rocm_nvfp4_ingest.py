"""GFX1201 NVFP4 checkpoint ingest for the MXFP4 W4A8 ABI."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import ml_dtypes
import numpy as np

from . import rocm_mxfp4 as mx

NVFP4_SCALE_BLOCK_K = 16
MXFP4_SCALE_BLOCK_K = 32
LOSSY_STEP = "e4m3_k16_to_e8m0_k32_scale_requantization"
_E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class NVFP4Projection:
    """One independently scaled checkpoint projection."""
    name: str
    packed_codes: np.ndarray
    e4m3_scales: np.ndarray
    global_scale: float


@dataclass(frozen=True)
class NVFP4IngestMetadata:
    name: str
    global_scale: float
    relative_rms_error: float
    sqnr_db: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "projection": self.name,
            "source_global_scale": self.global_scale,
            "relative_rms_error": self.relative_rms_error,
            "sqnr_db": self.sqnr_db,
            "lossy_steps": [LOSSY_STEP],
        }


@dataclass(frozen=True)
class MXFP4IngestedWeights:
    """Packed MXFP4 W4A8 weights, with projection boundaries preserved."""
    packed_codes: np.ndarray           # [N,K/2], low nibble is even K.
    scale_exponents: np.ndarray        # [K/32,N], E8M0.
    projection_names: tuple[str, ...]
    row_offsets: tuple[int, ...]
    metadata: tuple[NVFP4IngestMetadata, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.packed_codes.shape[0]), int(self.packed_codes.shape[1] * 2))

    def numeric_policy(self) -> dict[str, object]:
        return {
            "source_format": "nvfp4_e2m1_e4m3_k16",
            "destination_format": "mxfp4_e2m1_e8m0_k32",
            "execution_mode": "explicit_scale_requantization",
            "lossy_steps": [LOSSY_STEP],
            "source_scale_application": "projection_global_times_e4m3",
            "destination_scale_order": "k_group_n",
            "projection_metadata": [item.as_dict() for item in self.metadata],
        }


def _validate_projection(projection: NVFP4Projection) -> tuple[int, int, np.ndarray, np.ndarray, float]:
    if not projection.name:
        raise ValueError("NVFP4 projection name must be non-empty")
    packed = np.asarray(projection.packed_codes)
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise TypeError("NVFP4 packed codes must be uint8 [N,K/2]")
    n, packed_k = packed.shape
    k = packed_k * 2
    if n <= 0 or k <= 0 or k % MXFP4_SCALE_BLOCK_K:
        raise ValueError("NVFP4 ingest requires positive N and K divisible by 32")
    scales = np.asarray(projection.e4m3_scales)
    if scales.dtype != np.dtype(ml_dtypes.float8_e4m3fn):
        raise TypeError("NVFP4 scales must be decoded float8_e4m3fn values")
    if scales.shape != (n, k // NVFP4_SCALE_BLOCK_K):
        raise ValueError(
            "NVFP4 scales must use [N,K/16] order; "
            f"got {scales.shape} for {(n, k)}"
        )
    global_scale = float(projection.global_scale)
    if not math.isfinite(global_scale) or global_scale <= 0.0:
        raise ValueError("NVFP4 projection global_scale must be finite and positive")
    scales_f32 = scales.astype(np.float32) * np.float32(global_scale)
    if not np.isfinite(scales_f32).all() or np.any(scales_f32 < 0.0):
        raise ValueError("NVFP4 E4M3 scales must be finite and non-negative")
    return n, k, packed, scales_f32, global_scale


def _choose_e8m0_exponent(code_values: np.ndarray, scale_values: np.ndarray) -> int:
    """Choose no-clip or one-binade-finer by decoded-weight squared error."""
    maximum = float(np.max(scale_values, initial=0.0))
    if maximum == 0.0:
        return 0
    no_clip = math.ceil(math.log2(maximum))
    candidates = [e for e in (no_clip, no_clip - 1) if -126 <= e <= 127]
    if not candidates:
        raise ValueError("NVFP4 scale is outside the representable E8M0 range")
    values = np.asarray(code_values, np.float64).reshape(2, 16)
    scales = np.asarray(scale_values, np.float64).reshape(2, 1)
    best_exp, best_error = candidates[0], math.inf
    for exponent in candidates:
        candidate = math.ldexp(1.0, exponent)
        error = float(np.square(values * (scales - candidate)).sum())
        if error < best_error:  # ties retain the no-clip candidate.
            best_exp, best_error = exponent, error
    return best_exp


def requantize_nvfp4_projection(
    projection: NVFP4Projection,
) -> tuple[np.ndarray, np.ndarray, NVFP4IngestMetadata]:
    """Keep E2M1 codes; quantize adjacent E4M3/K16 scales to one E8M0/K32 scale."""
    n, k, packed, scales, global_scale = _validate_projection(projection)
    codes = mx.unpack_e2m1_codes(packed)
    code_values = _E2M1[codes]
    source = code_values * scales.repeat(16, axis=1)
    exponents = np.empty((k // 32, n), dtype=np.uint8)
    for row in range(n):
        for group in range(k // 32):
            start = group * 32
            scale_pair = scales[row, group * 2:group * 2 + 2]
            exponent = _choose_e8m0_exponent(
                code_values[row, start:start + 32], scale_pair
            )
            exponents[group, row] = 0 if not np.any(scale_pair) else exponent + 127
    reconstructed = mx.exact_weights(codes, exponents)
    signal = float(np.square(source.astype(np.float64)).sum())
    error = float(np.square(
        source.astype(np.float64) - reconstructed.astype(np.float64)
    ).sum())
    relative_rms = math.sqrt(error / signal) if signal else 0.0
    sqnr = 10.0 * math.log10(signal / error) if signal and error else None
    metadata = NVFP4IngestMetadata(
        projection.name, global_scale, relative_rms, sqnr
    )
    return np.ascontiguousarray(packed.copy()), exponents, metadata


def ingest_nvfp4_projections(
    projections: Sequence[NVFP4Projection],
) -> MXFP4IngestedWeights:
    """Convert projections independently before row concatenation.

    This keeps gate/up projection scales independent instead of collapsing
    their distinct source global scales into one shared value.
    """
    if not projections:
        raise ValueError("at least one NVFP4 projection is required")
    names = [item.name for item in projections]
    if len(set(names)) != len(names):
        raise ValueError("NVFP4 projection names must be unique")
    converted = [requantize_nvfp4_projection(item) for item in projections]
    if len({item[0].shape[1] for item in converted}) != 1:
        raise ValueError("merged NVFP4 projections must have matching K")
    offsets = [0]
    for codes, _, _ in converted:
        offsets.append(offsets[-1] + codes.shape[0])
    return MXFP4IngestedWeights(
        packed_codes=np.ascontiguousarray(
            np.concatenate([item[0] for item in converted], axis=0)
        ),
        scale_exponents=np.ascontiguousarray(
            np.concatenate([item[1] for item in converted], axis=1)
        ),
        projection_names=tuple(names),
        row_offsets=tuple(offsets),
        metadata=tuple(item[2] for item in converted),
    )


__all__ = [
    "LOSSY_STEP",
    "MXFP4IngestedWeights",
    "NVFP4IngestMetadata",
    "NVFP4Projection",
    "ingest_nvfp4_projections",
    "requantize_nvfp4_projection",
]
