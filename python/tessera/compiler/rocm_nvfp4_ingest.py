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
LOSSY_STEP = "nvfp4_e4m3_k16_to_mxfp4_e8m0_k32_scale_and_code_requantization"
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
            "destination_code_selection": "nearest_signed_e2m1_by_weight_sse",
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
    scales_scaled = scales.astype(np.float64) * global_scale
    if not np.isfinite(scales_scaled).all() or np.any(scales_scaled < 0.0):
        raise ValueError("NVFP4 E4M3 scales must be finite and non-negative")
    return n, k, packed, scales_scaled, global_scale


def _choose_e8m0_exponent(code_values: np.ndarray, scale_values: np.ndarray) -> int:
    """Choose the representable power of two minimizing decoded-weight SSE."""
    maximum = float(np.max(scale_values, initial=0.0))
    if maximum == 0.0:
        return 0
    minimum_e8m0 = math.ldexp(1.0, -126)
    if maximum < minimum_e8m0:
        raise ValueError("NVFP4 scale is outside the representable E8M0 range")

    values = np.asarray(code_values, np.float64).reshape(2, 16)
    scales = np.asarray(scale_values, np.float64).reshape(2, 1)
    weights = np.square(values)
    total_weight = float(weights.sum())
    # The squared-error objective is sum(c_i^2 * (s_i - q)^2), whose
    # unconstrained minimizer is the code-energy-weighted mean source scale.
    mean_scale = (
        float((weights * scales).sum()) / total_weight
        if total_weight else maximum
    )
    if mean_scale <= 0.0:
        return 0
    floor_exp = math.floor(math.log2(mean_scale))
    candidates = sorted({
        min(127, max(-126, floor_exp)),
        min(127, max(-126, floor_exp + 1)),
    })
    best_exp, best_error = candidates[0], math.inf
    for exponent in candidates:
        candidate = math.ldexp(1.0, exponent)
        error = float((weights * np.square(scales - candidate)).sum())
        if error < best_error:
            best_exp, best_error = exponent, error
    return best_exp


def _requantize_e2m1_block(source: np.ndarray, seed_exponent: int) -> tuple[np.ndarray, int]:
    """Jointly choose one E8M0 scale and nearest signed E2M1 codes for K32."""
    source64 = np.asarray(source, np.float64).reshape(32)
    if not np.any(source64):
        return np.zeros(32, dtype=np.uint8), 0

    candidates = range(max(-126, seed_exponent - 4), min(127, seed_exponent + 4) + 1)
    levels = _E2M1[:8].astype(np.float64)
    best_codes = None
    best_exponent = seed_exponent
    best_key = (math.inf, math.inf, math.inf)
    for exponent in candidates:
        scale = math.ldexp(1.0, exponent)
        normalized = np.abs(source64) / scale
        indices = np.abs(normalized[:, None] - levels[None, :]).argmin(axis=1)
        signed_codes = indices.astype(np.uint8)
        signed_codes[source64 < 0.0] |= np.uint8(8)
        decoded = _E2M1[signed_codes].astype(np.float64) * scale
        error = float(np.square(source64 - decoded).sum())
        key = (error, abs(exponent - seed_exponent), exponent)
        if key < best_key:
            best_key = key
            best_codes = signed_codes
            best_exponent = exponent
    assert best_codes is not None
    return best_codes, best_exponent


def _choose_e8m0_exponents(
    code_values: np.ndarray, scale_values: np.ndarray
) -> np.ndarray:
    """Vectorized form of _choose_e8m0_exponent over [..., 32] K32 blocks."""
    values = np.asarray(code_values, np.float64)
    scales = np.asarray(scale_values, np.float64)
    weights = np.square(values)
    maximum = np.max(scales, axis=-1)
    minimum_e8m0 = math.ldexp(1.0, -126)
    if np.any((maximum > 0.0) & (maximum < minimum_e8m0)):
        raise ValueError("NVFP4 scale is outside the representable E8M0 range")
    total_weight = np.sum(weights, axis=-1)
    weighted_scale = np.sum(weights * scales, axis=-1)
    mean_scale = np.divide(
        weighted_scale, total_weight,
        out=maximum.copy(), where=total_weight != 0.0,
    )
    positive = mean_scale > 0.0
    safe_mean = np.where(positive, mean_scale, 1.0)
    floor_exp = np.floor(np.log2(safe_mean)).astype(np.int64)
    lower_exp = np.clip(floor_exp, -126, 127)
    upper_exp = np.clip(floor_exp + 1, -126, 127)
    lower_scale = np.exp2(lower_exp.astype(np.float64))
    upper_scale = np.exp2(upper_exp.astype(np.float64))
    lower_error = np.sum(weights * np.square(scales - lower_scale[..., None]), axis=-1)
    upper_error = np.sum(weights * np.square(scales - upper_scale[..., None]), axis=-1)
    selected = np.where(upper_error < lower_error, upper_exp, lower_exp)
    return np.where(positive, selected, 0).astype(np.int64)


def _requantize_e2m1_blocks(
    source: np.ndarray, seed_exponents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized joint E2M1/E8M0 search over a batch of K32 blocks."""
    values = np.asarray(source, np.float64)
    seeds = np.asarray(seed_exponents, np.int64)
    if values.shape[:-1] != seeds.shape or values.shape[-1] != MXFP4_SCALE_BLOCK_K:
        raise ValueError("joint NVFP4 requantization expects [...,32] blocks")

    levels = _E2M1[:8].astype(np.float64)
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    best_error = np.full(seeds.shape, np.inf, dtype=np.float64)
    best_distance = np.full(seeds.shape, np.iinfo(np.int64).max, dtype=np.int64)
    best_exponent = seeds.copy()
    best_codes = np.zeros(values.shape, dtype=np.uint8)

    for delta in range(-4, 5):
        exponent = seeds + delta
        valid = (exponent >= -126) & (exponent <= 127)
        scale = np.exp2(np.clip(exponent, -126, 127).astype(np.float64))
        normalized = np.abs(values) / scale[..., None]
        magnitude_codes = np.searchsorted(
            midpoints, normalized.reshape(-1), side="left"
        ).reshape(values.shape)
        codes = magnitude_codes.astype(np.uint8)
        codes |= np.where(values < 0.0, np.uint8(8), np.uint8(0))
        decoded = _E2M1[codes] * scale[..., None]
        error = np.sum(np.square(values - decoded), axis=-1)
        distance = abs(delta)
        better = valid & (
            (error < best_error)
            | ((error == best_error) & (distance < best_distance))
            | ((error == best_error) & (distance == best_distance)
               & (exponent < best_exponent))
        )
        best_error = np.where(better, error, best_error)
        best_distance = np.where(better, distance, best_distance)
        best_exponent = np.where(better, exponent, best_exponent)
        best_codes = np.where(better[..., None], codes, best_codes)

    nonzero = np.any(values != 0.0, axis=-1)
    best_codes = np.where(nonzero[..., None], best_codes, np.uint8(0))
    best_exponent = np.where(nonzero, best_exponent, 0)
    return best_codes, best_exponent.astype(np.int64)


def requantize_nvfp4_projection(
    projection: NVFP4Projection,
) -> tuple[np.ndarray, np.ndarray, NVFP4IngestMetadata]:
    """Requantize NVFP4 weights into the MXFP4 E2M1/E8M0 K32 contract."""
    n, k, packed, scales, global_scale = _validate_projection(projection)
    codes = mx.unpack_e2m1_codes(packed)
    code_values = _E2M1[codes]
    source = code_values * scales.repeat(16, axis=1)
    groups = k // MXFP4_SCALE_BLOCK_K
    destination_codes = np.empty_like(codes)
    exponents = np.empty((groups, n), dtype=np.uint8)

    # The scalar reference performs Python dispatch for every row and K32
    # block. Work in bounded row batches so model-sized checkpoints keep
    # temporaries small while NumPy evaluates the same candidate search in
    # vectorized arrays.
    rows_per_batch = 64
    code_blocks = code_values.reshape(n, groups, MXFP4_SCALE_BLOCK_K)
    scale_pairs = scales.reshape(n, groups, 2)
    for row_start in range(0, n, rows_per_batch):
        row_end = min(n, row_start + rows_per_batch)
        batch_codes = code_blocks[row_start:row_end]
        batch_scales = np.repeat(scale_pairs[row_start:row_end], 16, axis=-1)
        batch_source = batch_codes * batch_scales
        seed_exponents = _choose_e8m0_exponents(batch_codes, batch_scales)
        converted_codes, converted_exponents = _requantize_e2m1_blocks(
            batch_source, seed_exponents
        )
        destination_codes[row_start:row_end] = converted_codes.reshape(
            row_end - row_start, k
        )
        has_scale = np.any(scale_pairs[row_start:row_end] != 0.0, axis=-1)
        exponents[:, row_start:row_end] = np.where(
            has_scale, converted_exponents + 127, 0
        ).T.astype(np.uint8)
    packed_destination = mx.pack_e2m1_codes(destination_codes)
    reconstructed = mx.exact_weights(destination_codes, exponents, dtype=np.float64)
    signal = float(np.square(source.astype(np.float64)).sum())
    error = float(np.square(
        source.astype(np.float64) - reconstructed.astype(np.float64)
    ).sum())
    relative_rms = math.sqrt(error / signal) if signal else 0.0
    sqnr = 10.0 * math.log10(signal / error) if signal and error else None
    metadata = NVFP4IngestMetadata(
        projection.name, global_scale, relative_rms, sqnr
    )
    return np.ascontiguousarray(packed_destination), exponents, metadata


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
    "nvfp4_requantization_policy",
    "reference_nvfp4_requantize",
]


def nvfp4_requantization_policy() -> dict[str, object]:
    return {
        "source_format": "nvfp4_e2m1_e4m3_k16",
        "destination_format": "mxfp4_e2m1_e8m0_k32",
        "execution_mode": "explicit_scale_requantization",
        "lossy_steps": [LOSSY_STEP],
        "source_scale_application": "projection_global_times_e4m3",
        "destination_code_selection": "nearest_signed_e2m1_by_weight_sse",
        "destination_scale_order": "k_group_n",
    }


def reference_nvfp4_requantize(codes, scales, globals_, *, row_offsets, numeric_policy):
    """Target-neutral eager oracle; native packages own physical conversion."""
    if numeric_policy != nvfp4_requantization_policy():
        raise ValueError("NVFP4 ingest requires its exact explicit numeric policy")
    codes = np.asarray(codes)
    scales = np.asarray(scales)
    globals_ = np.asarray(globals_)
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise TypeError("NVFP4 ingest codes must be uint8 rank two")
    n, packed_k = codes.shape
    k = packed_k*2
    offsets = tuple(row_offsets)
    if (len(offsets)<2 or any(type(x) is not int for x in offsets)
            or offsets[0]!=0 or offsets[-1]!=n
            or any(a>=b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("NVFP4 projection row boundaries must cover increasing rows")
    if globals_.dtype != np.float64 or globals_.shape != (len(offsets)-1,):
        raise TypeError("NVFP4 ingest globals must be f64 per projection")
    if scales.dtype != np.dtype(ml_dtypes.float8_e4m3fn) or scales.shape != (n,k//16):
        raise TypeError("NVFP4 ingest scales must be E4M3 [N,K/16]")
    projections = [
        NVFP4Projection(f"projection_{i}",codes[a:b],scales[a:b],float(globals_[i]))
        for i,(a,b) in enumerate(zip(offsets,offsets[1:]))]
    converted = ingest_nvfp4_projections(projections)
    stats = np.empty((n,k//32,2),np.float64)
    for i,(a,b) in enumerate(zip(offsets,offsets[1:])):
        source = _E2M1[mx.unpack_e2m1_codes(codes[a:b])].astype(np.float64)
        source *= (scales[a:b].astype(np.float64)*globals_[i]).repeat(16,axis=1)
        decoded = mx.exact_weights(
            mx.unpack_e2m1_codes(converted.packed_codes[a:b]),
            converted.scale_exponents[:,a:b]).astype(np.float64)
        stats[a:b,:,0] = np.square(source).reshape(b-a,k//32,32).sum(axis=-1)
        stats[a:b,:,1] = np.square(source-decoded).reshape(b-a,k//32,32).sum(axis=-1)
    return converted.packed_codes,converted.scale_exponents,stats
