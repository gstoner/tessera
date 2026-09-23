"""Physical MXFP4/W4A8 contracts for the gfx1201 ROCm route.

``mxfp4`` is still a planned/gated public dtype.  This module therefore does
not add it to Graph IR or claim executable support.  It defines the physical
format that a future ``ROCM-MXFP4-W4A8-1`` lowering must consume below Graph
IR, together with a bit-accurate host reference.

The two numerical modes are deliberately distinct:

* ``exact_per_block`` keeps the OCP MX scale group at 32 elements.  A kernel
  accumulates the two 16-wide gfx12 FP8 WMMA steps for one group in FP32,
  applies that group's E8M0 scale, and only then adds to the running result.
* ``folded_row_reference`` chooses one reference exponent per output row and
  folds each block's exponent difference into E4M3 weight bytes.  It removes
  the per-group multiply, but it is lossless only when every non-zero shifted
  E2M1 value remains representable in E4M3.  With subnormals that is guaranteed
  for exponent deltas through eight, not for arbitrary MXFP4 tensors.

The scale group is a format semantic.  It is independent of the WMMA
instruction K (16), the scheduled macro K tile (commonly 64 or 128), K unroll,
and split-K decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import ml_dtypes
import numpy as np


MXFP4_GROUP_K = 32
GFX12_FP8_WMMA_K = 16
MXFP4_NIBBLE_ORDER = "low_even_high_odd"
MXFP4_SCALE_ORDER = "k_group_n"
MXFP4_FRAGMENT_ORDER = "n_tile_k_step_half_row_bytes"

# Stable physical-layout identities.  These are part of the package/launch
# contract rather than informal names for NumPy views: changing byte order
# requires a new versioned identity and ABI.
MXFP4_CHECKPOINT_LAYOUT_V1 = "mxfp4.checkpoint_n_k2.low_even.v1"
MXFP4_TRANSPOSED_LAYOUT_V1 = "mxfp4.runtime_k2_n.low_even.v1"
MXFP4_GFX12_FRAGMENT_LAYOUT_V1 = "mxfp4.gfx12.n16_k16_lane_u32.v1"
MXFP4_FOLDED_ROW_LAYOUT_V1 = "mxfp4.gfx12.folded_e4m3.nk_row_major.v1"
MXFP4_AITER_SHUFFLED_LAYOUT_V1 = "mxfp4.aiter_shuffled.opaque.v1"
MXFP4_QUARK_REORDER_LAYOUT_V1 = "mxfp4.quark_reorder.opaque.v1"

MXFP4Mode = Literal["exact_per_block", "folded_row_reference"]

_E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _as_u8(name: str, values: np.ndarray, *, maximum: int) -> np.ndarray:
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must use an integer element type")
    if np.any(raw < 0) or np.any(raw > maximum):
        raise ValueError(f"{name} values must be in [0,{maximum}]")
    return raw.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class MXFP4WeightLayout:
    """One versioned packed-weight storage identity.

    ``conversion_supported`` means Tessera owns a checked conversion from the
    canonical checkpoint representation.  Recognizing a third-party layout is
    intentionally weaker than claiming that its permutation is implemented.
    """

    layout_id: str
    logical_shape: str
    lane_read: str
    conversion_supported: bool


MXFP4_WEIGHT_LAYOUTS: dict[str, MXFP4WeightLayout] = {
    MXFP4_CHECKPOINT_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_CHECKPOINT_LAYOUT_V1,
        "[N,K/2]",
        "strided_checkpoint_rows",
        True,
    ),
    MXFP4_TRANSPOSED_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_TRANSPOSED_LAYOUT_V1,
        "[K/2,N]",
        "legacy_scalar_k_major",
        True,
    ),
    MXFP4_GFX12_FRAGMENT_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
        "[N,K/2]",
        "one_contiguous_uint32_per_lane_per_n16_k16_step",
        True,
    ),
    MXFP4_FOLDED_ROW_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_FOLDED_ROW_LAYOUT_V1,
        "[N,K]",
        "e4m3_bytes_with_separate_row_reference",
        False,  # Requires E8M0 scales and explicit approximate-policy consent.
    ),
    MXFP4_AITER_SHUFFLED_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_AITER_SHUFFLED_LAYOUT_V1,
        "opaque",
        "third_party_incompatible",
        False,
    ),
    MXFP4_QUARK_REORDER_LAYOUT_V1: MXFP4WeightLayout(
        MXFP4_QUARK_REORDER_LAYOUT_V1,
        "[N,K/2]",
        "third_party_byte_order_unverified",
        False,
    ),
}


def mxfp4_weight_layout(layout_id: str) -> MXFP4WeightLayout:
    """Resolve a stable MXFP4 layout identity or fail closed."""
    try:
        return MXFP4_WEIGHT_LAYOUTS[layout_id]
    except KeyError as exc:
        raise ValueError(f"unknown MXFP4 weight layout {layout_id!r}") from exc


@dataclass(frozen=True)
class MXFP4PhysicalContract:
    """The launch-visible storage and scale contract for gfx1201 W4A8."""

    weight_storage: str = "packed_e2m1"
    weight_layout: str = MXFP4_CHECKPOINT_LAYOUT_V1
    weight_container: str = "uint8"
    elements_per_container: int = 2
    nibble_order: str = MXFP4_NIBBLE_ORDER
    scale_storage: str = "e8m0"
    scale_group_k: int = MXFP4_GROUP_K
    scale_order: str = MXFP4_SCALE_ORDER
    activation_storage: str = "fp8_e4m3"
    activation_scale: str = "per_token_fp32"
    accumulator: str = "fp32"
    output: str = "bf16"
    instruction_k: int = GFX12_FP8_WMMA_K

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "weight_storage": self.weight_storage,
            "weight_layout": self.weight_layout,
            "weight_container": self.weight_container,
            "elements_per_container": self.elements_per_container,
            "nibble_order": self.nibble_order,
            "scale_storage": self.scale_storage,
            "scale_group_k": self.scale_group_k,
            "scale_order": self.scale_order,
            "activation_storage": self.activation_storage,
            "activation_scale": self.activation_scale,
            "accumulator": self.accumulator,
            "output": self.output,
            "instruction_k": self.instruction_k,
        }


@dataclass(frozen=True)
class FoldedRowReference:
    """E4M3 bytes and row exponents produced by the fast MXFP4 fold."""

    weight_bytes: np.ndarray
    row_reference: np.ndarray
    exponent_delta: np.ndarray
    lossless: bool
    inexact_value_count: int
    max_normalized_abs_error: float
    max_normalized_relative_error: float
    approximate_policy: str = "explicit_allow"

    def __post_init__(self) -> None:
        if self.weight_bytes.dtype != np.uint8:
            raise TypeError("folded MXFP4 weights must be raw E4M3 uint8 bytes")
        if self.row_reference.dtype != np.uint8:
            raise TypeError("MXFP4 row reference must use E8M0 uint8 exponents")


def numeric_policy(mode: MXFP4Mode) -> dict[str, object]:
    """Return the numerical policy for one physical MXFP4 execution mode."""
    if mode not in ("exact_per_block", "folded_row_reference"):
        raise ValueError(f"unknown MXFP4 execution mode {mode!r}")
    return {
        **MXFP4PhysicalContract().as_metadata_dict(),
        "format": "mxfp4",
        "execution_mode": mode,
        "scale_application": (
            "fp32_partial_per_32_k"
            if mode == "exact_per_block"
            else "e4m3_weight_then_row_epilogue"
        ),
        "lossless_for_all_inputs": mode == "exact_per_block",
        "fold_exact_max_exponent_delta": None if mode == "exact_per_block" else 8,
    }


def pack_e2m1_codes(codes: np.ndarray) -> np.ndarray:
    """Pack ``[N,K]`` E2M1 codes with low nibble = even K."""
    c = _as_u8("MXFP4 E2M1 codes", codes, maximum=15)
    if c.ndim != 2 or c.shape[1] % 2:
        raise ValueError("MXFP4 codes must have shape [N,K] with even K")
    return np.ascontiguousarray(c[:, 0::2] | (c[:, 1::2] << np.uint8(4)))


def unpack_e2m1_codes(packed: np.ndarray) -> np.ndarray:
    """Unpack checkpoint-order ``[N,K/2]`` bytes into ``[N,K]`` codes."""
    p = _as_u8("packed MXFP4 weights", packed, maximum=255)
    if p.ndim != 2:
        raise ValueError("packed MXFP4 weights must have shape [N,K/2]")
    out = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = p & np.uint8(0x0F)
    out[:, 1::2] = p >> np.uint8(4)
    return out


def to_fragment_order(packed: np.ndarray) -> np.ndarray:
    """Permute checkpoint-order weights to the gfx12 WMMA lane-read order.

    The output retains shape ``[N,K/2]`` but changes physical byte order.  One
    wave reads 32 adjacent uint32 lane slots for each 16x16 N/K step.
    """
    p = _as_u8("packed MXFP4 weights", packed, maximum=255)
    if p.ndim != 2:
        raise ValueError("packed MXFP4 weights must have shape [N,K/2]")
    n, packed_k = p.shape
    k = packed_k * 2
    if n % 16 or k % 16:
        raise ValueError("fragment-order MXFP4 weights require N and K divisible by 16")
    tiled = p.reshape(n // 16, 16, k // 16, 2, 4)
    return np.ascontiguousarray(tiled.transpose(0, 2, 3, 1, 4).reshape(n, packed_k))


def from_fragment_order(fragment_order: np.ndarray) -> np.ndarray:
    """Invert :func:`to_fragment_order`."""
    p = _as_u8("fragment-order MXFP4 weights", fragment_order, maximum=255)
    if p.ndim != 2:
        raise ValueError("fragment-order MXFP4 weights must have shape [N,K/2]")
    n, packed_k = p.shape
    k = packed_k * 2
    if n % 16 or k % 16:
        raise ValueError("fragment-order MXFP4 weights require N and K divisible by 16")
    tiled = p.reshape(n // 16, k // 16, 2, 16, 4)
    return np.ascontiguousarray(tiled.transpose(0, 3, 1, 2, 4).reshape(n, packed_k))


def convert_weight_layout(
    weights: np.ndarray,
    *,
    source: str,
    destination: str,
) -> np.ndarray:
    """Convert packed weights once at model/package load time.

    The canonical checkpoint layout is the hub.  Third-party source layouts
    remain recognized but refused until Tessera owns byte-level conversions.
    """
    src = mxfp4_weight_layout(source)
    dst = mxfp4_weight_layout(destination)
    if MXFP4_FOLDED_ROW_LAYOUT_V1 in (source, destination):
        raise ValueError(
            "folded MXFP4 conversion requires E8M0 scales and explicit "
            "approximate-policy consent; use prepare_folded_weights"
        )
    if not src.conversion_supported or not dst.conversion_supported:
        raise ValueError(
            f"MXFP4 layout conversion {source!r} -> {destination!r} "
            "is recognized but has no proved Tessera conversion contract"
        )
    packed = _as_u8("packed MXFP4 weights", weights, maximum=255)
    if packed.ndim != 2:
        raise ValueError("packed MXFP4 weights must be rank 2")
    if source == destination:
        return np.ascontiguousarray(packed)
    if source == MXFP4_CHECKPOINT_LAYOUT_V1:
        checkpoint = packed
    elif source == MXFP4_TRANSPOSED_LAYOUT_V1:
        checkpoint = np.ascontiguousarray(packed.T)
    elif source == MXFP4_GFX12_FRAGMENT_LAYOUT_V1:
        checkpoint = from_fragment_order(packed)
    else:  # Registry totality makes this defensive.
        raise ValueError(f"unsupported MXFP4 source layout {source!r}")
    if destination == MXFP4_CHECKPOINT_LAYOUT_V1:
        return np.ascontiguousarray(checkpoint)
    if destination == MXFP4_TRANSPOSED_LAYOUT_V1:
        return np.ascontiguousarray(checkpoint.T)
    if destination == MXFP4_GFX12_FRAGMENT_LAYOUT_V1:
        return to_fragment_order(checkpoint)
    raise ValueError(f"unsupported MXFP4 destination layout {destination!r}")


def _validate_scale_plane(codes: np.ndarray, scale_exponents: np.ndarray) -> tuple[int, int]:
    if codes.ndim != 2:
        raise ValueError("MXFP4 codes must have shape [N,K]")
    n, k = codes.shape
    if k % MXFP4_GROUP_K:
        raise ValueError("MXFP4 K must be divisible by the 32-element scale group")
    if scale_exponents.shape != (k // MXFP4_GROUP_K, n):
        raise ValueError(
            "MXFP4 scales must have [K/32,N] order; "
            f"got {scale_exponents.shape} for weights {(n, k)}"
        )
    return n, k


def _e8m0_factor(exponents: np.ndarray) -> np.ndarray:
    e = np.asarray(exponents, dtype=np.uint8)
    # Encoding zero is reserved for a zero block in the physical route.
    return np.where(e == 0, np.float32(0.0), np.exp2(e.astype(np.float32) - 127.0))


def exact_weights(codes: np.ndarray, scale_exponents: np.ndarray) -> np.ndarray:
    """Decode MXFP4 to FP32 using its exact per-32 E8M0 scale contract."""
    c = _as_u8("MXFP4 E2M1 codes", codes, maximum=15)
    s = _as_u8("MXFP4 E8M0 exponents", scale_exponents, maximum=254)
    n, k = _validate_scale_plane(c, s)
    values = _E2M1[c]
    scale = _e8m0_factor(s).T.repeat(MXFP4_GROUP_K, axis=1)
    return np.ascontiguousarray(values.reshape(n, k) * scale)


def fold_to_row_reference(codes: np.ndarray,
                          scale_exponents: np.ndarray, *,
                          allow_approximate: bool = False) -> FoldedRowReference:
    """Fold block exponents into E4M3 bytes relative to one exponent per row.

    This is an approximate execution route even when one particular payload
    happens to fold exactly, so callers must opt in with
    ``allow_approximate=True``. ``lossless`` is computed from the actual codes,
    not only the largest delta:
    zero values remain exact at every delta, while non-zero values may round or
    underflow once the shifted magnitude leaves the E4M3 grid.
    """
    if not allow_approximate:
        raise ValueError(
            "folded_row_reference requires an explicit approximate numerical policy"
        )
    c = _as_u8("MXFP4 E2M1 codes", codes, maximum=15)
    s = _as_u8("MXFP4 E8M0 exponents", scale_exponents, maximum=254)
    n, k = _validate_scale_plane(c, s)
    row_ref = s.max(axis=0).astype(np.uint8, copy=False)
    delta = row_ref[None, :].astype(np.int16) - s.astype(np.int16)
    if np.any(delta < 0):  # defensive: max() should make this impossible
        raise ValueError("MXFP4 row reference must not be below a block exponent")
    expanded_delta = delta.T.repeat(MXFP4_GROUP_K, axis=1)
    normalized = _E2M1[c] * np.exp2(-expanded_delta.astype(np.float32))
    # E8M0 code zero is a reserved zero-block marker, not exponent zero.
    # Preserve that semantic before conversion and losslessness comparison;
    # otherwise a non-zero E2M1 payload becomes a tiny non-zero folded weight.
    reserved_zero = (s.T == 0).repeat(MXFP4_GROUP_K, axis=1)
    normalized = np.where(reserved_zero, np.float32(0.0), normalized)
    folded = normalized.astype(ml_dtypes.float8_e4m3fn)
    folded_f32 = folded.astype(np.float32)
    different = folded_f32 != normalized
    abs_error = np.abs(
        folded_f32.astype(np.float64) - normalized.astype(np.float64)
    )
    nonzero = normalized != 0.0
    relative_error = np.zeros_like(abs_error)
    np.divide(
        abs_error,
        np.abs(normalized.astype(np.float64)),
        out=relative_error,
        where=nonzero,
    )
    return FoldedRowReference(
        weight_bytes=np.ascontiguousarray(folded.view(np.uint8).reshape(n, k)),
        row_reference=np.ascontiguousarray(row_ref),
        exponent_delta=np.ascontiguousarray(delta),
        lossless=not bool(np.any(different)),
        inexact_value_count=int(np.count_nonzero(different)),
        max_normalized_abs_error=float(abs_error.max(initial=0.0)),
        max_normalized_relative_error=float(relative_error.max(initial=0.0)),
    )


def folded_weights(folded: FoldedRowReference) -> np.ndarray:
    """Reconstruct FP32 weights represented by a folded-row-reference payload."""
    values = folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return np.ascontiguousarray(values * _e8m0_factor(folded.row_reference)[:, None])
