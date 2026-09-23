"""Packed-weight model-load payload and manual gfx1201 folded-prefill route.

The expanded folded HSACO must reject this payload. The separate packed-decode
kernel is a manually launchable candidate, not an automatically selected route.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .rocm_mxfp4 import (
    FoldedRowReference,
    MXFP4_CHECKPOINT_LAYOUT_V1,
    MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    convert_weight_layout,
)
from .rocm_mxfp4_folded import prepare_folded_weights
from .rocm_mxfp4_folded import emit_mxfp4_folded_prefill_hip
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)
from .rocm_mxfp4_native import (
    _extract_gfx1201_hsaco, _rocm_hipcc, _schedule_hash,
    _target_string_attr,
)
from .rocm_native import (
    ROCMNativePackage, _driver_selected_device_libraries, _rocm_path,
    _version_fingerprint,
)


PACKED_FOLDED_SCALE_PLANE_V1 = "mxfp4.e8m0.k32_plus_row_reference.last_row.v1"
PACKED_FOLDED_WEIGHT_LAYOUT_V1 = MXFP4_GFX12_FRAGMENT_LAYOUT_V1
PACKED_FOLDED_PHYSICAL_V1 = "rocm_mxfp4_w4a8_packed_folded_prefill_v1"
PACKED_FOLDED_TARGET_ABI_V1 = (
    "tessera.rocm.mxfp4_w4a8.a_bpacked_sa_scaleplane_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.approx_bm256_tm4.v1"
)

_PACKED_HELPERS = r'''
// E2M1 magnitude folded to E4M3FN at each exponent difference. Values at
// delta >= 13 round to signed zero; E8M0 code zero is a zero block.
__constant__ unsigned char tessera_fold_e2m1_e4m3[13][16] = {
  {0,48,56,60,64,68,72,76,128,176,184,188,192,196,200,204},
  {0,40,48,52,56,60,64,68,128,168,176,180,184,188,192,196},
  {0,32,40,44,48,52,56,60,128,160,168,172,176,180,184,188},
  {0,24,32,36,40,44,48,52,128,152,160,164,168,172,176,180},
  {0,16,24,28,32,36,40,44,128,144,152,156,160,164,168,172},
  {0,8,16,20,24,28,32,36,128,136,144,148,152,156,160,164},
  {0,4,8,12,16,20,24,28,128,132,136,140,144,148,152,156},
  {0,2,4,6,8,12,16,20,128,130,132,134,136,140,144,148},
  {0,1,2,3,4,6,8,12,128,129,130,131,132,134,136,140},
  {0,0,1,2,2,3,4,6,128,128,129,130,130,131,132,134},
  {0,0,0,1,1,2,2,3,128,128,128,129,129,130,130,131},
  {0,0,0,0,0,1,1,2,128,128,128,128,128,129,129,130},
  {0,0,0,0,0,0,0,1,128,128,128,128,128,128,128,129},
};

__device__ __forceinline__ unsigned char tessera_fold_code_integer(
    int code, int delta, unsigned char block_scale) {
  if (block_scale == 0) return 0;
  const int sign = (code & 8) << 4;
  if (delta >= 13) return (unsigned char)sign;
  const int magnitude = code & 7;
  const unsigned long long base_codes = 0x4C4844403C383000ULL;
  const int base = (int)((base_codes >> (magnitude * 8)) & 255);
  const int normal = base - delta * 8;
  if (normal >= 8) return (unsigned char)(normal | sign);
  const unsigned int units = (0xC8643210U >> (magnitude * 4)) & 15;
  unsigned int subnormal;
  if (delta <= 8) {
    subnormal = units << (8 - delta);
  } else {
    const int shift = delta - 8;
    const unsigned int quotient = units >> shift;
    const unsigned int remainder = units & ((1U << shift) - 1);
    const unsigned int half = 1U << (shift - 1);
    subnormal = quotient + (remainder > half ||
                            (remainder == half && (quotient & 1)));
  }
  return (unsigned char)(subnormal | sign);
}
'''

_PERMUTE_HELPERS = r'''
// Four E2M1 magnitudes per v_perm_b32; sign remains the nibble's high bit.
// Keep all subnormal and signed-zero encodings identical to the scalar oracle.
__constant__ unsigned int tessera_fold_magnitudes[13][2] = {
  {0x3c383000u,0x4c484440u}, {0x34302800u,0x44403c38u},
  {0x2c282000u,0x3c383430u}, {0x24201800u,0x34302c28u},
  {0x1c181000u,0x2c282420u}, {0x14100800u,0x24201c18u},
  {0x0c080400u,0x1c181410u}, {0x06040200u,0x14100c08u},
  {0x03020100u,0x0c080604u}, {0x02010000u,0x06040302u},
  {0x01000000u,0x03020201u}, {0x00000000u,0x02010100u},
  {0x00000000u,0x01000000u},
};
__device__ __forceinline__ unsigned long long tessera_fold_word_permute(
    unsigned int word, int delta, unsigned char block_scale) {
  if (block_scale == 0) return 0;
  const unsigned int t0 = delta < 13 ? tessera_fold_magnitudes[delta][0] : 0;
  const unsigned int t1 = delta < 13 ? tessera_fold_magnitudes[delta][1] : 0;
  const unsigned int ev = word & 0x0f0f0f0fu;
  const unsigned int od = (word >> 4) & 0x0f0f0f0fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                        | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                        | ((od & 0x08080808u) << 4);
  const unsigned int low = __builtin_amdgcn_perm(bo, be, 0x05010400u);
  const unsigned int high = __builtin_amdgcn_perm(bo, be, 0x07030602u);
  return (unsigned long long)low | ((unsigned long long)high << 32);
}
'''

_EXPANDED_B_STAGE = r'''    {
      const long row = n0 + tid / 4;
      const int off = (tid & 3) * 16;
      copy_u32x4 value = {};
      if constexpr (__FULL_K64__) {
        const long safe = row < N ? row : N - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);
      } else if (kb + off < K) {
        const long safe = row < N ? row : N - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);
      }
      *reinterpret_cast<copy_u32x4 *>(sB + (tid / 4) * 80 + off) = value;
    }'''

_EXPANDED_A_STAGE = r'''#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = tid + q * 256;
      const long row = m0 + slot / 4;
      const int off = (slot & 3) * 16;
      copy_u32x4 value = {};
      if constexpr (__FULL_K64__) {
        const long safe = row < M ? row : M - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);
      } else if (kb + off < K) {
        const long safe = row < M ? row : M - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);
      }
      *reinterpret_cast<copy_u32x4 *>(sA + (slot / 4) * 80 + off) = value;
    }'''

_BATCHED_A_STAGE = r'''    // Issue the four independent A vectors before any LDS store. The
    // ragged-M clamp is uniform in both producer passes.
    copy_u32x4 staged_a[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = tid + q * 256;
      const long row = m0 + slot / 4;
      const int off = (slot & 3) * 16;
      const long safe = row < M ? row : M - 1;
      staged_a[q] = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = tid + q * 256;
      const int off = (slot & 3) * 16;
      *reinterpret_cast<copy_u32x4 *>(sA + (slot / 4) * 80 + off) = staged_a[q];
    }'''

_PACKED_B_STAGE = r'''    {
      // Each wave reads two contiguous 16x16 fragment tiles. Eight waves
      // cover four N tiles and four K16 steps in one K64 LDS stage.
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int tile = wave * 2 + q;
        const int local_n_tile = tile >> 2;
        const int local_k_step = tile & 3;
        const long global_n_tile = (n0 >> 4) + local_n_tile;
        const long safe_n_tile = global_n_tile < N / 16 ? global_n_tile : N / 16 - 1;
        const long safe_n = safe_n_tile * 16 + col;
        const long k_step = kb / 16 + local_k_step;
        const long word_slot = (safe_n_tile * (K / 16) + k_step) * 32 + lane;
        const unsigned int word = reinterpret_cast<const unsigned int *>(B)[word_slot];
        const unsigned char block_scale = Ref[(kb / 32 + local_k_step / 2) * N + safe_n];
        const unsigned char row_ref = Ref[(K / 32) * N + safe_n];
        const int delta = (int)row_ref - (int)block_scale;
        unsigned long long decoded = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int code = (word >> (e * 4)) & 15;
          const unsigned char value = __DECODE_EXPRESSION__;
          decoded |= (unsigned long long)value << (e * 8);
        }
        const int row = local_n_tile * 16 + col;
        const int off = local_k_step * 16 + (lane >> 4) * 8;
        *reinterpret_cast<unsigned long long *>(sB + row * 80 + off) = decoded;
      }
    }'''

_PACKED_B_STAGE_BATCHED = r'''    {
      // Issue both independent fragment-word and scale loads before either
      // decode. The q=0 arithmetic must not force a loadcnt drain before q=1.
      unsigned int words[2];
      unsigned char block_scales[2], row_refs[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int tile = wave * 2 + q;
        const int local_n_tile = tile >> 2;
        const int local_k_step = tile & 3;
        const long global_n_tile = (n0 >> 4) + local_n_tile;
        const long safe_n_tile = global_n_tile < N / 16 ? global_n_tile : N / 16 - 1;
        const long safe_n = safe_n_tile * 16 + col;
        const long k_step = kb / 16 + local_k_step;
        const long word_slot = (safe_n_tile * (K / 16) + k_step) * 32 + lane;
        words[q] = reinterpret_cast<const unsigned int *>(B)[word_slot];
        block_scales[q] = Ref[(kb / 32 + local_k_step / 2) * N + safe_n];
        row_refs[q] = Ref[(K / 32) * N + safe_n];
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int tile = wave * 2 + q;
        const int local_n_tile = tile >> 2;
        const int local_k_step = tile & 3;
        const int delta = (int)row_refs[q] - (int)block_scales[q];
        const unsigned char block_scale = block_scales[q];
        const unsigned int word = words[q];
        unsigned long long decoded = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int code = (word >> (e * 4)) & 15;
          const unsigned char value = __DECODE_EXPRESSION__;
          decoded |= (unsigned long long)value << (e * 8);
        }
        const int row = local_n_tile * 16 + col;
        const int off = local_k_step * 16 + (lane >> 4) * 8;
        *reinterpret_cast<unsigned long long *>(sB + row * 80 + off) = decoded;
      }
    }'''

_PACKED_B_STAGE_SHARED_SCALE = r'''    {
      // A wave owns two adjacent K16 words from one N16 tile. They share
      // one K32 scale code and one row reference; load both words first,
      // then decode against the same verified exponent delta.
      const int local_n_tile = wave >> 1;
      const int first_k_step = (wave & 1) * 2;
      const long global_n_tile = (n0 >> 4) + local_n_tile;
      const long safe_n_tile = global_n_tile < N / 16 ? global_n_tile : N / 16 - 1;
      const long safe_n = safe_n_tile * 16 + col;
      const unsigned char block_scale = Ref[(kb / 32 + first_k_step / 2) * N + safe_n];
      const unsigned char row_ref = Ref[(K / 32) * N + safe_n];
      const int delta = (int)row_ref - (int)block_scale;
      unsigned int words[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const long k_step = kb / 16 + first_k_step + q;
        const long word_slot = (safe_n_tile * (K / 16) + k_step) * 32 + lane;
        words[q] = reinterpret_cast<const unsigned int *>(B)[word_slot];
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const unsigned int word = words[q];
        unsigned long long decoded = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int code = (word >> (e * 4)) & 15;
          const unsigned char value = __DECODE_EXPRESSION__;
          decoded |= (unsigned long long)value << (e * 8);
        }
        const int row = local_n_tile * 16 + col;
        const int off = (first_k_step + q) * 16 + (lane >> 4) * 8;
        *reinterpret_cast<unsigned long long *>(sB + row * 80 + off) = decoded;
      }
    }'''

_PACKED_B_STAGE_VECTOR_PAIR = r'''    {
      // The physical fragment contract puts adjacent rows in adjacent lane
      // words. Assign each thread two adjacent slots so one 64-bit request
      // fetches both words, and read their E8M0 codes as one aligned pair.
      const int first_slot = tid * 2;
      const int local_lane = first_slot & 31;
      const int tile = first_slot >> 5;
      const int local_n_tile = tile >> 2;
      const int local_k_step = tile & 3;
      const long global_n_tile = (n0 >> 4) + local_n_tile;
      const long safe_n_tile = global_n_tile < N / 16 ? global_n_tile : N / 16 - 1;
      const long safe_n = safe_n_tile * 16 + (local_lane & 15);
      const long k_step = kb / 16 + local_k_step;
      const long word_slot = (safe_n_tile * (K / 16) + k_step) * 32 + local_lane;
      const unsigned long long packed =
          *reinterpret_cast<const unsigned long long *>(
              reinterpret_cast<const unsigned int *>(B) + word_slot);
      const unsigned short block_scales =
          *reinterpret_cast<const unsigned short *>(
              Ref + (kb / 32 + local_k_step / 2) * N + safe_n);
      const unsigned short row_refs =
          *reinterpret_cast<const unsigned short *>(Ref + (K / 32) * N + safe_n);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const unsigned int word = (unsigned int)(packed >> (q * 32));
        const unsigned char block_scale = (unsigned char)(block_scales >> (q * 8));
        const unsigned char row_ref = (unsigned char)(row_refs >> (q * 8));
        const int delta = (int)row_ref - (int)block_scale;
        const unsigned long long decoded =
            tessera_fold_word_permute(word, delta, block_scale);
        const int row = local_n_tile * 16 + (local_lane & 15) + q;
        const int off = local_k_step * 16 + (local_lane >> 4) * 8;
        *reinterpret_cast<unsigned long long *>(sB + row * 80 + off) = decoded;
      }
    }'''


@dataclass(frozen=True)
class PackedFoldedPayload:
    """Load-time packed B plus K32 E8M0 scales and one row reference.

    `weight_bytes` is fragment-order `[N,K/2]` with low nibble = even K.
    `scale_plane` is `[K/32+1,N]` in row-major uint8. Its last row is the
    E8M0 row reference used by the approximate folded epilogue; earlier rows
    are the original per-K32 exponents, including reserved zero-block codes.
    """

    weight_bytes: np.ndarray
    scale_plane: np.ndarray
    lossless: bool
    inexact_value_count: int
    max_normalized_abs_error: float
    max_normalized_relative_error: float
    approximate_policy: str = "explicit_allow"
    weight_layout: str = PACKED_FOLDED_WEIGHT_LAYOUT_V1
    scale_layout: str = PACKED_FOLDED_SCALE_PLANE_V1

    def __post_init__(self) -> None:
        # Direct public construction has no trusted fold result. Recompute its
        # numerical claims before a receipt can be issued.
        self._validate_and_freeze()

    @classmethod
    def _from_precomputed(
        cls, weight: np.ndarray, scales: np.ndarray, folded: FoldedRowReference,
    ) -> PackedFoldedPayload:
        """Factory-only path: reuse the fold calculated from the same inputs."""
        payload = object.__new__(cls)
        for name, value in (
            ("weight_bytes", weight), ("scale_plane", scales),
            ("lossless", folded.lossless),
            ("inexact_value_count", folded.inexact_value_count),
            ("max_normalized_abs_error", folded.max_normalized_abs_error),
            ("max_normalized_relative_error", folded.max_normalized_relative_error),
            ("approximate_policy", "explicit_allow"),
            ("weight_layout", PACKED_FOLDED_WEIGHT_LAYOUT_V1),
            ("scale_layout", PACKED_FOLDED_SCALE_PLANE_V1),
        ):
            object.__setattr__(payload, name, value)
        payload._validate_and_freeze(derived=folded)
        return payload

    def _validate_and_freeze(
        self, *, derived: FoldedRowReference | None = None,
    ) -> None:
        weight = self.weight_bytes
        scales = self.scale_plane
        if self.approximate_policy != "explicit_allow":
            raise ValueError("packed folded payload requires explicit approximate policy")
        if self.weight_layout != PACKED_FOLDED_WEIGHT_LAYOUT_V1:
            raise ValueError("packed folded payload requires gfx12 fragment-order weights")
        if self.scale_layout != PACKED_FOLDED_SCALE_PLANE_V1:
            raise ValueError("packed folded payload requires the versioned scale plane")
        if weight.dtype != np.uint8 or weight.ndim != 2 or not weight.flags.c_contiguous:
            raise TypeError("packed folded weights must be contiguous uint8 [N,K/2]")
        n, half_k = weight.shape
        k = half_k * 2
        if n <= 0 or n % 16 or k <= 0 or k % 64:
            raise ValueError("packed folded weights require N%16=0 and K%64=0")
        if (scales.dtype != np.uint8 or scales.shape != (k // 32 + 1, n)
                or not scales.flags.c_contiguous):
            raise TypeError("packed folded scale plane must be contiguous uint8 [K/32+1,N]")
        if np.any(scales[:-1] == 255) or np.any(scales[-1] == 255):
            raise ValueError("packed folded E8M0 code 255 is reserved")
        if not np.array_equal(scales[-1], scales[:-1].max(axis=0)):
            raise ValueError("packed folded row reference must be the block-exponent maximum")
        if derived is None:
            # The public constructor cannot trust caller-supplied loss claims.
            checkpoint = convert_weight_layout(
                weight, source=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
                destination=MXFP4_CHECKPOINT_LAYOUT_V1,
            )
            derived = prepare_folded_weights(
                checkpoint, scales[:-1], allow_approximate=True,
            )
        if not np.array_equal(derived.row_reference, scales[-1]):
            raise ValueError("packed folded precomputed row reference disagrees with plane")
        if (self.lossless != derived.lossless or
                self.inexact_value_count != derived.inexact_value_count or
                self.max_normalized_abs_error != derived.max_normalized_abs_error or
                self.max_normalized_relative_error != derived.max_normalized_relative_error):
            raise ValueError("packed folded loss metadata disagrees with payload bytes")
        # Package/model loading owns these buffers. Do not retain mutable
        # checkpoint views whose bytes can change after a receipt is issued.
        stable_weight = np.array(weight, copy=True, order="C")
        stable_scales = np.array(scales, copy=True, order="C")
        stable_weight.setflags(write=False)
        stable_scales.setflags(write=False)
        object.__setattr__(self, "weight_bytes", stable_weight)
        object.__setattr__(self, "scale_plane", stable_scales)

    @property
    def shape(self) -> tuple[int, int]:
        return self.weight_bytes.shape[0], self.weight_bytes.shape[1] * 2

    def receipt(self) -> dict[str, object]:
        """Bind the candidate layout, scale plane, numerical loss and bytes."""
        return {
            "weight_layout": self.weight_layout,
            "scale_layout": self.scale_layout,
            "execution_mode": "folded_row_reference_explicit_approximate",
            "approximate_policy": self.approximate_policy,
            "shape": self.shape,
            "weight_sha256": hashlib.sha256(self.weight_bytes.tobytes()).hexdigest(),
            "scale_plane_sha256": hashlib.sha256(self.scale_plane.tobytes()).hexdigest(),
            "fold_lossless": self.lossless,
            "fold_inexact_value_count": self.inexact_value_count,
            "fold_max_normalized_abs_error": self.max_normalized_abs_error,
            "fold_max_normalized_relative_error": self.max_normalized_relative_error,
            "execution_state": "artifact_only",
        }


def prepare_packed_folded_payload(
    packed_checkpoint: np.ndarray, scale_exponents: np.ndarray, *,
    allow_approximate: bool = False,
) -> PackedFoldedPayload:
    """Convert once at model load while preserving the exact K32 scale plane."""
    # Snapshot mutable caller arrays so both fold metadata and fragment bytes
    # describe the same checkpoint, even if the caller reuses its buffers.
    checkpoint = np.array(packed_checkpoint, copy=True, order="C")
    exponent_codes = np.array(scale_exponents, copy=True, order="C")
    folded = prepare_folded_weights(
        checkpoint, exponent_codes,
        allow_approximate=allow_approximate,
    )
    fragment = convert_weight_layout(
        checkpoint,
        source=MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    )
    # prepare_folded_weights range-checks all integer inputs. Use its
    # canonical uint8 representation for the physical plane as well.
    canonical_scales = np.ascontiguousarray(np.asarray(exponent_codes, dtype=np.uint8))
    plane = np.ascontiguousarray(
        np.concatenate((canonical_scales, folded.row_reference[None, :]), axis=0)
    )
    return PackedFoldedPayload._from_precomputed(fragment, plane, folded)


def folded_oracle_from_packed(payload: PackedFoldedPayload) -> FoldedRowReference:
    """Reconstruct the declared approximate oracle from the packed payload."""
    checkpoint = convert_weight_layout(
        payload.weight_bytes,
        source=MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
        destination=MXFP4_CHECKPOINT_LAYOUT_V1,
    )
    result = prepare_folded_weights(
        checkpoint, payload.scale_plane[:-1], allow_approximate=True,
    )
    if not np.array_equal(result.row_reference, payload.scale_plane[-1]):
        raise ValueError("packed folded reference row disagrees with the oracle")
    return result


def emit_mxfp4_packed_folded_prefill_hip(
    entry: str = "tessera_mxfp4_packed_folded_prefill",
    *, integer_decode: bool = False, batched_loads: bool = False,
    batched_a_loads: bool = False, reuse_pair_scales: bool = False,
    permute_decode: bool = False, vector_pair_loads: bool = False,
) -> str:
    """Emit the K64 packed-fragment LDS decode and folded WMMA schedule."""
    source = emit_mxfp4_folded_prefill_hip(entry, full_k64=True)
    if reuse_pair_scales and not batched_loads:
        raise ValueError("paired K16 scale reuse requires batched B loads")
    if permute_decode and integer_decode:
        raise ValueError("select one packed-word decode implementation")
    if vector_pair_loads and not permute_decode:
        raise ValueError("paired lane loads require permute decode")
    if vector_pair_loads and (batched_loads or batched_a_loads or reuse_pair_scales):
        raise ValueError("paired lane loads are a separate staging ablation")
    def replace_once(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("expanded folded HIP template changed; review packed decode")
        source = source.replace(old, new)

    replace_once(
        "using copy_u32x4 = unsigned int __attribute__((ext_vector_type(4)));",
        "using copy_u32x4 = unsigned int __attribute__((ext_vector_type(4)));"
        + _PACKED_HELPERS + (_PERMUTE_HELPERS if permute_decode else ""),
    )
    if batched_a_loads:
        replace_once(
            _EXPANDED_A_STAGE.replace("__FULL_K64__", "true"),
            _BATCHED_A_STAGE,
        )
    expression = (
        "tessera_fold_code_integer(code, delta, block_scale)"
        if integer_decode else
        "(block_scale == 0 ? 0 : delta >= 13 ? "
        "(unsigned char)((code & 8) << 4) : "
        "tessera_fold_e2m1_e4m3[delta][code])"
    )
    stage = (
        _PACKED_B_STAGE_VECTOR_PAIR if vector_pair_loads else
        _PACKED_B_STAGE_SHARED_SCALE if reuse_pair_scales else
        _PACKED_B_STAGE_BATCHED if batched_loads else _PACKED_B_STAGE
    )
    replace_once(
        _EXPANDED_B_STAGE.replace("__FULL_K64__", "true"),
        stage.replace("__DECODE_EXPRESSION__", expression),
    )
    if permute_decode and not vector_pair_loads:
        replace_once(
            """        unsigned long long decoded = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int code = (word >> (e * 4)) & 15;
          const unsigned char value = """ + expression + """;
          decoded |= (unsigned long long)value << (e * 8);
        }""",
            """        const unsigned long long decoded =
            tessera_fold_word_permute(word, delta, block_scale);""",
        )
    replace_once(
        "const unsigned char exponent = Ref[n < N ? n : 0];",
        "const unsigned char exponent = Ref[(K / 32) * N + (n < N ? n : 0)];",
    )
    return source


def package_mxfp4_packed_folded_prefill(
    m: int, payload: PackedFoldedPayload, *,
    entry: str = "tessera_mxfp4_packed_folded_prefill",
    integer_decode: bool = False,
    batched_loads: bool = False,
    batched_a_loads: bool = False,
    reuse_pair_scales: bool = False,
    permute_decode: bool = False, vector_pair_loads: bool = False,
) -> ROCMNativePackage:
    """Compile the distinct packed ABI, without admitting it to selection."""
    n, k = payload.shape
    if m <= 64:
        raise ValueError("packed folded prefill requires M>64")
    source = emit_mxfp4_packed_folded_prefill_hip(
        entry, integer_decode=integer_decode, batched_loads=batched_loads,
        batched_a_loads=batched_a_loads, reuse_pair_scales=reuse_pair_scales,
        permute_decode=permute_decode, vector_pair_loads=vector_pair_loads,
    )
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("packed folded MXFP4 requires the HIP compiler driver")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-packed-folded-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler), "-x", "hip", "-O3", "--genco",
            "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
            str(source_path), "-o", str(bundle_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle_path.is_file():
            raise RuntimeError(
                "packed folded HSACO compilation failed: "
                + (result.stderr.strip() or f"hipcc exited {result.returncode}")
            )
        image_bytes = _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=hashlib.sha256(
            (str(rocm_path) + "|gfx1201|packed_folded_bm256_tm4_v1").encode()
        ).hexdigest(),
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco", payload=image_bytes,
        entry_points=(NativeEntryPoint(entry, PACKED_FOLDED_TARGET_ABI_V1),),
        compile_state="cold",
        device_libraries=_driver_selected_device_libraries(arch="gfx1201"),
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry,
        abi_id=PACKED_FOLDED_TARGET_ABI_V1,
        buffers=(
            BufferBinding(0, "a", "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, "b_packed", "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, "a_scale", "input", "fp32", 1, "row_major", 4),
            BufferBinding(3, "scale_plane", "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, "output", "output", "bf16", 2, "row_major", 2),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"), ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard("a", 0, "eq", m), ShapeGuard("a", 1, "eq", k),
            ShapeGuard("b_packed", 0, "eq", n),
            ShapeGuard("b_packed", 1, "eq", k // 2),
            ShapeGuard("a_scale", 0, "eq", m),
            ShapeGuard("scale_plane", 0, "eq", k // 32 + 1),
            ShapeGuard("scale_plane", 1, "eq", n),
            ShapeGuard("output", 0, "eq", m),
            ShapeGuard("output", 1, "eq", n),
        ),
        geometry=LaunchGeometry(
            grid=((n + 63) // 64, (m + 255) // 256, 1),
            workgroup=(256, 1, 1),
        ),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-MXFP4-W4A8-1",
            "sync_key": (
                "GFX1201-PACKED-PERMUTE-DECODE-2026-09-23"
                if permute_decode else
                "GFX1201-PACKED-STAGING-ABLATION-2026-09-23"
                if batched_loads or batched_a_loads or reuse_pair_scales
                else "GFX1201-PACKED-FOLDED-DECODE-2026-09-23"
            ),
            "route": "packed_folded_fragment_bm256_tm4",
            "decode_strategy": "permute_word" if permute_decode else (
                "integer_nibble" if integer_decode else "table_nibble"
            ),
            "architecture": "gfx1201",
            "physical_contract": PACKED_FOLDED_PHYSICAL_V1,
            "numeric_policy": "folded_row_reference_explicit_approximate",
            "block_m": 256, "block_n": 64, "block_k": 64,
            "tile_m_per_wave": 4, "tile_n_per_wave": 2,
            "staging_policy": (
                "packed_fragment_decode_k64_vector_pair" if vector_pair_loads
                else
                "packed_fragment_decode_k64_pair_scale_reuse" if reuse_pair_scales
                else
                "packed_fragment_decode_k64_batched_ab" if batched_a_loads and batched_loads
                else "packed_fragment_decode_k64_batched_a" if batched_a_loads
                else "packed_fragment_decode_k64_batched_b" if batched_loads
                else "packed_fragment_decode_k64"
            ),
            "decode_policy": "integer_register" if integer_decode else "constant_table",
            **payload.receipt(),
            "execution_state": "manual_executable_candidate",
        },
    )
    return ROCMNativePackage(
        f"rocm.mxfp4_w4a8 packed_folded M={m} N={n} K={k}",
        source, " ".join(command[:-2]), image, descriptor,
    )


def author_packed_folded_graph(m: int, payload: PackedFoldedPayload) -> str:
    """Author the versioned packed physical Graph contract for this payload."""
    n, k = payload.shape
    if m <= 64:
        raise ValueError("packed folded prefill requires M>64")
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @packed_folded_w4a8(%a: tensor<{m}x{k}xui8>,
                                %b: tensor<{n}x{k // 2}xui8>,
                                %sa: tensor<{m}xf32>,
                                %plane: tensor<{k // 32 + 1}x{n}xui8>) -> tensor<{m}x{n}xbf16> {{
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %plane) {{
      physical_contract = "{PACKED_FOLDED_PHYSICAL_V1}",
      numeric_policy = {{accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"}},
      scale_layout = {{granularity = "output_column", block = [1, {k}], format = "e8m0_k32_plus_row_reference"}}
    }} : (tensor<{m}x{k}xui8>, tensor<{n}x{k // 2}xui8>, tensor<{m}xf32>,
         tensor<{k // 32 + 1}x{n}xui8>) -> tensor<{m}x{n}xbf16>
    return %0 : tensor<{m}x{n}xbf16>
  }}
}}
'''


def _lower_packed_folded_carrier(
    m: int, payload: PackedFoldedPayload, *, tessera_opt: Path,
) -> tuple[str, str, str, dict[str, object]]:
    """Lower authored Graph and verify its distinct Tile/Target ABI."""
    if not tessera_opt.is_file():
        raise FileNotFoundError(f"packed folded compiler not found: {tessera_opt}")
    graph_ir = author_packed_folded_graph(m, payload)

    def lower(*, target: bool) -> str:
        cmd = [str(tessera_opt), "--tessera-graph-to-schedule", "--tessera-schedule-to-tile"]
        if target:
            cmd.append("--lower-tile-to-rocm=arch=gfx1201")
        result = subprocess.run(cmd, input=graph_ir, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError("packed folded lowering failed: " + result.stderr.strip())
        return result.stdout

    tile_ir = lower(target=False)
    target_ir = lower(target=True)
    tiles = [line.strip() for line in tile_ir.splitlines() if "tile.scaled_matmul_kernel" in line]
    targets = [line.strip() for line in target_ir.splitlines()
               if "tessera_rocm.scaled_wmma_gemm" in line]
    if len(tiles) != 1 or len(targets) != 1:
        raise ValueError("packed folded lowering requires one Tile and Target carrier")
    tile, target = tiles[0], targets[0]
    for operation in (tile, target):
        if _target_string_attr(operation, "physical_contract") != PACKED_FOLDED_PHYSICAL_V1:
            raise ValueError("packed folded physical contract was lost during lowering")
    for key, expected in {
        "abi": "a_bpacked_sa_scaleplane_d_m_n_k",
        "package_abi": PACKED_FOLDED_TARGET_ABI_V1,
        "scale_format": "e8m0_k32_plus_row_reference",
    }.items():
        if _target_string_attr(target, key) != expected:
            raise ValueError(f"packed folded Target requires {key}={expected!r}")
    if _schedule_hash(tile, carrier="packed folded Tile IR") != _schedule_hash(
        target, carrier="packed folded Target IR",
    ):
        raise ValueError("packed folded Tile/Target schedule hashes disagree")
    receipt = {
        **payload.receipt(),
        "physical_contract": PACKED_FOLDED_PHYSICAL_V1,
        "target_abi": PACKED_FOLDED_TARGET_ABI_V1,
        "schedule_hash": _schedule_hash(target, carrier="packed folded Target IR"),
        "graph_ir_sha256": hashlib.sha256(graph_ir.encode()).hexdigest(),
        "tile_ir_sha256": hashlib.sha256(tile_ir.encode()).hexdigest(),
        "target_ir_sha256": hashlib.sha256(target_ir.encode()).hexdigest(),
        "hsaco_sha256": None,
    }
    return graph_ir, tile_ir, target_ir, receipt


def lower_packed_folded_artifact(
    m: int, payload: PackedFoldedPayload, *, tessera_opt: Path,
) -> dict[str, object]:
    """Issue a Graph→Target receipt without compiling or selecting a kernel."""
    return _lower_packed_folded_carrier(
        m, payload, tessera_opt=tessera_opt,
    )[3]


@dataclass(frozen=True)
class PackedFoldedScaledMatmulProgram:
    """Manually selected packed HSACO bound to its authored Graph carrier."""

    package: ROCMNativePackage
    graph_ir: str

    @property
    def route_receipt(self) -> dict[str, object]:
        package = self.package
        return {
            **package.descriptor.provenance,
            "abi_id": package.descriptor.abi_id,
            "entry_symbol": package.descriptor.entry_symbol,
            "hsaco_sha256": package.image.payload_digest,
            "artifact_image_digest": package.image.image_digest,
        }


def compile_packed_folded_scaled_matmul(
    a: np.ndarray, a_scale: np.ndarray, payload: PackedFoldedPayload, *,
    tessera_opt: Path, integer_decode: bool = True,
) -> PackedFoldedScaledMatmulProgram:
    """Materialize the explicit packed Graph→Target ABI as a manual package.

    This is not an automatic selector. The caller supplies the opt-in payload
    and retains the package for exact-device launch with matching buffers.
    """
    if a.dtype != np.uint8 or a.ndim != 2 or not a.flags.c_contiguous:
        raise ValueError("packed folded A must be contiguous raw E4M3 [M,K]")
    m, k = a.shape
    if payload.shape[1] != k or a_scale.shape != (m,) or a_scale.dtype != np.float32:
        raise ValueError("packed folded A, token scales, and B K must agree")
    graph_ir, tile_ir, target_ir, receipt = _lower_packed_folded_carrier(
        m, payload, tessera_opt=tessera_opt,
    )
    package = package_mxfp4_packed_folded_prefill(
        m, payload, integer_decode=integer_decode,
    )
    image = replace(
        package.image,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
    )
    descriptor = replace(
        package.descriptor,
        image_digest=image.image_digest,
        provenance={
            **package.descriptor.provenance,
            **receipt,
            "execution_state": "manual_executable_candidate",
            "materializer": "tessera_rocm.scaled_wmma_gemm",
            "decode_policy": "integer_register" if integer_decode else "constant_table",
            "hsaco_sha256": image.payload_digest,
        },
    )
    bound_package = ROCMNativePackage(
        tile_ir=tile_ir, target_ir=target_ir, backend_ir=package.target_ir,
        image=image, descriptor=descriptor,
    )
    return PackedFoldedScaledMatmulProgram(bound_package, graph_ir)


__all__ = [
    "PACKED_FOLDED_SCALE_PLANE_V1", "PACKED_FOLDED_WEIGHT_LAYOUT_V1",
    "PACKED_FOLDED_PHYSICAL_V1", "PACKED_FOLDED_TARGET_ABI_V1",
    "PackedFoldedPayload", "prepare_packed_folded_payload",
    "folded_oracle_from_packed", "author_packed_folded_graph",
    "lower_packed_folded_artifact", "emit_mxfp4_packed_folded_prefill_hip",
    "package_mxfp4_packed_folded_prefill", "PackedFoldedScaledMatmulProgram",
    "compile_packed_folded_scaled_matmul",
]
