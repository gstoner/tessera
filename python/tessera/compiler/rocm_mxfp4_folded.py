"""Opt-in folded MXFP4 prefill for the exact gfx1201 device.

The checkpoint conversion is a load-time operation.  The launch ABI consumes
its E4M3 bytes and one E8M0 reference exponent per output column; it cannot
be confused with the exact packed-E2M1/K32 ABI.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
)
from .rocm_mxfp4 import (
    FoldedRowReference, MXFP4_FOLDED_ROW_LAYOUT_V1,
    fold_to_row_reference,
)
from .rocm_mxfp4_native import _rocm_hipcc, _extract_gfx1201_hsaco
from .rocm_native import (
    ROCMNativePackage, _driver_selected_device_libraries, _rocm_path,
    _version_fingerprint,
)


GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k."
    "e4m3_e4m3_e8m0_bf16.approx_bm256_tm4.v1"
)
FOLDED_WEIGHT_LAYOUT = MXFP4_FOLDED_ROW_LAYOUT_V1


def prepare_folded_weights(
    packed_checkpoint: np.ndarray, scale_exponents: np.ndarray, *,
    allow_approximate: bool = False,
) -> FoldedRowReference:
    """Convert canonical [N,K/2] weights once, with explicit policy consent."""
    from .rocm_mxfp4 import unpack_e2m1_codes

    return fold_to_row_reference(
        unpack_e2m1_codes(packed_checkpoint), scale_exponents,
        allow_approximate=allow_approximate,
    )


_FOLDED_PREFILL_HIP = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
using floatx8 = float __attribute__((ext_vector_type(8)));
using fragment_i32x2 = int __attribute__((ext_vector_type(2)));
using copy_u32x4 = unsigned int __attribute__((ext_vector_type(4)));

// Four row fragments and two column fragments per wave: 4 x 2 waves form
// a 256 x 64 output block.  Both operands are padded in LDS by 16 bytes.
extern "C" __global__ __launch_bounds__(256) void __ENTRY__(
    const unsigned char *__restrict__ A,
    const unsigned char *__restrict__ B,
    const float *__restrict__ As,
    const unsigned char *__restrict__ Ref,
    __bf16 *__restrict__ O, long M, long N, long K
#ifdef TESSERA_FOLDED_PHASE_TRACE
    , unsigned long long *__restrict__ Trace
#endif
    ) {
  __shared__ alignas(16) unsigned char sA[256 * 80];
  __shared__ alignas(16) unsigned char sB[64 * 80];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int wm = wave >> 1;
  const int wn = wave & 1;
  const int col = lane & 15;
  const int half = (lane >> 4) * 8;
  const long m0 = (long)blockIdx.y * 256;
  const long n0 = (long)blockIdx.x * 64;
#ifdef TESSERA_FOLDED_PHASE_TRACE
  unsigned long long copy_ticks = 0, compute_ticks = 0;
  unsigned long long first_tick = 0, last_tick = 0;
#endif
  floatx8 acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      acc[i][j] = {};

  for (long kb = 0; kb < K; kb += 64) {
#ifdef TESSERA_FOLDED_PHASE_TRACE
    unsigned long long copy_start = 0;
    if (tid == 0) {
      copy_start = (unsigned long long)wall_clock64();
      if (kb == 0) first_tick = copy_start;
    }
#endif
    // Uniform, clamped vector copies keep every wave on the barrier path.
    // Four A vectors and one B vector per thread cover the two LDS tiles.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int slot = tid + q * 256;
      const long row = m0 + slot / 4;
      const int off = (slot & 3) * 16;
      copy_u32x4 value = {};
      if (kb + off < K) {
        const long safe = row < M ? row : M - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);
      }
      *reinterpret_cast<copy_u32x4 *>(sA + (slot / 4) * 80 + off) = value;
    }
    {
      const long row = n0 + tid / 4;
      const int off = (tid & 3) * 16;
      copy_u32x4 value = {};
      if (kb + off < K) {
        const long safe = row < N ? row : N - 1;
        value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);
      }
      *reinterpret_cast<copy_u32x4 *>(sB + (tid / 4) * 80 + off) = value;
    }
    __syncthreads();
#ifdef TESSERA_FOLDED_PHASE_TRACE
    unsigned long long compute_start = 0;
    if (tid == 0) {
      compute_start = (unsigned long long)wall_clock64();
      copy_ticks += compute_start - copy_start;
    }
#endif
    for (int step = 0; step < 4 && kb + step * 16 < K; ++step) {
      fragment_i32x2 af[4], bf[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned char *p = sA + (wm * 64 + i * 16 + col) * 80 + step * 16 + half;
        af[i][0] = *reinterpret_cast<const int *>(p);
        af[i][1] = *reinterpret_cast<const int *>(p + 4);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const unsigned char *p = sB + (wn * 32 + j * 16 + col) * 80 + step * 16 + half;
        bf[j][0] = *reinterpret_cast<const int *>(p);
        bf[j][1] = *reinterpret_cast<const int *>(p + 4);
      }
      __builtin_amdgcn_sched_barrier(6);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
              af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
#ifdef TESSERA_FOLDED_PHASE_TRACE
    if (tid == 0) {
      last_tick = (unsigned long long)wall_clock64();
      compute_ticks += last_tick - compute_start;
    }
#endif
  }
#ifdef TESSERA_FOLDED_PHASE_TRACE
  if (tid == 0) {
    const long slot = ((long)blockIdx.y * gridDim.x + blockIdx.x) * 4;
    Trace[slot] = first_tick;
    Trace[slot + 1] = last_tick;
    Trace[slot + 2] = copy_ticks;
    Trace[slot + 3] = compute_ticks;
  }
#endif

#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const long n = n0 + wn * 32 + j * 16 + col;
    const unsigned char exponent = Ref[n < N ? n : 0];
    const float row_scale = exponent ? __builtin_bit_cast(float, (unsigned int)exponent << 23) : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const long m = m0 + wm * 64 + i * 16 + half + e;
        if (m < M && n < N) {
          const float combined_scale = row_scale * As[m];
          O[m * N + n] = (__bf16)(acc[i][j][e] * combined_scale);
        }
      }
  }
}
'''


def emit_mxfp4_folded_prefill_hip(
    entry: str = "tessera_mxfp4_folded_prefill",
) -> str:
    if not entry.isidentifier():
        raise ValueError("folded MXFP4 entry must be a C identifier")
    return _FOLDED_PREFILL_HIP.replace("__ENTRY__", entry)


def package_mxfp4_folded_prefill(
    m: int, n: int, k: int, folded: FoldedRowReference, *,
    allow_approximate: bool = False,
    entry: str = "tessera_mxfp4_folded_prefill",
) -> ROCMNativePackage:
    """Package the opt-in BM256/TM4 route with payload-bound error metadata."""
    if not allow_approximate or folded.approximate_policy != "explicit_allow":
        raise ValueError("folded MXFP4 requires explicit approximate policy")
    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("folded MXFP4 requires positive M/N and K divisible by 32")
    if m <= 64:
        raise ValueError("folded BM256/TM4 route requires prefill M > 64")
    if folded.weight_bytes.shape != (n, k) or folded.row_reference.shape != (n,):
        raise ValueError("folded payload disagrees with M/N/K")
    if folded.weight_bytes.dtype != np.uint8 or folded.row_reference.dtype != np.uint8:
        raise TypeError("folded payload requires raw E4M3 and E8M0 uint8 arrays")
    if not folded.weight_bytes.flags.c_contiguous or not folded.row_reference.flags.c_contiguous:
        raise ValueError("folded payload must be contiguous at package load")
    if np.any(folded.row_reference == 255):
        raise ValueError("folded E8M0 row reference code 255 is reserved")
    source = emit_mxfp4_folded_prefill_hip(entry)
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("folded MXFP4 requires the HIP compiler driver")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-folded-") as directory:
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
                "folded MXFP4 HSACO compilation failed: "
                + (result.stderr.strip() or f"hipcc exited {result.returncode}")
            )
        payload = _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=hashlib.sha256(
            (str(rocm_path) + "|gfx1201|folded_bm256_tm4_v1").encode()
        ).hexdigest(),
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco", payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI),),
        compile_state="cold",
        device_libraries=_driver_selected_device_libraries(arch="gfx1201"),
    )
    provenance = {
        "work_item": "ROCM-MXFP4-W4A8-1",
        "sync_key": "ROCM-MXFP4-FOLDED-PREFILL-2026-09-22",
        "route": "folded_row_reference_bm256_tm4",
        "architecture": "gfx1201",
        "weight_layout": FOLDED_WEIGHT_LAYOUT,
        "numeric_policy": "folded_row_reference_explicit_approximate",
        "weight_sha256": hashlib.sha256(folded.weight_bytes.tobytes()).hexdigest(),
        "row_reference_sha256": hashlib.sha256(folded.row_reference.tobytes()).hexdigest(),
        "fold_lossless": folded.lossless,
        "fold_inexact_value_count": folded.inexact_value_count,
        "fold_max_normalized_abs_error": folded.max_normalized_abs_error,
        "fold_max_normalized_relative_error": folded.max_normalized_relative_error,
        "block_m": 256, "block_n": 64, "block_k": 64,
        "tile_m_per_wave": 4, "tile_n_per_wave": 2,
        "accum": "fp32", "output": "bf16",
    }
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry,
        abi_id=GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI,
        buffers=(
            BufferBinding(0, "a", "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, "b_folded", "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, "a_scale", "input", "fp32", 1, "row_major", 4),
            BufferBinding(3, "row_reference", "input", "uint8", 1, "row_major", 1),
            BufferBinding(4, "output", "output", "bf16", 2, "row_major", 2),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"), ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard("a", 0, "eq", m), ShapeGuard("a", 1, "eq", k),
            ShapeGuard("b_folded", 0, "eq", n),
            ShapeGuard("b_folded", 1, "eq", k),
            ShapeGuard("a_scale", 0, "eq", m),
            ShapeGuard("row_reference", 0, "eq", n),
            ShapeGuard("output", 0, "eq", m), ShapeGuard("output", 1, "eq", n),
        ),
        geometry=LaunchGeometry(
            grid=((n + 63) // 64, (m + 255) // 256, 1),
            workgroup=(256, 1, 1),
        ),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="none", synchronization=("completion",),
        ),
        provenance=provenance,
    )
    return ROCMNativePackage(
        f"rocm.mxfp4_w4a8 folded_row_reference M={m} N={n} K={k}",
        source, " ".join(command[:-2]), image, descriptor,
    )


__all__ = [
    "FOLDED_WEIGHT_LAYOUT", "GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI",
    "emit_mxfp4_folded_prefill_hip", "package_mxfp4_folded_prefill",
    "prepare_folded_weights",
]
