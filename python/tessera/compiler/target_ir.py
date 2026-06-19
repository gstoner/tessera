"""Target IR object model and Tile IR lowering.

Target IR is the backend-specific contract layer below Tile IR. This module
keeps hardware-free CPU/x86, NVIDIA/CUDA, Apple Silicon, and ROCm artifacts
object-backed and verifiable while preserving the textual MLIR inspection
surface used by the Python compiler tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..diagnostics import DiagnosticLevel, DiagnosticWhere, TesseraDiagnostic, TesseraErrorCode
from .apple_target_descriptor import apple_target_descriptor as _apple_target_descriptor
from .capabilities import normalize_target
from .tile_ir import TILE_METADATA_OPS, TileIRModule, TileIRVerificationError, TileOp


APPLE_CPU_TARGET = "apple_cpu"
APPLE_GPU_TARGET = "apple_gpu"
CPU_TARGET = "cpu"
ROCM_TARGET = "rocm"
NVIDIA_TARGETS = {"nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120"}
SUPPORTED_TARGETS = {APPLE_CPU_TARGET, APPLE_GPU_TARGET, CPU_TARGET, ROCM_TARGET, *NVIDIA_TARGETS}


# Phase 8.4 — embedded MSL source for the rope custom kernel. Carried as a
# StringAttr on `tessera_apple.gpu.msl_kernel` so the Target IR module is a
# self-contained, replayable artifact. The runtime compiles via
# `[device newLibraryWithSource:options:error:]` and caches by the
# concatenation of (msl_source, entry_point) — this string is therefore the
# canonical cache identity at the IR layer, mirrored by `cache_key` below.
_APPLE_GPU_ROPE_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rope_f32(\n"
    "    device const float* x      [[buffer(0)]],\n"
    "    device const float* theta  [[buffer(1)]],\n"
    "    device float*       out    [[buffer(2)]],\n"
    "    constant int&       M      [[buffer(3)]],\n"
    "    constant int&       K      [[buffer(4)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.x >= (uint)(K / 2) || gid.y >= (uint)M) return;\n"
    "    int row = (int)gid.y;\n"
    "    int pair = (int)gid.x;\n"
    "    int idx_even = row * K + pair * 2;\n"
    "    int idx_odd  = idx_even + 1;\n"
    "    float xe = x[idx_even];\n"
    "    float xo = x[idx_odd];\n"
    "    float c = cos(theta[idx_even]);\n"
    "    float s = sin(theta[idx_even]);\n"
    "    out[idx_even] = xe * c - xo * s;\n"
    "    out[idx_odd]  = xe * s + xo * c;\n"
    "}\n"
)


def _sha256_short(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_APPLE_GPU_ROPE_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_ROPE_MSL_SOURCE)


# Phase 8.4.1 — embedded MSL source for the flash-attention forward kernel.
# Same online-softmax algorithm as the runtime's apple_gpu_runtime.mm shim;
# carrying it inline here keeps the Target IR module a self-contained,
# replayable record of what the runtime will actually compile and execute.
_APPLE_GPU_FLASH_ATTN_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void flash_attn_f32(\n"
    "    device const float* Q       [[buffer(0)]],\n"
    "    device const float* K       [[buffer(1)]],\n"
    "    device const float* V       [[buffer(2)]],\n"
    "    device float*       O       [[buffer(3)]],\n"
    "    constant int&       B       [[buffer(4)]],\n"
    "    constant int&       Sq      [[buffer(5)]],\n"
    "    constant int&       Sk      [[buffer(6)]],\n"
    "    constant int&       D       [[buffer(7)]],\n"
    "    constant float&     scale   [[buffer(8)]],\n"
    "    constant int&       causal  [[buffer(9)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;\n"
    "    int batch = (int)gid.y;\n"
    "    int q_row = (int)gid.x;\n"
    "    if (D > 256) return;\n"
    "    int q_off = batch * Sq * D + q_row * D;\n"
    "    int kv_base = batch * Sk * D;\n"
    "    float m = -INFINITY;\n"
    "    float l = 0.0f;\n"
    "    float o[256];\n"
    "    for (int d = 0; d < D; ++d) o[d] = 0.0f;\n"
    "    for (int k_row = 0; k_row < Sk; ++k_row) {\n"
    "        if (causal != 0 && k_row > q_row) break;\n"
    "        int k_off = kv_base + k_row * D;\n"
    "        float score = 0.0f;\n"
    "        for (int d = 0; d < D; ++d) score += Q[q_off + d] * K[k_off + d];\n"
    "        score *= scale;\n"
    "        float new_m = max(m, score);\n"
    "        float exp_old = exp(m - new_m);\n"
    "        float exp_score = exp(score - new_m);\n"
    "        float new_l = l * exp_old + exp_score;\n"
    "        for (int d = 0; d < D; ++d) o[d] = o[d] * exp_old + V[k_off + d] * exp_score;\n"
    "        m = new_m;\n"
    "        l = new_l;\n"
    "    }\n"
    "    if (l == 0.0f) {\n"
    "        for (int d = 0; d < D; ++d) O[q_off + d] = 0.0f;\n"
    "    } else {\n"
    "        float inv_l = 1.0f / l;\n"
    "        for (int d = 0; d < D; ++d) O[q_off + d] = o[d] * inv_l;\n"
    "    }\n"
    "}\n"
)


_APPLE_GPU_FLASH_ATTN_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_FLASH_ATTN_MSL_SOURCE)


# Phase 8.4.4.2 — fp16 native flash-attention kernel. Mixed-precision design:
# `half` I/O, `float` per-thread accumulators (m, l, o[]) for softmax/online-
# update precision. Same algorithm as the f32 source.
_APPLE_GPU_FLASH_ATTN_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void flash_attn_f16(\n"
    "    device const half*  Q       [[buffer(0)]],\n"
    "    device const half*  K       [[buffer(1)]],\n"
    "    device const half*  V       [[buffer(2)]],\n"
    "    device half*        O       [[buffer(3)]],\n"
    "    constant int&       B       [[buffer(4)]],\n"
    "    constant int&       Sq      [[buffer(5)]],\n"
    "    constant int&       Sk      [[buffer(6)]],\n"
    "    constant int&       D       [[buffer(7)]],\n"
    "    constant float&     scale   [[buffer(8)]],\n"
    "    constant int&       causal  [[buffer(9)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;\n"
    "    int batch = (int)gid.y;\n"
    "    int q_row = (int)gid.x;\n"
    "    if (D > 256) return;\n"
    "    int q_off = batch * Sq * D + q_row * D;\n"
    "    int kv_base = batch * Sk * D;\n"
    "    float m = -INFINITY;\n"
    "    float l = 0.0f;\n"
    "    float o[256];\n"
    "    for (int d = 0; d < D; ++d) o[d] = 0.0f;\n"
    "    for (int k_row = 0; k_row < Sk; ++k_row) {\n"
    "        if (causal != 0 && k_row > q_row) break;\n"
    "        int k_off = kv_base + k_row * D;\n"
    "        float score = 0.0f;\n"
    "        for (int d = 0; d < D; ++d) score += float(Q[q_off + d]) * float(K[k_off + d]);\n"
    "        score *= scale;\n"
    "        float new_m = max(m, score);\n"
    "        float exp_old = exp(m - new_m);\n"
    "        float exp_score = exp(score - new_m);\n"
    "        float new_l = l * exp_old + exp_score;\n"
    "        for (int d = 0; d < D; ++d) o[d] = o[d] * exp_old + float(V[k_off + d]) * exp_score;\n"
    "        m = new_m;\n"
    "        l = new_l;\n"
    "    }\n"
    "    if (l == 0.0f) {\n"
    "        for (int d = 0; d < D; ++d) O[q_off + d] = half(0.0f);\n"
    "    } else {\n"
    "        float inv_l = 1.0f / l;\n"
    "        for (int d = 0; d < D; ++d) O[q_off + d] = half(o[d] * inv_l);\n"
    "    }\n"
    "}\n"
)
_APPLE_GPU_FLASH_ATTN_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_FLASH_ATTN_MSL_SOURCE_F16)


# Phase 8.4.2 — embedded MSL source for the softmax kernel. Standard 3-pass
# axis=-1 softmax: row max -> subtract+exp+sum -> divide. One thread per row.
_APPLE_GPU_SOFTMAX_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void softmax_f32(\n"
    "    device const float* x   [[buffer(0)]],\n"
    "    device float*       out [[buffer(1)]],\n"
    "    constant int&       M   [[buffer(2)]],\n"
    "    constant int&       K   [[buffer(3)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    int row = (int)gid;\n"
    "    int row_off = row * K;\n"
    "    float row_max = -INFINITY;\n"
    "    for (int j = 0; j < K; ++j) row_max = max(row_max, x[row_off + j]);\n"
    "    float denom = 0.0f;\n"
    "    for (int j = 0; j < K; ++j) {\n"
    "        float e = exp(x[row_off + j] - row_max);\n"
    "        out[row_off + j] = e;\n"
    "        denom += e;\n"
    "    }\n"
    "    float inv = 1.0f / denom;\n"
    "    for (int j = 0; j < K; ++j) out[row_off + j] *= inv;\n"
    "}\n"
)

_APPLE_GPU_SOFTMAX_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_SOFTMAX_MSL_SOURCE)


# Phase 8.4.2 — embedded MSL source for the gelu kernel. Tanh-approximation,
# matching the numpy reference. Elementwise; one thread per element.
_APPLE_GPU_GELU_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void gelu_f32(\n"
    "    device const float* x   [[buffer(0)]],\n"
    "    device float*       out [[buffer(1)]],\n"
    "    constant int&       N   [[buffer(2)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)N) return;\n"
    "    float v = x[gid];\n"
    "    float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);\n"
    "    out[gid] = 0.5f * v * (1.0f + tanh(t));\n"
    "}\n"
)

_APPLE_GPU_GELU_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_GELU_MSL_SOURCE)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.4.1 — fp16 / bf16 MSL source constants for the simple kernels.
# fp16: native MSL `half` kernels with `float` internal compute for accuracy.
# bf16: emit a `bf16` cache_key marker; the runtime shim dispatches the
#       fp32-conversion path internally (no native MSL bf16 source).
# ─────────────────────────────────────────────────────────────────────────────

_APPLE_GPU_ROPE_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rope_f16(\n"
    "    device const half*  x      [[buffer(0)]],\n"
    "    device const half*  theta  [[buffer(1)]],\n"
    "    device half*        out    [[buffer(2)]],\n"
    "    constant int&       M      [[buffer(3)]],\n"
    "    constant int&       K      [[buffer(4)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.x >= (uint)(K / 2) || gid.y >= (uint)M) return;\n"
    "    int row = (int)gid.y;\n"
    "    int pair = (int)gid.x;\n"
    "    int idx_even = row * K + pair * 2;\n"
    "    int idx_odd  = idx_even + 1;\n"
    "    float xe = float(x[idx_even]);\n"
    "    float xo = float(x[idx_odd]);\n"
    "    float c = cos(float(theta[idx_even]));\n"
    "    float s = sin(float(theta[idx_even]));\n"
    "    out[idx_even] = half(xe * c - xo * s);\n"
    "    out[idx_odd]  = half(xe * s + xo * c);\n"
    "}\n"
)
_APPLE_GPU_ROPE_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_ROPE_MSL_SOURCE_F16)

_APPLE_GPU_SOFTMAX_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void softmax_f16(\n"
    "    device const half*  x   [[buffer(0)]],\n"
    "    device half*        out [[buffer(1)]],\n"
    "    constant int&       M   [[buffer(2)]],\n"
    "    constant int&       K   [[buffer(3)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    int row = (int)gid;\n"
    "    int row_off = row * K;\n"
    "    float row_max = -INFINITY;\n"
    "    for (int j = 0; j < K; ++j) row_max = max(row_max, float(x[row_off + j]));\n"
    "    float denom = 0.0f;\n"
    "    for (int j = 0; j < K; ++j) {\n"
    "        float e = exp(float(x[row_off + j]) - row_max);\n"
    "        out[row_off + j] = half(e);\n"
    "        denom += e;\n"
    "    }\n"
    "    float inv = 1.0f / denom;\n"
    "    for (int j = 0; j < K; ++j) out[row_off + j] = half(float(out[row_off + j]) * inv);\n"
    "}\n"
)
_APPLE_GPU_SOFTMAX_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_SOFTMAX_MSL_SOURCE_F16)

_APPLE_GPU_GELU_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void gelu_f16(\n"
    "    device const half*  x   [[buffer(0)]],\n"
    "    device half*        out [[buffer(1)]],\n"
    "    constant int&       N   [[buffer(2)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)N) return;\n"
    "    float v = float(x[gid]);\n"
    "    float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);\n"
    "    out[gid] = half(0.5f * v * (1.0f + tanh(t)));\n"
    "}\n"
)
_APPLE_GPU_GELU_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_GELU_MSL_SOURCE_F16)


# bf16 doesn't get native MSL kernels in Phase 8.4.4.1 — the runtime shim
# does fp32 conversion at the boundary then dispatches the existing f32
# kernel. The IR-level marker reuses the f32 source text but flips the
# entry_point + cache_key + dtype attr so downstream tooling can tell the
# difference. Same shape as Phase 8.4.4 bf16 matmul (no native MPS bf16).
def _apple_gpu_kernel_msl_for_dtype(
    kernel: str, dtype: str
) -> tuple[str, str, str, str]:
    """Return (msl_source, entry_point, cache_key, dtype) for the given
    (kernel, dtype) pair. dtype is one of {"f32", "f16", "bf16"}."""

    if kernel == "rope":
        if dtype == "f16":
            return (_APPLE_GPU_ROPE_MSL_SOURCE_F16, "rope_f16",
                    _APPLE_GPU_ROPE_MSL_CACHE_KEY_F16, "f16")
        if dtype == "bf16":
            return (_APPLE_GPU_ROPE_MSL_SOURCE, "rope_bf16",
                    _APPLE_GPU_ROPE_MSL_CACHE_KEY, "bf16")
        return (_APPLE_GPU_ROPE_MSL_SOURCE, "rope_f32",
                _APPLE_GPU_ROPE_MSL_CACHE_KEY, "f32")
    if kernel == "softmax":
        if dtype == "f16":
            return (_APPLE_GPU_SOFTMAX_MSL_SOURCE_F16, "softmax_f16",
                    _APPLE_GPU_SOFTMAX_MSL_CACHE_KEY_F16, "f16")
        if dtype == "bf16":
            return (_APPLE_GPU_SOFTMAX_MSL_SOURCE, "softmax_bf16",
                    _APPLE_GPU_SOFTMAX_MSL_CACHE_KEY, "bf16")
        return (_APPLE_GPU_SOFTMAX_MSL_SOURCE, "softmax_f32",
                _APPLE_GPU_SOFTMAX_MSL_CACHE_KEY, "f32")
    if kernel == "gelu":
        if dtype == "f16":
            return (_APPLE_GPU_GELU_MSL_SOURCE_F16, "gelu_f16",
                    _APPLE_GPU_GELU_MSL_CACHE_KEY_F16, "f16")
        if dtype == "bf16":
            return (_APPLE_GPU_GELU_MSL_SOURCE, "gelu_bf16",
                    _APPLE_GPU_GELU_MSL_CACHE_KEY, "bf16")
        return (_APPLE_GPU_GELU_MSL_SOURCE, "gelu_f32",
                _APPLE_GPU_GELU_MSL_CACHE_KEY, "f32")
    if kernel == "flash_attn":
        if dtype == "f16":
            return (_APPLE_GPU_FLASH_ATTN_MSL_SOURCE_F16, "flash_attn_f16",
                    _APPLE_GPU_FLASH_ATTN_MSL_CACHE_KEY_F16, "f16")
        if dtype == "bf16":
            return (_APPLE_GPU_FLASH_ATTN_MSL_SOURCE, "flash_attn_bf16",
                    _APPLE_GPU_FLASH_ATTN_MSL_CACHE_KEY, "bf16")
        return (_APPLE_GPU_FLASH_ATTN_MSL_SOURCE, "flash_attn_f32",
                _APPLE_GPU_FLASH_ATTN_MSL_CACHE_KEY, "f32")
    if kernel == "matmul_softmax":
        # SYNTHESIZED for all dtypes (Optimizing-Compiler Plan F2 — the
        # threadgroup-tiled + half-precision synthesizer subsumes
        # matmul_softmax_{f32,f16,bf16} + tiled variants). f16 embeds the half
        # source; bf16 host-converts to f32, so it embeds the f32 source.
        from tessera.compiler.fusion import (
            _ENTRY, FusedRegion, synthesize_matmul_epilogue_msl,
        )
        region = FusedRegion((), reduction="softmax")
        synth_dtype = "f16" if dtype == "f16" else "f32"
        src = synthesize_matmul_epilogue_msl(region, dtype=synth_dtype)
        return (src, _ENTRY, _sha256_short(src), dtype)
    if kernel == "matmul_softmax_matmul":
        if dtype == "f16":
            return (_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE_F16,
                    "matmul_softmax_matmul_f16",
                    _APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_CACHE_KEY_F16, "f16")
        if dtype == "bf16":
            return (_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE,
                    "matmul_softmax_matmul_bf16",
                    _APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_CACHE_KEY, "bf16")
        return (_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE,
                "matmul_softmax_matmul_f32",
                _APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_CACHE_KEY, "f32")
    if kernel in ("matmul_gelu", "matmul_rmsnorm"):
        # Optimizing-Compiler Plan F2 (catalog retirement + half precision) —
        # the matmul-epilogue MSL is SYNTHESIZED from one generator (single
        # source of truth). The Target IR embeds the synthesized source for the
        # requested dtype: f16 embeds the half source; bf16 host-converts to f32,
        # so it embeds the f32 source (matching the matmul_softmax branch above).
        from tessera.compiler.fusion import (
            _ENTRY, FusedRegion, synthesize_matmul_epilogue_msl,
        )
        region = (FusedRegion(("gelu",)) if kernel == "matmul_gelu"
                  else FusedRegion((), reduction="rmsnorm"))
        synth_dtype = "f16" if dtype == "f16" else "f32"
        src = synthesize_matmul_epilogue_msl(region, dtype=synth_dtype)
        return (src, _ENTRY, _sha256_short(src), dtype)
    raise ValueError(f"unknown apple_gpu kernel: {kernel!r}")


def _apple_gpu_dtype_from_op(op) -> str:
    """Pick a runtime-supported dtype from a tile op's attrs. Defaults to f32
    when the attr is absent or not in the supported envelope."""

    raw = str(op.attrs.get("dtype", "f32"))
    if raw in {"f32", "f16", "bf16"}:
        return raw
    return "f32"


# Phase 8.4.3 — embedded MSL source for the fused matmul -> softmax(axis=-1)
# kernel. One thread per output row: computes the row of A@B into a stack
# array, then row-wise softmax in place. Cap N <= 256 to keep the stack
# array bounded. Carrying the source inline here keeps the Target IR a
# self-contained record of what the runtime actually compiles.
_APPLE_GPU_MATMUL_SOFTMAX_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_softmax_f32(\n"
    "    device const float* A   [[buffer(0)]],\n"
    "    device const float* B   [[buffer(1)]],\n"
    "    device float*       O   [[buffer(2)]],\n"
    "    constant int&       M   [[buffer(3)]],\n"
    "    constant int&       N   [[buffer(4)]],\n"
    "    constant int&       K   [[buffer(5)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    if (N > 256) return;\n"
    "    int row = (int)gid;\n"
    "    float scores[256];\n"
    "    int a_off = row * K;\n"
    "    for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        float a = A[a_off + k];\n"
    "        int b_off = k * N;\n"
    "        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];\n"
    "    }\n"
    "    float row_max = -INFINITY;\n"
    "    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);\n"
    "    float denom = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        scores[n] = exp(scores[n] - row_max);\n"
    "        denom += scores[n];\n"
    "    }\n"
    "    int o_off = row * N;\n"
    "    if (denom == 0.0f) {\n"
    "        for (int n = 0; n < N; ++n) O[o_off + n] = 0.0f;\n"
    "    } else {\n"
    "        float inv = 1.0f / denom;\n"
    "        for (int n = 0; n < N; ++n) O[o_off + n] = scores[n] * inv;\n"
    "    }\n"
    "}\n"
)

_APPLE_GPU_MATMUL_SOFTMAX_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_MATMUL_SOFTMAX_MSL_SOURCE)


# Phase 8.4.4.2 — fp16 native fused matmul -> softmax kernel. `half` I/O,
# `float` per-thread `scores[256]` accumulator (mixed precision matches
# production flash-attn implementations).
_APPLE_GPU_MATMUL_SOFTMAX_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_softmax_f16(\n"
    "    device const half*  A   [[buffer(0)]],\n"
    "    device const half*  B   [[buffer(1)]],\n"
    "    device half*        O   [[buffer(2)]],\n"
    "    constant int&       M   [[buffer(3)]],\n"
    "    constant int&       N   [[buffer(4)]],\n"
    "    constant int&       K   [[buffer(5)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    if (N > 256) return;\n"
    "    int row = (int)gid;\n"
    "    float scores[256];\n"
    "    int a_off = row * K;\n"
    "    for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        float a = float(A[a_off + k]);\n"
    "        int b_off = k * N;\n"
    "        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);\n"
    "    }\n"
    "    float row_max = -INFINITY;\n"
    "    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);\n"
    "    float denom = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        scores[n] = exp(scores[n] - row_max);\n"
    "        denom += scores[n];\n"
    "    }\n"
    "    int o_off = row * N;\n"
    "    if (denom == 0.0f) {\n"
    "        for (int n = 0; n < N; ++n) O[o_off + n] = half(0.0f);\n"
    "    } else {\n"
    "        float inv = 1.0f / denom;\n"
    "        for (int n = 0; n < N; ++n) O[o_off + n] = half(scores[n] * inv);\n"
    "    }\n"
    "}\n"
)
_APPLE_GPU_MATMUL_SOFTMAX_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_MATMUL_SOFTMAX_MSL_SOURCE_F16)


# Phase 8.4.5 — embedded MSL source for the 3-op fusion (matmul -> softmax
# -> matmul). One thread per output row; two stack arrays per thread
# (`scores[256]` for the softmax intermediate, `out[256]` for the second
# matmul accumulator). float accumulators throughout regardless of I/O dtype.
_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_softmax_matmul_f32(\n"
    "    device const float* A   [[buffer(0)]],\n"
    "    device const float* B   [[buffer(1)]],\n"
    "    device const float* C   [[buffer(2)]],\n"
    "    device float*       O   [[buffer(3)]],\n"
    "    constant int&       M   [[buffer(4)]],\n"
    "    constant int&       K   [[buffer(5)]],\n"
    "    constant int&       N   [[buffer(6)]],\n"
    "    constant int&       P   [[buffer(7)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    if (N > 256) return;\n"
    "    if (P > 256) return;\n"
    "    int row = (int)gid;\n"
    "    float scores[256];\n"
    "    int a_off = row * K;\n"
    "    for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        float a = A[a_off + k];\n"
    "        int b_off = k * N;\n"
    "        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];\n"
    "    }\n"
    "    float row_max = -INFINITY;\n"
    "    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);\n"
    "    float denom = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        scores[n] = exp(scores[n] - row_max);\n"
    "        denom += scores[n];\n"
    "    }\n"
    "    if (denom == 0.0f) {\n"
    "        for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    } else {\n"
    "        float inv = 1.0f / denom;\n"
    "        for (int n = 0; n < N; ++n) scores[n] *= inv;\n"
    "    }\n"
    "    float out[256];\n"
    "    for (int p = 0; p < P; ++p) out[p] = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        float sn = scores[n];\n"
    "        int c_off = n * P;\n"
    "        for (int p = 0; p < P; ++p) out[p] += sn * C[c_off + p];\n"
    "    }\n"
    "    int o_off = row * P;\n"
    "    for (int p = 0; p < P; ++p) O[o_off + p] = out[p];\n"
    "}\n"
)
_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE)


_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE_F16 = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_softmax_matmul_f16(\n"
    "    device const half*  A   [[buffer(0)]],\n"
    "    device const half*  B   [[buffer(1)]],\n"
    "    device const half*  C   [[buffer(2)]],\n"
    "    device half*        O   [[buffer(3)]],\n"
    "    constant int&       M   [[buffer(4)]],\n"
    "    constant int&       K   [[buffer(5)]],\n"
    "    constant int&       N   [[buffer(6)]],\n"
    "    constant int&       P   [[buffer(7)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= (uint)M) return;\n"
    "    if (N > 256) return;\n"
    "    if (P > 256) return;\n"
    "    int row = (int)gid;\n"
    "    float scores[256];\n"
    "    int a_off = row * K;\n"
    "    for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        float a = float(A[a_off + k]);\n"
    "        int b_off = k * N;\n"
    "        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);\n"
    "    }\n"
    "    float row_max = -INFINITY;\n"
    "    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);\n"
    "    float denom = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        scores[n] = exp(scores[n] - row_max);\n"
    "        denom += scores[n];\n"
    "    }\n"
    "    if (denom == 0.0f) {\n"
    "        for (int n = 0; n < N; ++n) scores[n] = 0.0f;\n"
    "    } else {\n"
    "        float inv = 1.0f / denom;\n"
    "        for (int n = 0; n < N; ++n) scores[n] *= inv;\n"
    "    }\n"
    "    float out[256];\n"
    "    for (int p = 0; p < P; ++p) out[p] = 0.0f;\n"
    "    for (int n = 0; n < N; ++n) {\n"
    "        float sn = scores[n];\n"
    "        int c_off = n * P;\n"
    "        for (int p = 0; p < P; ++p) out[p] += sn * float(C[c_off + p]);\n"
    "    }\n"
    "    int o_off = row * P;\n"
    "    for (int p = 0; p < P; ++p) O[o_off + p] = half(out[p]);\n"
    "}\n"
)
_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_CACHE_KEY_F16 = _sha256_short(_APPLE_GPU_MATMUL_SOFTMAX_MATMUL_MSL_SOURCE_F16)


# Optimizing-Compiler Plan F2 (catalog retirement) — the matmul -> gelu and
# matmul -> rmsnorm MSL kernels are no longer hand-written constants here; they
# are synthesized by tessera.compiler.fusion.synthesize_matmul_epilogue_msl
# (single source of truth) and embedded by _apple_gpu_msl_kernel_info above.


def _diagnostic_level(severity: str) -> DiagnosticLevel:
    return {
        "fatal": DiagnosticLevel.FATAL,
        "error": DiagnosticLevel.ERROR,
        "warning": DiagnosticLevel.WARNING,
        "info": DiagnosticLevel.INFO,
        "note": DiagnosticLevel.NOTE,
    }.get(severity.lower(), DiagnosticLevel.ERROR)


@dataclass(frozen=True)
class TargetIRDiagnostic:
    severity: str
    message: str
    code: str = "TARGET_IR"

    def format(self) -> str:
        structured = self.to_tessera_diagnostic()
        return f"{structured.level.value.upper()} [{structured.code.value}] [{self.code}]: {self.message}\n  where: {structured.where}"

    def to_tessera_diagnostic(self) -> TesseraDiagnostic:
        return TesseraDiagnostic(
            level=_diagnostic_level(self.severity),
            message=self.message,
            code=TesseraErrorCode.TARGET_CODEGEN,
            where=DiagnosticWhere(ir_level="target-ir", pass_name="verifier"),
            hints=["inspect Target IR, target profile, and backend feature requirements"],
        )


@dataclass(frozen=True)
class TargetIRVerificationResult:
    diagnostics: tuple[TargetIRDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)

    def structured_diagnostics(self) -> tuple[TesseraDiagnostic, ...]:
        return tuple(d.to_tessera_diagnostic() for d in self.diagnostics)


class TargetIRVerificationError(ValueError):
    pass


@dataclass
class TargetOp:
    op_name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    operands: list[str] = field(default_factory=list)
    result: Optional[str] = None

    def to_mlir(self, indent: str = "  ") -> str:
        result_text = f"%{self.result} = " if self.result else ""
        operands = ", ".join(self.operands)
        return f"{indent}{result_text}\"{self.op_name}\"({operands}) {_format_attr_dict(self.attrs)} : () -> ()"


@dataclass
class TargetFunction:
    name: str
    body: list[TargetOp] = field(default_factory=list)
    target: str = "cpu"

    def to_mlir(self, indent: str = "  ") -> str:
        func_op = {
            APPLE_CPU_TARGET: "tessera_apple.cpu.func",
            APPLE_GPU_TARGET: "tessera_apple.gpu.func",
            CPU_TARGET: "tessera.cpu.func",
            ROCM_TARGET: "tessera_rocm.func",
        }.get(self.target, "tessera_nvidia.func" if self.target in NVIDIA_TARGETS else "tessera.target.func")
        lines = [f"{indent}\"{func_op}\"() ({{"]
        for op in self.body:
            lines.append(op.to_mlir(indent + "  "))
        lines.append(f"{indent}}}) {{sym_name = {json.dumps(self.name)}}} : () -> ()")
        return "\n".join(lines)


@dataclass
class TargetIRModule:
    functions: list[TargetFunction] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=lambda: {"tessera.ir.level": "target"})

    def verify(self) -> TargetIRVerificationResult:
        return TargetIRVerifier().verify_module(self)

    def to_mlir(self, *, verify: bool = True) -> str:
        if verify:
            result = self.verify()
            if not result.ok:
                raise TargetIRVerificationError(result.format())
        lines = [f"module attributes {_format_attr_dict(self.attrs)} {{"]
        for fn in self.functions:
            lines.append(fn.to_mlir())
        lines.append("}")
        return "\n".join(lines)


class TargetIRVerifier:
    def verify_module(self, module: TargetIRModule) -> TargetIRVerificationResult:
        diagnostics: list[TargetIRDiagnostic] = []
        target = module.attrs.get("target")
        if target not in SUPPORTED_TARGETS:
            diagnostics.append(TargetIRDiagnostic("error", f"unsupported target {target!r}", "TARGET_IR_TARGET"))
        for fn in module.functions:
            diagnostics.extend(self.verify_function(fn, target=str(target)).diagnostics)
        return TargetIRVerificationResult(tuple(diagnostics))

    def verify_function(self, fn: TargetFunction, *, target: str) -> TargetIRVerificationResult:
        diagnostics: list[TargetIRDiagnostic] = []
        for op in fn.body:
            if target == APPLE_CPU_TARGET:
                self._verify_apple_cpu_op(op, diagnostics)
            elif target == APPLE_GPU_TARGET:
                self._verify_apple_gpu_op(op, diagnostics)
            elif target == CPU_TARGET:
                self._verify_cpu_op(op, diagnostics)
            elif target == ROCM_TARGET:
                self._verify_rocm_op(op, diagnostics)
            elif target in NVIDIA_TARGETS:
                self._verify_nvidia_op(op, diagnostics)
        return TargetIRVerificationResult(tuple(diagnostics))

    def _verify_cpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if not op.op_name.startswith("tessera.cpu."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid CPU op {op.op_name!r}", "TARGET_IR_CPU_OP"))
            return
        self._require(op, diagnostics, "source", "result", "ordinal", "abi")

    def _verify_apple_cpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera_apple.diagnostic":
            self._require(op, diagnostics, "severity", "reason")
            return
        if not op.op_name.startswith("tessera_apple.cpu."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid Apple CPU op {op.op_name!r}", "TARGET_IR_APPLE_CPU_OP"))
        self._require(op, diagnostics, "framework", "abi", "dtype")

    def _verify_apple_gpu_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera_apple.diagnostic":
            self._require(op, diagnostics, "severity", "reason")
            return
        if op.op_name == "tessera_apple.gpu.metal_kernel":
            self._require(op, diagnostics, "kernel", "framework", "status", "dtype")
        elif op.op_name == "tessera_apple.gpu.dispatch":
            self._require(op, diagnostics, "queue", "artifact", "execution_mode")
        elif op.op_name == "tessera_apple.gpu.mps_matmul":
            self._require(op, diagnostics, "framework", "abi", "dtype")
        elif op.op_name == "tessera_apple.gpu.mps_dispatch":
            self._require(op, diagnostics, "queue", "framework", "execution_mode")
        elif op.op_name == "tessera_apple.gpu.msl_kernel":
            # Phase 8.4 — custom MSL kernel artifact. Must carry the entry point
            # name and the MSL source itself so the IR is self-contained.
            self._require(op, diagnostics, "entry_point", "msl_source", "framework", "dtype")
        else:
            diagnostics.append(TargetIRDiagnostic("error", f"invalid Apple GPU op {op.op_name!r}", "TARGET_IR_APPLE_GPU_OP"))

    def _verify_rocm_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if op.op_name == "tessera.target.diagnostic":
            self._require(op, diagnostics, "target", "severity", "reason")
            return
        if not op.op_name.startswith("tessera_rocm."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid ROCm op {op.op_name!r}", "TARGET_IR_ROCM_OP"))
            return
        if op.op_name == "tessera_rocm.mfma":
            self._require(op, diagnostics, "arch", "shape", "accum")
        elif op.op_name == "tessera_rocm.async_copy":
            self._require(op, diagnostics, "src_space", "dst_space", "bytes")
        elif op.op_name == "tessera_rocm.wait":
            self._require(op, diagnostics, "ordinal")
        elif op.op_name == "tessera_rocm.elementwise":
            self._require(op, diagnostics, "arch")

    def _verify_nvidia_op(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic]) -> None:
        if not op.op_name.startswith("tessera_nvidia."):
            diagnostics.append(TargetIRDiagnostic("error", f"invalid NVIDIA op {op.op_name!r}", "TARGET_IR_NVIDIA_OP"))
            return
        if op.op_name == "tessera_nvidia.wgmma":
            self._require(op, diagnostics, "arch", "shape", "dtype_ab", "dtype_c", "warpgroup")
        elif op.op_name == "tessera_nvidia.tma_async_copy":
            self._require(op, diagnostics, "arch", "src_space", "dst_space", "bytes")
        elif op.op_name == "tessera_nvidia.mbarrier":
            self._require(op, diagnostics, "arch", "scope", "ordinal")
        elif op.op_name == "tessera_nvidia.tmem_alloc":
            self._require(op, diagnostics, "arch", "columns")
        elif op.op_name == "tessera_nvidia.tcgen05_mma":
            self._require(op, diagnostics, "arch", "shape", "accum", "cta_group")
        elif op.op_name == "tessera_nvidia.cuda_kernel":
            self._require(op, diagnostics, "arch", "kernel", "status")

    def _require(self, op: TargetOp, diagnostics: list[TargetIRDiagnostic], *attrs: str) -> None:
        missing = [attr for attr in attrs if attr not in op.attrs]
        if missing:
            diagnostics.append(TargetIRDiagnostic("error", f"{op.op_name} missing attrs: {', '.join(missing)}", "TARGET_IR_MISSING_ATTR"))


def lower_tile_to_target_ir(tile_module: TileIRModule, *, target_kind: str) -> TargetIRModule:
    tile_result = tile_module.verify()
    if not tile_result.ok:
        raise TileIRVerificationError(tile_result.format())
    target_kind = normalize_target(target_kind)
    if target_kind not in SUPPORTED_TARGETS:
        raise ValueError(f"target_ir does not support target {target_kind!r}")
    # Mixed-value attribute bag — values include strings, nested
    # dicts (target_features), and bools.  Widen the value type.
    attrs: dict[str, Any] = {"tessera.ir.level": "target", "target": target_kind}
    if target_kind == CPU_TARGET:
        attrs.update({"arch": "x86_64", "execution_mode": "numpy", "target_features": {"family": "cpu", "wall_clock": True, "device_timers": False}})
    elif target_kind in NVIDIA_TARGETS:
        attrs.update({"arch": _nvidia_arch(target_kind), "target_features": {"family": "nvidia", "tensor_cores": True, "device_timers": False}})
    elif target_kind == ROCM_TARGET:
        attrs.update({"arch": "gfx90a", "target_features": {"family": "rocm", "mfma": True, "device_timers": False}})
    elif target_kind == APPLE_CPU_TARGET:
        attrs.update({"arch": "arm64-apple-silicon", "execution_mode": "cpu_accelerate", "target_features": {"family": "apple", "accelerate": True, "device_timers": False}})
    else:
        # Phase 8.3: when the entire program is a single rank-2 f32 matmul, the
        # apple_gpu module is executable through MPSMatrixMultiplication. The
        # module-level execution_mode flips to "metal_runtime" so downstream
        # tooling (and Python tests) can distinguish artifact-only from
        # runtime-bound modules without re-walking the body.
        apple_gpu_runtime = _apple_gpu_module_is_mps_runtime(tile_module)
        execution_mode = "metal_runtime" if apple_gpu_runtime else "metal_artifact"
        # Phase 3: emit the explicit Apple GPU target descriptor. execution_mode
        # IS the execution_contract here — the existing lane is classic MPS
        # (metal_runtime) or artifact-only (metal_artifact); never mtl4_runtime
        # (the MTL4 cooperative-tensor lane is a separate, capability-gated
        # surface and is not claimed by this MPS path).
        attrs.update({
            "arch": "apple-metal",
            "execution_mode": execution_mode,
            "target_features": {"family": "apple", "metal": True, "device_timers": False},
            "target_descriptor": _apple_target_descriptor(execution_mode),
        })
    target_module = TargetIRModule(attrs=attrs)
    apple_gpu_runtime_flag = (
        target_kind == APPLE_GPU_TARGET
        and _apple_gpu_module_is_mps_runtime(tile_module)
    )
    apple_gpu_fusion_kind = (
        _apple_gpu_module_fusion_kind(tile_module)
        if apple_gpu_runtime_flag else None
    )
    for tile_fn in tile_module.functions:
        if apple_gpu_fusion_kind is not None:
            # Phase 8.4.3: a recognized fusion chain emits a single fused
            # msl_kernel + mps_dispatch pair, regardless of how many tile
            # ops the chain decomposed into. The per-op walk is bypassed
            # entirely for fusion modules so we don't double-emit.
            body = _lower_apple_gpu_fusion(tile_fn, apple_gpu_fusion_kind)
        else:
            body = _lower_tile_ops(
                tile_fn.body,
                target_kind=target_kind,
                apple_gpu_mps_runtime=apple_gpu_runtime_flag,
            )
        target_module.functions.append(TargetFunction(
            name=tile_fn.name,
            target=target_kind,
            body=body,
        ))
    return target_module


def _apple_gpu_module_fusion_kind(tile_module: TileIRModule) -> str | None:
    """Phase 8.4.3 + 8.4.5 + 8.4.7 — classify the Tile IR module as one of
    the recognized fusion chains:
      - "matmul_softmax_matmul"  (3 ops, full attention block)
      - "matmul_softmax"         (2 ops)
      - "matmul_gelu"            (2 ops, MLP block activation)
      - "matmul_rmsnorm"         (2 ops, transformer normalization)
      - None                     (no fusion)

    Distinct Graph IR ops carry distinct `ordinal` attrs through the
    Schedule -> Tile lowering, so counting unique ordinals tells us how
    many original ops are in the chain. The set of `source` strings tells
    us the chain shape.
    """

    if len(tile_module.functions) != 1:
        return None
    flat_ops = list(_flatten_tile_ops(tile_module.functions[0].body))
    compute_ops = [
        op for op in flat_ops
        if op.op_name not in {"tile.debug_artifact", "tile.debug_barrier"}
        and not (
            op.op_name.startswith("tessera.queue.")
            or op.op_name in {"tile.async_copy", "tile.wait_async"}
        )
    ]
    if not compute_ops:
        return None

    sources = {str(op.attrs.get("source", "")) for op in compute_ops}
    matmul_sources = {"tessera.matmul", "tessera.gemm"}
    softmax_sources = {"tessera.softmax", "tessera.softmax_safe"}
    rmsnorm_sources = {"tessera.rmsnorm", "tessera.rmsnorm_safe"}

    has_matmul = bool(sources & matmul_sources)
    if not has_matmul:
        return None

    # 3-op fusion: matmul -> softmax -> matmul
    if (sources & softmax_sources) and (sources <= (matmul_sources | softmax_sources)):
        matmul_ordinals = {
            op.attrs.get("ordinal")
            for op in compute_ops
            if str(op.attrs.get("source", "")) in matmul_sources
        }
        if len(matmul_ordinals) >= 2:
            return "matmul_softmax_matmul"
        return "matmul_softmax"

    # 2-op fusion: matmul -> gelu
    if "tessera.gelu" in sources and (sources <= (matmul_sources | {"tessera.gelu"})):
        return "matmul_gelu"

    # 2-op fusion: matmul -> rmsnorm[_safe]
    if (sources & rmsnorm_sources) and (sources <= (matmul_sources | rmsnorm_sources)):
        return "matmul_rmsnorm"

    return None


def _lower_apple_gpu_fusion(tile_fn, fusion_kind: str) -> list[TargetOp]:
    """Phase 8.4.3 + 8.4.4.2 + 8.4.5 — emit the fused msl_kernel +
    mps_dispatch pair for a recognized fusion chain. Picks the dtype-
    specific MSL source + entry_point + cache_key from the matmul "head"
    op's dtype attr. The kernel_key in the helper call corresponds to
    the chain shape: "matmul_softmax" or "matmul_softmax_matmul".
    """

    if fusion_kind not in {
        "matmul_softmax",
        "matmul_softmax_matmul",
        "matmul_gelu",
        "matmul_rmsnorm",
    }:
        raise ValueError(f"unknown apple_gpu fusion kind: {fusion_kind!r}")

    flat_ops = list(_flatten_tile_ops(tile_fn.body))
    head = next(
        (op for op in flat_ops
         if op.op_name not in {"tile.debug_artifact", "tile.debug_barrier"}
         and str(op.attrs.get("source", "")) in {"tessera.matmul", "tessera.gemm"}),
        None,
    )
    if head is None:
        return []
    base = _base_attrs(head)
    dtype = _apple_gpu_dtype_from_op(head)
    msl_source, entry_point, cache_key, dtype_attr = (
        _apple_gpu_kernel_msl_for_dtype(fusion_kind, dtype)
    )
    return [
        TargetOp("tessera_apple.gpu.msl_kernel", {
            **base,
            "framework": "Metal",
            "dtype": dtype_attr,
            "entry_point": entry_point,
            "msl_source": msl_source,
            "cache_key": cache_key,
            "fusion": fusion_kind,
            "grid": "rows",
            "threadgroup": "?x1x1",
        }),
        TargetOp("tessera_apple.gpu.mps_dispatch", {
            "ordinal": base["ordinal"],
            "queue": "MTLCommandQueue",
            "framework": "Metal",
            "execution_mode": "metal_runtime",
        }),
    ]


def _apple_gpu_module_is_mps_runtime(tile_module: TileIRModule) -> bool:
    """Return True when an apple_gpu Tile IR module qualifies for the runtime
    path (MPS for matmul, MSL for rope/flash_attn/softmax/gelu, or a fused
    chain like matmul -> softmax as of Phase 8.4.3).

    Two cases are accepted:
      1. Single-source: every compute op carries the same `source`, and that
         source is in the per-op envelope. Covers Graph IR ops that decompose
         into multiple Tile IR ops (rope -> tile.rope + tile.rotary_pair).
      2. Recognized fusion chain (Phase 8.4.3): the compute ops resolve to a
         pair like {matmul, softmax} when grouped by source, with both in the
         runtime envelope. The lowering will collapse them to a single
         msl_kernel emission.

    Name kept for backward compatibility — it now covers MPS + MSL +
    fusion paths.
    """

    if len(tile_module.functions) != 1:
        return False
    flat_ops = list(_flatten_tile_ops(tile_module.functions[0].body))
    compute_ops = [
        op for op in flat_ops
        if op.op_name not in {"tile.debug_artifact", "tile.debug_barrier"}
        and not (
            op.op_name.startswith("tessera.queue.")
            or op.op_name in {"tile.async_copy", "tile.wait_async"}
        )
    ]
    if not compute_ops:
        return False

    single_op_envelope = {
        "tessera.matmul",
        "tessera.gemm",
        "tessera.rope",
        "tessera.flash_attn",
        "tessera.softmax",
        "tessera.softmax_safe",
        "tessera.gelu",
    }

    sources = {str(op.attrs.get("source", "")) for op in compute_ops}
    if len(sources) == 1:
        (source,) = sources
        return source in single_op_envelope

    # Fusion shape check (covers matmul -> softmax, Phase 8.4.5 3-op chain,
    # and Phase 8.4.7 MLP patterns matmul -> gelu, matmul -> rmsnorm). Tile
    # IR carries the original Graph IR `source` attr on each op; ordinals
    # preserve Graph IR op order, so we can check both shape and order.
    matmul_sources = {"tessera.matmul", "tessera.gemm"}
    softmax_sources = {"tessera.softmax", "tessera.softmax_safe"}
    rmsnorm_sources = {"tessera.rmsnorm", "tessera.rmsnorm_safe"}

    if not (sources & matmul_sources):
        return False

    # The first compute op (lowest ordinal) must be a matmul — otherwise the
    # chain is in the wrong order (e.g., softmax -> matmul) and no fusion
    # applies.
    sorted_ops = sorted(
        compute_ops,
        key=lambda op: int(op.attrs.get("ordinal", 0)),
    )
    if str(sorted_ops[0].attrs.get("source", "")) not in matmul_sources:
        return False

    # 2-op or 3-op softmax-bearing chain (Phase 8.4.3 / 8.4.5)
    if (sources & softmax_sources) and (sources <= (matmul_sources | softmax_sources)):
        return True
    # 2-op matmul -> gelu (Phase 8.4.7)
    if "tessera.gelu" in sources and (sources <= (matmul_sources | {"tessera.gelu"})):
        return True
    # 2-op matmul -> rmsnorm[_safe] (Phase 8.4.7)
    if (sources & rmsnorm_sources) and (sources <= (matmul_sources | rmsnorm_sources)):
        return True
    return False


def _lower_tile_ops(
    ops: Iterable[TileOp],
    *,
    target_kind: str,
    apple_gpu_mps_runtime: bool = False,
) -> list[TargetOp]:
    lowered: list[TargetOp] = []
    # Phase 8.4 — when in runtime mode some Graph IR ops (e.g. rope, flash_attn)
    # decompose into multiple Tile IR ops sharing the same (source, ordinal).
    # The artifact-only path emits both for inspection/diagnostics; the runtime
    # path must emit exactly one func.call site, so we suppress duplicates.
    #
    # The dedup key is only consumed when the op actually produces a runtime
    # emission. Filter ops (tile.async_copy, tessera.queue.*, tile.wait_async,
    # debug_barriers) carry the same source/ordinal but emit nothing — if we
    # consumed the key for them, the real compute op would be skipped.
    seen_runtime_keys: set[tuple[str, object]] = set()
    for tile_op in _flatten_tile_ops(ops):
        if target_kind == ROCM_TARGET:
            lowered.extend(_lower_rocm_op(tile_op))
        elif target_kind == APPLE_CPU_TARGET:
            lowered.extend(_lower_apple_cpu_op(tile_op))
        elif target_kind == APPLE_GPU_TARGET:
            if apple_gpu_mps_runtime:
                key = (
                    str(tile_op.attrs.get("source", _source_from_tile_op(tile_op))),
                    tile_op.attrs.get("ordinal"),
                )
                if key in seen_runtime_keys:
                    continue
                emitted = _lower_apple_gpu_op(tile_op, mps_runtime=apple_gpu_mps_runtime)
                if emitted:
                    seen_runtime_keys.add(key)
                lowered.extend(emitted)
            else:
                lowered.extend(_lower_apple_gpu_op(tile_op, mps_runtime=apple_gpu_mps_runtime))
        elif target_kind == CPU_TARGET:
            lowered.extend(_lower_cpu_op(tile_op))
        elif target_kind in NVIDIA_TARGETS:
            lowered.extend(_lower_nvidia_op(tile_op, target_kind=target_kind))
    return lowered


def _flatten_tile_ops(ops: Iterable[TileOp]) -> Iterable[TileOp]:
    for op in ops:
        # tile.group + the placement/layout/schedule-plan metadata markers
        # (TILE_METADATA_OPS) carry no compute: drop the marker itself but still
        # recurse into its body so the compute ops inside tile.mesh.region reach
        # Target lowering. Without this, the catch-all below would mis-lower a
        # metadata marker into a spurious compute target op.
        if op.op_name == "tile.group" or op.op_name in TILE_METADATA_OPS:
            yield from _flatten_tile_ops(op.body)
        else:
            yield op
            yield from _flatten_tile_ops(op.body)


def _lower_rocm_op(op: TileOp) -> list[TargetOp]:
    if op.op_name in {"tile.debug_artifact", "tile.debug_barrier"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op, target="rocm")
    if op.op_name == "tile.mma":
        return [
            TargetOp("tessera_rocm.mfma", {**base, "arch": "gfx90a", "shape": "m16n16k16", "accum": "f32"}),
            TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
            TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
        ]
    if source == "tessera.flash_attn" or op.op_name.startswith("tessera.attn."):
        return [TargetOp("tessera.target.diagnostic", {
            **base,
            "target": "rocm",
            "severity": "unsupported",
            "reason": "flash_attn target kernel contract is not implemented for ROCm in this phase",
        })]
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera.target.diagnostic", {
            **base,
            "target": "rocm",
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for ROCm in this phase",
        })]
    if op.op_name == "tile.async_copy":
        return [
            TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
            TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
        ]
    if op.op_name.startswith("tessera.queue.") or op.op_name == "tile.wait_async":
        return []
    return [
        TargetOp("tessera_rocm.elementwise", {**base, "arch": "gfx90a"}),
        TargetOp("tessera_rocm.async_copy", {**base, "src_space": "global", "dst_space": "lds", "bytes": 16}),
        TargetOp("tessera_rocm.wait", {"ordinal": base["ordinal"]}),
    ]


def _lower_cpu_op(op: TileOp) -> list[TargetOp]:
    if op.op_name in {"tile.debug_artifact", "tile.debug_barrier"}:
        return []
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    return [TargetOp(_cpu_target_op_name(source), {**base, "abi": "numpy"})]


def _lower_nvidia_op(op: TileOp, *, target_kind: str) -> list[TargetOp]:
    if op.op_name in {"tile.debug_artifact", "tile.debug_barrier"}:
        return []
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    arch = _nvidia_arch(target_kind)
    if op.op_name == "tile.mma":
        if target_kind in {"nvidia_sm100", "nvidia_sm120"}:
            return [
                TargetOp("tessera_nvidia.tmem_alloc", {**base, "arch": arch, "columns": 128}),
                TargetOp("tessera_nvidia.tcgen05_mma", {
                    **base,
                    "arch": arch,
                    "shape": "m128n128k32",
                    "accum": "tmem_f32",
                    "cta_group": 2,
                    "block_scaled": True,
                }),
            ]
        return [
            TargetOp("tessera_nvidia.wgmma", {
                **base,
                "arch": arch,
                "shape": "m64n64k16",
                "dtype_ab": "bf16",
                "dtype_c": "f32",
                "warpgroup": 4,
            }),
            TargetOp("tessera_nvidia.tma_async_copy", {
                **base,
                "arch": arch,
                "src_space": "global",
                "dst_space": "shared",
                "bytes": 16,
            }),
            TargetOp("tessera_nvidia.mbarrier", {"ordinal": base["ordinal"], "arch": arch, "scope": "cta"}),
        ]
    if op.op_name == "tessera.attn.msa_kv_outer_sparse" or source == "tessera.msa_sparse_attention":
        return [TargetOp("tessera_nvidia.cuda_kernel", {
            **base,
            "arch": arch,
            "kernel": "msa_kv_outer_sparse",
            "status": "artifact_only",
            "mode": op.attrs.get("mode", "prefill"),
            "block_ids_layout": op.attrs.get("selected_block_layout", "B,Hkv,Sq,top_k"),
            "gqa_group_size": int(op.attrs.get("gqa_group_size", 1)),
            "tile_q": int(op.attrs.get("tile_q", 64)),
            "tile_kv": int(op.attrs.get("tile_kv", 128)),
            "kv_traversal": "kv_outer",
        })]
    kernel = "flash_attn_contract" if source == "tessera.flash_attn" or op.op_name.startswith("tessera.attn.") else "elementwise_contract"
    return [TargetOp("tessera_nvidia.cuda_kernel", {**base, "arch": arch, "kernel": kernel, "status": "artifact_only"})]


def _lower_apple_cpu_op(op: TileOp) -> list[TargetOp]:
    if op.op_name in {"tile.debug_artifact", "tile.debug_barrier"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera_apple.diagnostic", {
            **base,
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for Apple CPU in this phase",
        })]
    if op.op_name == "tile.mma":
        return [TargetOp("tessera_apple.cpu.accelerate_gemm", {**base, "framework": "Accelerate", "abi": "cblas_sgemm", "dtype": "f32"})]
    # Sprint 6: batched matmul artifact lane — route to the real (batched) GEMM
    # artifact, not the generic vector_op fallback. Matches the C++ TileToApple
    # artifact path (abi="cblas_sgemm_batched_loop").
    if source == "tessera.batched_gemm" or op.op_name == "tile.batched_gemm":
        return [TargetOp("tessera_apple.cpu.accelerate_gemm", {
            **base, "framework": "Accelerate",
            "abi": "cblas_sgemm_batched_loop", "dtype": "f32"})]
    if source in {"tessera.matmul", "tessera.gemm"}:
        return [TargetOp("tessera_apple.cpu.accelerate_gemm", {
            **base, "framework": "Accelerate", "abi": "cblas_sgemm", "dtype": "f32"})]
    if source == "tessera.moe":
        return [TargetOp("tessera_apple.cpu.moe_solver", {
            **base,
            "framework": "Accelerate",
            "abi": "moe_top1_round_robin",
            "dtype": "f32",
            "routing": "deterministic_top1",
        })]
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return [TargetOp("tessera_apple.cpu.vector_reduce", {**base, "framework": "Accelerate", "abi": "vDSP", "dtype": "f32"})]
    if source == "tessera.rope":
        return [TargetOp("tessera_apple.cpu.vector_op", {**base, "framework": "Accelerate", "abi": "vecLib", "pattern": "rotary_pairs", "dtype": "f32"})]
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    return [TargetOp("tessera_apple.cpu.vector_op", {**base, "framework": "Accelerate", "abi": "vecLib", "dtype": "f32"})]


def _lower_apple_gpu_op(op: TileOp, *, mps_runtime: bool = False) -> list[TargetOp]:
    if op.op_name in {"tile.debug_artifact", "tile.debug_barrier"}:
        return []
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    base = _base_attrs(op)
    if source.startswith("tessera.kv_cache.") or op.op_name == "tile.kv_cache":
        return [TargetOp("tessera_apple.diagnostic", {
            **base,
            "severity": "unsupported",
            "reason": "KV-cache target lowering is not implemented for Apple GPU in this phase",
        })]
    if op.op_name.startswith("tessera.queue.") or op.op_name in {"tile.async_copy", "tile.wait_async"}:
        return []
    # Phase 8.3 MPS runtime path: a single-matmul module is lowered to
    # mps_matmul + mps_dispatch with execution_mode="metal_runtime". Phase
    # 8.4.4 — the dtype attr now reflects the element type the runtime will
    # dispatch to (f32, f16, or bf16). The MatmulToAppleGPU lowering pass
    # picks the matching runtime symbol; the IR-level dtype attr is the
    # introspection mirror.
    if mps_runtime and op.op_name == "tile.mma" and source in {"tessera.matmul", "tessera.gemm"}:
        # Resolve dtype from the tile op's attrs if present (Phase 8.4.4),
        # else default to f32 (preserves Phase 8.3 contract).
        tile_dtype = str(op.attrs.get("dtype", "f32"))
        if tile_dtype not in {"f32", "f16", "bf16"}:
            tile_dtype = "f32"
        return [
            TargetOp("tessera_apple.gpu.mps_matmul", {
                **base,
                "framework": "MetalPerformanceShaders",
                "abi": "MPSMatrixMultiplication",
                "dtype": tile_dtype,
            }),
            TargetOp("tessera_apple.gpu.mps_dispatch", {
                "ordinal": base["ordinal"],
                "queue": "MTLCommandQueue",
                "framework": "MetalPerformanceShaders",
                "execution_mode": "metal_runtime",
            }),
        ]
    # Phase 8.4 + 8.4.4.1 custom MSL path: a single-rope module is lowered to
    # msl_kernel + mps_dispatch carrying the MSL source as a StringAttr. The
    # RopeToAppleGPU pass and the apple_gpu_runtime.mm shim consume this op.
    # dtype attr (Phase 8.4.4.1) picks between f32 / f16 / bf16 source +
    # entry point.
    if mps_runtime and (
        op.op_name in {"tile.rotary_pair", "tile.rope"}
        or source == "tessera.rope"
    ):
        dtype = _apple_gpu_dtype_from_op(op)
        msl_source, entry_point, cache_key, dtype_attr = (
            _apple_gpu_kernel_msl_for_dtype("rope", dtype)
        )
        return [
            TargetOp("tessera_apple.gpu.msl_kernel", {
                **base,
                "framework": "Metal",
                "dtype": dtype_attr,
                "entry_point": entry_point,
                "msl_source": msl_source,
                "cache_key": cache_key,
                "grid": "tokens_pairs",
                "threadgroup": "32x?",
            }),
            TargetOp("tessera_apple.gpu.mps_dispatch", {
                "ordinal": base["ordinal"],
                "queue": "MTLCommandQueue",
                "framework": "Metal",
                "execution_mode": "metal_runtime",
            }),
        ]
    # Phase 8.4.1 + 8.4.4.2 custom MSL path: single-flash_attn module with
    # dtype-aware MSL source selection (f32 / f16 / bf16).
    if mps_runtime and source == "tessera.flash_attn":
        dtype = _apple_gpu_dtype_from_op(op)
        msl_source, entry_point, cache_key, dtype_attr = (
            _apple_gpu_kernel_msl_for_dtype("flash_attn", dtype)
        )
        return [
            TargetOp("tessera_apple.gpu.msl_kernel", {
                **base,
                "framework": "Metal",
                "dtype": dtype_attr,
                "entry_point": entry_point,
                "msl_source": msl_source,
                "cache_key": cache_key,
                "grid": "batch_query_rows",
                "threadgroup": "32x?",
            }),
            TargetOp("tessera_apple.gpu.mps_dispatch", {
                "ordinal": base["ordinal"],
                "queue": "MTLCommandQueue",
                "framework": "Metal",
                "execution_mode": "metal_runtime",
            }),
        ]
    # Phase 8.4.2 + 8.4.4.1 custom MSL path: single-softmax (axis=-1) module
    # with dtype-aware MSL source selection.
    if mps_runtime and source in {"tessera.softmax", "tessera.softmax_safe"}:
        dtype = _apple_gpu_dtype_from_op(op)
        msl_source, entry_point, cache_key, dtype_attr = (
            _apple_gpu_kernel_msl_for_dtype("softmax", dtype)
        )
        return [
            TargetOp("tessera_apple.gpu.msl_kernel", {
                **base,
                "framework": "Metal",
                "dtype": dtype_attr,
                "entry_point": entry_point,
                "msl_source": msl_source,
                "cache_key": cache_key,
                "grid": "rows",
                "threadgroup": "?x1x1",
            }),
            TargetOp("tessera_apple.gpu.mps_dispatch", {
                "ordinal": base["ordinal"],
                "queue": "MTLCommandQueue",
                "framework": "Metal",
                "execution_mode": "metal_runtime",
            }),
        ]
    # Phase 8.4.2 + 8.4.4.1 custom MSL path: single-gelu (elementwise) module
    # with dtype-aware MSL source selection.
    if mps_runtime and source == "tessera.gelu":
        dtype = _apple_gpu_dtype_from_op(op)
        msl_source, entry_point, cache_key, dtype_attr = (
            _apple_gpu_kernel_msl_for_dtype("gelu", dtype)
        )
        return [
            TargetOp("tessera_apple.gpu.msl_kernel", {
                **base,
                "framework": "Metal",
                "dtype": dtype_attr,
                "entry_point": entry_point,
                "msl_source": msl_source,
                "cache_key": cache_key,
                "grid": "elements",
                "threadgroup": "?x1x1",
            }),
            TargetOp("tessera_apple.gpu.mps_dispatch", {
                "ordinal": base["ordinal"],
                "queue": "MTLCommandQueue",
                "framework": "Metal",
                "execution_mode": "metal_runtime",
            }),
        ]
    kernel, framework, extra = _apple_gpu_kernel_contract(source)
    return [
        TargetOp("tessera_apple.gpu.metal_kernel", {**base, "kernel": kernel, "framework": framework, "dtype": "f32", **extra}),
        TargetOp("tessera_apple.gpu.dispatch", {
            "ordinal": base["ordinal"],
            "queue": "MTLCommandQueue",
            "artifact": "metallib",
            "execution_mode": "metal_artifact",
        }),
    ]


def _apple_gpu_kernel_contract(source: str) -> tuple[str, str, dict[str, Any]]:
    if source == "tessera.flash_attn":
        return "flash_attn_contract", "Metal", {"status": "artifact_only", "grid": "bhn", "threadgroup": "64x1x1", "temporary_memory": "scores_lse"}
    if source in {"tessera.matmul", "tessera.gemm"}:
        return "matmul_contract", "MPSGraph", {"status": "artifact_only", "grid": "mn_tiles", "threadgroup": "16x16x1", "temporary_memory": "none"}
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return "softmax_contract", "MPSGraph", {"status": "artifact_only", "grid": "rows", "threadgroup": "256x1x1", "temporary_memory": "row_max_sum"}
    if source == "tessera.gelu":
        return "gelu_contract", "MPSGraph", {"status": "artifact_only", "grid": "elements", "threadgroup": "256x1x1", "temporary_memory": "none"}
    if source == "tessera.rope":
        return "rope_contract", "Metal", {"status": "artifact_only", "grid": "tokens_heads", "threadgroup": "128x1x1", "temporary_memory": "none"}
    if source == "tessera.moe":
        return "moe_contract", "MPSGraph", {"status": "artifact_only", "grid": "tokens_experts", "threadgroup": "128x1x1", "temporary_memory": "routing"}
    return "elementwise_contract", "Metal", {"status": "artifact_only", "grid": "elements", "threadgroup": "256x1x1", "temporary_memory": "none"}


def _base_attrs(op: TileOp, *, target: Optional[str] = None) -> dict[str, Any]:
    attrs = {
        "source": op.attrs.get("source", _source_from_tile_op(op)),
        "result": op.attrs.get("result", "value"),
        "ordinal": int(op.attrs.get("ordinal", 0)),
        "launch": _launch_metadata(op),
    }
    if "resource" in op.attrs:
        attrs["resource"] = op.attrs["resource"]
    if target is not None:
        attrs["target"] = target
    return attrs


def _launch_metadata(op: TileOp) -> dict[str, Any]:
    source = str(op.attrs.get("source", _source_from_tile_op(op)))
    if op.op_name == "tile.mma" or source in {"tessera.matmul", "tessera.gemm"}:
        return {
            "kernel_id": "matmul",
            "grid": "mn_tiles",
            "block": "warpgroup",
            "measurement": "wall_clock_pending",
        }
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return {"kernel_id": "softmax", "grid": "rows", "block": "256", "measurement": "wall_clock_pending"}
    return {"kernel_id": source.removeprefix("tessera.").replace(".", "_"), "grid": "elements", "measurement": "wall_clock_pending"}


def _source_from_tile_op(op: TileOp) -> str:
    if op.op_name == "tile.mma":
        return "tessera.matmul"
    if op.op_name == "tessera.attn.msa_kv_outer_sparse":
        return "tessera.msa_sparse_attention"
    if op.op_name.startswith("tessera.attn."):
        return "tessera.flash_attn"
    if op.op_name.startswith("tile."):
        return "tessera." + op.op_name.removeprefix("tile.")
    return op.op_name


def _cpu_target_op_name(source: str) -> str:
    bare = source.removeprefix("tessera.").replace(".", "_")
    if source in {"tessera.matmul", "tessera.gemm"}:
        bare = "matmul"
    elif source in {"tessera.conv2d", "tessera.conv2d_nhwc"}:
        bare = "conv2d_nhwc"
    return f"tessera.cpu.{bare}"


def _nvidia_arch(target_kind: str) -> str:
    return {
        "nvidia_sm80": "sm_80",
        "nvidia_sm90": "sm_90a",
        "nvidia_sm100": "sm_100a",
        "nvidia_sm120": "sm_120",
    }[target_kind]


def _format_attr_dict(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "{}"
    return "{" + ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in attrs.items()) + "}"


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "none"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_attr_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(json.dumps(value, sort_keys=True))
    return json.dumps(str(value))
