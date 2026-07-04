"""Apple GPU MSL kernel synthesis + Metal runtime dispatch (the F2 emitter).

B1 split (COMPILER_REFACTOR_PLAN Workstream B, the keystone): the
Metal-specific half of the former ``fusion.py`` — ``synthesize_*_msl`` string
emitters, the ``run_*`` runtime-dispatch functions, ctypes symbol loaders, and
the measured autotune loop + corpus. It consumes the arch-agnostic region model
from :mod:`tessera.compiler.fusion_core`.

B2 (COMPILER_REFACTOR_PLAN Workstream B2): :class:`AppleMSLEmitter` is the
reference implementation of the :class:`~tessera.compiler.emit.kernel_emitter.KernelEmitter`
plugin protocol — it *wraps* the ``synthesize_*_msl`` functions below (does not
rewrite them) so a non-Apple backend reuses the whole synthesizer by implementing
the same interface. Vocab snippets are now requested through ``EpilogueOp.emit``/
``ReductionOp.emit`` (target-parametric) rather than a Metal-only field.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from tessera.compiler.emit.kernel_emitter import (
    EmitError,
    KernelEmitter,
    KernelRunner,
    KernelSource,
    SpecPolicy,
    register_emitter,
    register_runner,
)
from tessera.compiler.fusion_core import (
    AttentionRegion,
    EPILOGUE_OPS,
    FusedRegion,
    GatedMatmulRegion,
    NormChainRegion,
    POINTWISE_OPS,
    PointwiseGraphRegion,
    PointwiseReduceRegion,
    REDUCTION_OPS,
    SYNTH_GATED_MAX_H,
    SYNTH_MAX_D,
    SYNTH_MAX_N,
    SYNTH_MAX_N_TILED,
    _PW_MAX_INPUTS,
    _PW_REDUCE_KINDS,
    _is_trailing_feature,
    select_attention_lowering,
    should_fuse_region,
)

#: The target id this module emits for — passed to the target-parametric
#: ``EpilogueOp.emit`` / ``ReductionOp.emit`` vocab accessors (B2).
_MSL_TARGET = "apple_gpu"

_ENTRY = "synth_matmul_epi"


# ─────────────────────────────────────────────────────────────────────────────
# F2a — MSL synthesis
# ─────────────────────────────────────────────────────────────────────────────


#: The matmul-into-``scores[]`` inner loop, parametrized for the F5 autotuner.
#: ``broadcast`` streams one A element across the whole N row (B read
#: contiguously); ``dot`` accumulates each output column as a K-dot (B read
#: strided).  Both fill ``scores[]`` identically — the oracle proves it — but
#: have different memory access, so the fastest depends on the shape.
SYNTH_VARIANTS = ("broadcast", "dot")


def _prologue_msl(region: "FusedRegion", indent: str) -> str:
    """MSL that transforms the loaded A element ``a`` in place — each prologue op
    wrapped as ``{ float v = a; <op.msl>; a = v; }`` so the EPILOGUE_OPS bodies
    (which operate on ``v``) are reused verbatim.  Empty when no prologue."""
    return "".join(
        f"{indent}{{ float v = a; {EPILOGUE_OPS[op].emit(_MSL_TARGET)} a = v; }}\n"
        for op in region.prologue)


def _matmul_body(variant: str, prologue: str = "") -> str:
    # ``prologue`` (possibly empty) transforms the loaded A element ``a`` before
    # the multiply — ``matmul(act(A), B)``.  When empty, the bodies are the
    # original byte-identical forms so existing kernels are unperturbed.
    if variant == "broadcast":
        if not prologue:
            return """    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }"""
        return f"""    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {{
        float a = A[a_off + k];
{prologue}        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }}"""
    if variant == "dot":
        if not prologue:
            return """    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[a_off + k] * B[k * N + n];
        scores[n] = acc;
    }"""
        return f"""    for (int n = 0; n < N; ++n) {{
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {{
            float a = A[a_off + k];
{prologue}            acc += a * B[k * N + n];
        }}
        scores[n] = acc;
    }}"""
    raise ValueError(f"unknown synth variant {variant!r}")


#: Synthesizer I/O dtypes.  f16 emits a native ``half``-I/O kernel (fp32
#: accumulators throughout); f32 is the default.  bf16 has no MSL kernel — it
#: reuses the f32 kernel via host-side fp32 conversion (the 8.4.4.x convention),
#: handled in ``run_fused_region``.
SYNTH_DTYPES = ("f32", "f16")


def _io_type(dtype: str) -> str:
    if dtype == "f32":
        return "float"
    if dtype == "f16":
        return "half"
    if dtype == "bf16":
        # Native MSL `bfloat` (Metal 3.1+ / MTLDataType.bfloat, Apple6+ incl.
        # Apple7/M1 Max — see the Apple7 GPU feature-set memo). I/O in bfloat,
        # fp32 accumulators inside the kernel — no host f32 upcast.
        return "bfloat"
    raise ValueError(f"synthesizer dtype must be f32/f16/bf16, got {dtype!r}")


def synthesize_matmul_epilogue_msl(region: FusedRegion,
                                   variant: str = "broadcast",
                                   dtype: str = "f32") -> str:
    """Emit the MSL source for ``region`` — one row per thread, the matmul into a
    stack accumulator, the pointwise chain inlined, then (optionally) a terminal
    reduction (rmsnorm/softmax) over the row.  ``variant`` selects the matmul
    inner-loop schedule (see ``SYNTH_VARIANTS``); the F5 autotuner picks the
    fastest one that passes the oracle.  ``dtype`` selects the I/O type — f16
    emits ``half`` I/O with fp32 accumulators (MSL's implicit half↔float
    conversions keep the body identical); the ``scores`` accumulator is always
    fp32, so the math is bit-for-bit the same as f32."""
    io = _io_type(dtype)
    bias_param = (f"    device const {io}* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    # M4 residual: full (M,N) tensor at buffer 7, added per element after the
    # pointwise chain (the transformer x + sublayer(x)). Validated non-reduction.
    residual_param = (f"    device const {io}* residual [[buffer(7)]],\n"
                      if region.has_residual else "")
    residual_add = ("        v += float(residual[o_off + n]);\n"
                    if region.has_residual else "")
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].emit(_MSL_TARGET)}" for op in region.epilogue)
    matmul_body = _matmul_body(variant, _prologue_msl(region, "        "))

    if region.reduction is None:
        # pure pointwise (+ optional residual): one pass, write O directly.
        finalize = f"""    int o_off = row * N;
    for (int n = 0; n < N; ++n) {{
        float v = scores[n];
{pointwise}
{residual_add}        O[o_off + n] = ST(v);
    }}"""
    else:
        # pointwise modifies scores in place; then the reduction reads the whole
        # row and writes O.
        pw_pass = ""
        if region.epilogue:
            pw_pass = f"""    for (int n = 0; n < N; ++n) {{
        float v = scores[n];
{pointwise}
        scores[n] = v;
    }}
"""
        red = REDUCTION_OPS[region.reduction].emit(_MSL_TARGET).format(eps=region.eps)
        finalize = f"""    int o_off = row * N;
{pw_pass}{red}"""

    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_ENTRY}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* B   [[buffer(1)]],
    device {io}*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
{bias_param}{residual_param}    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)M) return;
    if (N > {SYNTH_MAX_N}) return;
    int row = (int)gid;
    float scores[{SYNTH_MAX_N}];
    int a_off = row * K;
{matmul_body}
{finalize}
}}
"""
_ENTRY_TILED = "synth_matmul_epi_tiled"
_TILED_THREADS = 32

#: Cooperative (tree-reduce over ``_TILED_THREADS`` partials) reduction blocks —
#: the tiled analogue of REDUCTION_OPS' serial blocks.  ``{eps}`` is substituted.
_TILED_REDUCTIONS: dict[str, str] = {
    "rmsnorm":
        "    float _local = 0.0f;\n"
        "    for (int n = lid_i; n < N; n += T) _local += tg_scores[n] * tg_scores[n];\n"
        "    tg_red[lid_i] = _local;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (int stride = T / 2; stride > 0; stride >>= 1) {{\n"
        "        if (lid_i < stride) tg_red[lid_i] += tg_red[lid_i + stride];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }}\n"
        "    float _inv = rsqrt(tg_red[0] / float(N) + {eps}f);\n"
        "    int o_off = row * N;\n"
        "    for (int n = lid_i; n < N; n += T) O[o_off + n] = ST(tg_scores[n] * _inv);\n",
    "softmax":
        "    float _lmax = -INFINITY;\n"
        "    for (int n = lid_i; n < N; n += T) _lmax = max(_lmax, tg_scores[n]);\n"
        "    tg_red[lid_i] = _lmax;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (int stride = T / 2; stride > 0; stride >>= 1) {{\n"
        "        if (lid_i < stride) tg_red[lid_i] = max(tg_red[lid_i], tg_red[lid_i + stride]);\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }}\n"
        "    float _mx = tg_red[0];\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    float _lsum = 0.0f;\n"
        "    for (int n = lid_i; n < N; n += T) {{ float e = exp(tg_scores[n] - _mx);"
        " tg_scores[n] = e; _lsum += e; }}\n"
        "    tg_red[lid_i] = _lsum;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (int stride = T / 2; stride > 0; stride >>= 1) {{\n"
        "        if (lid_i < stride) tg_red[lid_i] += tg_red[lid_i + stride];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }}\n"
        "    float _sm = tg_red[0];\n"
        "    int o_off = row * N;\n"
        "    if (_sm == 0.0f) {{ for (int n = lid_i; n < N; n += T) O[o_off + n] = ST(0.0f); }}\n"
        "    else {{ float _inv = 1.0f / _sm;"
        " for (int n = lid_i; n < N; n += T) O[o_off + n] = ST(tg_scores[n] * _inv); }}\n",
}


def synthesize_matmul_epilogue_msl_tiled(region: FusedRegion,
                                         dtype: str = "f32") -> str:
    """Emit a THREADGROUP-TILED MSL kernel for ``region`` — one row per
    threadgroup, ``_TILED_THREADS`` threads cooperating, the row of scores in
    dynamic threadgroup memory (no per-thread stack), so N can reach
    ``SYNTH_MAX_N_TILED``.  Mirrors the proven hand-written
    ``matmul_softmax_tiled_f32`` structure, generalized over the epilogue +
    reduction vocabulary.  ``dtype`` selects the I/O type (the ``tg_scores``
    accumulator stays fp32 regardless)."""
    io = _io_type(dtype)
    bias_param = (f"    device const {io}* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].emit(_MSL_TARGET)}" for op in region.epilogue)

    # M4 prologue: transform the loaded A element before the multiply. Empty form
    # keeps the original single-line accumulate byte-identical.
    if region.has_prologue:
        pro = _prologue_msl(region, "            ")
        matmul_loop = f"""    for (int n = lid_i; n < N; n += T) {{
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {{
            float a = A[a_off + k];
{pro}            s += a * B[k * N + n];
        }}
        tg_scores[n] = s;
    }}"""
    else:
        matmul_loop = """    for (int n = lid_i; n < N; n += T) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[a_off + k] * B[k * N + n];
        tg_scores[n] = s;
    }"""

    if region.reduction is None:
        # pure pointwise: cooperative epilogue writes O directly, no reduction.
        body = f"""    int o_off = row * N;
    for (int n = lid_i; n < N; n += T) {{
        float v = tg_scores[n];
{pointwise}
        O[o_off + n] = ST(v);
    }}"""
        red_scratch = ""
    else:
        pw_pass = ""
        if region.epilogue:
            pw_pass = f"""    for (int n = lid_i; n < N; n += T) {{
        float v = tg_scores[n];
{pointwise}
        tg_scores[n] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
        red = _TILED_REDUCTIONS[region.reduction].format(eps=region.eps)
        body = pw_pass + red
        red_scratch = f"    threadgroup float tg_red[{_TILED_THREADS}];\n"

    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
#define T {_TILED_THREADS}
kernel void {_ENTRY_TILED}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* B   [[buffer(1)]],
    device {io}*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
{bias_param}    threadgroup float* tg_scores [[threadgroup(0)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]])
{{
    int row = (int)tg_pos;
    if (row >= M) return;
    int lid_i = (int)lid;
{red_scratch}    int a_off = row * K;
{matmul_loop}
    threadgroup_barrier(mem_flags::mem_threadgroup);
{body}
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# F2d — cooperative-matrix synthesis (simdgroup_matrix MMA + fused epilogue)
#
# The stack/tiled kernels do the matmul as scalar fp32 FMA in the general ALU —
# measured ~12 GF/s and ~no f16 speedup (the matrix units are never touched).
# This emits a `simdgroup_matrix` matmul (f16 multiply, fp32 accumulate — the
# M-series "tensor core" path that runs ~2× for f16), stores the accumulator
# tile to threadgroup memory, then runs the pointwise epilogue fused after.
# Mirrors the proven hand-written `mtl4_matmul_sg` kernel (32×32 output tile,
# 4 simdgroups / 128 threads, double-buffered K-slabs, bounds-checked).
#
# v1: POINTWISE epilogue only.  A terminal reduction (softmax/rmsnorm) needs a
# cross-tile row reduction (a row spans ceil(N/32) threadgroups), so reduction
# regions stay on the scalar tiled path until F2d-v2.
# ─────────────────────────────────────────────────────────────────────────────

_ENTRY_COOPMAT = "synth_matmul_epi_coopmat"


def coopmat_eligible(region: FusedRegion) -> bool:
    """v1 covers pointwise-epilogue regions (no terminal reduction). Residual and
    prologue regions route to the scalar kernel (the coopmat kernel has no
    residual buffer and no per-element A-load hook for an input prologue)."""
    return (region.reduction is None and not region.has_residual
            and not region.has_prologue)


#: Coopmat tile configs: tile -> (BM, BN, SGCOLS, NR, NC, THREADS).
#: 32x32 = a 2x2 simdgroup grid (128 threads), each owning a 2x2 array of 8x8
#: accumulators.  64x64 = a 2x4 grid (256 threads), each owning a 4x2 array (8
#: accumulators) — the mtl4_matmul_sg_fast register-blocked structure, more
#: arithmetic intensity per threadgroup (~80% of MPS vs the 32x32's ~45%).
SYNTH_COOPMAT_TILES = (32, 64)
_COOPMAT_TILE_CFG: dict[int, tuple[int, int, int, int, int, int]] = {
    32: (32, 32, 2, 2, 2, 128),
    64: (64, 64, 4, 4, 2, 256),
}


def coopmat_threads(tile: int) -> int:
    return _COOPMAT_TILE_CFG[tile][5]


def synthesize_matmul_epilogue_coopmat_msl(region: FusedRegion,
                                           dtype: str = "f16",
                                           tile: int = 32) -> str:
    """Emit a cooperative-matrix (``simdgroup_matrix``) matmul + fused pointwise
    epilogue.  ``dtype`` selects the MMA input type (f16 taps the matrix units
    at ~2×; f32 is the simdgroup f32 ceiling).  ``tile`` selects the output-tile
    size (32 or 64); 64x64 register-blocks 8 accumulators per simdgroup for more
    arithmetic intensity.  The accumulator is always fp32, so the epilogue sees
    full-precision matmul results.  Staging is bounds-checked, so arbitrary
    shapes work (no alignment requirement)."""
    if region.reduction is not None:
        raise ValueError("coopmat v1 does not support reduction epilogues")
    if tile not in _COOPMAT_TILE_CFG:
        raise ValueError(f"coopmat tile must be one of {SYNTH_COOPMAT_TILES}, got {tile}")
    BM, BN, SGCOLS, NR, NC, THREADS = _COOPMAT_TILE_CFG[tile]
    io = _io_type(dtype)
    bias_param = (f"    device const {io}* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].emit(_MSL_TARGET)}" for op in region.epilogue)
    sg_in = f"simdgroup_matrix<{io}, 8, 8>"

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant constexpr int BM = {BM};
constant constexpr int BN = {BN};
constant constexpr int BK = 16;
constant constexpr int SGCOLS = {SGCOLS};
constant constexpr int NR = {NR};
constant constexpr int NC = {NC};
constant constexpr int THREADS = {THREADS};
kernel void {_ENTRY_COOPMAT}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* B   [[buffer(1)]],
    device {io}*       O   [[buffer(2)]],
    constant int&      M   [[buffer(3)]],
    constant int&      N   [[buffer(4)]],
    constant int&      K   [[buffer(5)]],
{bias_param}    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{{
    int brow = int(tg.y) * BM;
    int bcol = int(tg.x) * BN;
    threadgroup {io} As[BM * BK];
    threadgroup {io} Bs[BK * BN];
    threadgroup float Cs[BM * BN];
    int sg_row = int(sgid) / SGCOLS, sg_col = int(sgid) % SGCOLS;
    int r0 = sg_row * (NR * 8), c0 = sg_col * (NC * 8);
    simdgroup_float8x8 acc[NR][NC];
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NC; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Cooperative K-slab staging (THREADS threads, zero-padded out of range).
    // Single-buffered: double-buffering was measured a wash on Apple (no
    // cp.async).  The throughput lever is the tile size / accumulator count.
    for (int k0 = 0; k0 < K; k0 += BK) {{
        for (int e = int(tid); e < BM * BK; e += THREADS) {{
            int r = e / BK, kk = e % BK;
            int gr = brow + r, gk = k0 + kk;
            As[e] = (gr < M && gk < K) ? A[gr * K + gk] : ({io})0;
        }}
        for (int e = int(tid); e < BK * BN; e += THREADS) {{
            int kk = e / BN, c = e % BN;
            int gk = k0 + kk, gc = bcol + c;
            Bs[e] = (gk < K && gc < N) ? B[gk * N + gc] : ({io})0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {{
            {sg_in} a[NR], b[NC];
            for (int i = 0; i < NR; ++i)
                simdgroup_load(a[i], As + (r0 + i * 8) * BK + kk, BK);
            for (int j = 0; j < NC; ++j)
                simdgroup_load(b[j], Bs + kk * BN + (c0 + j * 8), BN);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NC; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store the fp32 accumulator tile to threadgroup memory, then run the
    // pointwise epilogue per element (bias[n] indexes the global column).
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NC; ++j)
            simdgroup_store(acc[i][j], Cs + (r0 + i * 8) * BN + (c0 + j * 8), BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < BM * BN; e += THREADS) {{
        int r = e / BN, c = e % BN;
        int gr = brow + r, gc = bcol + c;
        if (gr < M && gc < N) {{
            float v = Cs[e];
            int n = gc;
{pointwise}
            O[gr * N + gc] = ({io})v;
        }}
    }}
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# F2d-v2 — cooperative-matrix matmul + cross-tile REDUCTION (softmax/rmsnorm)
#
# v1 leaves reduction regions on the scalar path: a row spans ceil(N/32)
# threadgroups, so the matmul can't both run on the matrix units AND reduce in
# one kernel — unless one threadgroup owns a whole row-block.  This kernel does
# exactly that: one threadgroup (one simdgroup) computes BM rows × the FULL N via
# simdgroup MMA into threadgroup memory, applies the pointwise chain, then runs
# the row reduction (each of BM rows by one thread) and writes O.  N is capped by
# the threadgroup budget (BM * N fp32 scores); larger N stays on the scalar/
# compose paths.
# ─────────────────────────────────────────────────────────────────────────────

_ENTRY_COOPMAT_REDUCE = "synth_matmul_reduce_coopmat"
_COOPMAT_REDUCE_BM = 8
#: Static N cap for the v2 kernel (Cs[BM*N] fp32 in threadgroup memory).
SYNTH_COOPMAT_REDUCE_MAX_N = 512

_COOPMAT_REDUCTIONS: dict[str, str] = {
    "rmsnorm":
        "        float _ss = 0.0f;\n"
        "        for (int c = 0; c < N; ++c) _ss += Cs[rr * N + c] * Cs[rr * N + c];\n"
        "        float _invr = rsqrt(_ss / float(N) + {eps}f);\n"
        "        for (int c = 0; c < N; ++c)\n"
        "            O[(brow + rr) * N + c] = ({io})(Cs[rr * N + c] * _invr);\n",
    "softmax":
        "        float _mx = -INFINITY;\n"
        "        for (int c = 0; c < N; ++c) _mx = max(_mx, Cs[rr * N + c]);\n"
        "        float _sm = 0.0f;\n"
        "        for (int c = 0; c < N; ++c) {{ float e = exp(Cs[rr * N + c] - _mx);"
        " Cs[rr * N + c] = e; _sm += e; }}\n"
        "        float _inv = (_sm > 0.0f) ? (1.0f / _sm) : 0.0f;\n"
        "        for (int c = 0; c < N; ++c)\n"
        "            O[(brow + rr) * N + c] = ({io})(Cs[rr * N + c] * _inv);\n",
}


def coopmat_reduce_eligible(region: FusedRegion, N: int) -> bool:
    """v2 covers a terminal reduction (softmax/rmsnorm) when the row fits the
    threadgroup-memory cap and N is a multiple of 8 (the simdgroup 8x8 stores)."""
    return (region.reduction in _COOPMAT_REDUCTIONS
            and 0 < N <= SYNTH_COOPMAT_REDUCE_MAX_N
            and N % 8 == 0)


def synthesize_matmul_reduction_coopmat_msl(region: FusedRegion,
                                            dtype: str = "f16") -> str:
    """Emit a cooperative-matrix matmul + fused row reduction (softmax/rmsnorm).
    One threadgroup (one simdgroup, 32 threads) computes ``BM`` rows × the full N
    via ``simdgroup_matrix`` MMA into ``Cs`` (fp32, threadgroup), applies the
    pointwise chain, then each of the BM rows is reduced by one thread."""
    if region.reduction not in _COOPMAT_REDUCTIONS:
        raise ValueError(f"coopmat reduce: unsupported reduction {region.reduction!r}")
    io = _io_type(dtype)
    bias_param = (f"    device const {io}* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].emit(_MSL_TARGET)}" for op in region.epilogue)
    reduce_block = _COOPMAT_REDUCTIONS[region.reduction].format(io=io, eps=region.eps)
    sg_in = f"simdgroup_matrix<{io}, 8, 8>"
    bm = _COOPMAT_REDUCE_BM
    maxn = SYNTH_COOPMAT_REDUCE_MAX_N

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant constexpr int BM = {bm};
constant constexpr int BK = 16;
constant constexpr int MAXN = {maxn};
kernel void {_ENTRY_COOPMAT_REDUCE}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* B   [[buffer(1)]],
    device {io}*       O   [[buffer(2)]],
    constant int&      M   [[buffer(3)]],
    constant int&      N   [[buffer(4)]],
    constant int&      K   [[buffer(5)]],
{bias_param}    uint  tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{{
    int brow = int(tg) * BM;
    if (brow >= M || N > MAXN) return;
    threadgroup {io} As[BM * BK];
    threadgroup {io} Bs[BK * 32];
    threadgroup float Cs[BM * MAXN];

    // Compute the BM × N score strip in 32-col blocks, holding 4 simdgroup
    // accumulators per block (v2.1) — the A row-block is staged once per K-slab
    // and reused across 4 output tiles, ~4x less A traffic than 8-col tiles.
    // (N is a multiple of 8 by eligibility, so each 8x8 store stays in bounds.)
    for (int jb = 0; jb < N; jb += 32) {{
        simdgroup_float8x8 acc[4];
        for (int j = 0; j < 4; ++j)
            acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k0 = 0; k0 < K; k0 += BK) {{
            for (int e = int(tid); e < BM * BK; e += 32) {{
                int r = e / BK, kk = e % BK;
                int gr = brow + r, gk = k0 + kk;
                As[e] = (gr < M && gk < K) ? A[gr * K + gk] : ({io})0;
            }}
            for (int e = int(tid); e < BK * 32; e += 32) {{
                int kk = e / 32, c = e % 32;
                int gk = k0 + kk, gc = jb + c;
                Bs[e] = (gk < K && gc < N) ? B[gk * N + gc] : ({io})0;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < BK; kk += 8) {{
                {sg_in} a;
                simdgroup_load(a, As + kk, BK);
                for (int j = 0; j < 4; ++j) {{
                    {sg_in} b;
                    simdgroup_load(b, Bs + kk * 32 + j * 8, 32);
                    simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (int j = 0; j < 4; ++j) {{
            int col = jb + j * 8;
            if (col < N) simdgroup_store(acc[j], Cs + col, N);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Pointwise chain in place (bias[n] indexes the column).
    for (int e = int(tid); e < BM * N; e += 32) {{
        int c = e % N;
        float v = Cs[e];
        int n = c;
{pointwise}
        Cs[e] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row reduction: each of the BM rows reduced by one thread.
    for (int rr = int(tid); rr < BM; rr += 32) {{
        if (brow + rr < M) {{
{reduce_block}        }}
    }}
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# F2a — GPU dispatch (compile + run the synthesized MSL)
# ─────────────────────────────────────────────────────────────────────────────


def _synth_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_matmul_epilogue_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int32]  # residual, has_residual
    sym.restype = ctypes.c_int32
    return sym


def _synth_tiled_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_matmul_epilogue_tiled_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _synth_coopmat_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_matmul_epilogue_coopmat", None)
    if sym is None:
        return None
    vp = ctypes.c_void_p
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, vp, vp, vp, vp,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def coopmat_tile_for(M: int, N: int, K: int,
                     region: FusedRegion | None = None) -> int:
    """Pick the coopmat output-tile size for (M,N,K).  An autotuned decision in
    ``_COOPMAT_TILE_CORPUS`` (per region-class × shape-bucket, from
    ``autotune_coopmat_tile``) wins when present; otherwise the shape heuristic.

    Heuristic (measured on M1 Max): the 64x64 register-blocked kernel (8
    simdgroups, 256 threads, 8 accumulators) wins for large-N matmuls (1.42x at
    2048x1024x512, where it also beats MPS-compose 1.18x) and ties the 32x32 for
    small/narrow ones — so gate 64 on a wide, deep matmul."""
    if region is not None:
        tuned = _COOPMAT_TILE_CORPUS.get(_corpus_key(region, M, N, K))
        if tuned is not None:
            return tuned
    if M >= 128 and N >= 384 and K >= 128:
        return 64
    return 32


def run_fused_region_coopmat(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                             bias: np.ndarray | None = None, tile: int | None = None
                             ) -> tuple[np.ndarray, str]:
    """Run the region via the COOPERATIVE-MATRIX synthesized kernel (F2d) —
    ``simdgroup_matrix`` MMA (f16 multiply / fp32 accumulate) with the pointwise
    epilogue fused after.  ``tile`` (32/64) selects the output-tile size; when
    ``None`` it is chosen by shape (``coopmat_tile_for``).  f16/f32 inputs only;
    reduction regions and other dtypes fall back to the scalar
    ``run_fused_region``.  Returns ``(output, execution)``."""
    if not coopmat_eligible(region):
        return run_fused_region(region, A, B, bias)
    in_dtype = np.asarray(A).dtype
    bf16 = _bf16_dtype()
    np_dt: Any
    if in_dtype == np.float16:
        np_dt, elem_size, dtype = np.float16, 2, "f16"
    elif in_dtype == np.float32:
        np_dt, elem_size, dtype = np.float32, 4, "f32"
    elif bf16 is not None and in_dtype == bf16:
        # Native bf16 simdgroup_matrix MMA (Apple7 MTLDataType.bfloat). The
        # synthesized `simdgroup_matrix<bfloat,8,8>` taps the matrix units like
        # f16 (fp32 accumulate); the C ABI is dtype-generic (void* + elem_size=2).
        np_dt, elem_size, dtype = bf16, 2, "bf16"
    else:
        return run_fused_region(region, A, B, bias)   # other → scalar path
    A = np.ascontiguousarray(A, np_dt)
    B = np.ascontiguousarray(B, np_dt)
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"matmul shape mismatch: A {A.shape}, B {B.shape}")
    if tile is None:
        tile = coopmat_tile_for(M, N, K, region)
    bias_arr = None
    if region.has_bias:
        if bias is None:
            raise ValueError("region needs a bias")
        bias_arr = np.ascontiguousarray(bias, np_dt).reshape(N)

    sym = _synth_coopmat_symbol()
    if sym is not None:
        source = synthesize_matmul_epilogue_coopmat_msl(
            region, dtype=dtype, tile=tile).encode("utf-8")
        out = np.zeros((M, N), np_dt)
        vp = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        bias_ptr = vp(bias_arr) if bias_arr is not None else None
        rc = sym(source, _ENTRY_COOPMAT.encode("utf-8"), vp(A), vp(B), bias_ptr,
                 vp(out), M, N, K, 1 if region.has_bias else 0, elem_size, tile)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, B, bias_arr).astype(np_dt), "reference"


def _synth_coopmat_reduce_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_matmul_reduce_coopmat", None)
    if sym is None:
        return None
    vp = ctypes.c_void_p
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, vp, vp, vp, vp,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_fused_region_coopmat_reduce(region: FusedRegion, A: np.ndarray,
                                    B: np.ndarray, bias: np.ndarray | None = None
                                    ) -> tuple[np.ndarray, str]:
    """Run a reduction region (matmul -> pointwise -> softmax/rmsnorm) via the
    cooperative-matrix kernel (F2d-v2): the matmul runs on the matrix units and
    the row reduction is fused in one kernel.  f16/f32 only, N within the
    threadgroup cap; everything else falls back to the scalar
    ``run_fused_region``.  Returns ``(output, execution)``."""
    in_dtype = np.asarray(A).dtype
    N = np.asarray(B).shape[1]
    if not (coopmat_reduce_eligible(region, N)
            and in_dtype in (np.float16, np.float32)):
        return run_fused_region(region, A, B, bias)
    np_dt, elem_size, dtype = ((np.float16, 2, "f16") if in_dtype == np.float16
                               else (np.float32, 4, "f32"))
    A = np.ascontiguousarray(A, np_dt)
    B = np.ascontiguousarray(B, np_dt)
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"matmul shape mismatch: A {A.shape}, B {B.shape}")
    bias_arr = None
    if region.has_bias:
        if bias is None:
            raise ValueError("region needs a bias")
        bias_arr = np.ascontiguousarray(bias, np_dt).reshape(N)

    sym = _synth_coopmat_reduce_symbol()
    if sym is not None:
        source = synthesize_matmul_reduction_coopmat_msl(
            region, dtype=dtype).encode("utf-8")
        out = np.zeros((M, N), np_dt)
        vp = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        bias_ptr = vp(bias_arr) if bias_arr is not None else None
        rc = sym(source, _ENTRY_COOPMAT_REDUCE.encode("utf-8"), vp(A), vp(B),
                 bias_ptr, vp(out), M, N, K, 1 if region.has_bias else 0, elem_size)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, B, bias_arr).astype(np_dt), "reference"


def _bf16_dtype() -> Any:
    try:
        import ml_dtypes
        return ml_dtypes.bfloat16
    except Exception:                          # noqa: BLE001 - optional dependency
        return None


def _synth_f16_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_matmul_epilogue_f16", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, u16, u16, u16, u16,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, u16, ctypes.c_int32]  # is_tiled, residual, has_residual
    sym.restype = ctypes.c_int32
    return sym


def _run_fused_region_bf16(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                           bias: np.ndarray | None,
                           variant: str,
                           residual: np.ndarray | None = None
                           ) -> tuple[np.ndarray | None, str]:
    """Native bf16 path — emit `bfloat`-typed MSL (Apple7 MTLDataType.bfloat)
    and reuse the f16 synth symbol's uint16 ABI (bf16 is 2-byte raw storage; the
    MSL `bfloat` element type gives the bits meaning). fp32 accumulators inside
    the kernel — no host f32 upcast. Returns ``(out, "metal_runtime")`` on GPU
    success, else ``(None, "fallback")`` so the caller can f32-emulate (e.g. if
    the runtime's MSL version predates `bfloat`)."""
    bf16 = _bf16_dtype()
    if bf16 is None:
        return None, "fallback"
    sym = _synth_f16_symbol()                  # same uint16 ABI as f16
    if sym is None:
        return None, "fallback"
    A = np.ascontiguousarray(A, bf16)
    B = np.ascontiguousarray(B, bf16)
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"matmul shape mismatch: A {A.shape}, B {B.shape}")
    # The tiled kernel has no residual buffer → residual stays on the stack path.
    n_cap = SYNTH_MAX_N if region.has_residual else SYNTH_MAX_N_TILED
    if N > n_cap:
        return None, "fallback"
    bias_arr = None
    has_bias = 1 if region.has_bias else 0
    has_residual = 1 if region.has_residual else 0
    u16p = lambda a: a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    bias_ptr = None
    if region.has_bias:
        if bias is None:
            raise ValueError("region needs a bias")
        bias_arr = np.ascontiguousarray(bias, bf16).reshape(N)
        bias_ptr = u16p(bias_arr)
    res_arr = None
    res_ptr = None
    if region.has_residual:
        if residual is None:
            raise ValueError("region needs a residual")
        res_arr = np.ascontiguousarray(residual, bf16).reshape(M, N)
        res_ptr = u16p(res_arr)
    is_tiled = 0 if N <= SYNTH_MAX_N else 1
    if is_tiled:
        source = synthesize_matmul_epilogue_msl_tiled(region, dtype="bf16")
        entry = _ENTRY_TILED
    else:
        source = synthesize_matmul_epilogue_msl(region, variant, dtype="bf16")
        entry = _ENTRY
    out = np.zeros((M, N), bf16)
    rc = sym(source.encode("utf-8"), entry.encode("utf-8"), u16p(A), u16p(B),
             bias_ptr, u16p(out), M, N, K, has_bias, is_tiled,
             res_ptr, has_residual)
    if rc == 1:
        return out, "metal_runtime"
    return None, "fallback"


def _run_fused_region_f16(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                          bias: np.ndarray | None,
                          variant: str,
                          residual: np.ndarray | None = None
                          ) -> tuple[np.ndarray, str]:
    """f16 path — native ``half``-I/O kernel (fp32 accumulators), stack for
    N<=1024 and threadgroup-tiled for N<=8192.  Reference (f32 math, cast to
    f16) when Metal/the symbol is unavailable."""
    A = np.ascontiguousarray(A, np.float16)
    B = np.ascontiguousarray(B, np.float16)
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"matmul shape mismatch: A {A.shape}, B {B.shape}")
    bias_arr = None
    if region.has_bias:
        if bias is None:
            raise ValueError("region needs a bias")
        bias_arr = np.ascontiguousarray(bias, np.float16).reshape(N)
    res_arr = None
    if region.has_residual:
        if residual is None:
            raise ValueError("region needs a residual")
        res_arr = np.ascontiguousarray(residual, np.float16).reshape(M, N)
    u16p = lambda a: a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    bias_ptr = u16p(bias_arr) if bias_arr is not None else None
    res_ptr = u16p(res_arr) if res_arr is not None else None
    has_bias = 1 if region.has_bias else 0
    has_residual = 1 if region.has_residual else 0

    sym = _synth_f16_symbol()
    # The tiled kernel has no residual buffer → residual stays on the stack path.
    n_cap = SYNTH_MAX_N if region.has_residual else SYNTH_MAX_N_TILED
    if sym is not None and N <= n_cap:
        is_tiled = 0 if N <= SYNTH_MAX_N else 1
        if is_tiled:
            source = synthesize_matmul_epilogue_msl_tiled(region, dtype="f16")
            entry = _ENTRY_TILED
        else:
            source = synthesize_matmul_epilogue_msl(region, variant, dtype="f16")
            entry = _ENTRY
        out = np.zeros((M, N), np.float16)
        rc = sym(source.encode("utf-8"), entry.encode("utf-8"), u16p(A), u16p(B),
                 bias_ptr, u16p(out), M, N, K, has_bias, is_tiled,
                 res_ptr, has_residual)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, B, bias_arr, res_arr).astype(np.float16), "reference"


def run_fused_region(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                     bias: np.ndarray | None = None,
                     variant: str = "broadcast",
                     residual: np.ndarray | None = None
                     ) -> tuple[np.ndarray, str]:
    """Run the region as ONE synthesized fused kernel on Metal.  Returns
    ``(output, execution)`` where execution is ``"metal_runtime"`` (the
    synthesized kernel ran) or ``"reference"`` (numpy fallback — no Metal, N too
    large, or compile/dispatch failed).  Either way the numbers are correct.
    ``variant`` selects the matmul inner-loop schedule (F5 autotuner).  The
    output dtype follows the input: f16 runs a native half kernel; bf16 (no MSL
    type) converts to f32, runs, and converts back."""
    in_dtype = np.asarray(A).dtype
    bf16 = _bf16_dtype()
    # F2d — pointwise regions run on the matrix units (simdgroup_matrix MMA),
    # ~55-98x the scalar kernel and capturing the f16 throughput. Reduction
    # regions stay on the scalar stack/tiled path (cross-tile row reduce is v2).
    _is_bf16 = bf16 is not None and in_dtype == bf16
    if coopmat_eligible(region) and (in_dtype in (np.float16, np.float32)
                                     or _is_bf16):
        # Pointwise regions run on the matrix units for f16/f32 AND bf16
        # (simdgroup_matrix<bfloat> — M2). A reduction region or an MSL-bfloat
        # miss falls through to the scalar bf16/f32 path below.
        out, ex = run_fused_region_coopmat(region, A, B, bias)
        if ex == "metal_runtime":
            return out, ex
    if _is_bf16:
        # Native bfloat scalar kernel next (Apple7 MTLDataType.bfloat); fall back
        # to f32 emulation only if the runtime's MSL `bfloat` is unavailable.
        out_bf, ex_bf = _run_fused_region_bf16(region, A, B, bias, variant,
                                               residual)
        if ex_bf == "metal_runtime" and out_bf is not None:
            return out_bf, ex_bf
        out32, ex32 = run_fused_region(
            region, np.asarray(A, np.float32), np.asarray(B, np.float32),
            None if bias is None else np.asarray(bias, np.float32), variant,
            None if residual is None else np.asarray(residual, np.float32))
        return out32.astype(bf16), ex32
    if in_dtype == np.float16:
        return _run_fused_region_f16(region, A, B, bias, variant, residual)

    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError(f"matmul shape mismatch: A {A.shape}, B {B.shape}")
    bias_arr = None
    if region.has_bias:
        if bias is None:
            raise ValueError("region needs a bias")
        bias_arr = np.ascontiguousarray(bias, np.float32).reshape(N)
    res_arr = None
    if region.has_residual:
        if residual is None:
            raise ValueError("region needs a residual")
        res_arr = np.ascontiguousarray(residual, np.float32).reshape(M, N)

    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = fp(bias_arr) if bias_arr is not None else None
    res_ptr = fp(res_arr) if res_arr is not None else None
    has_bias = 1 if region.has_bias else 0
    has_residual = 1 if region.has_residual else 0

    sym = _synth_symbol()
    if sym is not None and N <= SYNTH_MAX_N:
        source = synthesize_matmul_epilogue_msl(region, variant).encode("utf-8")
        out = np.zeros((M, N), np.float32)
        rc = sym(source, _ENTRY.encode("utf-8"), fp(A), fp(B), bias_ptr, fp(out),
                 M, N, K, has_bias, res_ptr, has_residual)
        if rc == 1:
            return out, "metal_runtime"

    # Large N (over the per-thread stack cap): the threadgroup-tiled kernel keeps
    # the score row in dynamic threadgroup memory, lifting the bound to ~8192.
    # The tiled kernel has no residual buffer, so residual regions skip it.
    tiled = _synth_tiled_symbol()
    if (tiled is not None and not region.has_residual
            and SYNTH_MAX_N < N <= SYNTH_MAX_N_TILED):
        source = synthesize_matmul_epilogue_msl_tiled(region).encode("utf-8")
        out = np.zeros((M, N), np.float32)
        rc = tiled(source, _ENTRY_TILED.encode("utf-8"), fp(A), fp(B), bias_ptr,
                   fp(out), M, N, K, has_bias)
        if rc == 1:
            return out, "metal_runtime"

    return region.reference(A, B, bias_arr, res_arr), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# M2 — norm_chain synthesis (NON-matmul-rooted): elementwise pre-ops -> norm.
# The pre-norm transformer pattern `normed = rmsnorm(x + residual)` fused into
# ONE kernel — one row per thread, the row materialized from the input (+ an
# optional residual add) into a stack accumulator, then the row reduction
# (rmsnorm/layer_norm, reused verbatim from REDUCTION_OPS). Eliminates the
# host round-trip between the residual add and the norm. See
# docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M2).
# ─────────────────────────────────────────────────────────────────────────────

_NORM_CHAIN_ENTRY = "synth_norm_chain"


def synthesize_norm_chain_msl(region: NormChainRegion, dtype: str = "f32") -> str:
    """Emit MSL for a norm_chain region — one row per thread, materialize the
    row (+ optional residual) into a fp32 ``scores`` accumulator, then the
    reused reduction block. ``dtype`` selects I/O (float/half/bfloat); the
    accumulator + math are always fp32."""
    io = _io_type(dtype)
    residual_param = (f"    device const {io}* residual [[buffer(4)]],\n"
                      if region.add_residual else "")
    residual_add = ("        v += float(residual[o_off + n]);\n"
                    if region.add_residual else "")
    # Post-norm affine: γ (per-feature weight) at buffer 5, β (bias) at buffer 6.
    gamma_param = (f"    device const {io}* gamma [[buffer(5)]],\n"
                   if region.weight else "")
    beta_param = (f"    device const {io}* beta [[buffer(6)]],\n"
                  if region.bias else "")
    affine = ""
    if region.weight or region.bias:
        ops = []
        if region.weight:
            ops.append("        o *= float(gamma[n]);")
        if region.bias:
            ops.append("        o += float(beta[n]);")
        body = "\n".join(ops)
        # The reduction wrote the normalized value to O; apply the affine in a
        # second pass (fp32 math, store back to the storage dtype).
        affine = f"""
    for (int n = 0; n < N; ++n) {{
        float o = float(O[o_off + n]);
{body}
        O[o_off + n] = ST(o);
    }}"""
    red = REDUCTION_OPS[region.norm].emit(_MSL_TARGET).format(eps=region.eps)
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_NORM_CHAIN_ENTRY}(
    device const {io}* X   [[buffer(0)]],
    device {io}*       O   [[buffer(1)]],
    constant int&       M   [[buffer(2)]],
    constant int&       N   [[buffer(3)]],
{residual_param}{gamma_param}{beta_param}    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)M) return;
    if (N > {SYNTH_MAX_N}) return;
    int row = (int)gid;
    int o_off = row * N;
    float scores[{SYNTH_MAX_N}];
    for (int n = 0; n < N; ++n) {{
        float v = float(X[o_off + n]);
{residual_add}        scores[n] = v;
    }}
{red}{affine}
}}
"""


def _synth_norm_chain_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_norm_chain_f32", None)
    if sym is None:
        return None
    fptr = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, fptr, fptr, fptr,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    fptr, fptr, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _synth_norm_chain_f16_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_norm_chain_f16", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, u16, u16, u16,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    u16, u16, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_norm_chain_region(region: NormChainRegion, X: np.ndarray,
                          residual: np.ndarray | None = None,
                          gamma: np.ndarray | None = None,
                          beta: np.ndarray | None = None
                          ) -> tuple[np.ndarray, str]:
    """Run a norm_chain region as ONE synthesized Metal kernel. f32 / native f16 /
    native bf16 I/O (Apple7 `bfloat`), fp32 math throughout; the post-norm affine
    (``gamma``/``beta``, per-feature) fuses in. Returns ``(out, "metal_runtime")``
    on GPU success, else ``(out, "reference")`` (numpy — no Metal / N too large /
    compile-or-dispatch failed). Either way correct."""
    bf16 = _bf16_dtype()
    in_dtype = np.asarray(X).dtype
    elem: str
    dt: Any  # numpy or ml_dtypes scalar type (f16 / bf16 / f32)
    if in_dtype == np.float16:
        elem, dt = "f16", np.float16
    elif bf16 is not None and in_dtype == bf16:
        elem, dt = "bf16", bf16
    else:
        elem, dt = "f32", np.float32

    def _ref():
        return region.reference(X, residual, gamma, beta).astype(dt)

    X = np.ascontiguousarray(X, dt)
    if X.ndim != 2:
        return _ref(), "reference"
    M, N = X.shape
    res_arr = None
    if region.add_residual:
        if residual is None:
            raise ValueError("region needs a residual")
        res_arr = np.ascontiguousarray(residual, dt)
        if res_arr.shape != X.shape:
            return _ref(), "reference"
    g_arr = (np.ascontiguousarray(np.asarray(gamma, dt).reshape(-1))
             if region.weight else None)
    b_arr = (np.ascontiguousarray(np.asarray(beta, dt).reshape(-1))
             if region.bias else None)
    if (region.weight and (g_arr is None or g_arr.shape != (N,))) or \
       (region.bias and (b_arr is None or b_arr.shape != (N,))):
        return _ref(), "reference"

    has_res = 1 if region.add_residual else 0
    has_w = 1 if region.weight else 0
    has_b = 1 if region.bias else 0
    src = synthesize_norm_chain_msl(region, dtype=elem).encode("utf-8")
    entry = _NORM_CHAIN_ENTRY.encode("utf-8")
    if elem == "f32":
        sym = _synth_norm_chain_symbol()
        if sym is not None and N <= SYNTH_MAX_N:
            fp = lambda a: (a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if a is not None else None)
            out = np.zeros((M, N), np.float32)
            rc = sym(src, entry, fp(X), fp(res_arr), fp(out), M, N, has_res,
                     fp(g_arr), fp(b_arr), has_w, has_b)
            if rc == 1:
                return out, "metal_runtime"
    else:
        sym = _synth_norm_chain_f16_symbol()
        if sym is not None and N <= SYNTH_MAX_N:
            u16p = lambda a: (a.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16)) if a is not None else None)
            out = np.zeros((M, N), dt)
            rc = sym(src, entry, u16p(X), u16p(res_arr), u16p(out), M, N, has_res,
                     u16p(g_arr), u16p(b_arr), has_w, has_b)
            if rc == 1:
                return out, "metal_runtime"
    return _ref(), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# M4 — whole-graph pointwise MSL emitter (the GPU `tessera_jit` foundation).
# An arbitrary connected DAG of elementwise ops compiles to ONE Metal kernel run
# in a single dispatch (one thread per element), instead of N separate MPSGraph
# dispatches with host round-trips between them. The GPU analogue of the CPU
# `run_graph_ops` lane. First cut: same-shape (no broadcast) pointwise ops; a
# broadcast operand or a non-pointwise op bounds the region. See
# docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M4).
# ─────────────────────────────────────────────────────────────────────────────

_PW_ENTRY = "synth_pointwise"


def synthesize_pointwise_graph_msl(region: PointwiseGraphRegion,
                                   dtype: str = "f32",
                                   broadcast: tuple[bool, ...] | None = None
                                   ) -> str:
    """Emit ONE MSL kernel computing the whole pointwise DAG per element. Each
    external input is a `device const <io>*` buffer (indices 0..k-1); the output
    is buffer k. fp32 temps throughout; the store goes through `ST(...)`.

    ``broadcast`` (per-input, aligned with ``region.inputs``) marks per-feature
    operands (bias/scale): a broadcast input is indexed ``[gid % C]`` (C = the
    last-dim width, buffer k+2) instead of ``[gid]``, so a length-``cols`` vector
    fuses in place. ``None`` ⇒ all full (the original same-shape behavior)."""
    io = _io_type(dtype)
    n_in = len(region.inputs)
    bc = tuple(broadcast) if broadcast is not None else (False,) * n_in
    params = "".join(
        f"    device const {io}* in{i} [[buffer({i})]],\n"
        for i in range(n_in))
    out_buf = n_in
    # value-id → MSL temp expression: broadcast inputs index [gid % C], full [gid].
    temp: dict[str, str] = {
        v: f"float(in{i}[{'gid % (uint)C' if bc[i] else 'gid'}])"
        for i, v in enumerate(region.inputs)}
    lines = []
    for k, (key, ins, out) in enumerate(region.ops):
        _arity, expr, _ref = POINTWISE_OPS[key]
        operands = [temp[i] for i in ins]
        lines.append(f"    float t{k} = {expr.format(*operands)};")
        temp[out] = f"t{k}"
    body = "\n".join(lines)
    # Declare the broadcast-modulus buffer only when some input needs it.
    cols_param = (f"    constant int&       C   [[buffer({out_buf + 2})]],\n"
                  if any(bc) else "")
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_PW_ENTRY}(
{params}    device {io}*       O   [[buffer({out_buf})]],
    constant int&       N   [[buffer({out_buf + 1})]],
{cols_param}    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)N) return;
{body}
    O[gid] = ST({temp[region.output]});
}}
"""


def _synth_pointwise_symbol(dtype: str) -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    name = ("tessera_apple_gpu_synth_pointwise_f32" if dtype == "f32"
            else "tessera_apple_gpu_synth_pointwise_f16")
    sym = getattr(rt, name, None)
    if sym is None:
        return None
    vpp = ctypes.POINTER(ctypes.c_void_p)
    ip = ctypes.POINTER(ctypes.c_int32)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, vpp, ip, ctypes.c_int32,
                    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_pointwise_graph(region: PointwiseGraphRegion, arrays: list[np.ndarray]
                        ) -> tuple[np.ndarray, str]:
    """Run the pointwise DAG as ONE Metal kernel. f32/f16 inputs; per-feature
    broadcast operands (bias/scale, shape ``(cols,)``/``(1,cols)``) fuse in place
    via ``gid % cols`` indexing (M4). Returns ``(out, "metal_runtime")`` on GPU
    success, else the numpy reference."""
    in_dtype = np.asarray(arrays[0]).dtype
    elem = "f16" if in_dtype == np.float16 else "f32"
    npdt = np.float16 if elem == "f16" else np.float32
    arrs = [np.ascontiguousarray(a, npdt) for a in arrays]
    try:
        out_shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:                          # incompatible shapes
        return region.reference(*arrays).astype(npdt), "reference"
    n = int(np.prod(out_shape)) if out_shape else 1
    cols = int(out_shape[-1]) if out_shape else 1

    # Classify each input: full (index [gid]) or per-feature broadcast ([gid%C]).
    # Anything else (per-row / internal broadcast) bails to the numpy reference.
    bc: list[bool] = []
    counts: list[int] = []
    for a in arrs:
        if a.shape == out_shape or a.size == n:
            bc.append(False)
            counts.append(int(a.size))
        elif a.size == cols and _is_trailing_feature(a.shape, out_shape):
            bc.append(True)
            counts.append(cols)
        else:
            return region.reference(*arrays).astype(npdt), "reference"

    sym = _synth_pointwise_symbol(elem)
    if sym is not None and len(arrs) <= _PW_MAX_INPUTS:
        in_ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        count_arr = (ctypes.c_int32 * len(arrs))(*counts)
        out = np.zeros(out_shape, npdt)
        rc = sym(synthesize_pointwise_graph_msl(
                     region, elem, tuple(bc)).encode("utf-8"),
                 _PW_ENTRY.encode("utf-8"),
                 ctypes.cast(in_ptrs, ctypes.POINTER(ctypes.c_void_p)),
                 ctypes.cast(count_arr, ctypes.POINTER(ctypes.c_int32)),
                 len(arrs), out.ctypes.data, n, cols)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(*arrays).astype(npdt), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# M5 — pointwise -> plain row-reduction fusion (sum/mean/amax/amin).
#
# A pointwise chain feeding a plain reduction over the last axis — sum(x*x) (L2²),
# mean(abs(x)) (L1), amax(exp(x)) — previously took TWO Metal kernels (the
# pointwise emitter + an MPSGraph reduce, with an intermediate DRAM round-trip).
# This collapses them into ONE: a thread per output row computes the pointwise
# chain per element and accumulates the reduction in a register. Output is the
# row-reduced shape (last axis dropped). Distinct from REDUCTION_OPS, which are
# shape-*preserving* normalizations (rmsnorm/softmax/layer_norm).
# ─────────────────────────────────────────────────────────────────────────────

_PW_REDUCE_ENTRY = "synth_pw_reduce"


def synthesize_pointwise_reduce_msl(region: PointwiseReduceRegion,
                                    dtype: str = "f32") -> str:
    """Emit ONE MSL kernel: a thread per row computes the pointwise chain per
    element and reduces over the row. fp32 accumulator; ``ST(...)`` store."""
    io = _io_type(dtype)
    n_in = len(region.inputs)
    params = "".join(
        f"    device const {io}* in{i} [[buffer({i})]],\n"
        for i in range(n_in))
    out_buf = n_in
    init, acc_expr, _fn = _PW_REDUCE_KINDS[region.reduce]
    temp: dict[str, str] = {v: f"float(in{i}[idx])"
                            for i, v in enumerate(region.inputs)}
    lines = []
    for k, (key, ins, out) in enumerate(region.ops):
        _arity, expr, _ref = POINTWISE_OPS[key]
        lines.append(f"        float t{k} = {expr.format(*[temp[i] for i in ins])};")
        temp[out] = f"t{k}"
    body = "\n".join(lines)
    finalize = "    acc /= float(cols);\n" if region.reduce == "mean" else ""
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_PW_REDUCE_ENTRY}(
{params}    device {io}*       O   [[buffer({out_buf})]],
    constant int&       rows [[buffer({out_buf + 1})]],
    constant int&       cols [[buffer({out_buf + 2})]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)rows) return;
    int base = (int)gid * cols;
    float acc = {init};
    for (int c = 0; c < cols; ++c) {{
        int idx = base + c;
{body}
        float v = {temp[region.output]};
        acc = {acc_expr};
    }}
{finalize}    O[gid] = ST(acc);
}}
"""


def _synth_pointwise_reduce_symbol(dtype: str) -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    name = ("tessera_apple_gpu_synth_pointwise_reduce_f32" if dtype == "f32"
            else "tessera_apple_gpu_synth_pointwise_reduce_f16")
    sym = getattr(rt, name, None)
    if sym is None:
        return None
    vpp = ctypes.POINTER(ctypes.c_void_p)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, vpp, ctypes.c_int32,
                    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_pointwise_reduce(region: PointwiseReduceRegion,
                         arrays: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """Run ``reduce(pointwise(inputs))`` as ONE Metal kernel. Inputs share one
    shape (rows = prod(shape[:-1]), cols = shape[-1]); output drops the last
    axis. Returns ``(out, "metal_runtime")`` on GPU success, else the reference."""
    in_dtype = np.asarray(arrays[0]).dtype
    elem = "f16" if in_dtype == np.float16 else "f32"
    npdt = np.float16 if elem == "f16" else np.float32
    arrs = [np.ascontiguousarray(a, npdt) for a in arrays]
    shape = arrs[0].shape
    if not shape or any(a.shape != shape for a in arrs):
        return region.reference(*arrays).astype(npdt), "reference"
    cols = int(shape[-1])
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    out_shape = shape[:-1]
    sym = _synth_pointwise_reduce_symbol(elem)
    if sym is not None and len(arrs) <= _PW_MAX_INPUTS:
        in_ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        out = np.zeros(out_shape if out_shape else (1,), npdt)
        rc = sym(synthesize_pointwise_reduce_msl(region, elem).encode("utf-8"),
                 _PW_REDUCE_ENTRY.encode("utf-8"),
                 ctypes.cast(in_ptrs, ctypes.POINTER(ctypes.c_void_p)),
                 len(arrs), out.ctypes.data, rows, cols)
        if rc == 1:
            return out.reshape(out_shape), "metal_runtime"
    return region.reference(*arrays).astype(npdt), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# F2c — attention synthesis (matmul -> softmax -> matmul)
# ─────────────────────────────────────────────────────────────────────────────

_ATTN_ENTRY = "synth_attention"
_ATTN_ONLINE_ENTRY = "synth_attention_online"


def synthesize_attention_msl(region: AttentionRegion = AttentionRegion(),
                             dtype: str = "f32") -> str:
    """Emit the MSL source for a fused attention block — one query row per thread.

    The source is shape- *and* scale/causal-independent (those are runtime
    buffers), so one cached pipeline serves every attention call. ``dtype``
    selects the I/O type (``float``/``half``/``bfloat``); reads cast to float,
    accumulators + softmax are fp32 throughout, the O-write goes through
    ``ST(...)`` (bfloat rejects implicit float→bfloat assignment)."""
    io = _io_type(dtype)
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_ATTN_ENTRY}(
    device const {io}* Q   [[buffer(0)]],
    device const {io}* K   [[buffer(1)]],
    device const {io}* V   [[buffer(2)]],
    device {io}*       O   [[buffer(3)]],
    constant int&       M   [[buffer(4)]],
    constant int&       Nk  [[buffer(5)]],
    constant int&       D   [[buffer(6)]],
    constant int&       Dv  [[buffer(7)]],
    constant float&     scale  [[buffer(8)]],
    constant int&       causal [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)M) return;
    if (Nk > {SYNTH_MAX_N}) return;
    int m = (int)gid;
    float scores[{SYNTH_MAX_N}];
    int q_off = m * D;
    float mx = -INFINITY;
    for (int n = 0; n < Nk; ++n) {{
        if (causal != 0 && n > m) {{ scores[n] = -INFINITY; continue; }}
        float s = 0.0f;
        int k_off = n * D;
        for (int d = 0; d < D; ++d) s += float(Q[q_off + d]) * float(K[k_off + d]);
        s *= scale;
        scores[n] = s;
        mx = max(mx, s);
    }}
    float sm = 0.0f;
    for (int n = 0; n < Nk; ++n) {{
        float e = exp(scores[n] - mx);
        scores[n] = e;
        sm += e;
    }}
    float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
    int o_off = m * Dv;
    for (int dv = 0; dv < Dv; ++dv) {{
        float acc = 0.0f;
        for (int n = 0; n < Nk; ++n) acc += scores[n] * float(V[n * Dv + dv]);
        O[o_off + dv] = ST(acc * inv);
    }}
}}
"""


def synthesize_attention_online_msl(region: AttentionRegion = AttentionRegion(),
                                    dtype: str = "f32") -> str:
    """Emit MSL for a fused attention block using ONLINE softmax — flash-attention
    style. One query row per thread; keys are streamed in a single pass holding
    only a running max ``m``, running denominator ``l``, and an ``acc[head_dim]``
    output accumulator (no ``scores[Nk]`` array). This lifts the SYNTH_MAX_N key
    cap entirely (Nk unbounded) at the cost of a head_dim ≤ SYNTH_MAX_D bound — the
    large-context attention case the materialized kernel can't reach.

    Same C-ABI buffer layout / entry signature as the materialized kernel, so it
    rides the existing ``tessera_apple_gpu_synth_attention_{f32,f16}`` symbols
    (which take the MSL source + entry name as parameters) — no new runtime
    symbol per kernel. ``dtype`` selects the I/O type; reads cast to float,
    accumulators are fp32, the O-write goes through ``ST(...)``."""
    io = _io_type(dtype)
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_ATTN_ONLINE_ENTRY}(
    device const {io}* Q   [[buffer(0)]],
    device const {io}* K   [[buffer(1)]],
    device const {io}* V   [[buffer(2)]],
    device {io}*       O   [[buffer(3)]],
    constant int&       M   [[buffer(4)]],
    constant int&       Nk  [[buffer(5)]],
    constant int&       D   [[buffer(6)]],
    constant int&       Dv  [[buffer(7)]],
    constant float&     scale  [[buffer(8)]],
    constant int&       causal [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)M) return;
    if (Dv > {SYNTH_MAX_D}) return;
    int m = (int)gid;
    int q_off = m * D;
    float acc[{SYNTH_MAX_D}];
    for (int d = 0; d < Dv; ++d) acc[d] = 0.0f;
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    for (int n = 0; n < Nk; ++n) {{
        // causal: every key n > m is masked, and so is every later key — stop.
        if (causal != 0 && n > m) break;
        float s = 0.0f;
        int k_off = n * D;
        for (int d = 0; d < D; ++d) s += float(Q[q_off + d]) * float(K[k_off + d]);
        s *= scale;
        float new_max = max(run_max, s);
        // first key: run_max = -inf -> corr = exp(-inf) = 0 (acc/sum start at 0).
        float corr = exp(run_max - new_max);
        float p = exp(s - new_max);
        run_sum = run_sum * corr + p;
        int v_off = n * Dv;
        for (int d = 0; d < Dv; ++d) acc[d] = acc[d] * corr + p * float(V[v_off + d]);
        run_max = new_max;
    }}
    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    int o_off = m * Dv;
    for (int d = 0; d < Dv; ++d) O[o_off + d] = ST(acc[d] * inv);
}}
"""


def _attn_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_attention_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, fp, fp, fp, fp,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _attn_f16_symbol() -> Any:
    """The uint16-I/O attention symbol — serves both half and native bfloat (the
    MSL source the caller emits selects which). Same arg layout as the f32 symbol
    but with uint16 buffers."""
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_attention_f16", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, u16, u16, u16, u16,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_float, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _attn_dtype_tag(dt: Any) -> str:
    """Map a numpy/ml_dtypes dtype to a synthesizer dtype tag. Non-16-bit floats
    use the f32 path (the inputs are cast to f32)."""
    if dt == np.float16:
        return "f16"
    bf = _bf16_dtype()
    if bf is not None and dt == bf:
        return "bf16"
    return "f32"


def run_fused_attention(region: AttentionRegion, Q: np.ndarray, K: np.ndarray,
                        V: np.ndarray) -> tuple[np.ndarray, str]:
    """Run the attention block as ONE synthesized fused kernel on Metal.  Returns
    ``(output, execution)`` — ``"metal_runtime"`` if the synthesized kernel ran,
    else ``"reference"`` (numpy).  Either way the numbers are correct.

    f16/bf16 inputs run a ``half``/``bfloat``-I/O kernel (fp32 accumulators) via
    the uint16 symbol; f32 (and any other dtype, cast to f32) uses the f32 symbol.
    Either dtype picks the materialized kernel for Nk ≤ SYNTH_MAX_N, else the
    online-softmax kernel for larger Nk (head_dim ≤ SYNTH_MAX_D), else reference."""
    tag = _attn_dtype_tag(np.asarray(V).dtype)
    # Orientation per the score matmul's transpose flags; preserve the storage
    # dtype on the half-precision path, cast to f32 otherwise.
    Qn, Kn = region._natural(Q, K, cast=(tag == "f32"))
    if tag == "f32":
        Qn = np.ascontiguousarray(Qn, np.float32)
        Kn = np.ascontiguousarray(Kn, np.float32)
        Vn = np.ascontiguousarray(V, np.float32)
    else:
        Qn = np.ascontiguousarray(Qn)
        Kn = np.ascontiguousarray(Kn)
        Vn = np.ascontiguousarray(V)
    M, D = Qn.shape
    Nk, Dk = Kn.shape
    Nv, Dv = Vn.shape
    if Dk != D:
        raise ValueError(f"Q/K head_dim mismatch: Q {Qn.shape}, K {Kn.shape}")
    if Nv != Nk:
        raise ValueError(f"K/V seqlen mismatch: K {Kn.shape}, V {Vn.shape}")

    # Pick the kernel by IO cost, not a hard threshold (Workstream C): the
    # selector scores materialized / online / reference by off-chip byte movement
    # and returns the minimum-byte feasible variant. For small Nk it returns
    # "materialized"; past the on-chip stack cap it crosses to "online"; beyond
    # both caps only "reference" is feasible — reproducing the old branch, now as
    # one scored decision point. Both kernels ride one symbol per dtype.
    sym = _attn_symbol() if tag == "f32" else _attn_f16_symbol()
    source: bytes | None = None
    entry = b""
    elt = 4 if tag == "f32" else 2
    choice = select_attention_lowering(M, Nk, D, Dv, elt_bytes=elt)
    if sym is not None and choice.variant == "materialized":
        source = synthesize_attention_msl(region, tag).encode("utf-8")
        entry = _ATTN_ENTRY.encode("utf-8")
    elif sym is not None and choice.variant == "online":
        source = synthesize_attention_online_msl(region, tag).encode("utf-8")
        entry = _ATTN_ONLINE_ENTRY.encode("utf-8")
    if sym is not None and source is not None:
        if tag == "f32":
            out = np.zeros((M, Dv), np.float32)
            p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            out = np.zeros((M, Dv), Vn.dtype)
            p = lambda a: a.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16))
        rc = sym(source, entry, p(Qn), p(Kn), p(Vn), p(out),
                 M, Nk, D, Dv, ctypes.c_float(region.scale),
                 1 if region.causal else 0)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(Q, K, V), "reference"
_GATED_ENTRY = "synth_gated_matmul"


def synthesize_gated_matmul_msl(region: GatedMatmulRegion = GatedMatmulRegion(),
                                dtype: str = "f32") -> str:
    """Emit the MSL source for ``O = f(A @ Wg) ⊙ (A @ Wu)`` — one A row per thread.

    Both projections accumulate in a single K-loop (A[k] loaded once, fanned to
    the gate and up rows), so the shared input is read K times total, not 2K.
    The gate activation runs in fp32; the O-write goes through ``ST(...)`` (bfloat
    rejects implicit float→bfloat). ``dtype`` selects the I/O type."""
    io = _io_type(dtype)
    act = EPILOGUE_OPS[region.gate_act].emit(_MSL_TARGET)        # operates on `v`
    return f"""#include <metal_stdlib>
using namespace metal;
using ST = {io};
kernel void {_GATED_ENTRY}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* Wg  [[buffer(1)]],
    device const {io}* Wu  [[buffer(2)]],
    device {io}*       O   [[buffer(3)]],
    constant int&       M   [[buffer(4)]],
    constant int&       H   [[buffer(5)]],
    constant int&       K   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= (uint)M) return;
    if (H > {SYNTH_GATED_MAX_H}) return;
    int row = (int)gid;
    float gate[{SYNTH_GATED_MAX_H}];
    float up[{SYNTH_GATED_MAX_H}];
    for (int n = 0; n < H; ++n) {{ gate[n] = 0.0f; up[n] = 0.0f; }}
    int a_off = row * K;
    for (int k = 0; k < K; ++k) {{
        float a = float(A[a_off + k]);
        int w_off = k * H;
        for (int n = 0; n < H; ++n) {{
            gate[n] += a * float(Wg[w_off + n]);
            up[n]   += a * float(Wu[w_off + n]);
        }}
    }}
    int o_off = row * H;
    for (int n = 0; n < H; ++n) {{
        float v = gate[n];
        {act}
        O[o_off + n] = ST(v * up[n]);
    }}
}}
"""


def _gated_symbol() -> Any:
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_gated_matmul_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, fp, fp, fp, fp,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _gated_f16_symbol() -> Any:
    """The uint16-I/O gated symbol — serves both half and native bfloat (the MSL
    source the caller emits selects which)."""
    from tessera.runtime import _load_apple_gpu_runtime
    rt = _load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_synth_gated_matmul_f16", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    sym.argtypes = [ctypes.c_char_p, ctypes.c_char_p, u16, u16, u16, u16,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_gated_matmul_region(region: GatedMatmulRegion, A: np.ndarray,
                            Wg: np.ndarray, Wu: np.ndarray
                            ) -> tuple[np.ndarray, str]:
    """Run ``O = f(A @ Wg) ⊙ (A @ Wu)`` as ONE synthesized Metal kernel. Returns
    ``(output, execution)`` — ``"metal_runtime"`` if the synthesized kernel ran,
    else ``"reference"`` (numpy). f16 runs a native ``half`` kernel; bf16 (no MSL
    type) converts to f32, runs, converts back (the 8.4.4.x convention)."""
    in_dtype = np.asarray(A).dtype
    bf16 = _bf16_dtype()
    if bf16 is not None and in_dtype == bf16:
        out32, ex = run_gated_matmul_region(
            region, np.asarray(A, np.float32), np.asarray(Wg, np.float32),
            np.asarray(Wu, np.float32))
        return out32.astype(bf16), ex
    tag = "f16" if in_dtype == np.float16 else "f32"
    if tag == "f16":
        A = np.ascontiguousarray(A, np.float16)
        Wg = np.ascontiguousarray(Wg, np.float16)
        Wu = np.ascontiguousarray(Wu, np.float16)
    else:
        A = np.ascontiguousarray(A, np.float32)
        Wg = np.ascontiguousarray(Wg, np.float32)
        Wu = np.ascontiguousarray(Wu, np.float32)
    M, K = A.shape
    Kg, H = Wg.shape
    if Kg != K or Wu.shape != (K, H):
        raise ValueError(
            f"gated matmul shape mismatch: A{A.shape} Wg{Wg.shape} Wu{Wu.shape}")
    sym = _gated_f16_symbol() if tag == "f16" else _gated_symbol()
    if sym is not None and H <= SYNTH_GATED_MAX_H:
        source = synthesize_gated_matmul_msl(region, tag).encode("utf-8")
        entry = _GATED_ENTRY.encode("utf-8")
        if tag == "f16":
            out = np.zeros((M, H), np.float16)
            p = lambda a: a.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16))
        else:
            out = np.zeros((M, H), np.float32)
            p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = sym(source, entry, p(A), p(Wg), p(Wu), p(out), M, H, K)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, Wg, Wu), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# F5 — autotune the synthesizer (gated behind F3 cost + F4 oracle)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AutotuneRecord:
    """The outcome of autotuning a region at a shape: per-variant latency +
    oracle verdict, and the chosen variant.  ``chosen`` is the fastest variant
    that *passed the oracle* — a fast-but-wrong variant is never chosen (the
    Sakana invariant, enforced by ``_pick_best_variant``)."""

    chosen: str | None
    latencies_ms: dict[str, float]
    correct: dict[str, bool]


def _pick_best_variant(latencies_ms: dict[str, float],
                       correct: dict[str, bool]) -> str | None:
    """F5's gate: the fastest variant *among those that pass the oracle*.  A
    faster variant that fails correctness is excluded — perf is gated behind
    correctness, never traded for it."""
    eligible = {v: t for v, t in latencies_ms.items() if correct.get(v, False)}
    if not eligible:
        return None
    return min(eligible, key=lambda v: eligible[v])


def _shape_bucket(n: int) -> int:
    """Coarsen a dimension to a power-of-two bucket so the corpus generalizes
    across nearby shapes instead of memorizing every exact size."""
    b = 1
    while b < n:
        b *= 2
    return b


def _corpus_key(region: FusedRegion, M: int, N: int, K: int) -> tuple:
    return (region.epilogue, region.reduction, region.has_bias,
            _shape_bucket(M), _shape_bucket(N), _shape_bucket(K))


#: Distilled best-variant decisions, keyed by (region-class, shape-bucket).
_AUTOTUNE_CORPUS: dict[tuple, str] = {}


def clear_autotune_corpus() -> None:
    _AUTOTUNE_CORPUS.clear()


def autotune_matmul_epilogue(region: FusedRegion, M: int, N: int, K: int, *,
                             variants: tuple[str, ...] = SYNTH_VARIANTS,
                             reps: int = 5, seed: int = 0
                             ) -> AutotuneRecord | None:
    """Measure each synthesis ``variant`` at (M,N,K) and record the fastest one
    that passes the oracle.  Returns ``None`` when the region is not fusible (F3)
    or no variant ran on Metal.  Every measured variant is checked against the
    numpy reference (F4) before its latency is allowed to count."""
    import time

    if not should_fuse_region(region, M, N, K):          # F3 cost gate
        return None
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = (rng.standard_normal((N,)).astype(np.float32)
            if region.has_bias else None)
    ref = region.reference(A, B, bias)

    latencies: dict[str, float] = {}
    correct: dict[str, bool] = {}
    for var in variants:
        out, execution = run_fused_region(region, A, B, bias, variant=var)
        if execution != "metal_runtime":
            continue                            # no Metal → can't measure this one
        ok = bool(np.allclose(out, ref, atol=1e-3))       # F4 oracle gate
        correct[var] = ok
        if not ok:
            continue                            # fast-but-wrong excluded
        best = float("inf")
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            run_fused_region(region, A, B, bias, variant=var)
            best = min(best, time.perf_counter() - t0)
        latencies[var] = best * 1e3
    if not correct:
        return None                             # nothing ran on Metal
    chosen = _pick_best_variant(latencies, correct)
    if chosen is not None:
        _AUTOTUNE_CORPUS[_corpus_key(region, M, N, K)] = chosen
    return AutotuneRecord(chosen, latencies, correct)


def best_variant_for(region: FusedRegion, M: int, N: int, K: int) -> str:
    """The distilled best variant for this region/shape from the autotune corpus,
    or the default ``"broadcast"`` when unseen — an O(1) lookup, no measurement."""
    return _AUTOTUNE_CORPUS.get(_corpus_key(region, M, N, K), "broadcast")


def autotune_enabled() -> bool:
    """Whether lazy measured-latency autotuning is on (env ``TESSERA_AUTOTUNE``).
    Off by default — autotuning measures every variant on first use of a shape,
    which is a one-time cost callers opt into."""
    import os
    return os.environ.get("TESSERA_AUTOTUNE", "").lower() in ("1", "true", "on", "yes")


def select_variant(region: FusedRegion, M: int, N: int, K: int, *,
                   autotune: bool | None = None) -> str:
    """Pick the synthesis variant for (region, shape) — Phase 3 'close the
    optimizing loop': when autotuning is enabled and this shape is unseen, MEASURE
    each variant on-device (`autotune_matmul_epilogue`, correctness-gated) and
    cache the fastest, so the choice is measured-best rather than the static
    default. Otherwise an O(1) corpus lookup (`best_variant_for`)."""
    if autotune is None:
        autotune = autotune_enabled()
    if autotune and _corpus_key(region, M, N, K) not in _AUTOTUNE_CORPUS:
        try:
            autotune_matmul_epilogue(region, M, N, K)   # measures + caches
        except Exception:                               # noqa: BLE001 - fall back to default
            pass
    return best_variant_for(region, M, N, K)


#: Distilled best coopmat *tile* (32/64) per (region-class, shape-bucket).
_COOPMAT_TILE_CORPUS: dict[tuple, int] = {}


def clear_coopmat_tile_corpus() -> None:
    _COOPMAT_TILE_CORPUS.clear()


def autotune_coopmat_tile(region: FusedRegion, M: int, N: int, K: int, *,
                          dtype: str = "f16", reps: int = 5, seed: int = 0
                          ) -> dict[int, float]:
    """Measure the 32x32 vs 64x64 coopmat tile at (M,N,K) and record the faster
    one that passes the oracle (perf gated behind correctness — the Sakana
    invariant).  Returns ``{tile: latency_ms}`` for the tiles that ran on Metal;
    the winner is stored in ``_COOPMAT_TILE_CORPUS`` so ``coopmat_tile_for``
    becomes an O(1) tuned lookup.  Returns ``{}`` when the region is not coopmat-
    eligible or no tile ran on Metal."""
    import time

    if not coopmat_eligible(region):
        return {}
    np_dt = np.float16 if dtype == "f16" else np.float32
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * 0.3).astype(np_dt)
    B = (rng.standard_normal((K, N)) * 0.3).astype(np_dt)
    bias = ((rng.standard_normal((N,)) * 0.3).astype(np_dt)
            if region.has_bias else None)
    ref = region.reference(A, B, bias)

    latencies: dict[int, float] = {}
    correct: dict[int, bool] = {}
    for tile in SYNTH_COOPMAT_TILES:
        out, execution = run_fused_region_coopmat(region, A, B, bias, tile=tile)
        if execution != "metal_runtime":
            continue
        atol = 1e-3 if np_dt == np.float32 else 3e-2
        ok = bool(np.allclose(out.astype(np.float32), ref, atol=atol, rtol=atol))
        correct[tile] = ok
        if not ok:
            continue                            # fast-but-wrong excluded
        best = float("inf")
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            run_fused_region_coopmat(region, A, B, bias, tile=tile)
            best = min(best, time.perf_counter() - t0)
        latencies[tile] = best * 1e3
    eligible = {t: ms for t, ms in latencies.items() if correct.get(t, False)}
    if eligible:
        winner = min(eligible, key=lambda t: eligible[t])
        _COOPMAT_TILE_CORPUS[_corpus_key(region, M, N, K)] = winner
    return latencies


# ─────────────────────────────────────────────────────────────────────────────
# B2 — KernelEmitter reference implementation (Apple MSL)
#
# AppleMSLEmitter adapts the synthesize_*_msl functions above to the generic
# KernelEmitter protocol. It is a thin dispatcher — NOT a reimplementation — so a
# non-Apple backend (Workstream C: x86 clang, NVIDIA PTX, ROCm AMDGCN) plugs into
# the same fusion_core region model by writing its own emitter, without forking
# the discovery/cost/oracle middle-end.
# ─────────────────────────────────────────────────────────────────────────────


class AppleMSLEmitter(KernelEmitter):
    """Reference :class:`KernelEmitter` — wraps the ``synthesize_*_msl`` bodies.

    Dispatches an arch-agnostic fused ``region`` to its Metal Shading Language
    source + entry-point name and returns a :class:`KernelSource`. Variant/tile
    selection (tiled / coopmat / online) stays in the ``run_*`` dispatch path;
    this emitter yields the canonical scalar source form for the region, which is
    what the generic synth→compile→cache→launch loop (Workstream B4) consumes.
    """

    target = _MSL_TARGET
    lang = "msl"

    #: region type → (synthesis callable, entry-point symbol). Ordered so a
    #: reduction-terminated pointwise DAG matches before the plain DAG.
    def _dispatch(self, region: Any):
        if isinstance(region, FusedRegion):
            return synthesize_matmul_epilogue_msl, _ENTRY
        if isinstance(region, NormChainRegion):
            return synthesize_norm_chain_msl, _NORM_CHAIN_ENTRY
        if isinstance(region, PointwiseReduceRegion):
            return synthesize_pointwise_reduce_msl, _PW_REDUCE_ENTRY
        if isinstance(region, PointwiseGraphRegion):
            return synthesize_pointwise_graph_msl, _PW_ENTRY
        if isinstance(region, AttentionRegion):
            return synthesize_attention_msl, _ATTN_ENTRY
        if isinstance(region, GatedMatmulRegion):
            return synthesize_gated_matmul_msl, _GATED_ENTRY
        return None

    def can_emit(self, region: Any) -> bool:
        return self._dispatch(region) is not None

    def emit(
        self, region: Any, *, spec: SpecPolicy = SpecPolicy.BUCKET, dtype: str = "f32"
    ) -> KernelSource:
        disp = self._dispatch(region)
        if disp is None:
            raise EmitError(
                f"AppleMSLEmitter cannot emit a region of type "
                f"{type(region).__name__}")
        synth, entry = disp
        source = synth(region, dtype=dtype)
        return KernelSource(source=source, entry=entry, lang=self.lang, spec=spec)


register_emitter(AppleMSLEmitter())


class AppleMSLRunner(KernelRunner):
    """Reference :class:`KernelRunner` (B2b) — executes a synthesized region via
    the Apple Metal runtime by delegating to this module's ``run_*`` functions.

    It is the execute-half twin of :class:`AppleMSLEmitter`: the F4 oracles in
    ``fusion_core`` call these through the injected-runner registry instead of a
    hard ``import apple_msl``, so a non-Apple backend registers its own runner and
    reuses the same oracle."""

    target = _MSL_TARGET

    def run_fused_region(self, region, *args, **kwargs):
        return run_fused_region(region, *args, **kwargs)

    def run_fused_attention(self, region, *args, **kwargs):
        return run_fused_attention(region, *args, **kwargs)

    def run_gated_matmul_region(self, region, *args, **kwargs):
        return run_gated_matmul_region(region, *args, **kwargs)

    def run_pointwise_graph(self, region, *args, **kwargs):
        return run_pointwise_graph(region, *args, **kwargs)


register_runner(AppleMSLRunner())
