"""General fusion middle-end — region IR, discovery, and MSL kernel synthesis.

Optimizing-Compiler Plan F0 + F1a + F2a (docs/audit/compiler/OPTIMIZING_COMPILER_PLAN.md).

Tessera's fusion was *enumerated*: ~8 hand-written passes that pattern-match a
named SSA chain and replace it with a func.call into one of 168 hand-authored MSL
kernels.  This module replaces that for the ``matmul -> pointwise-epilogue``
family with three general pieces:

* **F0 — FusedRegion**: a matmul root plus an ordered chain of pointwise epilogue
  ops, captured as one schedulable unit.
* **F1a — discover_fusable_regions**: walk an op list and grow maximal
  matmul -> pointwise chains (the intermediate must be single-use, else fusing
  would drop a value another op consumes).
* **F2a — synthesize_matmul_epilogue_msl**: emit the MSL *source* for the region
  — a row-per-thread matmul with the epilogue ops inlined — which the runtime
  compiles (cached) and runs via the generic
  ``tessera_apple_gpu_synth_matmul_epilogue_f32`` symbol.

One synthesizer + one dispatcher replace the catalog: ``matmul -> bias -> gelu``,
``matmul -> silu``, ``matmul -> sigmoid``, any pointwise chain — none of which
needs a hand-written kernel.  Correctness is gated by the Evaluator's horizontal
oracle (synthesized == unfused); cost by dlop_longtail_core dispatch counting.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

#: Cap on N (the per-thread ``scores[N]`` stack accumulator) — matches the
#: hand-written fused kernels' envelope.  Above this, fall back to unfused.
SYNTH_MAX_N = 1024
_ENTRY = "synth_matmul_epi"


@dataclass(frozen=True)
class EpilogueOp:
    """One pointwise epilogue op: how it lowers to MSL and to numpy."""

    name: str
    msl: str                                  # operates on `v`, may read bias[n]
    ref: Callable[[np.ndarray], np.ndarray]   # numpy reference (no-bias ops)
    needs_bias: bool = False


def _gelu(x: np.ndarray) -> np.ndarray:
    t = np.clip(0.7978845608028654 * (x + 0.044715 * x ** 3), -30.0, 30.0)
    return 0.5 * x * (1.0 + np.tanh(t))


#: The pointwise epilogue vocabulary.  Adding an activation here makes it fusible
#: into *any* matmul epilogue chain — no new kernel, no new pass.
EPILOGUE_OPS: dict[str, EpilogueOp] = {
    "bias":    EpilogueOp("bias", "v = v + bias[n];", lambda x: x, needs_bias=True),
    "relu":    EpilogueOp("relu", "v = max(v, 0.0f);", lambda x: np.maximum(x, 0.0)),
    "gelu":    EpilogueOp(
        "gelu",
        "{ float _t = clamp(0.7978845608028654f*(v+0.044715f*v*v*v), -30.0f, 30.0f);"
        " v = 0.5f*v*(1.0f+tanh(_t)); }",
        _gelu,
    ),
    "silu":    EpilogueOp("silu", "v = v / (1.0f + exp(-v));",
                          lambda x: x / (1.0 + np.exp(-x))),
    "sigmoid": EpilogueOp("sigmoid", "v = 1.0f / (1.0f + exp(-v));",
                          lambda x: 1.0 / (1.0 + np.exp(-x))),
    "tanh":    EpilogueOp("tanh", "v = tanh(v);", np.tanh),
}


@dataclass(frozen=True)
class ReductionOp:
    """A terminal *reduction* epilogue (rmsnorm/softmax): a row reduction over the
    matmul-row accumulator ``scores[N]``, then a per-element finalize into ``O``.
    Unlike pointwise ops, it needs the whole row — so it comes last, after any
    pointwise chain.  ``msl`` is a block (uses ``N``/``scores``/``O``/``o_off``,
    ``{eps}`` substituted); ``ref(x, eps)`` is the numpy ground truth."""

    name: str
    msl: str
    ref: Callable[[np.ndarray, float], np.ndarray]


def _rmsnorm_ref(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def _softmax_ref(x: np.ndarray, eps: float) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


#: Terminal reduction epilogues.  Adding one here retires the corresponding
#: hand-written matmul_<reduction> kernel.
REDUCTION_OPS: dict[str, ReductionOp] = {
    "rmsnorm": ReductionOp(
        "rmsnorm",
        "        float _ss = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) _ss += scores[n] * scores[n];\n"
        "        float _inv = rsqrt(_ss / float(N) + {eps}f);\n"
        "        for (int n = 0; n < N; ++n) O[o_off + n] = scores[n] * _inv;",
        _rmsnorm_ref,
    ),
    "softmax": ReductionOp(
        "softmax",
        "        float _mx = -INFINITY;\n"
        "        for (int n = 0; n < N; ++n) _mx = max(_mx, scores[n]);\n"
        "        float _sm = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) {{ scores[n] = exp(scores[n] - _mx);"
        " _sm += scores[n]; }}\n"
        "        for (int n = 0; n < N; ++n) O[o_off + n] = scores[n] / _sm;",
        _softmax_ref,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# F0 — fused region
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FusedRegion:
    """A matmul root + an ordered pointwise epilogue chain + an optional terminal
    reduction epilogue (rmsnorm/softmax).  The reduction, if present, runs last."""

    epilogue: tuple[str, ...]
    reduction: str | None = None
    eps: float = 1e-6
    a_name: str = "A"
    b_name: str = "B"
    bias_name: str | None = None

    def __post_init__(self) -> None:
        for op in self.epilogue:
            if op not in EPILOGUE_OPS:
                raise ValueError(f"unknown epilogue op {op!r}")
        bias_ops = [op for op in self.epilogue if EPILOGUE_OPS[op].needs_bias]
        if len(bias_ops) > 1:
            raise ValueError("at most one bias op per region")
        if self.reduction is not None and self.reduction not in REDUCTION_OPS:
            raise ValueError(f"unknown reduction epilogue {self.reduction!r}")
        if not self.epilogue and self.reduction is None:
            raise ValueError("a region must have at least one epilogue op")

    @property
    def has_bias(self) -> bool:
        return any(EPILOGUE_OPS[op].needs_bias for op in self.epilogue)

    def reference(self, A: np.ndarray, B: np.ndarray,
                  bias: np.ndarray | None = None) -> np.ndarray:
        """The *unfused* result: matmul, pointwise chain, then the reduction, in
        numpy — the horizontal-oracle ground truth the synthesized kernel matches."""
        out = np.asarray(A, np.float32) @ np.asarray(B, np.float32)
        for op in self.epilogue:
            spec = EPILOGUE_OPS[op]
            if spec.needs_bias:
                if bias is None:
                    raise ValueError(f"region needs a bias for {op!r}")
                out = out + np.asarray(bias, np.float32)
            else:
                out = spec.ref(out)
        if self.reduction is not None:
            out = REDUCTION_OPS[self.reduction].ref(out, self.eps)
        return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# F2a — MSL synthesis
# ─────────────────────────────────────────────────────────────────────────────


#: The matmul-into-``scores[]`` inner loop, parametrized for the F5 autotuner.
#: ``broadcast`` streams one A element across the whole N row (B read
#: contiguously); ``dot`` accumulates each output column as a K-dot (B read
#: strided).  Both fill ``scores[]`` identically — the oracle proves it — but
#: have different memory access, so the fastest depends on the shape.
SYNTH_VARIANTS = ("broadcast", "dot")


def _matmul_body(variant: str) -> str:
    if variant == "broadcast":
        return """    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }"""
    if variant == "dot":
        return """    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[a_off + k] * B[k * N + n];
        scores[n] = acc;
    }"""
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
    raise ValueError(f"synthesizer dtype must be f32 or f16, got {dtype!r}")


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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)
    matmul_body = _matmul_body(variant)

    if region.reduction is None:
        # pure pointwise: one pass, write O directly.
        finalize = f"""    int o_off = row * N;
    for (int n = 0; n < N; ++n) {{
        float v = scores[n];
{pointwise}
        O[o_off + n] = v;
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
        red = REDUCTION_OPS[region.reduction].msl.format(eps=region.eps)
        finalize = f"""    int o_off = row * N;
{pw_pass}{red}"""

    return f"""#include <metal_stdlib>
using namespace metal;
kernel void {_ENTRY}(
    device const {io}* A   [[buffer(0)]],
    device const {io}* B   [[buffer(1)]],
    device {io}*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
{bias_param}    uint gid [[thread_position_in_grid]])
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


# ─────────────────────────────────────────────────────────────────────────────
# F2b-tiled — threadgroup-tiled synthesis for large N (the stack kernel caps at
# SYNTH_MAX_N; this lifts it to SYNTH_MAX_N_TILED via dynamic threadgroup memory)
# ─────────────────────────────────────────────────────────────────────────────

#: Large-N cap for the tiled kernel: one row of N fp32 scores lives in dynamic
#: threadgroup memory (32 KB budget on current Apple arches → 8192 floats),
#: mirroring the hand-written ``matmul_softmax_tiled_f32`` envelope it retires.
SYNTH_MAX_N_TILED = 8192
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
        "    for (int n = lid_i; n < N; n += T) O[o_off + n] = tg_scores[n] * _inv;\n",
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
        "    if (_sm == 0.0f) {{ for (int n = lid_i; n < N; n += T) O[o_off + n] = 0.0f; }}\n"
        "    else {{ float _inv = 1.0f / _sm;"
        " for (int n = lid_i; n < N; n += T) O[o_off + n] = tg_scores[n] * _inv; }}\n",
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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)

    if region.reduction is None:
        # pure pointwise: cooperative epilogue writes O directly, no reduction.
        body = f"""    int o_off = row * N;
    for (int n = lid_i; n < N; n += T) {{
        float v = tg_scores[n];
{pointwise}
        O[o_off + n] = v;
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
    for (int n = lid_i; n < N; n += T) {{
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += A[a_off + k] * B[k * N + n];
        tg_scores[n] = s;
    }}
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
    """v1 covers pointwise-epilogue regions (no terminal reduction)."""
    return region.reduction is None


def synthesize_matmul_epilogue_coopmat_msl(region: FusedRegion,
                                           dtype: str = "f16") -> str:
    """Emit a cooperative-matrix (``simdgroup_matrix``) matmul + fused pointwise
    epilogue.  ``dtype`` selects the MMA input type (f16 taps the matrix units
    at ~2×; f32 is the simdgroup f32 ceiling).  The accumulator is always fp32,
    so the epilogue sees full-precision matmul results."""
    if region.reduction is not None:
        raise ValueError("coopmat v1 does not support reduction epilogues")
    io = _io_type(dtype)
    bias_param = (f"    device const {io}* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)
    sg_in = f"simdgroup_matrix<{io}, 8, 8>"

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant constexpr int BM = 32;
constant constexpr int BN = 32;
constant constexpr int BK = 16;
constant constexpr int SG = 16;
constant constexpr int NT = 2;
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
    threadgroup {io} As[2][BM * BK];
    threadgroup {io} Bs[2][BK * BN];
    threadgroup float Cs[BM * BN];
    int sg_row = int(sgid) / 2, sg_col = int(sgid) % 2;
    int r0 = sg_row * SG, c0 = sg_col * SG;
    simdgroup_float8x8 acc[NT][NT];
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < NT; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Cooperative K-slab staging (128 threads, zero-padded out of range).
    // Single-buffered: manual double-buffering was measured a wash on Apple
    // (no cp.async — the threadgroup prefetch doesn't overlap load with compute
    // the way it does on NVIDIA), so the simpler kernel is kept.  The real tile
    // upgrade is 64x64 register blocking (a deferred perf-only follow-up).
    for (int k0 = 0; k0 < K; k0 += BK) {{
        for (int e = int(tid); e < BM * BK; e += 128) {{
            int r = e / BK, kk = e % BK;
            int gr = brow + r, gk = k0 + kk;
            As[0][e] = (gr < M && gk < K) ? A[gr * K + gk] : ({io})0;
        }}
        for (int e = int(tid); e < BK * BN; e += 128) {{
            int kk = e / BN, c = e % BN;
            int gk = k0 + kk, gc = bcol + c;
            Bs[0][e] = (gk < K && gc < N) ? B[gk * N + gc] : ({io})0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {{
            {sg_in} a[NT], b[NT];
            for (int i = 0; i < NT; ++i)
                simdgroup_load(a[i], As[0] + (r0 + i * 8) * BK + kk, BK);
            for (int j = 0; j < NT; ++j)
                simdgroup_load(b[j], Bs[0] + kk * BN + (c0 + j * 8), BN);
            for (int i = 0; i < NT; ++i)
                for (int j = 0; j < NT; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store the fp32 accumulator tile to threadgroup memory, then run the
    // pointwise epilogue per element (bias[n] indexes the global column).
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < NT; ++j)
            simdgroup_store(acc[i][j], Cs + (r0 + i * 8) * BN + (c0 + j * 8), BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < BM * BN; e += 128) {{
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
    threadgroup-memory cap."""
    return (region.reduction in _COOPMAT_REDUCTIONS
            and 0 < N <= SYNTH_COOPMAT_REDUCE_MAX_N)


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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)
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
    threadgroup {io} Bs[BK * 8];
    threadgroup float Cs[BM * MAXN];

    // Compute the BM × N score strip, one 8-col tile at a time.
    for (int jb = 0; jb < N; jb += 8) {{
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k0 = 0; k0 < K; k0 += BK) {{
            for (int e = int(tid); e < BM * BK; e += 32) {{
                int r = e / BK, kk = e % BK;
                int gr = brow + r, gk = k0 + kk;
                As[e] = (gr < M && gk < K) ? A[gr * K + gk] : ({io})0;
            }}
            for (int e = int(tid); e < BK * 8; e += 32) {{
                int kk = e / 8, c = e % 8;
                int gk = k0 + kk, gc = jb + c;
                Bs[e] = (gk < K && gc < N) ? B[gk * N + gc] : ({io})0;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < BK; kk += 8) {{
                {sg_in} a, b;
                simdgroup_load(a, As + kk, BK);
                simdgroup_load(b, Bs + kk * 8, 8);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        simdgroup_store(acc, Cs + jb, N);   // 8 rows × 8 cols at column jb
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
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
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
                    ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def run_fused_region_coopmat(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                             bias: np.ndarray | None = None
                             ) -> tuple[np.ndarray, str]:
    """Run the region via the COOPERATIVE-MATRIX synthesized kernel (F2d) —
    ``simdgroup_matrix`` MMA (f16 multiply / fp32 accumulate) with the pointwise
    epilogue fused after.  f16/f32 inputs only; reduction regions and other
    dtypes fall back to the scalar ``run_fused_region``.  Returns
    ``(output, execution)``."""
    if not coopmat_eligible(region):
        return run_fused_region(region, A, B, bias)
    in_dtype = np.asarray(A).dtype
    np_dt: Any
    if in_dtype == np.float16:
        np_dt, elem_size, dtype = np.float16, 2, "f16"
    elif in_dtype == np.float32:
        np_dt, elem_size, dtype = np.float32, 4, "f32"
    else:
        return run_fused_region(region, A, B, bias)   # bf16/other → scalar path
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

    sym = _synth_coopmat_symbol()
    if sym is not None:
        source = synthesize_matmul_epilogue_coopmat_msl(
            region, dtype=dtype).encode("utf-8")
        out = np.zeros((M, N), np_dt)
        vp = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        bias_ptr = vp(bias_arr) if bias_arr is not None else None
        rc = sym(source, _ENTRY_COOPMAT.encode("utf-8"), vp(A), vp(B), bias_ptr,
                 vp(out), M, N, K, 1 if region.has_bias else 0, elem_size)
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
                    ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _run_fused_region_f16(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                          bias: np.ndarray | None,
                          variant: str) -> tuple[np.ndarray, str]:
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
    u16p = lambda a: a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    bias_ptr = u16p(bias_arr) if bias_arr is not None else None
    has_bias = 1 if region.has_bias else 0

    sym = _synth_f16_symbol()
    if sym is not None and N <= SYNTH_MAX_N_TILED:
        is_tiled = 0 if N <= SYNTH_MAX_N else 1
        if is_tiled:
            source = synthesize_matmul_epilogue_msl_tiled(region, dtype="f16")
            entry = _ENTRY_TILED
        else:
            source = synthesize_matmul_epilogue_msl(region, variant, dtype="f16")
            entry = _ENTRY
        out = np.zeros((M, N), np.float16)
        rc = sym(source.encode("utf-8"), entry.encode("utf-8"), u16p(A), u16p(B),
                 bias_ptr, u16p(out), M, N, K, has_bias, is_tiled)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, B, bias_arr).astype(np.float16), "reference"


def run_fused_region(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                     bias: np.ndarray | None = None,
                     variant: str = "broadcast") -> tuple[np.ndarray, str]:
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
    if coopmat_eligible(region) and in_dtype in (np.float16, np.float32):
        out, ex = run_fused_region_coopmat(region, A, B, bias)
        if ex == "metal_runtime":
            return out, ex
    if bf16 is not None and in_dtype == bf16:
        out, ex = run_fused_region(
            region, np.asarray(A, np.float32), np.asarray(B, np.float32),
            None if bias is None else np.asarray(bias, np.float32), variant)
        return out.astype(bf16), ex
    if in_dtype == np.float16:
        return _run_fused_region_f16(region, A, B, bias, variant)

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

    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = fp(bias_arr) if bias_arr is not None else None
    has_bias = 1 if region.has_bias else 0

    sym = _synth_symbol()
    if sym is not None and N <= SYNTH_MAX_N:
        source = synthesize_matmul_epilogue_msl(region, variant).encode("utf-8")
        out = np.zeros((M, N), np.float32)
        rc = sym(source, _ENTRY.encode("utf-8"), fp(A), fp(B), bias_ptr, fp(out),
                 M, N, K, has_bias)
        if rc == 1:
            return out, "metal_runtime"

    # Large N (over the per-thread stack cap): the threadgroup-tiled kernel keeps
    # the score row in dynamic threadgroup memory, lifting the bound to ~8192.
    tiled = _synth_tiled_symbol()
    if tiled is not None and SYNTH_MAX_N < N <= SYNTH_MAX_N_TILED:
        source = synthesize_matmul_epilogue_msl_tiled(region).encode("utf-8")
        out = np.zeros((M, N), np.float32)
        rc = tiled(source, _ENTRY_TILED.encode("utf-8"), fp(A), fp(B), bias_ptr,
                   fp(out), M, N, K, has_bias)
        if rc == 1:
            return out, "metal_runtime"

    return region.reference(A, B, bias_arr), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# F2c — attention synthesis (matmul -> softmax -> matmul)
# ─────────────────────────────────────────────────────────────────────────────

_ATTN_ENTRY = "synth_attention"


@dataclass(frozen=True)
class AttentionRegion:
    """A fused scaled-dot-product-attention block ``O = softmax(scale·Q·Kᵀ)·V``.

    The two matmuls + the softmax between them collapse into ONE synthesized
    kernel (one query row per thread): the score row in a stack accumulator, a
    numerically-stable two-pass softmax, then the ``P·V`` contraction.  ``causal``
    masks keys ``n > m`` (decoder self-attention).  This is the same shape as the
    hand-written ``matmul_softmax_matmul`` kernel — but synthesized, so scale /
    causal / shape variants need no new kernel."""

    scale: float = 1.0
    causal: bool = False
    q_name: str = "Q"
    k_name: str = "K"
    v_name: str = "V"

    def reference(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """The unfused result in numpy — the horizontal-oracle ground truth."""
        Q = np.asarray(Q, np.float32)
        K = np.asarray(K, np.float32)
        V = np.asarray(V, np.float32)
        scores = (Q @ K.T) * np.float32(self.scale)
        if self.causal:
            m = np.arange(scores.shape[0])[:, None]
            n = np.arange(scores.shape[1])[None, :]
            scores = np.where(n > m, np.float32(-np.inf), scores)
        scores = scores - scores.max(-1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(-1, keepdims=True)
        return (p @ V).astype(np.float32)


def synthesize_attention_msl(region: AttentionRegion = AttentionRegion()) -> str:
    """Emit the MSL source for a fused attention block — one query row per thread.

    The source is shape- *and* scale/causal-independent (those are runtime
    buffers), so one cached pipeline serves every attention call."""
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void {_ATTN_ENTRY}(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float*       O   [[buffer(3)]],
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
        for (int d = 0; d < D; ++d) s += Q[q_off + d] * K[k_off + d];
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
        for (int n = 0; n < Nk; ++n) acc += scores[n] * V[n * Dv + dv];
        O[o_off + dv] = acc * inv;
    }}
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


def run_fused_attention(region: AttentionRegion, Q: np.ndarray, K: np.ndarray,
                        V: np.ndarray) -> tuple[np.ndarray, str]:
    """Run the attention block as ONE synthesized fused kernel on Metal.  Returns
    ``(output, execution)`` — ``"metal_runtime"`` if the synthesized kernel ran,
    else ``"reference"`` (numpy).  Either way the numbers are correct."""
    Q = np.ascontiguousarray(Q, np.float32)
    K = np.ascontiguousarray(K, np.float32)
    V = np.ascontiguousarray(V, np.float32)
    M, D = Q.shape
    Nk, Dk = K.shape
    Nv, Dv = V.shape
    if Dk != D:
        raise ValueError(f"Q/K head_dim mismatch: Q {Q.shape}, K {K.shape}")
    if Nv != Nk:
        raise ValueError(f"K/V seqlen mismatch: K {K.shape}, V {V.shape}")

    sym = _attn_symbol()
    if sym is not None and Nk <= SYNTH_MAX_N:
        source = synthesize_attention_msl(region).encode("utf-8")
        out = np.zeros((M, Dv), np.float32)
        fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = sym(source, _ATTN_ENTRY.encode("utf-8"), fp(Q), fp(K), fp(V), fp(out),
                 M, Nk, D, Dv, ctypes.c_float(region.scale),
                 1 if region.causal else 0)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(Q, K, V), "reference"


# ─────────────────────────────────────────────────────────────────────────────
# F3 — fusion cost model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FusionCost:
    """The analytical profitability of fusing a region into one synthesized
    kernel.  ``fusible`` is the hard gate (does the row fit the stack
    accumulator? is there anything to fuse?); ``score`` ranks the profitable
    candidates by the work they save (dispatches collapsed + intermediate DRAM
    traffic kept in registers).  A non-fusible region scores ``-inf`` so the gate
    leaves it to the per-op path (where large-N matmul has tiled/MPS kernels)."""

    fusible: bool
    dispatches_unfused: int
    dispatches_fused: int
    bytes_saved: int
    reason: str = ""

    @property
    def dispatch_saved(self) -> int:
        return self.dispatches_unfused - self.dispatches_fused

    @property
    def score(self) -> float:
        if not self.fusible:
            return float("-inf")
        # one synthesized kernel is worth at least its collapsed dispatches; the
        # avoided intermediate round-trips (in MB) break ties between candidates.
        return self.dispatch_saved + self.bytes_saved / (1024.0 * 1024.0)


def fusion_cost(region: FusedRegion, M: int, N: int, K: int) -> FusionCost:
    """Cost of fusing a ``matmul -> pointwise(-> reduction)`` region at (M,N,K)."""
    n_chain = len(region.epilogue) + (1 if region.reduction is not None else 0)
    unfused = 1 + n_chain                      # the matmul + each epilogue dispatch
    if N > SYNTH_MAX_N_TILED:                   # beyond even the tiled kernel
        return FusionCost(False, unfused, unfused, 0,
                          f"N={N} exceeds tiled threadgroup cap {SYNTH_MAX_N_TILED}")
    if n_chain == 0:
        return FusionCost(False, unfused, unfused, 0, "nothing to fuse")
    # each unfused epilogue writes then re-reads an M×N intermediate; fusion keeps
    # it in the per-thread accumulator.
    bytes_saved = 2 * n_chain * M * N * 4
    return FusionCost(True, unfused, 1, bytes_saved)


def attention_cost(region: AttentionRegion, M: int, Nk: int, D: int,
                   Dv: int) -> FusionCost:
    """Cost of fusing a ``matmul -> softmax -> matmul`` attention block."""
    unfused = 3                                # QKᵀ + softmax + PV
    if Nk > SYNTH_MAX_N:
        return FusionCost(False, unfused, unfused, 0,
                          f"Nk={Nk} exceeds per-thread stack cap {SYNTH_MAX_N}")
    bytes_saved = 2 * M * Nk * 4               # the score matrix, kept in registers
    return FusionCost(True, unfused, 1, bytes_saved)


def should_fuse_region(region: FusedRegion, M: int, N: int, K: int) -> bool:
    return fusion_cost(region, M, N, K).score > 0.0


def should_fuse_attention(region: AttentionRegion, M: int, Nk: int, D: int,
                          Dv: int) -> bool:
    return attention_cost(region, M, Nk, D, Dv).score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# F4 — codegen-gated oracle (verify a synthesized kernel before trusting it)
# ─────────────────────────────────────────────────────────────────────────────

#: Cache of per-region-class verification verdicts (one probe per shape-class).
_VERIFY_CACHE: dict[Any, bool] = {}


def clear_verification_cache() -> None:
    _VERIFY_CACHE.clear()


def verify_synthesized_region(region: FusedRegion, *, seed: int = 0,
                              atol: float = 1e-3, force: bool = False) -> bool:
    """Codegen-gated oracle: run the *synthesized* kernel for ``region`` on a small
    probe and compare it to the unfused numpy reference.  Returns ``True`` only if
    the GPU result matches (a correct synthesizer) — or if no synthesized kernel
    ran (no Metal: the reference path is trusted by construction).  Returns
    ``False`` when a kernel ran and *diverged* — a synthesizer bug — so the caller
    falls back to the trusted per-op path.  This is the codegen analogue of the
    magellan/alphaevolve reward-hack rejection: a faster-but-wrong kernel is
    refused.  Verdicts are cached per region-class; pass ``force`` to re-probe."""
    key = ("R", region.epilogue, region.reduction, region.has_bias,
           round(region.eps, 9))
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    B = rng.standard_normal((12, 16)).astype(np.float32)
    bias = (rng.standard_normal((16,)).astype(np.float32)
            if region.has_bias else None)
    out, execution = run_fused_region(region, A, B, bias)
    if execution != "metal_runtime":
        verdict = True                         # no synthesized kernel to distrust
    else:
        verdict = bool(np.allclose(out, region.reference(A, B, bias), atol=atol))
    _VERIFY_CACHE[key] = verdict
    return verdict


def verify_synthesized_attention(region: AttentionRegion, *, seed: int = 0,
                                 atol: float = 1e-3, force: bool = False) -> bool:
    """Codegen-gated oracle for a synthesized attention block (see
    ``verify_synthesized_region``)."""
    key = ("A", round(region.scale, 9), region.causal)
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((8, 16)).astype(np.float32)
    K = rng.standard_normal((8, 16)).astype(np.float32)
    V = rng.standard_normal((8, 16)).astype(np.float32)
    out, execution = run_fused_attention(region, Q, K, V)
    if execution != "metal_runtime":
        verdict = True
    else:
        verdict = bool(np.allclose(out, region.reference(Q, K, V), atol=atol))
    _VERIFY_CACHE[key] = verdict
    return verdict


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


# ─────────────────────────────────────────────────────────────────────────────
# F1a — region discovery
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Op:
    """Minimal op for discovery: name + input value-ids + output value-id."""

    name: str
    inputs: tuple[str, ...]
    output: str
    attrs: dict[str, Any] = field(default_factory=dict)


#: Graph op names that are pointwise epilogue candidates → the EPILOGUE_OPS key.
_POINTWISE_ALIASES: dict[str, str] = {
    "tessera.relu": "relu", "tessera.gelu": "gelu", "tessera.silu": "silu",
    "tessera.sigmoid": "sigmoid", "tessera.tanh": "tanh",
    "tessera.add": "bias",                 # add of a matmul result + a vector
    "tessera.bias_add": "bias",
    "relu": "relu", "gelu": "gelu", "silu": "silu", "sigmoid": "sigmoid",
    "tanh": "tanh", "add": "bias", "bias_add": "bias",
}
_MATMUL_NAMES = {"tessera.matmul", "tessera.gemm", "matmul", "gemm"}
#: Terminal reduction epilogue op names → the REDUCTION_OPS key.
_REDUCTION_ALIASES: dict[str, str] = {
    "tessera.rmsnorm": "rmsnorm", "tessera.softmax": "softmax",
    "rmsnorm": "rmsnorm", "softmax": "softmax",
}


def discover_fusable_regions(ops: list[_Op]) -> list[tuple[int, list[int], FusedRegion]]:
    """Find maximal ``matmul -> pointwise-chain`` regions in ``ops``.

    Returns ``(matmul_index, [epilogue_indices], FusedRegion)`` per region.  An
    op only fuses into the chain if it is the *sole* consumer of the running
    value — otherwise fusing would drop an intermediate another op reads.
    """
    use_count: dict[str, int] = {}
    for op in ops:
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1
    by_input: dict[str, list[int]] = {}
    for i, op in enumerate(ops):
        for v in op.inputs:
            by_input.setdefault(v, []).append(i)

    regions: list[tuple[int, list[int], FusedRegion]] = []
    consumed: set[int] = set()
    for i, op in enumerate(ops):
        if i in consumed or op.name not in _MATMUL_NAMES:
            continue
        chain: list[int] = []
        epi: list[str] = []
        reduction: str | None = None
        cur = op.output
        while True:
            if use_count.get(cur, 0) != 1:
                break                          # intermediate has another consumer
            consumers = by_input.get(cur, [])
            if len(consumers) != 1:
                break
            j = consumers[0]
            nxt = ops[j]
            key = _POINTWISE_ALIASES.get(nxt.name)
            if key is None:
                break                          # not a pointwise epilogue op
            chain.append(j)
            epi.append(key)
            consumed.add(j)
            cur = nxt.output
        # a terminal reduction (rmsnorm/softmax) may close the chain — single-use.
        if use_count.get(cur, 0) == 1:
            consumers = by_input.get(cur, [])
            if len(consumers) == 1:
                j = consumers[0]
                rkey = _REDUCTION_ALIASES.get(ops[j].name)
                if rkey is not None:
                    reduction = rkey
                    chain.append(j)
                    consumed.add(j)
        if epi or reduction:
            a, b = op.inputs[0], op.inputs[1]
            regions.append((i, chain, FusedRegion(tuple(epi), reduction=reduction,
                                                  a_name=a, b_name=b)))
    return regions


_SOFTMAX_NAMES = {"tessera.softmax", "softmax"}
_SCALE_NAMES = {"tessera.mul", "tessera.scale", "mul", "scale"}


def discover_attention_regions(
    ops: list[_Op],
) -> list[tuple[list[int], AttentionRegion, tuple[str, str, str]]]:
    """Find ``matmul(Q,Kᵀ) -> [scale] -> softmax -> matmul(P,V)`` regions.

    Returns ``([op_indices], AttentionRegion, (q_value, k_value, v_value))`` per
    region.  All intermediates must be single-use (else fusing drops a value
    another op reads).  An optional scalar-multiply between the score matmul and
    the softmax sets ``region.scale`` when its factor is a constant attribute."""
    use_count: dict[str, int] = {}
    for op in ops:
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1
    by_input: dict[str, list[int]] = {}
    for i, op in enumerate(ops):
        for v in op.inputs:
            by_input.setdefault(v, []).append(i)

    def sole_consumer(val: str) -> int | None:
        if use_count.get(val, 0) != 1:
            return None
        cons = by_input.get(val, [])
        return cons[0] if len(cons) == 1 else None

    regions: list[tuple[list[int], AttentionRegion, tuple[str, str, str]]] = []
    consumed: set[int] = set()
    for i, op in enumerate(ops):
        if i in consumed or op.name not in _MATMUL_NAMES or len(op.inputs) < 2:
            continue
        q_val, k_val = op.inputs[0], op.inputs[1]
        chain = [i]
        scale = 1.0
        cur = op.output

        j = sole_consumer(cur)
        if j is None:
            continue
        # optional scalar-multiply applying the attention scale.
        if ops[j].name in _SCALE_NAMES:
            factor = ops[j].attrs.get("scale", ops[j].attrs.get("factor"))
            if isinstance(factor, (int, float)):
                scale = float(factor)
            chain.append(j)
            consumed.add(j)
            cur = ops[j].output
            j = sole_consumer(cur)
            if j is None:
                continue
        if ops[j].name not in _SOFTMAX_NAMES:
            continue
        chain.append(j)
        consumed.add(j)
        cur = ops[j].output

        j = sole_consumer(cur)
        if j is None or ops[j].name not in _MATMUL_NAMES or len(ops[j].inputs) < 2:
            continue
        # the softmax result must be the FIRST operand of the P@V matmul.
        if ops[j].inputs[0] != cur:
            continue
        v_val = ops[j].inputs[1]
        chain.append(j)
        consumed.add(j)
        regions.append((chain, AttentionRegion(scale=scale), (q_val, k_val, v_val)))
    return regions


__all__ = [
    "EPILOGUE_OPS",
    "EpilogueOp",
    "REDUCTION_OPS",
    "ReductionOp",
    "FusedRegion",
    "AttentionRegion",
    "FusionCost",
    "fusion_cost",
    "attention_cost",
    "should_fuse_region",
    "should_fuse_attention",
    "verify_synthesized_region",
    "verify_synthesized_attention",
    "clear_verification_cache",
    "SYNTH_VARIANTS",
    "SYNTH_DTYPES",
    "SYNTH_MAX_N_TILED",
    "synthesize_matmul_epilogue_msl_tiled",
    "synthesize_matmul_epilogue_coopmat_msl",
    "run_fused_region_coopmat",
    "coopmat_eligible",
    "synthesize_matmul_reduction_coopmat_msl",
    "run_fused_region_coopmat_reduce",
    "coopmat_reduce_eligible",
    "SYNTH_COOPMAT_REDUCE_MAX_N",
    "AutotuneRecord",
    "autotune_matmul_epilogue",
    "best_variant_for",
    "clear_autotune_corpus",
    "SYNTH_MAX_N",
    "synthesize_matmul_epilogue_msl",
    "synthesize_attention_msl",
    "run_fused_region",
    "run_fused_attention",
    "discover_fusable_regions",
    "discover_attention_regions",
]
