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

#: Cap on head_dim for the ONLINE-softmax attention kernel (M2): it streams keys
#: with no ``scores[Nk]`` array, keeping only an ``acc[head_dim]`` accumulator —
#: so it trades the SYNTH_MAX_N key cap for a head_dim cap, and handles Nk far
#: beyond SYNTH_MAX_N (the large-context / long-sequence attention case). Matches
#: the hand-written flash-attn kernel's head_dim envelope.
SYNTH_MAX_D = 256


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
#:
#: C2 close-out (2026-06-17): this set is deliberately NOT grown beyond the common
#: matmul-epilogue activations (bias/relu/gelu/silu/sigmoid/tanh).  EPILOGUE_OPS
#: fuses an activation *into* the matmul kernel (no intermediate write); rarer
#: activations are instead handled by the general pointwise-DAG path
#: (``POINTWISE_OPS`` / ``discover_pointwise_graph``), which fuses the matmul's
#: result tail into one kernel as a separate (still on-GPU) dispatch.  So the only
#: ops worth in-matmul fusion are the hot activations already here; growing this
#: further would be speculative.
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


def _layer_norm_ref(x: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)


#: Terminal reduction epilogues.  Adding one here retires the corresponding
#: hand-written matmul_<reduction> kernel.
# NB: O-writes go through ``ST(...)`` (a ``using ST = <io>;`` alias each kernel
# defines) — MSL's `bfloat` rejects *implicit* float→bfloat assignment (unlike
# `half`), so the store needs an explicit cast that is also valid for half/float.
REDUCTION_OPS: dict[str, ReductionOp] = {
    "rmsnorm": ReductionOp(
        "rmsnorm",
        "        float _ss = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) _ss += scores[n] * scores[n];\n"
        "        float _inv = rsqrt(_ss / float(N) + {eps}f);\n"
        "        for (int n = 0; n < N; ++n) O[o_off + n] = ST(scores[n] * _inv);",
        _rmsnorm_ref,
    ),
    "softmax": ReductionOp(
        "softmax",
        "        float _mx = -INFINITY;\n"
        "        for (int n = 0; n < N; ++n) _mx = max(_mx, scores[n]);\n"
        "        float _sm = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) {{ scores[n] = exp(scores[n] - _mx);"
        " _sm += scores[n]; }}\n"
        "        for (int n = 0; n < N; ++n) O[o_off + n] = ST(scores[n] / _sm);",
        _softmax_ref,
    ),
    "layer_norm": ReductionOp(
        "layer_norm",
        "        float _mean = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) _mean += scores[n];\n"
        "        _mean /= float(N);\n"
        "        float _var = 0.0f;\n"
        "        for (int n = 0; n < N; ++n) {{ float _d = scores[n] - _mean;"
        " _var += _d * _d; }}\n"
        "        float _inv = rsqrt(_var / float(N) + {eps}f);\n"
        "        for (int n = 0; n < N; ++n) O[o_off + n] = ST((scores[n] - _mean) * _inv);",
        _layer_norm_ref,
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
    # M4 residual: a full (M,N) tensor added to the matmul result after the
    # pointwise chain — the transformer ``x + sublayer(x)`` pattern (per-element,
    # distinct from the per-feature ``bias``). v1 is non-reduction only.
    residual: bool = False
    # M4 prologue: a pure-pointwise chain applied *elementwise to the A operand*
    # before the contraction — ``matmul(act(A), B)`` (e.g. project a GeLU'd
    # activation). Each op is the same EPILOGUE_OPS vocabulary minus bias, baked
    # into the kernel source at the A-load site (so NO extra buffer / ABI arg).
    # Because it acts per-element of A before the K-sum, it equals applying the
    # op to A then contracting — exact for any pointwise op.
    prologue: tuple[str, ...] = ()
    # Graph bookkeeping (like a_name/b_name; NOT synthesis-semantic): the op
    # indices of the prologue activation chain in the source graph, so the
    # orchestrator can mark them consumed. Empty unless discovery found a prologue.
    prologue_src_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for op in self.epilogue:
            if op not in EPILOGUE_OPS:
                raise ValueError(f"unknown epilogue op {op!r}")
        bias_ops = [op for op in self.epilogue if EPILOGUE_OPS[op].needs_bias]
        if len(bias_ops) > 1:
            raise ValueError("at most one bias op per region")
        for op in self.prologue:
            if op not in EPILOGUE_OPS:
                raise ValueError(f"unknown prologue op {op!r}")
            if EPILOGUE_OPS[op].needs_bias:
                raise ValueError(f"prologue op {op!r} cannot need a bias")
        if self.reduction is not None and self.reduction not in REDUCTION_OPS:
            raise ValueError(f"unknown reduction epilogue {self.reduction!r}")
        if self.residual and self.reduction is not None:
            raise ValueError("residual + reduction not supported (v1)")
        if (not self.epilogue and self.reduction is None and not self.residual
                and not self.prologue):
            raise ValueError(
                "a region must have a prologue, epilogue op, reduction, or residual")

    @property
    def has_bias(self) -> bool:
        return any(EPILOGUE_OPS[op].needs_bias for op in self.epilogue)

    @property
    def has_residual(self) -> bool:
        return self.residual

    @property
    def has_prologue(self) -> bool:
        return bool(self.prologue)

    def reference(self, A: np.ndarray, B: np.ndarray,
                  bias: np.ndarray | None = None,
                  residual: np.ndarray | None = None) -> np.ndarray:
        """The *unfused* result: matmul, pointwise chain, optional residual add,
        then the reduction, in numpy — the horizontal-oracle ground truth the
        synthesized kernel matches."""
        a = np.asarray(A, np.float32)
        for op in self.prologue:
            a = EPILOGUE_OPS[op].ref(a)          # pointwise on A before contraction
        out = a @ np.asarray(B, np.float32)
        for op in self.epilogue:
            spec = EPILOGUE_OPS[op]
            if spec.needs_bias:
                if bias is None:
                    raise ValueError(f"region needs a bias for {op!r}")
                out = out + np.asarray(bias, np.float32)
            else:
                out = spec.ref(out)
        if self.residual:
            if residual is None:
                raise ValueError("region needs a residual")
            out = out + np.asarray(residual, np.float32)
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


def _prologue_msl(region: "FusedRegion", indent: str) -> str:
    """MSL that transforms the loaded A element ``a`` in place — each prologue op
    wrapped as ``{ float v = a; <op.msl>; a = v; }`` so the EPILOGUE_OPS bodies
    (which operate on ``v``) are reused verbatim.  Empty when no prologue."""
    return "".join(
        f"{indent}{{ float v = a; {EPILOGUE_OPS[op].msl} a = v; }}\n"
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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)
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
        red = REDUCTION_OPS[region.reduction].msl.format(eps=region.eps)
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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)

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
    pointwise = "\n".join(f"            {EPILOGUE_OPS[op].msl}" for op in region.epilogue)
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


@dataclass(frozen=True)
class NormChainRegion:
    """A non-matmul-rooted row-reduction region:
    ``norm(x [+ residual]) [* gamma] [+ beta]``.

    ``norm`` is a ``REDUCTION_OPS`` key (``rmsnorm``/``layer_norm``/``softmax``).
    ``add_residual`` fuses a preceding residual add (pre-norm pattern); ``weight``
    (per-feature γ) and ``bias`` (per-feature β) fuse the post-norm affine — i.e.
    the real transformer RMSNorm/LayerNorm with affine. fp32 math throughout."""

    norm: str
    add_residual: bool = False
    weight: bool = False
    bias: bool = False
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.norm not in REDUCTION_OPS:
            raise ValueError(f"unknown norm {self.norm!r}")

    def reference(self, X: np.ndarray, residual: np.ndarray | None = None,
                  gamma: np.ndarray | None = None,
                  beta: np.ndarray | None = None) -> np.ndarray:
        """The unfused numpy ground truth the synthesized kernel must match."""
        v = np.asarray(X, np.float32)
        if self.add_residual:
            if residual is None:
                raise ValueError("region needs a residual")
            v = v + np.asarray(residual, np.float32)
        out = REDUCTION_OPS[self.norm].ref(v, self.eps).astype(np.float32)
        if self.weight:
            if gamma is None:
                raise ValueError("region needs a weight (gamma)")
            out = out * np.asarray(gamma, np.float32).reshape(-1)
        if self.bias:
            if beta is None:
                raise ValueError("region needs a bias (beta)")
            out = out + np.asarray(beta, np.float32).reshape(-1)
        return out.astype(np.float32)


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
    red = REDUCTION_OPS[region.norm].msl.format(eps=region.eps)
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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


_GELU_EXPR = ("(0.5f*({0})*(1.0f+tanh(clamp(0.7978845608028654f*"
              "(({0})+0.044715f*({0})*({0})*({0})), -30.0f, 30.0f))))")

#: pointwise op → (arity, MSL expr template over operand temps, numpy ref).
POINTWISE_OPS: dict[str, tuple[int, str, Any]] = {
    "add": (2, "({0} + {1})", lambda a, b: a + b),
    "sub": (2, "({0} - {1})", lambda a, b: a - b),
    "mul": (2, "({0} * {1})", lambda a, b: a * b),
    "div": (2, "({0} / {1})", lambda a, b: a / b),
    "relu": (1, "max({0}, 0.0f)", lambda a: np.maximum(a, 0.0)),
    "sigmoid": (1, "(1.0f/(1.0f+exp(-({0}))))", _sigmoid_np),
    "tanh": (1, "tanh({0})", np.tanh),
    "silu": (1, "(({0})/(1.0f+exp(-({0}))))", lambda a: a * _sigmoid_np(a)),
    "gelu": (1, _GELU_EXPR, _gelu),
    "neg": (1, "(-({0}))", lambda a: -np.asarray(a)),
    "abs": (1, "fabs({0})", np.abs),
    "exp": (1, "exp({0})", np.exp),
    # Phase C — real-valued ops that already have single-op GPU lanes; adding
    # them here enlarges fusable pointwise DAGs (fewer dispatches). Each is
    # codegen-gated by verify_synthesized_pointwise (equal_nan-aware, so the
    # domain-restricted ones below are safe).
    "sqrt": (1, "sqrt({0})", np.sqrt),
    "rsqrt": (1, "rsqrt({0})", lambda a: 1.0 / np.sqrt(a)),
    "log": (1, "log({0})", np.log),
    "log1p": (1, "log(1.0f + ({0}))", np.log1p),
    "expm1": (1, "(exp({0}) - 1.0f)", np.expm1),
    "reciprocal": (1, "(1.0f / ({0}))", lambda a: 1.0 / np.asarray(a)),
    # Numerically-stable softplus: max(x,0) + log1p(exp(-|x|)); MSL and ref share
    # the identical definition so the oracle compares like-for-like.
    "softplus": (1, "(max({0}, 0.0f) + log(1.0f + exp(-fabs({0}))))",
                 lambda a: np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))),
    # Phase C tail — binary min/max and unary sign (real catalog ops with
    # single-op lanes; adding them lets DAGs containing min/max/sign fuse rather
    # than bailing at those nodes). NaN-safe under the oracle's equal_nan compare.
    "maximum": (2, "max({0}, {1})", np.maximum),
    "minimum": (2, "min({0}, {1})", np.minimum),
    "sign": (1, "sign({0})", np.sign),
}
#: graph op-name → pointwise vocab key.
_POINTWISE_NAMES: dict[str, str] = {}
for _k in POINTWISE_OPS:
    _POINTWISE_NAMES[_k] = _k
    _POINTWISE_NAMES[f"tessera.{_k}"] = _k


def is_pointwise_op(op_name: str) -> bool:
    """True if ``op_name`` is in the pointwise-DAG synthesizer's vocabulary — the
    single source of truth shared by ``discover_pointwise_graph`` (runtime prepass)
    and the compile-time routing recognizer (``driver._apple_gpu_chain_kind``), so
    the two never drift."""
    return op_name in _POINTWISE_NAMES


@dataclass(frozen=True)
class PointwiseGraphRegion:
    """A connected DAG of same-shape pointwise ops. ``ops`` is topo-ordered
    ``(vocab_key, (input_value_ids,), out_value_id)``; ``inputs`` are the
    external value-ids (→ buffer indices, in order); ``output`` is the terminal
    value-id."""

    ops: tuple[tuple[str, tuple[str, ...], str], ...]
    inputs: tuple[str, ...]
    output: str

    def reference(self, *arrays: np.ndarray) -> np.ndarray:
        env = {v: np.asarray(a, np.float32)
               for v, a in zip(self.inputs, arrays)}
        for key, ins, out in self.ops:
            _arity, _expr, ref = POINTWISE_OPS[key]
            env[out] = np.asarray(ref(*[env[i] for i in ins]), np.float32)
        return env[self.output]


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


# Up to 16 input buffers (MSL allows ~31; small pointwise DAGs stay well under).
_PW_MAX_INPUTS = 16


def discover_pointwise_graph(ops: list[_Op], skip: set[int] | None = None
                             ) -> list[tuple[list[int], PointwiseGraphRegion]]:
    """Find maximal connected same-shape pointwise regions with a single exit.

    Returns ``([op_indices], PointwiseGraphRegion)`` per region. A region fires
    only if it has ≥2 ops (a single op is left to the MPSGraph elementwise lane)
    and one exit value (consumed outside the region or the terminal)."""
    skip = skip or set()
    cand = [i for i, op in enumerate(ops)
            if i not in skip and op.name in _POINTWISE_NAMES]
    if not cand:
        return []
    candset = set(cand)
    produced = {ops[i].output: i for i in cand}
    use_count: dict[str, int] = {}
    for op in ops:
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1

    # Union-find over candidate ops connected by an intra-candidate value edge.
    parent = {i: i for i in cand}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in cand:
        for v in ops[i].inputs:
            p = produced.get(v)
            if p is not None and p in candset:
                parent[find(i)] = find(p)

    comps: dict[int, list[int]] = {}
    for i in cand:
        comps.setdefault(find(i), []).append(i)

    regions: list[tuple[list[int], PointwiseGraphRegion]] = []
    for members in comps.values():
        if len(members) < 2 or len(members) > _PW_MAX_INPUTS * 2:
            continue
        members = sorted(members)
        mset = set(members)
        internal = {ops[i].output for i in members}
        # external inputs: operands not produced inside the region (ordered).
        ext: list[str] = []
        for i in members:
            for v in ops[i].inputs:
                if v not in internal and v not in ext:
                    ext.append(v)
        if len(ext) > _PW_MAX_INPUTS:
            continue
        # single exit: exactly one internal value used outside the region.
        exits = [ops[i].output for i in members
                 if any(c not in mset for c in _consumers(ops, ops[i].output))
                 or use_count.get(ops[i].output, 0) == 0]
        # the terminal (use_count 0) or the unique externally-consumed value.
        exits = list(dict.fromkeys(exits))
        if len(exits) != 1:
            continue
        out_v = exits[0]
        region_ops = tuple(
            (_POINTWISE_NAMES[ops[i].name], tuple(ops[i].inputs), ops[i].output)
            for i in members)
        regions.append((members,
                        PointwiseGraphRegion(region_ops, tuple(ext), out_v)))
    return regions


def _consumers(ops: list[_Op], value: str) -> list[int]:
    return [i for i, op in enumerate(ops) if value in op.inputs]


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


def _is_trailing_feature(a_shape: tuple[int, ...],
                         out_shape: tuple[int, ...]) -> bool:
    """True when ``a`` broadcasts to ``out`` purely along the *last* axis — its
    last dim equals out's last dim and every other dim is 1 (e.g. ``(cols,)`` /
    ``(1, cols)`` against ``(rows, cols)``). Exactly the case where flat index
    ``gid % cols`` selects the right element, so the GPU kernel can broadcast it.
    Per-row / internal broadcast (e.g. ``(rows, 1)``) returns False → reference."""
    if not a_shape or not out_shape:
        return False
    if a_shape[-1] != out_shape[-1]:
        return False
    return all(d == 1 for d in a_shape[:-1])


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
#: reduce-kind → (MSL init, MSL accumulate-expr template, numpy fn).
_PW_REDUCE_KINDS: dict[str, tuple[str, str, Any]] = {
    "sum": ("0.0f", "acc + v", lambda a: a.sum(-1)),
    "mean": ("0.0f", "acc + v", lambda a: a.mean(-1)),
    "amax": ("-INFINITY", "max(acc, v)", lambda a: a.max(-1)),
    "amin": ("INFINITY", "min(acc, v)", lambda a: a.min(-1)),
}


@dataclass(frozen=True)
class PointwiseReduceRegion:
    """A pointwise DAG (``ops`` topo-ordered, same vocab as PointwiseGraphRegion)
    terminated by a plain row reduction (``reduce`` ∈ sum/mean/amax/amin) over the
    last axis. ``output`` is the pointwise terminal value-id fed to the reduction;
    the result drops the last axis (rows → scalar)."""

    ops: tuple[tuple[str, tuple[str, ...], str], ...]
    inputs: tuple[str, ...]
    output: str
    reduce: str

    def __post_init__(self) -> None:
        if self.reduce not in _PW_REDUCE_KINDS:
            raise ValueError(f"unknown reduce {self.reduce!r}")

    def reference(self, *arrays: np.ndarray) -> np.ndarray:
        env = {v: np.asarray(a, np.float32)
               for v, a in zip(self.inputs, arrays)}
        for key, ins, out in self.ops:
            _arity, _expr, ref = POINTWISE_OPS[key]
            env[out] = np.asarray(ref(*[env[i] for i in ins]), np.float32)
        _init, _acc, fn = _PW_REDUCE_KINDS[self.reduce]
        return np.asarray(fn(env[self.output]), np.float32)


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
    # Orientation of the *raw* operands as the score matmul consumes them, read
    # from its transpose flags (M2 attention-orientation fix). The synthesizer
    # works on natural Q(M,D)/K(Nk,D); these say whether a raw operand is already
    # transposed and must be flipped first. Default False = the operand is
    # natural — so direct AttentionRegion() construction is unchanged.
    q_transposed: bool = False
    k_transposed: bool = False

    def _natural(self, Q: np.ndarray, K: np.ndarray, cast: bool = True):
        """Flip raw operands to natural Q(M,D)/K(Nk,D) per the transpose flags.
        ``cast`` (default) coerces to f32 for the numpy reference; ``cast=False``
        preserves the storage dtype (f16/bf16) for the half-precision GPU path."""
        Q = np.asarray(Q, np.float32) if cast else np.asarray(Q)
        K = np.asarray(K, np.float32) if cast else np.asarray(K)
        if self.q_transposed:
            Q = np.ascontiguousarray(Q.T)
        if self.k_transposed:
            K = np.ascontiguousarray(K.T)
        return Q, K

    def reference(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """The unfused result in numpy — the horizontal-oracle ground truth."""
        Q, K = self._natural(Q, K)
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


# ─────────────────────────────────────────────────────────────────────────────
# F2e — gated-matmul (SwiGLU gate) synthesis: O = f(A @ Wg) ⊙ (A @ Wu)
#
# Two matmuls sharing the A operand, an activation f on the gate branch, and an
# elementwise multiply — the transformer FFN gate written from PRIMITIVE ops
# (matmul, silu, mul). Complementary to the library ``gpu_swiglu`` (which also
# folds the down-projection): this fires only when a graph is expressed in
# primitives, never displacing the library op. One A row per thread fills BOTH
# score rows in a single K-loop (A read once, shared), then gate-act × up.
# ─────────────────────────────────────────────────────────────────────────────

#: Per-thread cap for the gated kernel. Two fp32 stack rows (gate + up) live
#: per thread, so the cap is half the single-row epilogue cap to stay within the
#: register/stack budget — larger H falls to the per-op (two-matmul) path.
SYNTH_GATED_MAX_H = SYNTH_MAX_N // 2
_GATED_ENTRY = "synth_gated_matmul"


@dataclass(frozen=True)
class GatedMatmulRegion:
    """A fused gated MLP projection ``O = f(A @ Wg) ⊙ (A @ Wu)``.

    Two matmuls share the A operand; ``gate_act`` (a unary EPILOGUE_OPS
    activation — ``silu`` for SwiGLU) is applied to the gate branch, then an
    elementwise multiply with the up branch.  ``A:(M,K)  Wg,Wu:(K,H) -> O:(M,H)``.
    This is the SwiGLU *gate* (no down-projection) from primitive ops; the library
    ``gpu_swiglu`` covers the whole block including the down-proj, so the
    discoverer only fires on primitive-op graphs that don't call the ``swiglu``
    op."""

    gate_act: str = "silu"
    a_name: str = "A"
    wg_name: str = "Wg"
    wu_name: str = "Wu"

    def __post_init__(self) -> None:
        if (self.gate_act not in EPILOGUE_OPS
                or EPILOGUE_OPS[self.gate_act].needs_bias):
            raise ValueError(
                f"gate activation must be a unary pointwise op, got "
                f"{self.gate_act!r}")

    def reference(self, A: np.ndarray, Wg: np.ndarray,
                  Wu: np.ndarray) -> np.ndarray:
        """The unfused result in numpy — the horizontal-oracle ground truth."""
        a = np.asarray(A, np.float32)
        gate = EPILOGUE_OPS[self.gate_act].ref(a @ np.asarray(Wg, np.float32))
        up = a @ np.asarray(Wu, np.float32)
        return (gate * up).astype(np.float32)


def synthesize_gated_matmul_msl(region: GatedMatmulRegion = GatedMatmulRegion(),
                                dtype: str = "f32") -> str:
    """Emit the MSL source for ``O = f(A @ Wg) ⊙ (A @ Wu)`` — one A row per thread.

    Both projections accumulate in a single K-loop (A[k] loaded once, fanned to
    the gate and up rows), so the shared input is read K times total, not 2K.
    The gate activation runs in fp32; the O-write goes through ``ST(...)`` (bfloat
    rejects implicit float→bfloat). ``dtype`` selects the I/O type."""
    io = _io_type(dtype)
    act = EPILOGUE_OPS[region.gate_act].msl        # operates on `v`
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


def should_fuse_gated(region: GatedMatmulRegion, M: int, H: int, K: int) -> bool:
    """F3 cost gate: the gated kernel keeps two fp32 rows of width H in the
    per-thread stack, so it only fuses when H fits the (halved) cap. Larger H is
    left to the per-op two-matmul path."""
    return H <= SYNTH_GATED_MAX_H


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


_GATED_VERIFY_CACHE: dict[Any, bool] = {}


def verify_synthesized_gated(region: GatedMatmulRegion, *, seed: int = 0,
                             atol: float = 1e-3, force: bool = False) -> bool:
    """Codegen-gated oracle for a gated-matmul region (see
    ``verify_synthesized_region``): run the synthesized kernel on a small probe
    and compare to the unfused numpy reference. ``True`` unless a kernel ran and
    diverged. Cached per gate-activation."""
    key = ("G", region.gate_act)
    if not force and key in _GATED_VERIFY_CACHE:
        return _GATED_VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 12)).astype(np.float32)
    Wg = rng.standard_normal((12, 16)).astype(np.float32)
    Wu = rng.standard_normal((12, 16)).astype(np.float32)
    out, execution = run_gated_matmul_region(region, A, Wg, Wu)
    if execution != "metal_runtime":
        verdict = True
    else:
        verdict = bool(np.allclose(out, region.reference(A, Wg, Wu), atol=atol))
    _GATED_VERIFY_CACHE[key] = verdict
    return verdict


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
# F3b — attention lowering SELECTOR (Workstream C)
# ─────────────────────────────────────────────────────────────────────────────
#
# FlashAttention's deeper lesson is "choose lowering by IO traffic," not "have a
# flash kernel." `attention_cost` above is a *gate* (fuse or not); this is a
# *selector* — it scores every feasible attention kernel variant by total
# off-chip byte movement (the FA currency) and picks the minimum, replacing the
# hard `Nk <= SYNTH_MAX_N` branch in `run_fused_attention`. Page-gather / offload
# staging bytes from a Workstream-A PagedKVState enter the score directly.


@dataclass(frozen=True)
class AttnLoweringCost:
    """The IO cost of one attention lowering variant.

    ``dram_bytes`` is the off-chip traffic that decides profitability: the fused
    kernels keep the M×Nk score matrix on-chip (no round-trip); the unfused
    reference writes and re-reads it. ``feasible`` encodes the hardware on-chip
    bound (stack/threadgroup capacity), not an arbitrary perf threshold.
    """

    variant: str            # "materialized" | "online" | "reference"
    feasible: bool
    dram_bytes: int
    flops: int
    reason: str = ""


def _attn_dram_bytes(M: int, Nk: int, D: int, Dv: int, elt: int, *,
                     score_roundtrip: bool, stage_bytes: int = 0) -> int:
    """Off-chip bytes: Q+K+V reads + O write (+ optional score round-trip and
    page-staging transfers)."""
    io = (M * D + Nk * D + Nk * Dv + M * Dv) * elt
    score = (2 * M * Nk * elt) if score_roundtrip else 0
    return io + score + stage_bytes


def attention_lowering_costs(
    M: int, Nk: int, D: int, Dv: int, *, elt_bytes: int = 4, stage_bytes: int = 0,
) -> tuple[AttnLoweringCost, ...]:
    """Score the three attention kernel variants at one shape.

    ``stage_bytes`` is the host→device transfer a paged/tiered KV gather adds
    (0 for resident/contiguous KV); it rides on the fused variants since they are
    what a paged consumer would dispatch.
    """
    qk = 2 * M * Nk * D          # QKᵀ
    pv = 2 * M * Nk * Dv         # P·V
    materialized = AttnLoweringCost(
        "materialized", Nk <= SYNTH_MAX_N,
        _attn_dram_bytes(M, Nk, D, Dv, elt_bytes, score_roundtrip=False,
                         stage_bytes=stage_bytes),
        qk + pv,
        "on-chip scores[Nk] stack" if Nk <= SYNTH_MAX_N
        else f"Nk={Nk} exceeds stack cap {SYNTH_MAX_N}",
    )
    online = AttnLoweringCost(
        "online", Dv <= SYNTH_MAX_D,
        _attn_dram_bytes(M, Nk, D, Dv, elt_bytes, score_roundtrip=False,
                         stage_bytes=stage_bytes),
        qk + pv + M * Nk,        # + streaming-softmax rescale
        "streaming softmax (Nk unbounded)" if Dv <= SYNTH_MAX_D
        else f"Dv={Dv} exceeds online head_dim cap {SYNTH_MAX_D}",
    )
    reference = AttnLoweringCost(
        "reference", True,
        _attn_dram_bytes(M, Nk, D, Dv, elt_bytes, score_roundtrip=True,
                         stage_bytes=stage_bytes),
        qk + pv,
        "unfused numpy — score matrix round-trips through DRAM",
    )
    return (materialized, online, reference)


def select_attention_lowering(
    M: int, Nk: int, D: int, Dv: int, *, elt_bytes: int = 4, stage_bytes: int = 0,
) -> AttnLoweringCost:
    """Pick the minimum-byte *feasible* attention lowering.

    Among feasible variants, rank by ``(dram_bytes, flops)`` — fewest off-chip
    bytes first (the FA objective), ties broken toward fewer FLOPs (materialized
    over online when both fit). The reference variant is always feasible but
    carries the score round-trip, so it loses whenever a fused kernel fits.
    """
    feasible = [c for c in attention_lowering_costs(
        M, Nk, D, Dv, elt_bytes=elt_bytes, stage_bytes=stage_bytes) if c.feasible]
    return min(feasible, key=lambda c: (c.dram_bytes, c.flops))


def paged_stage_bytes(kv_state: Any, token_indices: "list[int] | None" = None,
                      *, elt_bytes: int = 4) -> int:
    """Host→device staging bytes a gather over ``kv_state`` (a PagedKVState) adds.

    Connects Workstream A to the cost model: non-resident pages touched by the
    gather cost a transfer; resident/contiguous KV costs nothing. Returns 0 for
    states without a tiered page table.
    """
    try:
        from ..cache.paged_kv import as_paged_kv_state, PageTier
        state = as_paged_kv_state(kv_state)
        geo = state.kv_geometry()
        table = state.page_table()
    except Exception:
        return 0
    if token_indices is None:
        touched = {e.page_id for e in table}
    else:
        ps = geo.page_size
        touched = {int(t) // ps for t in token_indices}
    page_elems = geo.page_size * geo.num_heads * geo.head_dim
    cold = sum(1 for e in table
               if e.page_id in touched and e.tier is not PageTier.RESIDENT)
    return cold * page_elems * 2 * elt_bytes   # K + V per page


# ─────────────────────────────────────────────────────────────────────────────
# F4 — codegen-gated oracle (verify a synthesized kernel before trusting it)
# ─────────────────────────────────────────────────────────────────────────────

#: Cache of per-region-class verification verdicts (one probe per shape-class).
_VERIFY_CACHE: dict[Any, bool] = {}


def clear_verification_cache() -> None:
    _VERIFY_CACHE.clear()
    _GATED_VERIFY_CACHE.clear()


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
           region.prologue, region.has_residual, round(region.eps, 9))
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


def verify_synthesized_pointwise(region: PointwiseGraphRegion, *, seed: int = 0,
                                 atol: float = 1e-3, force: bool = False) -> bool:
    """Codegen-gated oracle for a synthesized pointwise-DAG kernel (see
    ``verify_synthesized_region``).  Runs the whole ``region`` on a small
    same-shape probe and compares to the unfused numpy reference; returns ``True``
    when the GPU result matches (or when no synthesized kernel ran — the reference
    path is trusted by construction) and ``False`` when a kernel ran and diverged,
    so the caller falls back to the per-op MPSGraph lane.  This brings the
    pointwise path to F4 parity with the matmul-epilogue / gated / attention
    region kinds — the gate that makes lane-by-lane numpy displacement safe.
    Verdicts are cached per region-class; pass ``force`` to re-probe."""
    key = ("P", region.ops, len(region.inputs))
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    # Same-shape full probes, one per external input (the broadcast classifier in
    # run_pointwise_graph treats equal shapes as full operands).
    probes = [rng.standard_normal((8, 16)).astype(np.float32)
              for _ in region.inputs]
    # Domain-restricted ops (sqrt/log/rsqrt) on a standard-normal probe produce
    # NaN/inf in both kernel and reference by design — silence the numpy warnings
    # and treat matched NaNs as agreement (equal_nan).
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out, execution = run_pointwise_graph(region, probes)
        if execution != "metal_runtime":
            verdict = True                     # no synthesized kernel to distrust
        else:
            verdict = bool(np.allclose(out, region.reference(*probes),
                                       atol=atol, equal_nan=True))
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
#: Norm op names recognized by the norm_chain discoverer → REDUCTION_OPS key.
#: (rmsnorm_safe shares the rmsnorm kernel; layer_norm is norm_chain-only today.)
_NORM_CHAIN_ALIASES: dict[str, str] = {
    "tessera.rmsnorm": "rmsnorm", "rmsnorm": "rmsnorm",
    "tessera.rmsnorm_safe": "rmsnorm", "rmsnorm_safe": "rmsnorm",
    "tessera.layer_norm": "layer_norm", "layer_norm": "layer_norm",
}
_ADD_NAMES = {"tessera.add", "add"}


def discover_fusable_regions(
    ops: list[_Op], skip: set[int] | None = None,
) -> list[tuple[int, list[int], FusedRegion]]:
    """Find maximal ``matmul -> pointwise-chain`` regions in ``ops``.

    Returns ``(matmul_index, [epilogue_indices], FusedRegion)`` per region.  An
    op only fuses into the chain if it is the *sole* consumer of the running
    value — otherwise fusing would drop an intermediate another op reads.
    ``skip`` carries op indices already claimed by an earlier (more specific)
    discoverer — e.g. a gated-matmul region claims its two projections + the gate
    activation before this matmul-epilogue pass runs, so they aren't re-fused.
    """
    use_count: dict[str, int] = {}
    for op in ops:
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1
    by_input: dict[str, list[int]] = {}
    producer: dict[str, int] = {}
    for i, op in enumerate(ops):
        producer[op.output] = i
        for v in op.inputs:
            by_input.setdefault(v, []).append(i)

    regions: list[tuple[int, list[int], FusedRegion]] = []
    consumed: set[int] = set(skip or ())
    for i, op in enumerate(ops):
        if i in consumed or op.name not in _MATMUL_NAMES:
            continue
        chain: list[int] = []
        epi: list[str] = []
        reduction: str | None = None
        # Backward: a single-use UNARY activation chain feeding the matmul's A
        # operand → a prologue, ``matmul(act(A), B)``. Each producer must be the
        # sole consumer of its output (fusing it in drops nothing) and a unary
        # pointwise op (bias/add are binary → not a prologue). Walk to the chain
        # root; the synthesized kernel applies the chain at the A-load site.
        prologue: list[str] = []
        prologue_idx: list[int] = []
        a_root = op.inputs[0]
        while True:
            if use_count.get(a_root, 0) != 1:
                break                          # A read elsewhere too → don't fuse
            p = producer.get(a_root)
            if p is None or p in consumed:
                break
            pkey = _POINTWISE_ALIASES.get(ops[p].name)
            if pkey is None or pkey == "bias" or len(ops[p].inputs) != 1:
                break                          # not a unary activation
            prologue.insert(0, pkey)           # source order: root op applied first
            prologue_idx.insert(0, p)
            consumed.add(p)
            a_root = ops[p].inputs[0]
        cur = op.output
        while True:
            if use_count.get(cur, 0) != 1:
                break                          # intermediate has another consumer
            consumers = by_input.get(cur, [])
            if len(consumers) != 1:
                break
            j = consumers[0]
            if j in consumed:
                break                          # claimed by an earlier discoverer
            nxt = ops[j]
            key = _POINTWISE_ALIASES.get(nxt.name)
            if key is None:
                break                          # not a pointwise epilogue op
            chain.append(j)
            epi.append(key)
            consumed.add(j)
            cur = nxt.output
        # a terminal reduction (rmsnorm/softmax) may close the chain — single-use.
        reduction_eps: float | None = None
        if use_count.get(cur, 0) == 1:
            consumers = by_input.get(cur, [])
            if len(consumers) == 1:
                j = consumers[0]
                rkey = _REDUCTION_ALIASES.get(ops[j].name)
                if rkey is not None:
                    reduction = rkey
                    # Carry the reduction op's eps into the fused region. Dropping
                    # it silently falls back to FusedRegion's 1e-6 default, which
                    # diverges from the unfused rmsnorm (canonical eps 1e-5) by a
                    # row-dependent ~1% — the kernel stays self-consistent with its
                    # own oracle but no longer matches the op the user wrote.
                    # (softmax ignores eps, so this is a no-op there.) Mirrors the
                    # eps resolution in discover_norm_chain_regions below.
                    eps_default = (1e-6 if ops[j].name.endswith("rmsnorm_safe")
                                   else 1e-5)
                    reduction_eps = float(ops[j].attrs.get("eps", eps_default))
                    chain.append(j)
                    consumed.add(j)
        if epi or reduction or prologue:
            b = op.inputs[1]
            regions.append((i, chain, FusedRegion(
                tuple(epi), reduction=reduction, a_name=a_root, b_name=b,
                prologue=tuple(prologue),
                prologue_src_indices=tuple(prologue_idx),
                eps=reduction_eps if reduction_eps is not None else 1e-6)))
    return regions


def discover_norm_chain_regions(
    ops: list[_Op], skip: set[int] | None = None,
) -> list[tuple[list[int], NormChainRegion, dict[str, str], str]]:
    """Find ``[add(x,residual) ->] rmsnorm/layer_norm [-> mul(γ)] [-> add(β)]``
    regions — the transformer norm (pre-norm residual + post-norm affine), NOT
    rooted at a matmul (a matmul→bias→norm chain is already a FusedRegion;
    ``skip`` carries those already-claimed op indices).

    Norm-centric scan: from each norm op, walk back to a single-use residual add
    and forward to the affine weight/bias. Returns
    ``([op_indices], NormChainRegion, {x[,residual][,gamma][,beta]}, out_value)``.
    Fires only when there is a fusion win beyond a bare norm (a preceding add or
    a post weight) — a lone norm is left to the MPSGraph rowop lane, per the
    boundary rule 'synthesize only the fusable glue; never displace a library
    call'. All fused intermediates must be single-use."""
    skip = skip or set()
    use_count: dict[str, int] = {}
    for op in ops:
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1
    by_input: dict[str, list[int]] = {}
    producer: dict[str, int] = {}
    for i, op in enumerate(ops):
        producer[op.output] = i
        for v in op.inputs:
            by_input.setdefault(v, []).append(i)

    def sole_consumer(val: str) -> int | None:
        if use_count.get(val, 0) != 1:
            return None
        cons = by_input.get(val, [])
        return cons[0] if len(cons) == 1 else None

    def other_operand(op: _Op, this: str) -> str:
        return op.inputs[1] if op.inputs[0] == this else op.inputs[0]

    regions: list[tuple[list[int], NormChainRegion, dict[str, str], str]] = []
    claimed: set[int] = set()
    for j, op in enumerate(ops):
        if j in skip or j in claimed:
            continue
        norm = _NORM_CHAIN_ALIASES.get(op.name)
        if norm is None or not op.inputs:
            continue
        idxs = [j]
        x_v = op.inputs[0]
        residual_v: str | None = None
        add_residual = False
        # Backward: a single-use residual add feeding the norm.
        if use_count.get(x_v, 0) == 1:
            p = producer.get(x_v)
            if (p is not None and p not in skip and ops[p].name in _ADD_NAMES
                    and len(ops[p].inputs) == 2):
                add_residual = True
                x_v, residual_v = ops[p].inputs[0], ops[p].inputs[1]
                idxs.append(p)
        # Forward: post-norm affine — mul(γ) then optional add(β).
        weight = bias = False
        gamma_v: str | None = None
        beta_v: str | None = None
        out_v = op.output
        c = sole_consumer(out_v)
        if (c is not None and c not in skip and ops[c].name in _SCALE_NAMES
                and len(ops[c].inputs) == 2):
            weight = True
            gamma_v = other_operand(ops[c], out_v)
            idxs.append(c)
            out_v = ops[c].output
            c2 = sole_consumer(out_v)
            if (c2 is not None and c2 not in skip and ops[c2].name in _ADD_NAMES
                    and len(ops[c2].inputs) == 2):
                bias = True
                beta_v = other_operand(ops[c2], out_v)
                idxs.append(c2)
                out_v = ops[c2].output
        if not (add_residual or weight):
            continue                           # bare norm → MPSGraph rowop lane
        eps_default = 1e-6 if op.name.endswith("rmsnorm_safe") else 1e-5
        eps = float(op.attrs.get("eps", eps_default))
        region = NormChainRegion(norm=norm, add_residual=add_residual,
                                 weight=weight, bias=bias, eps=eps)
        operands = {"x": x_v}
        if add_residual:
            operands["residual"] = residual_v  # type: ignore[assignment]
        if weight:
            operands["gamma"] = gamma_v         # type: ignore[assignment]
        if bias:
            operands["beta"] = beta_v           # type: ignore[assignment]
        regions.append((sorted(idxs), region, operands, out_v))
        claimed.update(idxs)
    return regions


_SOFTMAX_NAMES = {"tessera.softmax", "softmax"}
_SCALE_NAMES = {"tessera.mul", "tessera.scale", "mul", "scale"}


def _transpose_flag(op: _Op, axis: str) -> bool:
    """Read a matmul transpose flag (``axis`` = ``"a"``/``"b"``), accepting both
    the ``transpose_b`` and ``transposeB`` spellings. The flags live in the op's
    kwargs — the executor merges kwargs into ``_Op.attrs`` before discovery."""
    return bool(op.attrs.get(f"transpose_{axis}")
                or op.attrs.get(f"transpose{axis.upper()}"))


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
        # M2 orientation fix: resolve K's layout from the score matmul's
        # transpose flags (unambiguous, unlike value shapes when D==Nk). The
        # synthesizer wants natural Q(M,D)/K(Nk,D); `transpose_b=True` means the
        # K operand already IS natural (the matmul does Q·Kᵀ), so it does NOT
        # need flipping — `transpose_b=False` means the operand is Kᵀ and must.
        q_transposed = _transpose_flag(op, "a")
        k_transposed = not _transpose_flag(op, "b")
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
        # P@V must be a plain contraction (P(M,Nk) @ V(Nk,Dv)); a transpose on
        # either operand isn't the standard attention shape — leave it unfused.
        if _transpose_flag(ops[j], "a") or _transpose_flag(ops[j], "b"):
            continue
        v_val = ops[j].inputs[1]
        chain.append(j)
        consumed.add(j)
        regions.append((chain, AttentionRegion(
            scale=scale, q_transposed=q_transposed, k_transposed=k_transposed),
            (q_val, k_val, v_val)))
    return regions


#: Elementwise multiply op names for the gated-matmul combine. Distinct from
#: `_SCALE_NAMES` (affine scale-by-vector): the gate combine multiplies two
#: same-shape projection results.
_MUL_NAMES = {"tessera.mul", "mul", "tessera.multiply", "multiply"}


def discover_gated_matmul_regions(
    ops: list[_Op], skip: set[int] | None = None,
) -> list[tuple[list[int], GatedMatmulRegion, str]]:
    """Find ``f(A @ Wg) ⊙ (A @ Wu)`` — the SwiGLU gate from PRIMITIVE ops.

    Pattern: two matmuls sharing operand A (one is the gate, one the up
    projection), a unary activation on the gate branch, and an elementwise
    multiply combining them. Returns ``([op_indices], GatedMatmulRegion,
    out_value)``. Mul-centric scan: from each multiply, one operand must be a
    single-use activation of a matmul(A, Wg) and the other a single-use
    matmul(A, Wu) sharing the same A. All fused intermediates must be single-use,
    and nothing already claimed (``skip``) is reused — so this is complementary
    to the library ``swiglu`` op (which never lowers to these primitives)."""
    skip = skip or set()
    use_count: dict[str, int] = {}
    producer: dict[str, int] = {}
    for i, op in enumerate(ops):
        producer[op.output] = i
        for v in op.inputs:
            use_count[v] = use_count.get(v, 0) + 1

    regions: list[tuple[list[int], GatedMatmulRegion, str]] = []
    claimed: set[int] = set()
    for j, op in enumerate(ops):
        if j in skip or j in claimed or op.name not in _MUL_NAMES:
            continue
        if len(op.inputs) != 2:
            continue
        x, y = op.inputs
        # The activation branch is the gate; the bare-matmul branch is the up.
        # Try both operand orderings (multiply is commutative).
        for act_v, up_v in ((x, y), (y, x)):
            pa, pu = producer.get(act_v), producer.get(up_v)
            if pa is None or pu is None or pa in skip or pu in skip:
                continue
            act_key = _POINTWISE_ALIASES.get(ops[pa].name)
            if act_key is None or act_key == "bias" or len(ops[pa].inputs) != 1:
                continue                       # gate branch isn't a unary activation
            g_v = ops[pa].inputs[0]            # the activation's input = gate matmul
            pg = producer.get(g_v)
            if pg is None or pg in skip:
                continue
            if (ops[pg].name not in _MATMUL_NAMES
                    or ops[pu].name not in _MATMUL_NAMES):
                continue
            if len(ops[pg].inputs) < 2 or len(ops[pu].inputs) < 2:
                continue
            if ops[pg].inputs[0] != ops[pu].inputs[0]:
                continue                       # the two matmuls must share operand A
            # single-use: gate-matmul out -> act only; act out -> mul only;
            # up-matmul out -> mul only (fusing must drop nothing).
            if (use_count.get(g_v, 0) != 1 or use_count.get(act_v, 0) != 1
                    or use_count.get(up_v, 0) != 1):
                continue
            idxs = [pg, pa, pu, j]
            regions.append((idxs, GatedMatmulRegion(
                gate_act=act_key, a_name=ops[pg].inputs[0],
                wg_name=ops[pg].inputs[1], wu_name=ops[pu].inputs[1]),
                op.output))
            claimed.update(idxs)
            break
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
    "coopmat_tile_for",
    "coopmat_threads",
    "SYNTH_COOPMAT_TILES",
    "autotune_coopmat_tile",
    "clear_coopmat_tile_corpus",
    "synthesize_matmul_reduction_coopmat_msl",
    "run_fused_region_coopmat_reduce",
    "coopmat_reduce_eligible",
    "SYNTH_COOPMAT_REDUCE_MAX_N",
    "AutotuneRecord",
    "autotune_matmul_epilogue",
    "best_variant_for",
    "select_variant",
    "autotune_enabled",
    "autotune_matmul_epilogue",
    "clear_autotune_corpus",
    "SYNTH_MAX_N",
    "SYNTH_MAX_D",
    "synthesize_matmul_epilogue_msl",
    "synthesize_attention_msl",
    "synthesize_attention_online_msl",
    "run_fused_region",
    "run_fused_attention",
    "discover_fusable_regions",
    "discover_attention_regions",
    "GatedMatmulRegion",
    "synthesize_gated_matmul_msl",
    "run_gated_matmul_region",
    "should_fuse_gated",
    "verify_synthesized_gated",
    "discover_gated_matmul_regions",
    "is_pointwise_op",
]
