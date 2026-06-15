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


# ─────────────────────────────────────────────────────────────────────────────
# F0 — fused region
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FusedRegion:
    """A matmul root + an ordered pointwise epilogue chain (op names)."""

    epilogue: tuple[str, ...]
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

    @property
    def has_bias(self) -> bool:
        return any(EPILOGUE_OPS[op].needs_bias for op in self.epilogue)

    def reference(self, A: np.ndarray, B: np.ndarray,
                  bias: np.ndarray | None = None) -> np.ndarray:
        """The *unfused* result: matmul then each epilogue op, in numpy.  This is
        the horizontal-oracle ground truth the synthesized kernel must match."""
        out = np.asarray(A, np.float32) @ np.asarray(B, np.float32)
        for op in self.epilogue:
            spec = EPILOGUE_OPS[op]
            if spec.needs_bias:
                if bias is None:
                    raise ValueError(f"region needs a bias for {op!r}")
                out = out + np.asarray(bias, np.float32)
            else:
                out = spec.ref(out)
        return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# F2a — MSL synthesis
# ─────────────────────────────────────────────────────────────────────────────


def synthesize_matmul_epilogue_msl(region: FusedRegion) -> str:
    """Emit the MSL source for ``region`` — one row per thread, the matmul into a
    stack accumulator, then the epilogue chain inlined per element."""
    bias_param = ("    device const float* bias [[buffer(6)]],\n"
                  if region.has_bias else "")
    body_lines = []
    for op in region.epilogue:
        body_lines.append(f"            {EPILOGUE_OPS[op].msl}")
    body = "\n".join(body_lines)
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void {_ENTRY}(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       O   [[buffer(2)]],
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
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {{
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }}
    int o_off = row * N;
    for (int n = 0; n < N; ++n) {{
        float v = scores[n];
{body}
        O[o_off + n] = v;
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


def run_fused_region(region: FusedRegion, A: np.ndarray, B: np.ndarray,
                     bias: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Run the region as ONE synthesized fused kernel on Metal.  Returns
    ``(output, execution)`` where execution is ``"metal_runtime"`` (the
    synthesized kernel ran) or ``"reference"`` (numpy fallback — no Metal, N too
    large, or compile/dispatch failed).  Either way the numbers are correct."""
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

    sym = _synth_symbol()
    if sym is not None and N <= SYNTH_MAX_N:
        source = synthesize_matmul_epilogue_msl(region).encode("utf-8")
        out = np.zeros((M, N), np.float32)
        fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        bias_ptr = fp(bias_arr) if bias_arr is not None else None
        rc = sym(source, _ENTRY.encode("utf-8"), fp(A), fp(B), bias_ptr, fp(out),
                 M, N, K, 1 if region.has_bias else 0)
        if rc == 1:
            return out, "metal_runtime"
    return region.reference(A, B, bias_arr), "reference"


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
        if epi:
            a, b = op.inputs[0], op.inputs[1]
            bias_name = None
            regions.append((i, chain, FusedRegion(tuple(epi), a, b, bias_name)))
    return regions


__all__ = [
    "EPILOGUE_OPS",
    "EpilogueOp",
    "FusedRegion",
    "SYNTH_MAX_N",
    "synthesize_matmul_epilogue_msl",
    "run_fused_region",
    "discover_fusable_regions",
]
