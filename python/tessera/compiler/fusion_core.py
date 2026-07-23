"""Arch-agnostic fusion middle-end — region IR, discovery, cost, F4 oracle.

B1 split (COMPILER_REFACTOR_PLAN Workstream B, the keystone): the
target-independent half of the former ``fusion.py`` — ``FusedRegion`` /
``EpilogueOp`` / ``ReductionOp`` semantics, ``discover_*``, ``*_cost`` /
``should_fuse_*``, and the numpy-reference ``verify_synthesized_*`` F4 oracles.
MSL synthesis + Metal runtime dispatch live in
``tessera.compiler.emit.apple_msl``; ``tessera.compiler.fusion`` re-exports both,
so importers see no change. Pure relocation — no behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# B2: the target-agnostic emitter/runner protocol. Importing only the
# (arch-neutral) target-alias set + runner accessor here keeps
# fusion_core -> emit.kernel_emitter one-directional (kernel_emitter has no
# runtime fusion_core dependency).
from tessera.compiler.emit.kernel_emitter import (
    KernelRunner,
    METAL_TARGETS,
    REFERENCE_EXECUTIONS,
    RunnerError,
    active_runner as _active_runner,
)


# --- runner resolution (B2b/B3 seam) ---------------------------------------
# The F4 oracles below are arch-agnostic (compare a synthesized kernel to the
# numpy reference), but they need an *executor* (a KernelRunner) to produce the
# "actual". B3: each verify_* takes an explicit ``runner`` so the SAME oracle
# gates any backend's kernel; when omitted it resolves the registered active
# runner here. Whichever backend registered a runner (Apple's AppleMSLRunner
# today) executes. If nothing has registered yet — e.g. fusion_core is imported
# directly, without the Apple emitter — we import apple_msl once (it
# self-registers), preserving B1's import-safety while keeping core -> emit soft.
def _runner() -> KernelRunner:
    r = _active_runner()
    if r is None:
        from tessera.compiler.emit import apple_msl  # noqa: F401 — self-registers
        r = _active_runner()
    if r is None:
        raise RunnerError(
            "no KernelRunner registered to execute a synthesized region")
    return r


def _effective_atol(runner: KernelRunner, atol: float) -> float:
    """Oracle tolerance for ``runner``: the caller's ``atol`` widened to the
    backend's declared precision budget (``runner.accuracy_atol``), so an f16 lead
    kernel's rounding is not misread as a miscompile while an O(1) miscompile is
    still caught. A ``None`` budget (f32/exact backends — Apple, x86) leaves
    ``atol`` unchanged. Simplest slice of the accuracy-budgeted arbiter
    (Decision #28 / plan D2)."""
    budget = getattr(runner, "accuracy_atol", None)
    return atol if budget is None else max(atol, budget)


def _effective_rtol(runner: KernelRunner) -> float:
    """Backend-declared relative budget, preserving NumPy's strict default."""
    budget = getattr(runner, "accuracy_rtol", None)
    return 1e-5 if budget is None else float(budget)


SYNTH_MAX_N = 1024

#: Cap on head_dim for the ONLINE-softmax attention kernel (M2): it streams keys
#: with no ``scores[Nk]`` array, keeping only an ``acc[head_dim]`` accumulator —
#: so it trades the SYNTH_MAX_N key cap for a head_dim cap, and handles Nk far
#: beyond SYNTH_MAX_N (the large-context / long-sequence attention case). Matches
#: the hand-written flash-attn kernel's head_dim envelope.
SYNTH_MAX_D = 256


@dataclass(frozen=True)
class EpilogueOp:
    """One pointwise epilogue op: how it lowers to a target kernel and to numpy.

    B2: the lowering is target-parametric — consumers request a snippet via
    :meth:`emit` (``target``) instead of reading a Metal-only field. ``_msl`` is
    the backing Metal Shading Language body (the reference target); Workstream C
    backends add more languages behind the same :meth:`emit` seam."""

    name: str
    _msl: str                                 # metal body: operates on `v`, may read bias[n]
    ref: Callable[[np.ndarray], np.ndarray]   # numpy reference (no-bias ops)
    needs_bias: bool = False

    def emit(self, target: str = "metal") -> str:
        """Kernel snippet for this op on ``target`` (operates on ``v``; may read
        ``bias[n]``). Unknown target raises — never silently emit the wrong
        language (Decision #21)."""
        if target in METAL_TARGETS:
            return self._msl
        raise ValueError(
            f"EpilogueOp {self.name!r}: no kernel snippet for target {target!r} "
            f"(known targets: {sorted(METAL_TARGETS)})")


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

# Stable shared-contract diagnostics. Keeping the code separate from the detail
# lets registry tooling discover the emission site and callers match the code.
E_FUSED_EPILOGUE_BAD_DTYPE = "E_FUSED_EPILOGUE_BAD_DTYPE"
E_FUSED_EPILOGUE_BAD_OP = "E_FUSED_EPILOGUE_BAD_OP"
E_FUSED_EPILOGUE_BAD_ORDER = "E_FUSED_EPILOGUE_BAD_ORDER"
E_FUSED_EPILOGUE_MISSING_OPERAND = "E_FUSED_EPILOGUE_MISSING_OPERAND"


@dataclass(frozen=True)
class ReductionOp:
    """A terminal *reduction* epilogue (rmsnorm/softmax): a row reduction over the
    matmul-row accumulator ``scores[N]``, then a per-element finalize into ``O``.
    Unlike pointwise ops, it needs the whole row — so it comes last, after any
    pointwise chain.  :meth:`emit` returns a block (uses ``N``/``scores``/``O``/
    ``o_off``; the caller substitutes ``{eps}``); ``ref(x, eps)`` is the numpy
    ground truth. B2: target-parametric like :class:`EpilogueOp`."""

    name: str
    _msl: str
    ref: Callable[[np.ndarray, float], np.ndarray]

    def emit(self, target: str = "metal") -> str:
        """Reduction block for ``target`` (uses ``N``/``scores``/``O``/``o_off``;
        caller substitutes ``{eps}``). Unknown target raises (Decision #21)."""
        if target in METAL_TARGETS:
            return self._msl
        raise ValueError(
            f"ReductionOp {self.name!r}: no kernel snippet for target {target!r} "
            f"(known targets: {sorted(METAL_TARGETS)})")


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
    # B2c symbolic-dim carrier (dynamic-shapes decision): the Graph-IR dim names
    # for the region's shape when it is symbolic — e.g. ("batch", "d_ff",
    # "d_model"). NOT synthesis-semantic (the synth path derives concrete M/N/K
    # from operands); it lets a SpecPolicy.DYNAMIC emitter key one kernel by
    # symbolic identity rather than concrete values. ``None`` = anonymous/static.
    dim_names: tuple[str, ...] | None = None
    # Storage policy for target candidates. The reference remains f32; tensor-
    # core candidates quantize operands at this explicit ABI boundary.
    storage_dtype: str = "f16"
    # Executable physical input contracts for generic host emitters. Logical
    # matmul semantics remain A[M,K] @ B[K,N]; these fields only select the
    # pointer-stride formula and launch-time C/Fortran materialization.
    a_layout: str = "row_major"
    b_layout: str = "row_major"

    def __post_init__(self) -> None:
        if self.a_layout not in ("row_major", "col_major"):
            raise ValueError("a_layout must be row_major or col_major")
        if self.b_layout not in ("row_major", "col_major"):
            raise ValueError("b_layout must be row_major or col_major")
        if self.storage_dtype not in (
                "f16", "bf16", "f32", "fp8_e4m3", "fp8_e5m2"):
            raise ValueError(
                f"{E_FUSED_EPILOGUE_BAD_DTYPE}: storage_dtype must be f16, bf16, "
                "f32, fp8_e4m3, or fp8_e5m2")
        for op in self.epilogue:
            if op not in EPILOGUE_OPS:
                raise ValueError(
                    f"{E_FUSED_EPILOGUE_BAD_OP}: unknown epilogue op {op!r}")
        bias_ops = [op for op in self.epilogue if EPILOGUE_OPS[op].needs_bias]
        if len(bias_ops) > 1:
            raise ValueError(
                f"{E_FUSED_EPILOGUE_BAD_ORDER}: at most one bias op per region")
        for op in self.prologue:
            if op not in EPILOGUE_OPS:
                raise ValueError(
                    f"{E_FUSED_EPILOGUE_BAD_OP}: unknown prologue op {op!r}")
            if EPILOGUE_OPS[op].needs_bias:
                raise ValueError(
                    f"{E_FUSED_EPILOGUE_BAD_ORDER}: prologue op {op!r} cannot "
                    "need a bias")
        if self.reduction is not None and self.reduction not in REDUCTION_OPS:
            raise ValueError(
                f"{E_FUSED_EPILOGUE_BAD_OP}: unknown reduction epilogue "
                f"{self.reduction!r}")
        if self.residual and self.reduction is not None:
            raise ValueError(
                f"{E_FUSED_EPILOGUE_BAD_ORDER}: residual + reduction not supported "
                "(v1)")
        if (not self.epilogue and self.reduction is None and not self.residual
                and not self.prologue):
            raise ValueError(
                f"{E_FUSED_EPILOGUE_BAD_OP}: a region must have a prologue, "
                "epilogue op, reduction, or residual")

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
                    raise ValueError(
                        f"{E_FUSED_EPILOGUE_MISSING_OPERAND}: region needs a bias "
                        f"for {op!r}")
                out = out + np.asarray(bias, np.float32)
            else:
                out = spec.ref(out)
        if self.residual:
            if residual is None:
                raise ValueError(
                    f"{E_FUSED_EPILOGUE_MISSING_OPERAND}: region needs a residual")
            out = out + np.asarray(residual, np.float32)
        if self.reduction is not None:
            out = REDUCTION_OPS[self.reduction].ref(out, self.eps)
        return out.astype(np.float32)


def _round_to_storage(x: "np.ndarray", dtype: str) -> "np.ndarray":
    """Round f32 ``x`` to the 16-bit storage ``dtype`` and back to f32, so a
    reference sees the same rounded operands the 16-bit GEMM kernels do."""
    a = np.ascontiguousarray(x, np.float32)
    if dtype in ("float16", "f16"):
        return a.astype(np.float16).astype(np.float32)
    if dtype in ("bfloat16", "bf16"):
        try:
            import ml_dtypes
            return a.astype(ml_dtypes.bfloat16).astype(np.float32)
        except Exception:                      # round-to-nearest-even fallback
            u = a.view(np.uint32).astype(np.uint64)
            bits = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32)
            return bits.view(np.float32)
    raise ValueError(f"unsupported matmul storage dtype {dtype!r}")


@dataclass(frozen=True)
class MatmulRegion:
    """A bare matmul ``D = A @ B`` (no fusion) — the region kind the D1 arbiter
    keys a plain-GEMM candidate on (there is no matmul feature in ``FusedRegion``,
    which always carries at least one fused op). ``dtype`` is the 16-bit storage
    (``bfloat16``/``float16``); the accumulate is f32, matching the emitted
    ``mma.sync`` and shipped GEMM kernels."""

    dtype: str = "bfloat16"
    # Orientation of the *raw* operands as the matmul consumes them, read from its
    # transposeA/transposeB flags (the `TransposeIntoMatmul` Graph-IR fold produces
    # these). The GEMM works on natural A(M,K)/B(K,N); these say whether a raw
    # operand is already transposed and must be flipped first. Default False = the
    # operand is natural, so direct `MatmulRegion()` construction is unchanged.
    # This is the backend consumer of the transpose contract (COMPILER_REFACTOR_PLAN
    # H): it closes the OPTIMIZING_COMPILER_PLAN §6 orientation note for plain GEMM
    # the same way `q_transposed`/`k_transposed` did for attention (M2) — resolving
    # orientation from the layout flag, NOT from value shapes (ambiguous at M==K==N).
    transpose_a: bool = False
    transpose_b: bool = False

    def _natural(self, A: np.ndarray, B: np.ndarray, cast: bool = True):
        """Flip raw operands to natural A(M,K)/B(K,N) per the transpose flags.
        ``cast`` (default) coerces to f32 for the numpy reference; ``cast=False``
        preserves the storage dtype for the half-precision GPU path."""
        A = np.asarray(A, np.float32) if cast else np.asarray(A)
        B = np.asarray(B, np.float32) if cast else np.asarray(B)
        if self.transpose_a:
            A = np.ascontiguousarray(A.T)
        if self.transpose_b:
            B = np.ascontiguousarray(B.T)
        return A, B

    def reference(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """The f32 result of ``A @ B`` with both operands oriented per the transpose
        flags then rounded to ``dtype`` — the horizontal-oracle ground truth the
        GEMM candidate matches."""
        A, B = self._natural(A, B)
        Aq = _round_to_storage(A, self.dtype)
        Bq = _round_to_storage(B, self.dtype)
        return (Aq @ Bq).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# F2b-tiled — threadgroup-tiled synthesis for large N (the stack kernel caps at
# SYNTH_MAX_N; this lifts it to SYNTH_MAX_N_TILED via dynamic threadgroup memory)
# ─────────────────────────────────────────────────────────────────────────────

#: Large-N cap for the tiled kernel: one row of N fp32 scores lives in dynamic
#: threadgroup memory (32 KB budget on current Apple arches → 8192 floats),
#: mirroring the hand-written ``matmul_softmax_tiled_f32`` envelope it retires.
SYNTH_MAX_N_TILED = 8192


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
    storage_dtype: str = "f16"

    def __post_init__(self) -> None:
        if self.storage_dtype not in (
                "f16", "bf16", "f32", "fp8_e4m3", "fp8_e5m2"):
            raise ValueError(
                "storage_dtype must be f16, bf16, f32, fp8_e4m3, or fp8_e5m2")

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
    storage_dtype: str = "f16"

    def __post_init__(self) -> None:
        if self.storage_dtype not in (
                "f16", "bf16", "f32", "fp8_e4m3", "fp8_e5m2"):
            raise ValueError(
                "storage_dtype must be f16, bf16, f32, fp8_e4m3, or fp8_e5m2")
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


def should_fuse_gated(region: GatedMatmulRegion, M: int, H: int, K: int) -> bool:
    """F3 cost gate: the gated kernel keeps two fp32 rows of width H in the
    per-thread stack, so it only fuses when H fits the (halved) cap. Larger H is
    left to the per-op two-matmul path."""
    return H <= SYNTH_GATED_MAX_H


_GATED_VERIFY_CACHE: dict[Any, bool] = {}


def verify_synthesized_gated(region: GatedMatmulRegion, *, seed: int = 0,
                             atol: float = 1e-3, force: bool = False,
                             runner: KernelRunner | None = None) -> bool:
    """Codegen-gated oracle for a gated-matmul region (see
    ``verify_synthesized_region``): run the synthesized kernel on a small probe
    and compare to the unfused numpy reference. ``True`` unless a kernel ran and
    diverged. Cached per gate-activation."""
    r = runner or _runner()
    # B3: key the verdict by backend identity too — a verdict from one runner
    # must NOT be reused for another, else a faithful backend's cached True lets a
    # wrong backend skip its own gate (and vice-versa).
    key = (r.target, "G", region.gate_act)
    if not force and key in _GATED_VERIFY_CACHE:
        return _GATED_VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    # Keep the probe in the bounded activation range used by the low-precision
    # execution contracts. Unscaled N(0,1) inputs amplify two FP8 projections
    # through the gate and test overflow/error growth rather than codegen.
    A = (rng.standard_normal((8, 12)) * .2).astype(np.float32)
    Wg = (rng.standard_normal((12, 16)) * .2).astype(np.float32)
    Wu = (rng.standard_normal((12, 16)) * .2).astype(np.float32)
    out, execution = r.run_gated_matmul_region(region, A, Wg, Wu)
    if execution in REFERENCE_EXECUTIONS:
        verdict = True
    else:
        verdict = bool(np.allclose(
            out, region.reference(A, Wg, Wu), atol=_effective_atol(r, atol),
            rtol=_effective_rtol(r)))
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
                              atol: float = 1e-3, force: bool = False,
                              runner: KernelRunner | None = None) -> bool:
    """Codegen-gated oracle: run the *synthesized* kernel for ``region`` on a small
    probe and compare it to the unfused numpy reference.  Returns ``True`` only if
    the GPU result matches (a correct synthesizer) — or if no synthesized kernel
    ran (no Metal: the reference path is trusted by construction).  Returns
    ``False`` when a kernel ran and *diverged* — a synthesizer bug — so the caller
    falls back to the trusted per-op path.  This is the codegen analogue of the
    magellan/alphaevolve reward-hack rejection: a faster-but-wrong kernel is
    refused.  Verdicts are cached per region-class; pass ``force`` to re-probe.

    B3: ``runner`` injects a specific backend's :class:`KernelRunner` so the same
    numpy-reference oracle gates *any* backend's synthesized kernel — the F4 gate
    is universal, not Apple-only. ``None`` uses the registered active runner."""
    r = runner or _runner()
    # B3: key by backend identity too (see verify_synthesized_gated) so one
    # runner's verdict is never reused for another.
    key = (r.target, "R", region.epilogue, region.reduction, region.has_bias,
           region.prologue, region.has_residual, round(region.eps, 9))
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((8, 12)) * .2).astype(np.float32)
    B = (rng.standard_normal((12, 16)) * .2).astype(np.float32)
    bias = ((rng.standard_normal((16,)) * .2).astype(np.float32)
            if region.has_bias else None)
    # A residual region needs its (M,N) residual probe too, else the runner's
    # required-buffer guard raises instead of exercising the synthesized kernel
    # (which supports residuals). None for a non-residual region — unchanged.
    residual = ((rng.standard_normal((8, 16)) * .2).astype(np.float32)
                if region.has_residual else None)
    out, execution = r.run_fused_region(region, A, B, bias, residual=residual)
    if execution in REFERENCE_EXECUTIONS:
        verdict = True                         # no synthesized kernel to distrust
    else:
        verdict = bool(np.allclose(
            out, region.reference(A, B, bias, residual),
            atol=_effective_atol(r, atol), rtol=_effective_rtol(r)))
    _VERIFY_CACHE[key] = verdict
    return verdict


def verify_synthesized_matmul(region: "MatmulRegion", *, seed: int = 0,
                              atol: float = 1e-3, force: bool = False,
                              runner: KernelRunner | None = None) -> bool:
    """Codegen-gated oracle for a bare-matmul candidate (see
    ``verify_synthesized_region``): run the candidate's GEMM on an aligned probe
    and compare to the dtype-rounded ``A @ B`` reference. Arbiter-only — the
    ``runner`` is a candidate adapter exposing ``run_matmul``; a runner without it
    (a real backend runner) trusts the reference, since nothing device-emitted is
    in play for this op."""
    r = runner or _runner()
    run = getattr(r, "run_matmul", None)
    if run is None:
        return True
    key = (r.target, "M", region.dtype)
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    M, N, K = 32, 16, 32                        # aligned probe (M%16, N%8, K%16)
    A = (rng.standard_normal((M, K)) * 0.4).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.4).astype(np.float32)
    # Feed the probe in the region's RAW orientation: a transposed region's
    # candidate flips its operands back to natural via `_natural`, so a natural
    # (K,N) probe would be double-flipped into a shape mismatch. Transpose here so
    # run() + reference() both re-derive the same natural (M,K)@(K,N) product.
    if getattr(region, "transpose_a", False):
        A = np.ascontiguousarray(A.T)           # raw (K, M)
    if getattr(region, "transpose_b", False):
        B = np.ascontiguousarray(B.T)           # raw (N, K)
    out, execution = run(region, A, B)
    if execution in REFERENCE_EXECUTIONS:
        verdict = True
    else:
        verdict = bool(np.allclose(out, region.reference(A, B),
                                   atol=_effective_atol(r, atol)))
    _VERIFY_CACHE[key] = verdict
    return verdict


def verify_synthesized_attention(region: AttentionRegion, *, seed: int = 0,
                                 atol: float = 1e-3, force: bool = False,
                                 runner: KernelRunner | None = None) -> bool:
    """Codegen-gated oracle for a synthesized attention block (see
    ``verify_synthesized_region``)."""
    r = runner or _runner()
    key = (r.target, "A", round(region.scale, 9), region.causal)
    if not force and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((8, 16)) * .2).astype(np.float32)
    K = (rng.standard_normal((8, 16)) * .2).astype(np.float32)
    V = (rng.standard_normal((8, 16)) * .2).astype(np.float32)
    out, execution = r.run_fused_attention(region, Q, K, V)
    if execution in REFERENCE_EXECUTIONS:
        verdict = True
    else:
        verdict = bool(np.allclose(
            out, region.reference(Q, K, V), atol=_effective_atol(r, atol),
            rtol=_effective_rtol(r)))
    _VERIFY_CACHE[key] = verdict
    return verdict


def verify_synthesized_pointwise(region: PointwiseGraphRegion, *, seed: int = 0,
                                 atol: float = 1e-3, force: bool = False,
                                 runner: KernelRunner | None = None) -> bool:
    """Codegen-gated oracle for a synthesized pointwise-DAG kernel (see
    ``verify_synthesized_region``).  Runs the whole ``region`` on a small
    same-shape probe and compares to the unfused numpy reference; returns ``True``
    when the GPU result matches (or when no synthesized kernel ran — the reference
    path is trusted by construction) and ``False`` when a kernel ran and diverged,
    so the caller falls back to the per-op MPSGraph lane.  This brings the
    pointwise path to F4 parity with the matmul-epilogue / gated / attention
    region kinds — the gate that makes lane-by-lane numpy displacement safe.
    Verdicts are cached per region-class; pass ``force`` to re-probe."""
    r = runner or _runner()
    key = (r.target, "P", region.ops, len(region.inputs))
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
        out, execution = r.run_pointwise_graph(region, probes)
        if execution in REFERENCE_EXECUTIONS:
            verdict = True                     # no synthesized kernel to distrust
        else:
            verdict = bool(np.allclose(out, region.reference(*probes),
                                       atol=_effective_atol(r, atol),
                                       equal_nan=True))
    _VERIFY_CACHE[key] = verdict
    return verdict


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
