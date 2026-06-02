"""JIT Phase 2 substrate — encode-session chain registry + planner + executor.

The single-cb decode plan (``docs/audit/backend/apple/APPLE_AUDIT.md``)
maps Stage 3 onto two phases:

* **Phase 1 (landed)** — ``@decode_chain`` decorator. User calls the
  ``_enc`` helpers explicitly under a ``with batched_session():``;
  the decorator opens / commits the session.

* **Phase 2 (this module + jit.py hookup)** — the JIT inspects a
  function's op trace, identifies maximal straight-line subgraphs of
  encode-session-compatible ops with chain dependency, and routes
  them through ``batched_session()`` automatically. The user writes
  natural code; the JIT batches.

This module ships the **substrate** for Phase 2 — the components
``compiler/jit.py`` will consume:

1. ``ENCODE_OP_REGISTRY`` — canonical metadata for every
   encode-session-aware op. Maps op-name + dtype to the encode helper
   callable and the input/output tensor positions (so the chain
   planner knows which args are "carrier" tensors vs. shape params).
2. ``ChainPlanner`` — walks a list of ``OpRecord`` traces and groups
   them into ``ChainSegment``s of consecutive encode-eligible ops
   that share data dependency. Falls back to per-op execution for
   non-encode-eligible ops.
3. ``ChainExecutor`` — given a ``ChainSegment``, opens a session,
   dispatches each op through its encode helper, commits + waits.

The jit.py integration is then mechanical: build an OpRecord list
from the function's emitted ops, plan it, execute. That hookup is a
follow-on PR — this module's tests prove the substrate is correct in
isolation so the JIT integration is a small lift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .apple_gpu_batched import (
    DeviceTensor,
    batched_session,
    bmm_enc,
    bmm_enc_bf16,
    bmm_enc_f16,
    conv2d_enc_no_bias,
    conv2d_enc_no_bias_bf16,
    conv2d_enc_no_bias_f16,
    device_tensor,
    flash_attn_enc,
    flash_attn_enc_bf16,
    flash_attn_enc_f16,
    gelu_enc,
    gelu_enc_bf16,
    gelu_enc_f16,
    layer_norm_enc,
    layer_norm_enc_bf16,
    layer_norm_enc_f16,
    rmsnorm_enc,
    rmsnorm_enc_bf16,
    rmsnorm_enc_f16,
    rope_enc,
    rope_enc_bf16,
    rope_enc_f16,
    silu_enc,
    silu_enc_bf16,
    silu_enc_f16,
    softmax_enc,
    softmax_enc_bf16,
    softmax_enc_f16,
)


# ---------------------------------------------------------------------
# Registry — canonical metadata for every encode-session-aware op.
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EncodeOpSpec:
    """Metadata describing one encode-session-aware op.

    Fields:
    * ``name`` — canonical op name (matches the Python wrapper).
    * ``dtype`` — ``"f32"`` or ``"f16"``.
    * ``encode_fn`` — Python callable signature
      ``(session, *tensor_args, **shape_kwargs) -> DeviceTensor``.
    * ``input_tensor_args`` — positional positions of input
      DeviceTensors (excluding session). Used by the planner to
      track data dependency.
    * ``has_output`` — True iff the op produces a DeviceTensor result
      (every current encode op does; reserved for future void-return
      ops like a fused write).
    """
    name: str
    dtype: str
    encode_fn: Callable[..., DeviceTensor]
    input_tensor_args: tuple[int, ...]
    has_output: bool = True


# Position semantics for each op's input_tensor_args: indices into
# (positional-args-after-session, kwargs-by-name-not-counted). The
# planner uses these to thread output→input edges across ops.

_REGISTRY_ENTRIES: tuple[EncodeOpSpec, ...] = (
    # f32 ops
    EncodeOpSpec("bmm", "f32", bmm_enc, input_tensor_args=(0, 1)),
    EncodeOpSpec("layer_norm", "f32", layer_norm_enc,
                  input_tensor_args=(0, 1, 2)),
    EncodeOpSpec("rmsnorm", "f32", rmsnorm_enc,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("softmax", "f32", softmax_enc, input_tensor_args=(0,)),
    EncodeOpSpec("rope", "f32", rope_enc, input_tensor_args=(0, 1)),
    EncodeOpSpec("silu", "f32", silu_enc, input_tensor_args=(0,)),
    EncodeOpSpec("gelu", "f32", gelu_enc, input_tensor_args=(0,)),
    EncodeOpSpec("flash_attn", "f32", flash_attn_enc,
                  input_tensor_args=(0, 1, 2)),
    # Project 5 (2026-06-01) — conv2d encode-session integration.
    # Registry surface omits the optional bias parameter so the
    # ``input_tensor_args`` positional contract holds. Callers that
    # need bias use :func:`apple_gpu_batched.conv2d_enc` directly.
    EncodeOpSpec("conv2d", "f32", conv2d_enc_no_bias,
                  input_tensor_args=(0, 1)),
    # f16 ops
    EncodeOpSpec("bmm", "f16", bmm_enc_f16, input_tensor_args=(0, 1)),
    EncodeOpSpec("layer_norm", "f16", layer_norm_enc_f16,
                  input_tensor_args=(0, 1, 2)),
    EncodeOpSpec("rmsnorm", "f16", rmsnorm_enc_f16,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("softmax", "f16", softmax_enc_f16,
                  input_tensor_args=(0,)),
    EncodeOpSpec("rope", "f16", rope_enc_f16, input_tensor_args=(0, 1)),
    EncodeOpSpec("silu", "f16", silu_enc_f16, input_tensor_args=(0,)),
    EncodeOpSpec("gelu", "f16", gelu_enc_f16, input_tensor_args=(0,)),
    EncodeOpSpec("flash_attn", "f16", flash_attn_enc_f16,
                  input_tensor_args=(0, 1, 2)),
    # bf16 ops — Project-3 (2026-06-01) MPSGraph entries + Phase-3b
    # (2026-06-01) MSL-kernel entries via on-GPU bf16↔fp32 cast.
    EncodeOpSpec("bmm", "bf16", bmm_enc_bf16, input_tensor_args=(0, 1)),
    EncodeOpSpec("layer_norm", "bf16", layer_norm_enc_bf16,
                  input_tensor_args=(0, 1, 2)),
    EncodeOpSpec("rmsnorm", "bf16", rmsnorm_enc_bf16,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("softmax", "bf16", softmax_enc_bf16,
                  input_tensor_args=(0,)),
    EncodeOpSpec("silu", "bf16", silu_enc_bf16, input_tensor_args=(0,)),
    EncodeOpSpec("gelu", "bf16", gelu_enc_bf16, input_tensor_args=(0,)),
    # Phase 3b MSL bf16 via on-GPU cast.
    EncodeOpSpec("rope", "bf16", rope_enc_bf16,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("flash_attn", "bf16", flash_attn_enc_bf16,
                  input_tensor_args=(0, 1, 2)),
    # Sprint A (2026-06-01) — conv2d f16/bf16 encode lanes complete
    # the 3-dtype matrix; conv2d is no longer the only encode-eligible
    # op without bf16/f16 coverage.
    EncodeOpSpec("conv2d", "f16", conv2d_enc_no_bias_f16,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("conv2d", "bf16", conv2d_enc_no_bias_bf16,
                  input_tensor_args=(0, 1)),
)


ENCODE_OP_REGISTRY: dict[tuple[str, str], EncodeOpSpec] = {
    (e.name, e.dtype): e for e in _REGISTRY_ENTRIES
}


def is_encode_eligible(op_name: str, dtype: str) -> bool:
    """True iff ``(op_name, dtype)`` is encode-session-compatible."""
    return (op_name, dtype) in ENCODE_OP_REGISTRY


def encode_spec(op_name: str, dtype: str) -> EncodeOpSpec:
    """Look up the encode metadata. Raises if unregistered."""
    key = (op_name, dtype)
    if key not in ENCODE_OP_REGISTRY:
        raise KeyError(
            f"op {op_name!r} dtype {dtype!r} not in encode registry; "
            f"registered: {sorted(ENCODE_OP_REGISTRY.keys())}")
    return ENCODE_OP_REGISTRY[key]


# ---------------------------------------------------------------------
# OpRecord — minimal trace primitive.
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TraceRef:
    """A reference to the output of an earlier op in the same trace.

    Phase 2.1 (2026-06-01) — when ``auto_batch`` (or future
    ``@jit(target="apple_gpu", auto_batch=True)``) captures a function's
    op trace, each op's output becomes a ``TraceRef`` that downstream
    ops can consume as input. The executor resolves these to the
    actual :class:`DeviceTensor` at run time.

    ``op_index`` indexes into the captured trace (0 = first op).
    Negative values can't be used — refs are strictly forward
    (later ops consume earlier ops' outputs).
    """
    op_index: int

    def __post_init__(self) -> None:
        if self.op_index < 0:
            raise ValueError(
                f"TraceRef.op_index must be non-negative, got "
                f"{self.op_index}")


@dataclass
class OpRecord:
    """One op in a function trace. Inputs may be:

    * :class:`np.ndarray` — host buffer, uploaded lazily by the
      executor before the encoding pass.
    * :class:`DeviceTensor` — already device-resident (e.g., a
      pre-uploaded weight via ``ResidentWeights``); passed through.
    * :class:`TraceRef` — handle to the output of an earlier op in
      the same trace; resolved by the executor at run time.

    Each op produces ONE output tensor (stored as ``output`` after
    execution). ``shape_kwargs`` carry the non-tensor args (rows /
    cols / batch / etc).
    """
    op_name: str
    dtype: str
    inputs: list[Any] = field(default_factory=list)
    shape_kwargs: dict[str, Any] = field(default_factory=dict)
    output: Optional[DeviceTensor] = None


# ---------------------------------------------------------------------
# Chain planner — group consecutive encode-eligible ops.
# ---------------------------------------------------------------------

@dataclass
class ChainSegment:
    """A maximal run of consecutive encode-eligible ops in the trace.
    Has a ``kind`` of ``"encode"`` (run under one session) or
    ``"single"`` (one non-encode-eligible op; executed in isolation).
    """
    kind: str   # "encode" | "single"
    ops: list[OpRecord]


# Phase 5b (2026-06-01) — empirically-measured cap on MPSGraph
# encode calls per command buffer before the GPU dispatch hangs at
# the 30-second commit_and_wait timeout. The cliff sits around 30-40
# encodeToCommandBuffer calls per cb on this Mac (M-series, macOS
# 26). 30 leaves comfortable margin; mostly relevant for multi-layer
# transformer decode where 12+ ops/layer can easily exceed the
# per-cb limit at N≥3 layers.
#
# Callers can override via ``plan_chain(..., max_ops_per_cb=N)``.
# A future Tessera release may adjust this if Apple's MPSGraph
# raises the limit; the constant is the single source of truth.
DEFAULT_OPS_PER_CB: int = 30


def plan_chain(
    trace: list[OpRecord],
    *,
    max_ops_per_cb: int = DEFAULT_OPS_PER_CB,
) -> list[ChainSegment]:
    """Group a trace into chain segments.

    Rules:
    * Consecutive ops where ``is_encode_eligible(op_name, dtype)`` is
      True are merged into one ``ChainSegment(kind="encode")``.
    * Non-eligible ops become a ``ChainSegment(kind="single")`` of
      length 1.
    * A chain breaks on (a) dtype heterogeneity, (b) a non-eligible
      op, or (c) reaching ``max_ops_per_cb`` ops in the current
      encode segment (the multi-cb chunking — protects against the
      MPSGraph per-cb-encode cliff).

    Cross-segment data flow works transparently: a ``TraceRef`` in
    segment K+1 that references an output from segment K is
    resolved by the executor using the persistent ``results`` list
    (which spans all segments).
    """
    if max_ops_per_cb <= 0:
        raise ValueError(
            f"max_ops_per_cb must be >= 1, got {max_ops_per_cb}")

    segments: list[ChainSegment] = []
    current: Optional[ChainSegment] = None
    current_dtype: Optional[str] = None
    for op in trace:
        eligible = is_encode_eligible(op.op_name, op.dtype)
        if eligible:
            # Start a new encode segment if (a) no current segment,
            # (b) current segment is not encode, (c) dtype mismatch,
            # or (d) current segment has reached the per-cb budget.
            should_split = (
                current is None
                or current.kind != "encode"
                or current_dtype != op.dtype
                or len(current.ops) >= max_ops_per_cb
            )
            if should_split:
                if current is not None:
                    segments.append(current)
                current = ChainSegment(kind="encode", ops=[op])
                current_dtype = op.dtype
            else:
                # ``should_split`` is True whenever ``current is None``;
                # reaching this branch means current is a live segment.
                assert current is not None
                current.ops.append(op)
        else:
            if current is not None:
                segments.append(current)
                current = None
                current_dtype = None
            segments.append(ChainSegment(kind="single", ops=[op]))
    if current is not None:
        segments.append(current)
    return segments


# ---------------------------------------------------------------------
# Chain executor — open session, dispatch ops in order.
# ---------------------------------------------------------------------

def _materialize_tensor(t: Any) -> DeviceTensor:
    """Coerce a trace input to a DeviceTensor. ndarray → uploaded
    fresh; DeviceTensor → passthrough. Callers responsible for
    freeing the freshly-uploaded ones if they own them."""
    if isinstance(t, DeviceTensor):
        return t
    if isinstance(t, np.ndarray):
        return device_tensor(t)
    raise TypeError(
        f"trace input must be DeviceTensor or np.ndarray, got "
        f"{type(t).__name__}")


def execute_chain(segments: list[ChainSegment]
                  ) -> list[Optional[DeviceTensor]]:
    """Execute a planned chain. Returns the output DeviceTensor of
    each op (or None for ops that don't produce one). For encode
    segments, opens one ``batched_session()`` and dispatches every op
    into it; commits when the segment finishes.

    The caller owns the returned DeviceTensors (must ``free()`` when
    done). ndarray-uploaded inputs are tracked and freed on segment
    exit unless they appear as a later op's output (which they can't,
    since they're inputs — so all uploads are local).
    """
    results: list[Optional[DeviceTensor]] = []
    for seg in segments:
        if seg.kind == "encode":
            _exec_encode_segment(seg, results)
        else:
            _exec_single_segment(seg, results)
    return results


def _exec_encode_segment(seg: ChainSegment,
                         results: list[Optional[DeviceTensor]]) -> None:
    """Run one encode segment — single batched_session().

    Resolves ``TraceRef`` inputs by looking up the producing op's
    output in ``results`` (which we populate as we go, in trace
    order). Cross-segment refs (TraceRef pointing at an op in an
    earlier segment) work too — ``results`` is the full trace's
    output list.
    """
    locally_uploaded: list[DeviceTensor] = []
    with batched_session() as s:
        for op in seg.ops:
            spec = encode_spec(op.op_name, op.dtype)
            tensor_inputs: list[DeviceTensor] = []
            for inp in op.inputs:
                if isinstance(inp, np.ndarray):
                    t = device_tensor(inp)
                    locally_uploaded.append(t)
                    tensor_inputs.append(t)
                elif isinstance(inp, DeviceTensor):
                    tensor_inputs.append(inp)
                elif isinstance(inp, TraceRef):
                    # Resolve to the producing op's output.
                    if inp.op_index >= len(results):
                        raise IndexError(
                            f"op {op.op_name}: TraceRef(op_index="
                            f"{inp.op_index}) references a future op "
                            f"(only {len(results)} ops executed so far)")
                    resolved = results[inp.op_index]
                    if resolved is None:
                        raise RuntimeError(
                            f"op {op.op_name}: TraceRef(op_index="
                            f"{inp.op_index}) resolved to None — the "
                            f"producing op did not yield an output "
                            f"(non-eligible op in chain?)")
                    tensor_inputs.append(resolved)
                else:
                    raise TypeError(
                        f"op {op.op_name}: input must be ndarray, "
                        f"DeviceTensor, or TraceRef, got "
                        f"{type(inp).__name__}")
            out = spec.encode_fn(s, *tensor_inputs, **op.shape_kwargs)
            op.output = out
            results.append(out)
    # After commit_wait fires, locally-uploaded inputs are safe to
    # free (the GPU has finished reading them).
    for t in locally_uploaded:
        t.free()


def _exec_single_segment(seg: ChainSegment,
                         results: list[Optional[DeviceTensor]]) -> None:
    """Run one non-eligible op in isolation. For ops not in the
    encode registry, we don't try to be clever — defer the actual
    execution to a registered eager-call site (or raise). For now
    this branch is a sentinel: it appends ``None`` to results and
    flags op.output as unset.

    A full JIT integration would route non-encode ops through the
    existing apple_gpu eager dispatch (i.e., the non-session
    dispatcher for the same op). That hookup is out of scope for
    the substrate module — we ship the structural framework here
    and the JIT decides the eager fallback at integration time."""
    for op in seg.ops:
        op.output = None
        results.append(None)


# ---------------------------------------------------------------------
# Convenience: end-to-end plan + execute.
# ---------------------------------------------------------------------

def precompile_chain(
    trace: list[OpRecord],
    *,
    max_ops_per_cb: int = 1,
) -> int:
    """Phase 5c (2026-06-01) — warm the MPSGraph cache for every
    (op, dtype, shape) in the trace by running it once.

    ``max_ops_per_cb`` defaults to ``1`` (Glass-jaw #7, 2026-06-01:
    now tunable) so per-op cbs commit individually and the
    first-encounter MPSGraph compile cost amortizes across many small
    command buffers instead of stacking up in one big cb (which is
    where the shape × op-count cliff hits). Pass the SAME budget you
    will use in production (e.g. ``DEFAULT_OPS_PER_CB``) when you want
    the warm-up to exercise the exact chunking shape the production
    run will hit — warming at ``1`` still populates the per-(op, dtype,
    shape) MPSGraph cache, but the cb-segmentation cost is only
    amortized for the budget you actually warm at.

    After precompile, subsequent production runs hit the warm MPSGraph
    cache and run fast.

    Returns the number of ops actually executed (= length of
    encode-eligible subset of the trace). Resulting DeviceTensors
    are freed immediately — caller doesn't need them.
    """
    results = execute_chain(plan_chain(trace, max_ops_per_cb=max_ops_per_cb))
    # Free the warmup outputs — caller doesn't want them.
    freed = 0
    for r in results:
        if r is not None:
            r.free()
            freed += 1
    return freed


def run_trace(
    trace: list[OpRecord],
    *,
    max_ops_per_cb: int = DEFAULT_OPS_PER_CB,
) -> list[Optional[DeviceTensor]]:
    """Plan + execute in one call. Returns the per-op outputs in
    trace order (same order as the input list). The ``max_ops_per_cb``
    arg caps the number of encode-eligible ops per command buffer;
    chains longer than that split into K cb's transparently."""
    return execute_chain(plan_chain(trace, max_ops_per_cb=max_ops_per_cb))


__all__ = [
    "ChainSegment",
    "DEFAULT_OPS_PER_CB",
    "ENCODE_OP_REGISTRY",
    "EncodeOpSpec",
    "OpRecord",
    "TraceRef",
    "encode_spec",
    "execute_chain",
    "is_encode_eligible",
    "plan_chain",
    "precompile_chain",
    "run_trace",
]
