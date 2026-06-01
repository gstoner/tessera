"""JIT Phase 2 substrate — encode-session chain registry + planner + executor.

The single-cb decode plan (``docs/audit/single_command_buffer_decode_plan.md``)
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
    device_tensor,
    flash_attn_enc,
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
    # bf16 ops — Project-3 (2026-06-01). MPSGraph-routed only;
    # rope/flash_attn bf16 (MSL kernel paths) are the Phase-3b
    # follow-on and intentionally absent from the registry today.
    EncodeOpSpec("bmm", "bf16", bmm_enc_bf16, input_tensor_args=(0, 1)),
    EncodeOpSpec("layer_norm", "bf16", layer_norm_enc_bf16,
                  input_tensor_args=(0, 1, 2)),
    EncodeOpSpec("rmsnorm", "bf16", rmsnorm_enc_bf16,
                  input_tensor_args=(0, 1)),
    EncodeOpSpec("softmax", "bf16", softmax_enc_bf16,
                  input_tensor_args=(0,)),
    EncodeOpSpec("silu", "bf16", silu_enc_bf16, input_tensor_args=(0,)),
    EncodeOpSpec("gelu", "bf16", gelu_enc_bf16, input_tensor_args=(0,)),
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


def plan_chain(trace: list[OpRecord]) -> list[ChainSegment]:
    """Group a trace into chain segments.

    Rules:
    * Consecutive ops where ``is_encode_eligible(op_name, dtype)`` is
      True are merged into one ``ChainSegment(kind="encode")``.
    * Non-eligible ops become a ``ChainSegment(kind="single")`` of
      length 1.
    * A chain breaks ONLY on dtype heterogeneity (f32 ↔ f16 mid-
      chain forces a session boundary, because the encode helpers are
      typed). Same-dtype eligible runs of any length stay in one
      session.
    """
    segments: list[ChainSegment] = []
    current: Optional[ChainSegment] = None
    current_dtype: Optional[str] = None
    for op in trace:
        eligible = is_encode_eligible(op.op_name, op.dtype)
        if eligible:
            # Start or continue an encode segment, but only if dtype
            # matches the segment's running dtype (mixed dtypes need
            # separate sessions).
            if (current is not None and current.kind == "encode"
                    and current_dtype == op.dtype):
                current.ops.append(op)
            else:
                if current is not None:
                    segments.append(current)
                current = ChainSegment(kind="encode", ops=[op])
                current_dtype = op.dtype
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

def run_trace(trace: list[OpRecord]) -> list[Optional[DeviceTensor]]:
    """Plan + execute in one call. Returns the per-op outputs in
    trace order (same order as the input list)."""
    return execute_chain(plan_chain(trace))


__all__ = [
    "ChainSegment",
    "ENCODE_OP_REGISTRY",
    "EncodeOpSpec",
    "OpRecord",
    "TraceRef",
    "encode_spec",
    "execute_chain",
    "is_encode_eligible",
    "plan_chain",
    "run_trace",
]
