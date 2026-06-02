"""User-facing Apple GPU op surface with dual-mode dispatch.

Phase 2.1 of the single-cb decode plan
(``docs/audit/single_command_buffer_decode_plan.md``) — the missing
piece between the chain substrate (registry + planner + executor)
and a true JIT-level auto-batching experience.

Each op in this module has TWO execution modes:

* **Eager mode** (the default — no trace active) — calls the existing
  Apple-GPU dispatcher directly. Same semantics as e.g.
  ``tessera_apple_gpu_rmsnorm_f32``: host-data in, host-data out,
  one command buffer per op. Drop-in replacement for users who
  already call those C ABIs.
* **Trace-capturing mode** — active under ``@auto_batch`` (or a
  future ``@jit(target='apple_gpu', auto_batch=True)``). Each call
  appends an ``OpRecord`` to a context-local trace and returns a
  ``TraceRef`` — the actual GPU dispatch is deferred until the
  decorator's exit, where ``run_trace`` batches everything into one
  or more sessions.

The contract: code written against this surface runs correctly in
BOTH modes. ``@auto_batch`` is "free" — the user gets the single-cb
speedup without touching their code.

Today's surface (eager dispatch always supported; trace dispatch
covers the 16 ops in :data:`apple_gpu_chain.ENCODE_OP_REGISTRY`):

* ``rmsnorm`` / ``layer_norm`` / ``softmax`` (row ops)
* ``bmm`` (batched matmul)
* ``rope`` (rotary embedding)
* ``silu`` / ``gelu`` (unary activations)
* ``flash_attn`` (fused attention)
"""

from __future__ import annotations

import contextvars
import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np

from .apple_gpu_batched import (
    DeviceTensor,
    device_tensor,
    session_available,
)
from .apple_gpu_chain import (
    OpRecord,
    TraceRef,
    precompile_chain,
    run_trace,
)


# ---------------------------------------------------------------------
# Trace context — None when in eager mode, a list-of-OpRecord when
# under @auto_batch. Captured via contextvars so concurrent
# @auto_batch'd functions on different threads / async tasks get
# their own traces.
# ---------------------------------------------------------------------

_TRACE_CTX: contextvars.ContextVar[Optional[list[OpRecord]]] = (
    contextvars.ContextVar("_apple_gpu_op_trace", default=None))


def _active_trace() -> Optional[list[OpRecord]]:
    """Return the active trace list, or None when in eager mode."""
    return _TRACE_CTX.get()


# Types accepted as op inputs across both modes.
OpInput = Union[np.ndarray, DeviceTensor, TraceRef]


def _record_op(op_name: str, dtype: str, inputs: list[OpInput],
               shape_kwargs: dict[str, Any]) -> TraceRef:
    """Append an OpRecord to the active trace and return a TraceRef
    pointing at it. Caller has already verified a trace is active."""
    trace = _active_trace()
    assert trace is not None  # checked by caller
    idx = len(trace)
    trace.append(OpRecord(
        op_name=op_name, dtype=dtype,
        inputs=list(inputs), shape_kwargs=dict(shape_kwargs)))
    return TraceRef(op_index=idx)


# ---------------------------------------------------------------------
# Eager-mode fallback. Each op accepts the same inputs in eager mode
# as in trace mode — they're materialized to host arrays (downloading
# DeviceTensor inputs if needed) and routed to the existing
# ``apple_gpu_batched`` helper inside a one-shot session.
# ---------------------------------------------------------------------

def _materialize_eager(inputs: list[OpInput],
                       output_shape: tuple[int, ...],
                       dtype: str) -> tuple[
                           list[DeviceTensor], list[bool]]:
    """Convert ndarray inputs to DeviceTensors for one-shot eager
    dispatch. Returns (device_tensors, owned_flags) where owned_flags
    marks which DeviceTensors this call uploaded (and should free)
    vs. ones the caller passed (don't touch). The output_shape /
    dtype args are reserved for future input-type validation."""
    devs: list[DeviceTensor] = []
    owned: list[bool] = []
    for inp in inputs:
        if isinstance(inp, np.ndarray):
            devs.append(device_tensor(inp))
            owned.append(True)
        elif isinstance(inp, DeviceTensor):
            devs.append(inp)
            owned.append(False)
        elif isinstance(inp, TraceRef):
            raise RuntimeError(
                "TraceRef received in eager mode — TraceRefs should "
                "only appear inside an @auto_batch block")
        else:
            raise TypeError(
                f"input must be ndarray, DeviceTensor, or TraceRef, "
                f"got {type(inp).__name__}")
    return devs, owned


def _eager_run(op_name: str, dtype: str, inputs: list[OpInput],
               shape_kwargs: dict[str, Any]) -> DeviceTensor:
    """Eager mode — open a session, encode the op, commit. Caller
    receives a DeviceTensor (caller owns it; .free() when done).
    Locally-uploaded inputs are freed after dispatch."""
    # Late import: avoid circular at module load.
    from .apple_gpu_batched import batched_session
    from .apple_gpu_chain import encode_spec
    devs, owned = _materialize_eager(inputs, (), dtype)
    try:
        spec = encode_spec(op_name, dtype)
        with batched_session() as s:
            out = spec.encode_fn(s, *devs, **shape_kwargs)
        return out
    finally:
        for d, o in zip(devs, owned):
            if o:
                d.free()


# ---------------------------------------------------------------------
# Dual-mode helper — pick eager or trace dispatch from contextvar.
# Each user-facing op calls this with its (name, dtype, args, kwargs).
# ---------------------------------------------------------------------

def _dispatch(op_name: str, dtype: str,
              inputs: list[OpInput],
              shape_kwargs: dict[str, Any]
              ) -> Union[DeviceTensor, TraceRef]:
    if _active_trace() is not None:
        return _record_op(op_name, dtype, inputs, shape_kwargs)
    return _eager_run(op_name, dtype, inputs, shape_kwargs)


# ---------------------------------------------------------------------
# User-facing op functions. Each one is a thin wrapper that builds
# the input list + shape kwargs and dispatches.
# ---------------------------------------------------------------------

def rmsnorm(X: OpInput, gamma: OpInput, *,
            rows: int, cols: int, eps: float = 1e-5,
            dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Row-wise RMS normalization (Llama-style). Eager or trace
    dispatch chosen by context. ``dtype`` selects f32 / f16
    registry entry."""
    return _dispatch("rmsnorm", dtype, [X, gamma],
                      dict(rows=rows, cols=cols, eps=eps))


def layer_norm(X: OpInput, gamma: OpInput, beta: OpInput, *,
               rows: int, cols: int, eps: float = 1e-5,
               dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Row-wise layer normalization. Eager or trace dispatch."""
    return _dispatch("layer_norm", dtype, [X, gamma, beta],
                      dict(rows=rows, cols=cols, eps=eps))


def softmax(X: OpInput, *, rows: int, cols: int,
            dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Free-standing softmax (separate from flash_attn fusion)."""
    return _dispatch("softmax", dtype, [X],
                      dict(rows=rows, cols=cols))


def bmm(A: OpInput, B: OpInput, *,
        batch: int, M: int, N: int, K: int,
        b_broadcast: bool = False,
        dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Batched matrix multiply."""
    return _dispatch("bmm", dtype, [A, B],
                      dict(batch=batch, M=M, N=N, K=K,
                            b_broadcast=b_broadcast))


def rope(X: OpInput, Theta: OpInput, *,
         M: int, K: int,
         dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Rotary position embedding apply (pair-wise rotation)."""
    return _dispatch("rope", dtype, [X, Theta], dict(M=M, K=K))


def silu(X: OpInput, *, n: int,
         dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """SiLU activation (x * sigmoid(x))."""
    return _dispatch("silu", dtype, [X], dict(n=n))


def gelu(X: OpInput, *, n: int,
         dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """GELU activation (tanh approximation)."""
    return _dispatch("gelu", dtype, [X], dict(n=n))


def flash_attn(Q: OpInput, K: OpInput, V: OpInput, *,
               B: int, Sq: int, Sk: int, D: int,
               scale: Optional[float] = None,
               causal: bool = False,
               dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Flash-attention forward (Q, K, V → O)."""
    if scale is None:
        scale = 1.0 / (float(D) ** 0.5)
    return _dispatch("flash_attn", dtype, [Q, K, V],
                      dict(B=B, Sq=Sq, Sk=Sk, D=D,
                            scale=scale, causal=causal))


def conv2d(X: OpInput, Wt: OpInput, *,
            N: int, H: int, W: int, Cin: int, Cout: int,
            kH: int, kW: int, strideH: int = 1, strideW: int = 1,
            padH: int = 0, padW: int = 0,
            dilationH: int = 1, dilationW: int = 1,
            groups: int = 1,
            dtype: str = "f32") -> Union[DeviceTensor, TraceRef]:
    """Project 5 (2026-06-01) — NHWC conv2d on the encode-session
    chain. Source layout NHWC, weights HWIO, output NHWC. The
    no-bias variant is exposed here so the trace can route through
    :data:`ENCODE_OP_REGISTRY` (which tracks positional DeviceTensor
    args); bias-bearing conv2d goes through
    :func:`apple_gpu_batched.conv2d_enc` directly."""
    return _dispatch("conv2d", dtype, [X, Wt],
                      dict(N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                            kH=kH, kW=kW, strideH=strideH,
                            strideW=strideW, padH=padH, padW=padW,
                            dilationH=dilationH, dilationW=dilationW,
                            groups=groups))


# ---------------------------------------------------------------------
# @auto_batch decorator — capture the trace, execute it through
# run_trace, and return the (resolved) final DeviceTensor.
# ---------------------------------------------------------------------

@contextmanager
def _trace_scope() -> Iterator[list[OpRecord]]:
    """Activate trace-capturing mode. The yielded list IS the trace
    being built; the caller consumes it after exit."""
    trace: list[OpRecord] = []
    token = _TRACE_CTX.set(trace)
    try:
        yield trace
    finally:
        _TRACE_CTX.reset(token)


def _resolve_return(value: Any,
                    executed: list[Optional[DeviceTensor]]) -> Any:
    """The user's @auto_batch function returns a (possibly nested)
    structure of TraceRef + plain values. Walk the structure and
    replace each TraceRef with the actual DeviceTensor produced by
    the executor."""
    if isinstance(value, TraceRef):
        if value.op_index >= len(executed):
            raise IndexError(
                f"@auto_batch: TraceRef(op_index={value.op_index}) "
                f"out of range — only {len(executed)} ops in trace")
        resolved = executed[value.op_index]
        if resolved is None:
            raise RuntimeError(
                f"@auto_batch: TraceRef(op_index={value.op_index}) "
                f"resolved to None — non-eligible op in trace?")
        return resolved
    if isinstance(value, tuple):
        return tuple(_resolve_return(v, executed) for v in value)
    if isinstance(value, list):
        return [_resolve_return(v, executed) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_return(v, executed) for k, v in value.items()}
    return value


def auto_batch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator — capture every Apple-GPU op call inside ``fn`` into
    a single trace, then batch-execute it through ``run_trace``.

    Inside the wrapped function the user writes natural-looking code::

        @auto_batch
        def attention(x, gamma, wq, wk, wv, wo, theta, *, B, S, D, eps):
            n = apple_gpu_ops.rmsnorm(x, gamma, rows=B*S, cols=D, eps=eps)
            q = apple_gpu_ops.bmm(n, wq, batch=1, M=B*S, N=D, K=D)
            k = apple_gpu_ops.bmm(n, wk, batch=1, M=B*S, N=D, K=D)
            v = apple_gpu_ops.bmm(n, wv, batch=1, M=B*S, N=D, K=D)
            q_r = apple_gpu_ops.rope(q, theta, M=B*S, K=D)
            k_r = apple_gpu_ops.rope(k, theta, M=B*S, K=D)
            a = apple_gpu_ops.flash_attn(q_r, k_r, v,
                                          B=B, Sq=S, Sk=S, D=D)
            return apple_gpu_ops.bmm(a, wo, batch=1, M=B*S, N=D, K=D)

    Behind the scenes, each ``apple_gpu_ops.*`` call returns a
    :class:`TraceRef`; the decorator runs the trace through
    ``run_trace`` (one cb per encode-segment) and replaces the
    TraceRef in the return value with the actual DeviceTensor.

    Same code, called without the decorator, runs the same ops
    eagerly (one cb per op).
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _active_trace() is not None:
            # Nested @auto_batch — flatten into the outer trace.
            return fn(*args, **kwargs)
        with _trace_scope() as trace:
            ret = fn(*args, **kwargs)
        # Trace captured. Execute it.
        executed = run_trace(trace)
        return _resolve_return(ret, executed)

    def warmup(*args, **kwargs) -> int:
        """Phase 5c (2026-06-01) — warm the MPSGraph cache by running
        the wrapped function ONCE with per-op cbs (``max_ops_per_cb=1``).
        Each op's MPSGraph compile cost amortizes across many small
        command buffers, avoiding the shape × op-count cliff that
        catches the first big-chain encode.

        After ``warmup(*args, **kwargs)``, the regular call
        ``wrapped(*args, **kwargs)`` at the default budget hits the
        warm MPSGraph cache and runs fast even at shape × N combos
        that would otherwise hang.

        Returns the number of ops actually executed during warmup.
        The output DeviceTensors are freed immediately — the warmup
        is for cache-warming only.

        Caller-visible side-effects: any ``ResidentWeights.activation``
        slot used inside the function gets uploaded; weights stay
        resident (which is the point).
        """
        if _active_trace() is not None:
            # Already inside a trace — warmup is a no-op for the
            # outer trace's perspective. Run the inner function in
            # the existing trace.
            fn(*args, **kwargs)
            return 0
        with _trace_scope() as trace:
            fn(*args, **kwargs)
        return precompile_chain(trace)

    wrapped.warmup = warmup  # type: ignore[attr-defined]
    return wrapped


__all__ = [
    "auto_batch",
    "bmm",
    "conv2d",
    "flash_attn",
    "gelu",
    "layer_norm",
    "rmsnorm",
    "rope",
    "silu",
    "softmax",
]
