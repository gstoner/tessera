"""JIT / runtime bridge for Apple GPU dispatch.

This module is the single hop between the Python frontends (``tessera.ga.*``
/ ``tessera.ebm.*`` / future ``tessera.ops.*``) and the shared Apple GPU
runtime loader (``tessera._apple_gpu_dispatch``).  The contract is:

  1. **Frontend marks the op.** A public-API helper (e.g.
     ``tessera.ga.inner``) calls
     :func:`dispatch_via_manifest("clifford_inner", abi=..., args=...)`.
  2. **Bridge resolves through the manifest.** It calls
     ``backend_manifest.manifest_for(op_name)`` to pick the ``apple_gpu``
     entry, confirms the status is ``fused``, and reads the canonical
     C ABI symbol name from ``_CLIFFORD_APPLE_GPU_FUSED`` /
     ``_EBM_APPLE_GPU_FUSED``.
  3. **Bridge dispatches through the shared loader.**
     ``_apple_gpu_dispatch.bind_symbol(symbol, argtypes)`` returns the
     ctypes-bound function; the bridge invokes it with the operands.
  4. **Bridge records the route.** Each dispatch appends a
     :class:`JitBridgeRoute` to a thread-local trace.  Benchmarks /
     tests drain the trace via :func:`take_dispatch_trace` so they can
     prove each row really went through the bridge.

Why this matters: before this layer existed, every GA/EBM fast path
had its own ``getattr(handle, "...")`` + ``argtypes`` block, duplicating
manifest knowledge in user-facing code.  The bridge centralizes the
manifest lookup + ctypes binding so the frontends are dialect-blind:
they hand the bridge an op name + a list of ctypes args, and get back
the result.

The bridge is also the surface a future ``@jit(target="apple_gpu")``
artifact executor will use — when the JIT traces a user function and
emits an artifact with an op list, executing that artifact reduces to
walking the op list and calling :func:`dispatch_via_manifest` per op.
The :func:`jit_context` context manager marks a span as JIT-driven so
the trace can distinguish JIT-executed routes from direct frontend
calls.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from tessera._apple_gpu_dispatch import bind_symbol as _bind_apple_gpu_symbol
from tessera.compiler import backend_manifest as bm


__all__ = [
    "JitBridgeRoute",
    "JitBridgeMiss",
    "dispatch_via_manifest",
    "record_driver_route",
    "lookup_apple_gpu_symbol",
    "take_dispatch_trace",
    "current_dispatch_trace",
    "clear_dispatch_trace",
    "jit_context",
    "current_jit_context",
    "tracing_enabled",
    "set_tracing_enabled",
]


# ---------------------------------------------------------------------------
# Route trace types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JitBridgeRoute:
    """One row of the bridge's dispatch trace.

    Captured every time a public-API call resolves through the bridge.
    """
    op_name: str
    target: str                   # always "apple_gpu" for v1
    status: str                   # manifest status; should be "fused"
    symbol: str                   # C ABI symbol that ran on the GPU
    context: str                  # "direct" | "jit:<target>" | "test"
    latency_ms: float
    args_summary: tuple[str, ...] = field(default_factory=tuple)


class JitBridgeMiss(Exception):
    """Raised when the manifest doesn't have a ``apple_gpu=fused`` entry
    for the requested op.  Frontends catch this and fall back to numpy."""


# ---------------------------------------------------------------------------
# Thread-local trace + JIT-context stack
# ---------------------------------------------------------------------------

class _BridgeState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.trace: list[JitBridgeRoute] = []
        self.jit_stack: list[str] = []
        self.tracing_enabled: bool = False


_STATE = _BridgeState()


def tracing_enabled() -> bool:
    """``True`` iff trace recording is on for this thread."""
    return _STATE.tracing_enabled


def set_tracing_enabled(enabled: bool) -> None:
    """Toggle trace recording for this thread.

    Off by default — recording every dispatch on every call would add
    measurable overhead to small kernels.  Benchmarks and tests flip
    this on for the spans they care about.
    """
    _STATE.tracing_enabled = bool(enabled)


def current_jit_context() -> Optional[str]:
    """Return the innermost JIT context, e.g., ``"jit:apple_gpu"``, or
    ``None`` if the current call is outside any JIT span."""
    if not _STATE.jit_stack:
        return None
    return _STATE.jit_stack[-1]


@contextmanager
def jit_context(target: str) -> Iterator[None]:
    """Mark a span as JIT-driven.  Routes recorded inside the span
    carry ``context="jit:<target>"``.

    Usage::

        with jit_bridge.jit_context("apple_gpu"):
            out = tessera.ga.inner(a, b)        # dispatches through bridge
                                                 # trace row: context="jit:apple_gpu"
    """
    _STATE.jit_stack.append(f"jit:{target}")
    try:
        yield
    finally:
        _STATE.jit_stack.pop()


def take_dispatch_trace() -> list[JitBridgeRoute]:
    """Return + clear the accumulated trace for this thread."""
    rows = _STATE.trace
    _STATE.trace = []
    return rows


def current_dispatch_trace() -> tuple[JitBridgeRoute, ...]:
    """Read the current trace without clearing — useful for snapshot
    assertions in tests."""
    return tuple(_STATE.trace)


def clear_dispatch_trace() -> None:
    """Drop the accumulated trace.  Equivalent to
    ``take_dispatch_trace()`` but discards the result."""
    _STATE.trace = []


# ---------------------------------------------------------------------------
# Manifest → symbol resolution
# ---------------------------------------------------------------------------

def lookup_apple_gpu_symbol(op_name: str) -> Optional[str]:
    """Resolve ``op_name`` to its Apple-GPU C ABI symbol via the
    backend manifest.  Returns ``None`` when the manifest doesn't have
    a ``apple_gpu=fused`` entry (e.g., during the planning phase for
    a new op).

    The lookup is layered: ``clifford_*`` ops use
    ``_CLIFFORD_APPLE_GPU_FUSED``, ``ebm_*`` ops use
    ``_EBM_APPLE_GPU_FUSED``.  Both tables are the canonical source of
    truth for the per-op symbol name.
    """
    if op_name.startswith("clifford_"):
        spec = bm._CLIFFORD_APPLE_GPU_FUSED.get(op_name)
        if spec is None:
            return None
        prefix = str(spec["symbol_prefix"])
        # GA kernels are versioned per dtype; for the bridge's v1 we
        # only emit f32, mirroring the public-API guards.
        return f"{prefix}f32"
    if op_name.startswith("ebm_"):
        spec = bm._EBM_APPLE_GPU_FUSED.get(op_name)
        if spec is None:
            return None
        return str(spec["symbol"])
    if op_name.startswith("complex_"):
        # M7 follow-up (2026-05-18): the conformal primitives use a
        # parallel manifest table.  Only the 4 GPU-beneficial ops
        # have a fused entry; the rest stay CPU-only.
        spec = bm._COMPLEX_APPLE_GPU_FUSED.get(op_name)
        if spec is None:
            return None
        return str(spec["symbol"])
    return None


def _confirm_apple_gpu_fused(op_name: str) -> tuple[str, str]:
    """Look up the ``(symbol, status)`` for an op + raise
    :class:`JitBridgeMiss` if it isn't ``apple_gpu=fused``."""
    symbol = lookup_apple_gpu_symbol(op_name)
    if symbol is None:
        raise JitBridgeMiss(
            f"manifest has no apple_gpu fast path for {op_name!r}"
        )
    for entry in bm.manifest_for(op_name):
        if entry.target == "apple_gpu":
            if entry.status != "fused":
                raise JitBridgeMiss(
                    f"{op_name}: apple_gpu status={entry.status!r}, "
                    f"need 'fused'"
                )
            return symbol, entry.status
    raise JitBridgeMiss(f"{op_name}: no apple_gpu entry in manifest")


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def dispatch_via_manifest(
    op_name: str,
    *,
    argtypes: Sequence[Any],
    args: Sequence[Any],
    restype: Optional[Any] = None,
    args_summary: tuple[str, ...] = (),
) -> bool:
    """Dispatch one op through the bridge.

    ``op_name`` — the manifest op key (e.g., ``"clifford_inner"`` or
    ``"ebm_inner_step"``).

    ``argtypes`` — ctypes type list for the C ABI symbol, matching the
    manifest's documented signature (e.g.,
    ``(POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int32)``).

    ``args`` — the actual ctypes-wrapped argument values to pass to the
    bound function.  The bridge does not transform them — caller is
    responsible for ``ctypes.data_as`` / ``c_int32(...)`` / etc.

    ``args_summary`` — optional human-readable summary of the arg
    shapes/dtypes, recorded in the trace row for debuggability.

    Returns ``True`` when the dispatch fired on the GPU.  Returns
    ``False`` (and records nothing) when the runtime isn't available —
    the frontend falls back to numpy.  Raises :class:`JitBridgeMiss`
    only when the manifest entry is missing or in the wrong state
    (this is a programming error, not a runtime miss).
    """
    symbol, status = _confirm_apple_gpu_fused(op_name)
    fn = _bind_apple_gpu_symbol(symbol, tuple(argtypes), restype)
    if fn is None:
        # Runtime not loadable on this host — frontends fall back to
        # numpy.  No trace row (didn't actually dispatch anything).
        return False
    t0 = time.perf_counter_ns()
    fn(*args)
    latency_ms = (time.perf_counter_ns() - t0) / 1e6
    if _STATE.tracing_enabled:
        context = current_jit_context() or "direct"
        _STATE.trace.append(JitBridgeRoute(
            op_name=op_name,
            target="apple_gpu",
            status=status,
            symbol=symbol,
            context=context,
            latency_ms=latency_ms,
            args_summary=tuple(args_summary),
        ))
    return True


# ---------------------------------------------------------------------------
# Phase D (Apple plan, 2026-05-20) — unified proof envelope for the
# generic-tensor lane.  The GA/EBM/M7 path always goes through
# ``dispatch_via_manifest`` above, which appends a ``JitBridgeRoute``
# trace row on success.  The generic-tensor lane in
# ``runtime.py::_apple_gpu_dispatch_*`` loads symbols directly and
# previously emitted no trace row, so canonical reports came back
# with ``proof_routes=()``.  This helper closes the gap: the
# generic-tensor dispatch helpers call it after a successful native
# kernel invocation, and the resulting ``JitBridgeRoute`` carries
# the same five-field envelope (op_name, target, status, symbol,
# latency_ms + args_summary) as the manifest-dispatch path.
# ---------------------------------------------------------------------------


def record_driver_route(
    op_name: str,
    *,
    target: str,
    status: str,
    symbol: str,
    latency_ms: float,
    args_summary: tuple[str, ...] = (),
) -> None:
    """Append a ``JitBridgeRoute`` trace row from the generic-tensor
    driver path.  No-op if tracing isn't enabled.

    The ``context`` field is forced to ``"driver"`` so consumers can
    distinguish bridge-dispatch (``"direct"`` / ``"jit:<target>"`` /
    ``"test"``) from driver-dispatch (``"driver"``) when reading
    proof_routes back."""

    if not _STATE.tracing_enabled:
        return
    _STATE.trace.append(JitBridgeRoute(
        op_name=op_name,
        target=target,
        status=status,
        symbol=symbol,
        context="driver",
        latency_ms=latency_ms,
        args_summary=tuple(args_summary),
    ))


# ---------------------------------------------------------------------------
# Convenience: bind + dispatch from a Python frontend.  This wraps
# the common "compute output buffer in numpy, then pass everything as
# ctypes pointers" pattern used by every fast path.
# ---------------------------------------------------------------------------

def shaped_summary(*arrays: Any) -> tuple[str, ...]:
    """Produce a tuple of ``"shape:dtype"`` strings for trace rows."""
    out: list[str] = []
    for a in arrays:
        shape = getattr(a, "shape", None)
        dtype = getattr(a, "dtype", None)
        if shape is None or dtype is None:
            out.append(repr(type(a).__name__))
        else:
            out.append(f"{tuple(shape)}:{dtype}")
    return tuple(out)
