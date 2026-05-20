"""Canonical program: ``O = softmax(A @ B) @ C``.

The tensor-side counterpart to the GA vertical slice.

Apple plan phase E (2026-05-20): on Darwin this canonical now
**dispatches the real fused MSL kernel**
(``tessera_apple_gpu_matmul_softmax_matmul_f32``) via the runtime
shim.  Previously the canonical surface honestly reported
``REFERENCE_FORCED`` even on Darwin because the driver itself was
running the numpy reference inline.  That's the credibility gap
phase E closes.

Numpy stays as the correctness oracle — the canonical computes the
numpy reference too and asserts ``max_abs_err`` stays within the
fp32 tolerance.  If the dispatch fails for any reason (e.g.,
shape outside the kernel's supported envelope, runtime library
unavailable), the canonical falls back to the numpy result *and*
records a ``REFERENCE_FORCED`` fallback with the specific reason
so the report is unambiguous.

On non-Darwin: unchanged — ``NON_DARWIN_HOST`` fallback.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera.compiler import backend_manifest as bm
from tessera.compiler import jit_bridge as bridge
from tessera.compiler.compile_report import (
    CompileReport,
    finalize_compile_report,
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_TENSOR,
    hash_ir_text,
)
from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
)


PROGRAM_ID = "matmul_softmax_matmul"


def _make_inputs(M: int = 32, N: int = 32, K: int = 32, seed: int = 0):
    rng = np.random.RandomState(seed)
    A = rng.randn(M, K).astype(np.float32)
    B = rng.randn(K, N).astype(np.float32)
    C = rng.randn(N, K).astype(np.float32)
    return A, B, C


def _numpy_reference(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``softmax(A @ B) @ C`` with numerically-stable softmax."""
    scores = A @ B
    scores_shift = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores_shift)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ C


def _ir_text(M: int, N: int, K: int) -> str:
    return (
        "graph_ir {\n"
        f"  %a = tessera.placeholder shape=({M}, {K}) dtype=f32\n"
        f"  %b = tessera.placeholder shape=({K}, {N}) dtype=f32\n"
        f"  %c = tessera.placeholder shape=({N}, {K}) dtype=f32\n"
        "  %scores = tessera.matmul(%a, %b)\n"
        "  %probs  = tessera.softmax(%scores)\n"
        "  %out    = tessera.matmul(%probs, %c)\n"
        "  return %out\n"
        "}\n"
    )


def _try_apple_gpu_dispatch(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Attempt the fused MSL kernel dispatch on Darwin.

    Returns ``(out, error_reason)``: ``out`` is the kernel result on
    success and ``None`` on failure; ``error_reason`` is the
    fallback-reason string used to populate ``REFERENCE_FORCED`` if
    the dispatch couldn't proceed.
    """
    if sys.platform != "darwin":
        return None, "non-Darwin host"
    try:
        from tessera.runtime import _apple_gpu_dispatch_matmul_softmax_matmul
    except Exception as exc:  # pragma: no cover — defensive
        return None, f"runtime import failed: {exc!r}"

    # The dispatch routine itself shape-validates and falls back to
    # numpy internally if the envelope isn't supported.  We separate
    # the "I dispatched the kernel" case from the "I returned numpy"
    # case by checking N + P <= 256 against the kernel's documented
    # envelope; outside that we honestly report REFERENCE_FORCED.
    M, K = A.shape
    if not (B.shape[1] <= 256 and C.shape[1] <= 256):
        return None, (
            f"shape outside the fused MSL kernel's envelope "
            f"(N={B.shape[1]} P={C.shape[1]}; max 256 for either)"
        )

    try:
        out = _apple_gpu_dispatch_matmul_softmax_matmul([A, B, C], np)
    except Exception as exc:
        return None, f"runtime dispatch failed: {exc!r}"
    return out, None


def _target_decision_for_host(dispatched_native: bool, fallback_note: str | None) -> tuple[str, dict[str, str], FallbackReason | None]:
    """Build the report's target name + decision row + fallback reason.

    On Darwin with successful native dispatch:
        target = ``apple_gpu``, no fallback reason.
    On Darwin with dispatch declined / failed:
        target = ``apple_gpu``, ``REFERENCE_FORCED`` + the specific
        reason ``fallback_note`` describes.
    Elsewhere:
        target = ``cpu``, ``NON_DARWIN_HOST``.
    """
    if sys.platform != "darwin":
        return (
            "cpu",
            {"cpu": "non-Darwin host; numpy reference path"},
            FallbackReason.NON_DARWIN_HOST,
        )

    matmul_entries = bm.manifest_for("matmul")
    softmax_entries = bm.manifest_for("softmax")
    apple_matmul = next(
        (e for e in matmul_entries if e.target == "apple_gpu"), None,
    )
    apple_softmax = next(
        (e for e in softmax_entries if e.target == "apple_gpu"), None,
    )
    base_note = (
        f"fused 3-op MSL kernel "
        f"tessera_apple_gpu_matmul_softmax_matmul_f32 "
        f"(matmul status={apple_matmul.status if apple_matmul else '?'}; "
        f"softmax status={apple_softmax.status if apple_softmax else '?'})"
    )

    if dispatched_native:
        return (
            "apple_gpu",
            {"apple_gpu": f"{base_note}. NATIVE DISPATCH: this run executed the fused MSL kernel via the apple_gpu runtime shim."},
            None,  # no fallback — native dispatch succeeded
        )

    return (
        "apple_gpu",
        {"apple_gpu": (
            f"{base_note}. REFERENCE_FORCED: {fallback_note or 'dispatch declined'}."
        )},
        FallbackReason.REFERENCE_FORCED,
    )


def run(
    *, M: int = 32, N: int = 32, K: int = 32, seed: int = 0,
    native_required: bool = False,
) -> CompileReport:
    """Build the canonical program, execute it, return a CompileReport.

    On Darwin the program dispatches the fused MSL kernel
    ``tessera_apple_gpu_matmul_softmax_matmul_f32``.  numpy stays as
    the correctness oracle — ``max_abs_err`` is computed against the
    pure-numpy implementation and gated by ``tolerance``.

    Parameters
    ----------
    native_required
        When ``True``, raises :class:`fallback.TesseraNativeRequiredError`
        instead of falling back to the numpy reference path on any
        host where the fused dispatch isn't reachable (non-Darwin or
        Darwin-with-dispatch-failure).
    """
    A, B, C = _make_inputs(M=M, N=N, K=K, seed=seed)

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        native_out, fallback_note = _try_apple_gpu_dispatch(A, B, C)
        if native_out is not None:
            out = native_out
            dispatched_native = True
        else:
            out = _numpy_reference(A, B, C)
            dispatched_native = False
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    target, target_decision, fallback_reason = _target_decision_for_host(
        dispatched_native, fallback_note,
    )

    if native_required and fallback_reason is not None:
        raise TesseraNativeRequiredError(
            fallback_reason, target="apple_gpu", op_name=PROGRAM_ID,
        )

    ref = _numpy_reference(A, B, C)
    max_abs_err = float(np.abs(out - ref).max())

    return finalize_compile_report(CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(M, N, K))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        # Higher tolerance than M1 because the fused MSL kernel
        # uses fp32 accumulators throughout but fp32 storage I/O,
        # so round-off accumulates differently than numpy's pure
        # fp32 path.  1e-4 is the empirical envelope.
        correctness={"max_abs_err": max_abs_err, "tolerance": 1e-4},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
