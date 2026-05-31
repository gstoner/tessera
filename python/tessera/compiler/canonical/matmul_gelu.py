"""Canonical program: ``O = gelu(A @ B)``.

Apple follow-up #1 (2026-05-20): the second generic-tensor canonical
to dispatch the real fused MSL kernel on Darwin (after
``matmul_softmax_matmul``).  Mirrors the same Phase E pattern:

* On Darwin within envelope (N ≤ 256): dispatch
  ``tessera_apple_gpu_matmul_gelu_f32`` via the runtime shim, emit
  a unified ``JitBridgeRoute`` (``context="driver"``), report
  ``fallback_reason = None``.
* On Darwin outside envelope: numpy fallback with a precise
  ``REFERENCE_FORCED`` note.
* On non-Darwin: ``NON_DARWIN_HOST`` fallback.

Numpy stays as the correctness oracle.  The fused MSL kernel uses
fp32 storage + accumulation throughout, so ``max_abs_err`` against
the numpy reference should stay within ``1e-4``.
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


PROGRAM_ID = "matmul_gelu"


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _make_inputs(M: int = 32, N: int = 32, K: int = 32, seed: int = 0):
    """Build deterministic ``(A, B)``.

    Inputs are scaled by ``0.5`` to keep ``A @ B`` inside the
    numerical envelope the shipped MSL kernel was tested against
    (see ``test_apple_gpu_matmul_gelu_executes_through_fused_msl_kernel``
    in ``test_apple_backend_roadmap.py``).  Without the scale,
    ``randn(32, 32)``-style inputs produce ``A @ B`` entries large
    enough that the kernel's tanh approximation of the cubic term
    overflows to NaN in a small fraction of cells.  The ``0.5`` scale
    matches what's already being used in CI today.
    """
    rng = np.random.RandomState(seed)
    A = (rng.randn(M, K) * 0.5).astype(np.float32)
    B = (rng.randn(K, N) * 0.5).astype(np.float32)
    return A, B


def _numpy_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``gelu(A @ B)`` with the tanh-approximation gelu (matches the
    MSL kernel's formulation)."""
    scores = A @ B
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * scores * (1.0 + np.tanh(c * (scores + 0.044715 * scores ** 3)))


def _ir_text(M: int, N: int, K: int) -> str:
    return (
        "graph_ir {\n"
        f"  %a = tessera.placeholder shape=({M}, {K}) dtype=f32\n"
        f"  %b = tessera.placeholder shape=({K}, {N}) dtype=f32\n"
        "  %scores = tessera.matmul(%a, %b)\n"
        "  %out    = tessera.gelu(%scores)\n"
        "  return %out\n"
        "}\n"
    )


def _try_apple_gpu_dispatch(A: np.ndarray, B: np.ndarray):
    """Attempt the fused MSL kernel dispatch on Darwin.

    Returns ``(out, error_reason)``: ``out`` is the kernel result on
    success, ``None`` on failure; ``error_reason`` populates
    ``REFERENCE_FORCED`` when the dispatch can't proceed.
    """
    if not _is_darwin():
        return None, "non-Darwin host"
    try:
        from tessera.runtime import _apple_gpu_dispatch_matmul_gelu
    except Exception as exc:  # pragma: no cover
        return None, f"runtime import failed: {exc!r}"

    # Envelope check — N ≤ 256 (matches the MSL kernel's documented limit).
    if B.shape[1] > 256:
        return None, (
            f"shape outside the fused MSL kernel's envelope "
            f"(N={B.shape[1]}; max 256)"
        )

    try:
        out = _apple_gpu_dispatch_matmul_gelu([A, B], np)
    except Exception as exc:
        return None, f"runtime dispatch failed: {exc!r}"
    return out, None


def _target_decision_for_host(
    dispatched_native: bool, fallback_note: str | None,
) -> tuple[str, dict[str, str], FallbackReason | None]:
    if not _is_darwin():
        return (
            "cpu",
            {"cpu": "non-Darwin host; numpy reference path"},
            FallbackReason.NON_DARWIN_HOST,
        )
    matmul_entries = bm.manifest_for("matmul")
    gelu_entries = bm.manifest_for("gelu")
    apple_matmul = next(
        (e for e in matmul_entries if e.target == "apple_gpu"), None,
    )
    apple_gelu = next(
        (e for e in gelu_entries if e.target == "apple_gpu"), None,
    )
    base_note = (
        "fused 2-op MSL kernel tessera_apple_gpu_matmul_gelu_f32 "
        f"(matmul status={apple_matmul.status if apple_matmul else '?'}; "
        f"gelu status={apple_gelu.status if apple_gelu else '?'})"
    )
    if dispatched_native:
        return (
            "apple_gpu",
            {"apple_gpu": (
                f"{base_note}. NATIVE DISPATCH: this run executed the "
                "fused MSL kernel via the apple_gpu runtime shim."
            )},
            None,
        )
    return (
        "apple_gpu",
        {"apple_gpu": (
            f"{base_note}. REFERENCE_FORCED: "
            f"{fallback_note or 'dispatch declined'}."
        )},
        FallbackReason.REFERENCE_FORCED,
    )


def run(
    *, M: int = 32, N: int = 32, K: int = 32, seed: int = 0,
    native_required: bool = False,
) -> CompileReport:
    A, B = _make_inputs(M=M, N=N, K=K, seed=seed)

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        native_out, fallback_note = _try_apple_gpu_dispatch(A, B)
        if native_out is not None:
            out = native_out
            dispatched_native = True
        else:
            out = _numpy_reference(A, B)
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

    ref = _numpy_reference(A, B)
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
        correctness={"max_abs_err": max_abs_err, "tolerance": 1e-4},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
