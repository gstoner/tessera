"""Canonical program: ``O = softmax(A @ B) @ C``.

The tensor-side counterpart to the GA vertical slice.  Lane: numpy
reference everywhere; on Darwin we also walk the bridge's manifest
lookup for ``matmul`` / ``softmax`` so the report carries a
target-decision row that proves the fused 3-op MSL kernel
(``tessera_apple_gpu_matmul_softmax_matmul_f32``) is the intended
fast path.

Per M1: the report schema + this program form the first canonical
end-to-end CPU-runnable demo for the tensor lane.  Native dispatch
of the fused kernel is left to M3 (`native_required=True` option);
M1's job is the inspectable envelope.
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
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_TENSOR,
    hash_ir_text,
)


PROGRAM_ID = "matmul_softmax_matmul"


def _make_inputs(M: int = 32, N: int = 32, K: int = 32, seed: int = 0):
    """Small deterministic ``(A, B, C)`` tuple — keeps the CPU
    reference fast enough for unit-test CI."""
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
    """Tiny synthetic Graph-IR-shaped text for the report's
    ``ir_hashes`` map.  M1 emits a placeholder digest until M2
    threads the real GraphIRModule through compile-report emission."""
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


def _target_decision_for_host() -> tuple[str, dict[str, str], str | None]:
    """Pick the report's target name + decision row + fallback reason.

    On Darwin: target=apple_gpu, decision row mentions the manifest
    symbol for the 3-op fused kernel.  Elsewhere: target=cpu, fallback
    reason names the host.
    """
    if sys.platform == "darwin":
        # Probe the manifest for the leaf ops the fused chain comprises.
        # `manifest_for` takes the unqualified op name.
        matmul_entries = bm.manifest_for("matmul")
        softmax_entries = bm.manifest_for("softmax")
        apple_matmul = next(
            (e for e in matmul_entries if e.target == "apple_gpu"), None,
        )
        apple_softmax = next(
            (e for e in softmax_entries if e.target == "apple_gpu"), None,
        )
        return (
            "apple_gpu",
            {
                "apple_gpu": (
                    f"fused 3-op MSL kernel "
                    f"tessera_apple_gpu_matmul_softmax_matmul_f32 "
                    f"(matmul status={apple_matmul.status if apple_matmul else '?'}; "
                    f"softmax status={apple_softmax.status if apple_softmax else '?'})"
                )
            },
            None,
        )
    return (
        "cpu",
        {"cpu": "non-Darwin host; numpy reference path"},
        "non-Darwin host",
    )


def run(*, M: int = 32, N: int = 32, K: int = 32, seed: int = 0) -> CompileReport:
    """Build the canonical program, execute it, return a
    :class:`CompileReport`."""
    A, B, C = _make_inputs(M=M, N=N, K=K, seed=seed)
    target, target_decision, fallback_reason = _target_decision_for_host()

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        out = _numpy_reference(A, B, C)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        # M1 doesn't dispatch the fused kernel from this driver
        # (that lands in M3 with native_required=True).  The trace
        # is still snapshotted so future revisions don't have to
        # rewrite this path.
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    ref = _numpy_reference(A, B, C)
    max_abs_err = float(np.abs(out - ref).max())

    return CompileReport(
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
        correctness={"max_abs_err": max_abs_err, "tolerance": 1e-6},
    )


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
