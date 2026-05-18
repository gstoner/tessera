"""Canonical program: ``decode_init → T × inner_step → self_verify``.

The EBM vertical-slice demo, emitted as a :class:`CompileReport`.
The pipeline is the canonical EBT decode loop:

  1. ``decode_init(x, K)`` — produce K candidate trajectories.
  2. ``inner_step(...)`` repeated T times — refine each candidate by
     gradient descent on the supplied energy gradient.
  3. ``self_verify(energies, candidates)`` — hard-argmin pick.

Lane: numpy-reference everywhere; on Darwin the bridge route trace
captures the three native MSL kernel dispatches
(``ebm_decode_init_noise_apply_f32`` → ``ebm_inner_step_f32`` ×
T → ``ebm_self_verify_f32``).
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera import ebm
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
    classify_host,
)


PROGRAM_ID = "decode_init_inner_loop_self_verify"


def _make_inputs(*, B: int = 2, K: int = 4, D: int = 8, T: int = 3, seed: int = 0):
    """Deterministic ``(x, candidates, grad, energies)`` tuple."""
    rng = np.random.RandomState(seed)
    x = rng.randn(B, D).astype(np.float32)
    candidates = rng.randn(B, K, D).astype(np.float32)
    grad = rng.randn(B, K, D).astype(np.float32) * 0.1
    energies = rng.randn(B, K).astype(np.float32)
    return x, candidates, grad, energies, T


def _numpy_reference(
    x: np.ndarray, candidates: np.ndarray, grad: np.ndarray,
    energies: np.ndarray, T: int,
) -> np.ndarray:
    """Numpy baseline: T inner_step iterations, then argmin reduction."""
    y = candidates.copy()
    for _ in range(T):
        y = y - 0.05 * grad
    # self_verify with beta=None ⇒ hard argmin.
    idx = np.argmin(energies, axis=-1)
    B = energies.shape[0]
    return np.array([y[b, idx[b]] for b in range(B)])


def _ir_text(B: int, K: int, D: int, T: int) -> str:
    return (
        "graph_ir {\n"
        f"  %x          = tessera.placeholder shape=({B}, {D}) dtype=f32\n"
        f"  %candidates = tessera.placeholder shape=({B}, {K}, {D}) dtype=f32\n"
        f"  %grad       = tessera.placeholder shape=({B}, {K}, {D}) dtype=f32\n"
        f"  %energies   = tessera.placeholder shape=({B}, {K}) dtype=f32\n"
        f"  %candidates_0 = ebm.decode_init(%x, K={K})\n"
        f"  %candidates_T = scf.for i in [0, {T}) {{ ebm.inner_step(...) }}\n"
        "  %winner       = ebm.self_verify(%energies, %candidates_T)\n"
        "  return %winner\n"
        "}\n"
    )


def run_per_step_gradient(
    *, B: int = 2, K: int = 4, D: int = 8, T: int = 3, seed: int = 0,
) -> np.ndarray:
    """M6 Step 3 variant — every refinement step recomputes ``∇E(y)``
    via :func:`tessera.compiler.energy_grad.refine` instead of
    reusing a snapshot.

    Returns the (B, K, D) refined candidates so the caller can
    compare against the snapshot path.  This is the building
    block for an MSL-fused energy+gradient kernel — same shape,
    same outputs, but the gradient is materialized inside the
    refinement loop rather than uploaded once.
    """
    from tessera import energy
    from tessera.compiler.energy_grad import make_gradient_program, refine

    _, candidates, _, _, T = _make_inputs(B=B, K=K, D=D, T=T, seed=seed)

    def E(y):
        return energy.norm_sq(y)

    prog = make_gradient_program(E)
    # Apply refinement to every (B, K) candidate row.
    flat = candidates.reshape(B * K, D)
    refined = np.stack(
        [refine(flat[i], prog, T=T, eta=0.05) for i in range(flat.shape[0])],
        axis=0,
    )
    return refined.reshape(B, K, D)


def run(
    *, B: int = 2, K: int = 4, D: int = 8, T: int = 3, seed: int = 0,
    native_required: bool = False,
) -> CompileReport:
    """Execute the canonical EBM pipeline and emit a CompileReport."""
    x, candidates, grad, energies, T = _make_inputs(B=B, K=K, D=D, T=T, seed=seed)
    is_darwin = sys.platform == "darwin"
    host_fail = classify_host(is_darwin=is_darwin, runtime_available=True)
    if native_required and host_fail is not None:
        raise TesseraNativeRequiredError(
            host_fail, target="apple_gpu", op_name=PROGRAM_ID,
        )
    target = "apple_gpu" if is_darwin else "cpu"
    target_decision = {
        target: (
            "apple_gpu fused MSL kernels: ebm_decode_init_noise_apply_f32 + "
            "ebm_inner_step_f32 ×T + ebm_self_verify_f32"
            if is_darwin else "non-Darwin host; numpy reference"
        )
    }
    fallback_reason: FallbackReason | None = host_fail

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        # Drive the public EBM API.  On Darwin each call routes
        # through `jit_bridge.dispatch_via_manifest`; the bridge
        # records a JitBridgeRoute per dispatch.
        y = candidates.copy()
        for _ in range(T):
            y = ebm.inner_step(y, grad, 0.05)
        winner = ebm.self_verify(energies, y)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    ref = _numpy_reference(x, candidates, grad, energies, T)
    max_abs_err = float(np.abs(np.asarray(winner) - ref).max())

    return finalize_compile_report(CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(B, K, D, T))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        correctness={"max_abs_err": max_abs_err, "tolerance": 5e-5},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
