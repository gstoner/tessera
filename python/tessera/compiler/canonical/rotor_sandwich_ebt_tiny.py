"""Canonical program: ``rotor_sandwich → ebt_tiny`` — GA + EBM
composite.

The composite vertical slice: a GA rotor conditions the candidates,
then EBT-tiny refines + picks the argmin in one fused MSL dispatch.

  1. ``ga.rotor_sandwich(R, points)`` — rotate the candidate
     "points" multivectors via ``R x R†``.  Apple GPU fused MSL.
  2. ``ebm.ebt_tiny(y0, grad, eta, T, B, K, D)`` — fused
     refinement + per-row squared-norm energy + K-way argmin in
     one Metal dispatch.  Streaming closed-form (any ``D``;
     ``K ≤ 256``).

Both ops have shipped fused kernels today; this canonical does NOT
depend on M6 Step 3 (per-step gradient recomputation).  The
``grad`` supplied to ``ebt_tiny`` is a fixed snapshot, which
matches the kernel's documented contract.  Step 3 would replace
the snapshot with on-device gradient generation — orthogonal to
this canonical's correctness claim.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera import ebm, ga
from tessera.compiler import jit_bridge as bridge
from tessera.compiler.compile_report import (
    CompileReport,
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_MIXED,
    hash_ir_text,
)
from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
)


PROGRAM_ID = "rotor_sandwich_ebt_tiny"


def _make_inputs(
    *, B: int = 4, K: int = 16, D: int = 8, T: int = 4, seed: int = 0,
):
    """Deterministic ``(rotor, points, grad)`` inputs.

    ``points.shape == (B * K, D)`` — flattened candidate axis as
    required by ``ebt_tiny``.  ``rotor`` is the Cl(3,0) multivector
    that conditions each candidate before refinement.
    """
    rng = np.random.RandomState(seed)
    # Apply the rotor to points first, then reshape into the EBT layout.
    # For canonical purposes, both rotor and points are 8-component
    # Cl(3,0) vectors (B*K rows).
    R = (rng.randn(B * K, 8).astype(np.float32) * 0.3)
    V = rng.randn(B * K, 8).astype(np.float32)
    grad = rng.randn(B * K, D).astype(np.float32) * 0.1
    return R, V, grad, B, K, D, T


def _numpy_reference(
    R: np.ndarray, V: np.ndarray, grad: np.ndarray,
    B: int, K: int, D: int, T: int, eta: float,
) -> np.ndarray:
    """Pure-numpy version of the chain: rotor_sandwich → take first
    D coords → fused refinement+self_verify."""
    a = ga.Cl(3, 0)
    rotated = np.zeros_like(V)
    for i in range(V.shape[0]):
        out = ga.rotor_sandwich(ga.Multivector(R[i], a), ga.Multivector(V[i], a))
        rotated[i] = np.asarray(out._coefficients)
    # Take first D components as the EBT candidate state.
    y0 = rotated[:, :D].copy()
    # Replicate ebt_tiny's math: y_T = y0 - T*eta*grad; e_k = ||y_T||^2;
    # argmin over K per batch.
    y_T = y0 - (T * eta) * grad
    energies = np.sum(y_T * y_T, axis=1).reshape(B, K)
    candidates = y_T.reshape(B, K, D)
    return candidates[np.arange(B), energies.argmin(axis=1)]


def _ir_text(B: int, K: int, D: int, T: int) -> str:
    return (
        "graph_ir {\n"
        f"  %R     = tessera.placeholder shape=({B * K}, 8) "
        f"dtype=f32 kind=multivector algebra=Cl(3,0)\n"
        f"  %V     = tessera.placeholder shape=({B * K}, 8) "
        f"dtype=f32 kind=multivector algebra=Cl(3,0)\n"
        f"  %grad  = tessera.placeholder shape=({B * K}, {D}) dtype=f32\n"
        "  %rot    = ga.rotor_sandwich(%R, %V)        // fused MSL\n"
        f"  %y0     = ga.take_first(%rot, n={D})\n"
        f"  %winner = ebm.ebt_tiny(%y0, %grad, eta=0.05, T={T}, "
        f"B={B}, K={K}, D={D})                       // fused MSL\n"
        "  return %winner\n"
        "}\n"
    )


def run(
    *, B: int = 4, K: int = 16, D: int = 8, T: int = 4, seed: int = 0,
    eta: float = 0.05, native_required: bool = False,
) -> CompileReport:
    R, V, grad, B, K, D, T = _make_inputs(B=B, K=K, D=D, T=T, seed=seed)
    is_darwin = sys.platform == "darwin"
    host_fail = classify_host(is_darwin=is_darwin, runtime_available=True)
    if native_required and host_fail is not None:
        raise TesseraNativeRequiredError(
            host_fail, target="apple_gpu", op_name=PROGRAM_ID,
        )
    target = "apple_gpu" if is_darwin else "cpu"
    target_decision = {
        target: (
            "fused chain: clifford_rotor_sandwich + ebm_ebt_tiny "
            "(both shipped MSL kernels); takes first D rotor "
            "components as EBT candidates"
            if is_darwin else "non-Darwin host; numpy reference"
        )
    }
    fallback_reason: FallbackReason | None = host_fail

    a = ga.Cl(3, 0)
    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        # 1) Rotate every (rotor, candidate) pair.  Each call dispatches
        # through the bridge → fused MSL on Darwin.
        rotated = np.zeros_like(V)
        for i in range(V.shape[0]):
            out = ga.rotor_sandwich(
                ga.Multivector(R[i], a), ga.Multivector(V[i], a),
            )
            rotated[i] = np.asarray(out._coefficients)
        # 2) Take first D components as the EBT state.
        y0 = rotated[:, :D].copy()
        # 3) Fused refinement + self_verify via ebt_tiny.
        winner = ebm.ebt_tiny(y0, grad, eta=eta, T=T, B=B, K=K, D=D)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    ref = _numpy_reference(R, V, grad, B, K, D, T, eta)
    max_abs_err = float(np.abs(winner - ref).max())

    return CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_MIXED,  # GA + tensor in one program
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(B, K, D, T))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        correctness={"max_abs_err": max_abs_err, "tolerance": 5e-5},
    )


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
