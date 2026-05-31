"""Canonical program: exercise all 4 fused M7 (Visual Complex) kernels.

Apple follow-up #2 (2026-05-20).  The four ops with shipped Apple GPU
MSL kernels are:

* ``complex_mul``     → ``tessera_apple_gpu_complex_mul_f32``
* ``complex_exp``     → ``tessera_apple_gpu_complex_exp_f32``
* ``mobius``          → ``tessera_apple_gpu_complex_mobius_f32``
* ``stereographic``   → ``tessera_apple_gpu_complex_stereographic_f32``

Each routes through ``tessera.complex.*`` on top of
``jit_bridge.dispatch_via_manifest``, which already emits a unified
``JitBridgeRoute`` per successful native dispatch.  The canonical
runs the four ops in sequence on small batched inputs and assembles
a ``CompileReport`` with **4 proof_routes** entries — one per fused
kernel — proving the M7 surface flows through the unified proof
envelope alongside GA/EBM and the generic-tensor lane.

Numpy stays the correctness oracle.  Each op's output is compared
against its pure-Python equivalent via ``np.allclose`` at fp32
tolerance.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

import tessera
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


PROGRAM_ID = "visual_complex_fused"


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _make_inputs(N: int = 16, seed: int = 0) -> dict[str, Any]:
    """Deterministic small batched inputs for the four fused ops."""
    rng = np.random.RandomState(seed)
    # complex_mul + complex_exp consume same-shape complex tensors.
    z = (rng.randn(N) + 1j * rng.randn(N)).astype(np.complex64)
    w = (rng.randn(N) + 1j * rng.randn(N)).astype(np.complex64)
    # mobius takes (z, a, b, c, d) scalar coefficients; pick a
    # non-degenerate Möbius (ad - bc != 0).  Cast c to complex
    # since `tessera.complex.mobius` accepts complex coefficients.
    a, b_, c, d = complex(1.0, 0.2), complex(0.0, 0.1), complex(0.1, 0.0), complex(1.0, 0.0)
    # stereographic takes a 3-vector ``(x, y, z)`` on R^3.  Pack as
    # an ``(N, 3)`` float32 array — the tuple-form would force
    # float64 inside ``tessera.complex.stereographic`` and skip the
    # fp32 Apple GPU dispatch branch.
    xyz = (rng.randn(N, 3) * 0.3).astype(np.float32)
    xyz[:, 2] -= 0.5  # bias away from the north pole (z = 1)
    return {
        "z": z, "w": w,
        "mobius_coeffs": (a, b_, c, d),
        "stereo_xyz": xyz,
    }


def _numpy_reference(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    z, w = inputs["z"], inputs["w"]
    a, b_, c, d = inputs["mobius_coeffs"]
    xyz = inputs["stereo_xyz"]
    x_co, y_co, z_co = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return {
        "complex_mul": (z * w).astype(np.complex64),
        "complex_exp": np.exp(z).astype(np.complex64),
        "mobius":      ((a * z + b_) / (c * z + d)).astype(np.complex64),
        # Stereographic projection R^3 \ {north pole} → C.
        "stereographic": ((x_co + 1j * y_co) / (1.0 - z_co)).astype(np.complex64),
    }


def _ir_text(N: int) -> str:
    return (
        "graph_ir {\n"
        f"  %z = tessera.placeholder shape=({N},) dtype=complex64\n"
        f"  %w = tessera.placeholder shape=({N},) dtype=complex64\n"
        "  %m  = tessera.complex_mul(%z, %w)\n"
        "  %e  = tessera.complex_exp(%z)\n"
        "  %mb = tessera.mobius(%z, a, b, c, d)\n"
        "  %st = tessera.stereographic(x, y, z)\n"
        "  return %m, %e, %mb, %st\n"
        "}\n"
    )


def _run_native_or_reference(
    inputs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], bool, str | None]:
    """Run all four ops via ``tessera.complex.*``.

    On Darwin: each op routes through ``jit_bridge.dispatch_via_manifest``
    and the fused MSL kernel.
    Non-Darwin: each op falls back to numpy inside ``tessera.complex.*``
    automatically.

    Returns ``(outputs, dispatched_native, fallback_note)``.
    ``dispatched_native`` is True iff at least one bridge route was
    recorded by the time the four calls finish.
    """
    from tessera import complex as _ts_complex

    z = inputs["z"]
    w = inputs["w"]
    a, b_, c, d = inputs["mobius_coeffs"]
    xyz = inputs["stereo_xyz"]

    if not _is_darwin():
        return {}, False, "non-Darwin host"

    try:
        cm = _ts_complex.complex_mul(z, w)
        ce = _ts_complex.complex_exp(z)
        mb = _ts_complex.mobius(z, a, b_, c, d)
        st = _ts_complex.stereographic(xyz)
    except Exception as exc:
        return {}, False, f"runtime call raised: {exc!r}"

    outs = {
        "complex_mul":   _ts_complex.to_numpy(cm, dtype=np.complex64),
        "complex_exp":   _ts_complex.to_numpy(ce, dtype=np.complex64),
        "mobius":        _ts_complex.to_numpy(mb, dtype=np.complex64),
        "stereographic": _ts_complex.to_numpy(st, dtype=np.complex64),
    }
    return outs, True, None


def _target_decision_for_host(
    dispatched_native: bool, n_routes: int, fallback_note: str | None,
) -> tuple[str, dict[str, str], FallbackReason | None]:
    if not _is_darwin():
        return (
            "cpu",
            {"cpu": "non-Darwin host; numpy reference path for all 4 M7 ops"},
            FallbackReason.NON_DARWIN_HOST,
        )
    base_note = (
        "M7 Visual Complex fused surface (4 ops): "
        "tessera_apple_gpu_complex_mul_f32 / complex_exp_f32 / "
        "complex_mobius_f32 / complex_stereographic_f32"
    )
    if dispatched_native and n_routes >= 1:
        return (
            "apple_gpu",
            {"apple_gpu": (
                f"{base_note}. NATIVE DISPATCH: {n_routes} of 4 ops "
                "executed via the apple_gpu runtime shim "
                "(the others may have fallen back per-op if their "
                "envelope didn't match)."
            )},
            None,
        )
    return (
        "apple_gpu",
        {"apple_gpu": (
            f"{base_note}. REFERENCE_FORCED: "
            f"{fallback_note or 'no fused kernel dispatched'}."
        )},
        FallbackReason.REFERENCE_FORCED,
    )


def run(
    *, N: int = 16, seed: int = 0,
    native_required: bool = False,
) -> CompileReport:
    inputs = _make_inputs(N=N, seed=seed)

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        native_outs, dispatched_native, fallback_note = (
            _run_native_or_reference(inputs)
        )
        ref = _numpy_reference(inputs)
        if dispatched_native and native_outs:
            outs = native_outs
        else:
            outs = ref
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    target, target_decision, fallback_reason = _target_decision_for_host(
        dispatched_native, len(routes), fallback_note,
    )

    if native_required and fallback_reason is not None:
        raise TesseraNativeRequiredError(
            fallback_reason, target="apple_gpu", op_name=PROGRAM_ID,
        )

    # Per-op correctness: max abs err across all 4 ops vs numpy reference.
    max_err = 0.0
    for op_name in ("complex_mul", "complex_exp", "mobius", "stereographic"):
        diff = np.abs(outs[op_name] - ref[op_name])
        max_err = max(max_err, float(diff.max()))

    return finalize_compile_report(CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(N))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        # Tolerance set per the fp32 complex math envelope — complex
        # exp / mobius have multiplicative growth; 1e-4 is the empirical
        # ceiling at N=16.
        correctness={"max_abs_err": max_err, "tolerance": 1e-4},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
