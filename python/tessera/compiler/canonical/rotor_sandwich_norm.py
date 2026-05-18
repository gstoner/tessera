"""Canonical program: ``rotor_sandwich(R, V) → norm(.)``.

The GA vertical-slice demo, now emitted as a :class:`CompileReport`
rather than just a benchmark row.  Lane: ``@clifford_jit`` AST → IR
on Darwin; numpy reference on every other host.  Either way the
report carries:

  - the lowered IR digest (Clifford IR text → 16-char sha256);
  - the target decision and fallback reason;
  - the bridge proof routes (Apple GPU only);
  - the correctness envelope vs the numpy reference;
  - a deterministic ``report_hash`` excluding timing fields.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera import ga
from tessera.compiler import jit_bridge as bridge
from tessera.compiler.clifford_jit import clifford_jit
from tessera.compiler.compile_report import (
    CompileReport,
    finalize_compile_report,
    FRONTEND_CLIFFORD_JIT,
    VALUE_KIND_MULTIVECTOR,
    hash_ir_text,
)
from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
)


PROGRAM_ID = "rotor_sandwich_norm"


def _make_inputs(seed: int = 0):
    """Build a small, fully-deterministic ``(rotor, points)`` pair."""
    a = ga.Cl(3, 0)
    rng = np.random.RandomState(seed)
    R = (rng.randn(8, 8).astype(np.float32) * 0.3)
    V = rng.randn(8, 8).astype(np.float32)
    return a, R, V


def _numpy_reference(a, R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Per-batch ``||R V R†||`` via the public Python API.  This is
    the correctness baseline both for the Apple GPU path (where it
    becomes the comparator) and for non-Darwin hosts (where it IS
    the executed path)."""
    return np.array([
        float(np.asarray(ga.norm(ga.rotor_sandwich(
            ga.Multivector(R[i], a), ga.Multivector(V[i], a)
        ))))
        for i in range(R.shape[0])
    ])


def run(*, native_required: bool = False) -> CompileReport:
    """Build the canonical program, execute it, return a
    :class:`CompileReport`.

    Parameters
    ----------
    native_required
        **M3:** when ``True``, raises
        :class:`fallback.TesseraNativeRequiredError` instead of
        falling back to numpy on a non-Darwin host.  The default
        (``False``) preserves M1 behavior: report carries
        ``fallback_reason=NON_DARWIN_HOST`` and runs numpy.
    """
    a, R, V = _make_inputs(seed=0)

    @clifford_jit(target="apple_gpu", native_required=native_required)
    def point_cloud_rotor_invariant(rotor, points):
        rotated = ga.rotor_sandwich(rotor, points)
        return ga.norm(rotated)

    ir = point_cloud_rotor_invariant.artifact.ir
    ir_text = ir.text() if ir is not None else ""
    ir_hashes = {"graph_ir": hash_ir_text(ir_text)} if ir_text else {}

    is_darwin = sys.platform == "darwin"
    host_fail = classify_host(is_darwin=is_darwin, runtime_available=True)
    if native_required and host_fail is not None:
        # Don't run anything — let the caller see the diagnostic.
        raise TesseraNativeRequiredError(
            host_fail, target="apple_gpu", op_name=PROGRAM_ID,
        )
    target = "apple_gpu" if is_darwin else "cpu"
    target_decision = {
        target: (
            "apple_gpu fused MSL via @clifford_jit"
            if is_darwin else "non-Darwin host; numpy reference"
        )
    }
    fallback_reason: FallbackReason | None = host_fail

    rotor = ga.Multivector(R, a)
    points = ga.Multivector(V, a)

    # M2 step 4 (2026-05-18): nest a fresh capture scope while
    # invoking the @clifford_jit callable so its auto-emit lands
    # in a discarded sub-sink, not in the parent compile_session.
    # The driver itself is the canonical emit point for this
    # program; we don't want an inner double-emit.
    from tessera.compiler.compile_report import capture_compile_reports as _cap
    t0 = time.perf_counter_ns()
    if is_darwin:
        with _cap():
            out = point_cloud_rotor_invariant(rotor, points)
        out_arr = np.asarray([float(np.asarray(out[i])) for i in range(R.shape[0])])
        # @clifford_jit's __call__ swallows the bridge trace into
        # last_routes(); read it back from there rather than the
        # bridge's now-empty thread-local trace.
        routes = point_cloud_rotor_invariant.last_routes()
    else:
        # Direct numpy reference on non-Darwin — no bridge routes.
        out_arr = _numpy_reference(a, R, V)
        routes = ()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    ref = _numpy_reference(a, R, V)
    max_abs_err = float(np.abs(out_arr - ref).max())

    return finalize_compile_report(CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_CLIFFORD_JIT,
        value_kind=VALUE_KIND_MULTIVECTOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes=ir_hashes,
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        correctness={"max_abs_err": max_abs_err, "tolerance": 5e-5},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
