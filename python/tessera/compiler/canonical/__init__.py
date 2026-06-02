"""Tessera canonical-program drivers — M1 / M1.5 deliverable.

A canonical program is one tiny end-to-end demo per ambition surface
(tensor / GA / EBM / KV-cache / mixed) that emits a
:class:`CompileReport` instead of producing a benchmark row in
isolation.  The intent is a single inspectable format that proves
which layers of the compiler stack are real.

See :doc:`docs/audit/compiler_improvement_milestone_plan_2026_05_18`
§M1 + §M1.5 for the full sequencing.

This module is the **registry** for the 6-program suite.  All six
programs are shipped as of M1.5 (2026-05-18) — see ``CANONICAL_PROGRAMS``
below for the per-program description, and the milestone plan
``docs/audit/compiler/COMPILER_AUDIT.md`` §M1.5
for the honest-reporting notes (programs that don't yet have a fused
backend symbol emit ``fallback_reason=REFERENCE_FORCED`` so the
CompileReport stays accurate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..compile_report import CompileReport


@dataclass(frozen=True)
class CanonicalProgram:
    """One row of the canonical-program registry.

    ``program_id``    — stable identifier; embedded in the emitted
                         :class:`CompileReport`.
    ``family``        — bucket for the support-table view.
    ``owner_file``    — Python module that builds the report.
    ``run``           — callable ``() -> CompileReport``.  ``None``
                         when ``status="planned"``.
    ``status``        — one of {``"shipped"``, ``"planned"``}.
    ``description``   — one-line summary for the audit table.
    """
    program_id: str
    family: str
    owner_file: str
    run: Optional[Callable[[], CompileReport]]
    status: str
    description: str


# Lazy imports — keep this package light when only the registry is
# consulted (e.g., by the support-table audit).
def _run_rotor_sandwich_norm() -> CompileReport:
    from .rotor_sandwich_norm import run
    return run()


def _run_matmul_softmax_matmul() -> CompileReport:
    from .matmul_softmax_matmul import run
    return run()


def _run_conv2d_norm_activation() -> CompileReport:
    from .conv2d_norm_activation import run
    return run()


def _run_decode_init_inner_loop_self_verify() -> CompileReport:
    from .decode_init_inner_loop_self_verify import run
    return run()


def _run_kv_cache_append_prune_read() -> CompileReport:
    from .kv_cache_append_prune_read import run
    return run()


def _run_rotor_sandwich_ebt_tiny() -> CompileReport:
    from .rotor_sandwich_ebt_tiny import run
    return run()


def _run_matmul_gelu() -> CompileReport:
    from .matmul_gelu import run
    return run()


def _run_visual_complex_fused() -> CompileReport:
    from .visual_complex_fused import run
    return run()


CANONICAL_PROGRAMS: tuple[CanonicalProgram, ...] = (
    CanonicalProgram(
        program_id="rotor_sandwich_norm",
        family="geometric_algebra",
        owner_file="python/tessera/compiler/canonical/rotor_sandwich_norm.py",
        run=_run_rotor_sandwich_norm,
        status="shipped",
        description="rotor_sandwich(R, V) followed by norm(.); GA vertical slice via @clifford_jit.",
    ),
    CanonicalProgram(
        program_id="matmul_softmax_matmul",
        family="attention",
        owner_file="python/tessera/compiler/canonical/matmul_softmax_matmul.py",
        run=_run_matmul_softmax_matmul,
        status="shipped",
        description=(
            "O = softmax(A @ B) @ C — dispatches the fused 3-op MSL "
            "kernel ``tessera_apple_gpu_matmul_softmax_matmul_f32`` on "
            "Darwin within envelope (N + P ≤ 256).  Outside the envelope "
            "and on non-Darwin hosts, falls back to numpy with a "
            "precise REFERENCE_FORCED note."
        ),
    ),
    CanonicalProgram(
        program_id="conv2d_norm_activation",
        family="cnn",
        owner_file="python/tessera/compiler/canonical/conv2d_norm_activation.py",
        run=_run_conv2d_norm_activation,
        status="shipped",
        description="conv2d_nhwc → layer_norm → gelu; numpy reference; honest fallback_reason since conv2d has no fused MSL kernel yet.",
    ),
    CanonicalProgram(
        program_id="kv_cache_append_prune_read",
        family="kv_cache",
        owner_file="python/tessera/compiler/canonical/kv_cache_append_prune_read.py",
        run=_run_kv_cache_append_prune_read,
        status="shipped",
        description="KVCacheHandle append → prune → read; paged numpy storage today, FA-4 consumes this state.",
    ),
    CanonicalProgram(
        program_id="decode_init_inner_loop_self_verify",
        family="energy_based_models",
        owner_file="python/tessera/compiler/canonical/decode_init_inner_loop_self_verify.py",
        run=_run_decode_init_inner_loop_self_verify,
        status="shipped",
        description="EBM decode_init → T inner-loop steps → self_verify; argmin over K; native MSL on Apple GPU.",
    ),
    CanonicalProgram(
        program_id="rotor_sandwich_ebt_tiny",
        family="ga_ebm_composite",
        owner_file="python/tessera/compiler/canonical/rotor_sandwich_ebt_tiny.py",
        run=_run_rotor_sandwich_ebt_tiny,
        status="shipped",
        description="rotor_sandwich → ebt_tiny composite; both ops fused on Apple GPU; value_kind=mixed (GA + tensor).",
    ),
    # Apple follow-up #1 (2026-05-20) — second generic-tensor canonical
    # to dispatch the real fused MSL kernel on Darwin.
    CanonicalProgram(
        program_id="matmul_gelu",
        family="mlp",
        owner_file="python/tessera/compiler/canonical/matmul_gelu.py",
        run=_run_matmul_gelu,
        status="shipped",
        description=(
            "O = gelu(A @ B) — dispatches the fused 2-op MSL kernel "
            "``tessera_apple_gpu_matmul_gelu_f32`` on Darwin within "
            "envelope (N ≤ 256).  Emits a unified JitBridgeRoute via "
            "the same proof-route plumbing as matmul_softmax_matmul."
        ),
    ),
    # Apple follow-up #2 (2026-05-20) — Visual Complex canonical
    # exercising all 4 fused M7 kernels in one report.
    CanonicalProgram(
        program_id="visual_complex_fused",
        family="visual_complex",
        owner_file="python/tessera/compiler/canonical/visual_complex_fused.py",
        run=_run_visual_complex_fused,
        status="shipped",
        description=(
            "Sequential exercise of complex_mul / complex_exp / mobius / "
            "stereographic — the 4 fused M7 Apple GPU kernels.  Each "
            "step routes through ``tessera.complex.*`` on top of the "
            "shipped MSL symbols; the report carries one proof_routes "
            "row per op."
        ),
    ),
)


def program_for(program_id: str) -> CanonicalProgram:
    """Lookup helper — raises KeyError on unknown id."""
    for p in CANONICAL_PROGRAMS:
        if p.program_id == program_id:
            return p
    raise KeyError(f"unknown canonical program: {program_id!r}")


def shipped_programs() -> tuple[CanonicalProgram, ...]:
    return tuple(p for p in CANONICAL_PROGRAMS if p.status == "shipped")


def planned_programs() -> tuple[CanonicalProgram, ...]:
    return tuple(p for p in CANONICAL_PROGRAMS if p.status == "planned")


__all__ = [
    "CanonicalProgram",
    "CANONICAL_PROGRAMS",
    "program_for",
    "shipped_programs",
    "planned_programs",
]
