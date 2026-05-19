"""Tessera canonical-program drivers — M1 / M1.5 deliverable.

A canonical program is one tiny end-to-end demo per ambition surface
(tensor / GA / EBM / KV-cache / mixed) that emits a
:class:`CompileReport` instead of producing a benchmark row in
isolation.  The intent is a single inspectable format that proves
which layers of the compiler stack are real.

See :doc:`docs/audit/compiler_improvement_milestone_plan_2026_05_18`
§M1 + §M1.5 for the full sequencing.

This module is the **registry** for the 6-program suite.  M1 ships
the first two programs (`rotor_sandwich_norm` and
`matmul_softmax_matmul`); the remaining four are listed here with
explicit ``status="planned"`` so the suite cannot silently look
complete.
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
            "O = softmax(A @ B) @ C — intended Apple GPU 3-op fused symbol "
            "(matmul→softmax→matmul); driver currently runs the numpy "
            "reference on every host and reports REFERENCE_FORCED so the "
            "CompileReport stays honest about what executed."
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
