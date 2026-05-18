"""Canonical-program **stability gate** — the last unchecked M6
Step 3 prerequisite (per the 2026-05-18 post-reassessment plan).

For every shipped canonical program, two back-to-back runs must:

  1. Produce identical ``CompileReport.report_hash()`` values
     (timing-independent identity).
  2. Carry the correct, deterministic ``fallback_reason`` (``None``
     when the program ran natively; a typed :class:`FallbackReason`
     when it didn't).
  3. Surface a consistent ``frontend`` / ``value_kind`` / ``target``
     triple — these are part of M1's normative envelope and must
     not drift between runs.

Until this gate is green, **M6 Step 3 (energy-gradient generation
+ fused energy kernels) does not land** — Step 3 introduces new
proof surfaces and must ride on top of stable report identity, not
parallel to it.
"""

from __future__ import annotations

import sys

import pytest

from tessera.compiler import canonical
from tessera.compiler.compile_report import CompileReport
from tessera.compiler.fallback import FallbackReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_fallback_for_program(program_id: str) -> FallbackReason | None:
    """The canonical fallback_reason this program should emit on the
    current host.

    Programs whose entire dispatch path is fused on Apple GPU emit
    ``None`` on Darwin; programs that intentionally route through
    numpy reference (e.g., ``conv2d_norm_activation`` until conv2d
    gets a fused MSL kernel) emit ``REFERENCE_FORCED`` on Darwin and
    ``NON_DARWIN_HOST`` elsewhere.
    """
    is_darwin = sys.platform == "darwin"
    # Reference-forced even on Darwin: conv2d still has no fused
    # MSL kernel, so the canonical correctly reports REFERENCE_FORCED.
    reference_forced_on_darwin = {"conv2d_norm_activation"}
    if program_id in reference_forced_on_darwin:
        return (
            FallbackReason.REFERENCE_FORCED if is_darwin
            else FallbackReason.NON_DARWIN_HOST
        )
    # Native-on-Darwin programs.
    if is_darwin:
        return None
    return FallbackReason.NON_DARWIN_HOST


def _expected_frontend(program_id: str) -> str:
    return {
        "rotor_sandwich_norm": "clifford_jit",
    }.get(program_id, "tessera.jit")


def _expected_value_kind(program_id: str) -> str:
    return {
        "rotor_sandwich_norm": "multivector",
        # GA + tensor in one program → mixed per Decision #15a.
        "rotor_sandwich_ebt_tiny": "mixed",
    }.get(program_id, "tensor")


# ---------------------------------------------------------------------------
# Parametrize over every shipped canonical program
# ---------------------------------------------------------------------------

_SHIPPED = canonical.shipped_programs()


@pytest.mark.parametrize("program", _SHIPPED, ids=lambda p: p.program_id)
def test_shipped_canonical_has_stable_report_hash(program) -> None:
    """Two back-to-back runs must produce the same report hash.
    The hash explicitly excludes timing / version fields so the only
    drift sources are IR identity, target decision, proof routes
    (op names + symbols + context), correctness, and fallback reason.
    """
    a = program.run()
    b = program.run()
    assert isinstance(a, CompileReport) and isinstance(b, CompileReport)
    assert a.report_hash() == b.report_hash(), (
        f"{program.program_id} report_hash drifted between runs:\n"
        f"  run 1: {a.report_hash()}  fallback={a.fallback_reason}\n"
        f"  run 2: {b.report_hash()}  fallback={b.fallback_reason}"
    )


@pytest.mark.parametrize("program", _SHIPPED, ids=lambda p: p.program_id)
def test_shipped_canonical_fallback_reason_matches_host(program) -> None:
    """Each canonical's ``fallback_reason`` must be the value the
    M5 no-silent-native rule expects for this host."""
    expected = _expected_fallback_for_program(program.program_id)
    report = program.run()
    assert report.fallback_reason == expected, (
        f"{program.program_id}: fallback_reason={report.fallback_reason!r} "
        f"but host expected {expected!r}"
    )


@pytest.mark.parametrize("program", _SHIPPED, ids=lambda p: p.program_id)
def test_shipped_canonical_envelope_fields_are_stable(program) -> None:
    """Frontend / value_kind / target are part of M1's normative
    envelope.  They must not change between runs of the same
    program on the same host."""
    a = program.run()
    b = program.run()
    assert a.frontend == b.frontend == _expected_frontend(program.program_id)
    assert a.value_kind == b.value_kind == _expected_value_kind(program.program_id)
    assert a.target == b.target


@pytest.mark.parametrize("program", _SHIPPED, ids=lambda p: p.program_id)
def test_shipped_canonical_correctness_clears_tolerance(program) -> None:
    """Every shipped canonical must clear its own correctness
    tolerance.  A failure here means the reference baseline drifted
    or the native dispatch is wrong — both block M6 Step 3."""
    report = program.run()
    assert report.correctness is not None, program.program_id
    err = float(report.correctness["max_abs_err"])
    tol = float(report.correctness["tolerance"])
    assert err <= tol, (
        f"{program.program_id}: max_abs_err={err} > tolerance={tol}"
    )


# ---------------------------------------------------------------------------
# Aggregate gate — all shipped canonicals together
# ---------------------------------------------------------------------------

def test_all_shipped_canonicals_have_unique_report_hashes() -> None:
    """Two different shipped canonicals must NOT collide in their
    report hashes.  A collision would mean the audit machinery can't
    distinguish them — which would silently break the M5 schema's
    `report_hash` cross-reference."""
    hashes = {p.program_id: p.run().report_hash() for p in _SHIPPED}
    # Invert: hash -> list of program ids
    by_hash: dict[str, list[str]] = {}
    for pid, h in hashes.items():
        by_hash.setdefault(h, []).append(pid)
    collisions = {h: pids for h, pids in by_hash.items() if len(pids) > 1}
    assert not collisions, f"report hash collisions: {collisions}"


def test_stability_gate_enumerates_at_least_four_canonicals() -> None:
    """The gate must cover every shipped program — parametrize
    silently dropping rows would be invisible.  Cross-check the
    count so the gate stays meaningful."""
    assert len(_SHIPPED) >= 4, (
        f"only {len(_SHIPPED)} shipped programs — the stability "
        "gate covers each, but the M6 Step 3 prerequisite expects "
        "at least the four canonicals from the post-reassessment plan"
    )
