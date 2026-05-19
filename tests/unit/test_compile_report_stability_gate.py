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
    # Reference-forced even on Darwin — drivers that build a
    # CompileReport but execute pure numpy (no Metal dispatch).
    # 2026-05-18 P1 fix: matmul_softmax_matmul and
    # kv_cache_append_prune_read are added here because the driver
    # bodies don't actually dispatch any fused kernel today; the
    # honest report is REFERENCE_FORCED until they do.
    reference_forced_on_darwin = {
        "conv2d_norm_activation",
        "matmul_softmax_matmul",
        "kv_cache_append_prune_read",
    }
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


def _native_proof_strength(report) -> dict[str, bool]:
    """Decompose a CompileReport's native-proof signals.  Used by
    the no-silent-native gate; **deliberately excludes
    ``ir_hashes``** because every shipped canonical synthesizes a
    Graph IR digest regardless of whether anything was dispatched.

    Valid proofs:

      * ``proof_routes`` non-empty — the bridge actually fired.
      * ``plan_hash`` populated — a compiled artifact exists
        (e.g., ``@clifford_jit.artifact.plan_hash``).
      * A literal ``tessera_*`` C ABI symbol in
        ``target_decision`` values — names what the kernel will be.
    """
    decision_text = " ".join(report.target_decision.values())
    return {
        "has_routes": len(report.proof_routes) > 0,
        "has_plan_hash": bool(report.plan_hash),
        "has_symbol": "tessera_" in decision_text,
    }


@pytest.mark.parametrize("program", _SHIPPED, ids=lambda p: p.program_id)
def test_shipped_canonical_no_silent_native(program) -> None:
    """**M5 no-silent-native rule** applied at the CompileReport
    layer (P1 reviewer fix + reviewer follow-up, 2026-05-18).

    If a report claims native execution — ``target != "cpu"`` and
    ``fallback_reason is None`` — it MUST carry at least one of:

      * a non-empty ``proof_routes`` tuple (bridge dispatch trace),
      * a populated ``plan_hash`` (compiled artifact attached),
      * a literal ``tessera_*`` C ABI symbol mentioned in
        ``target_decision`` (the manifest fast path the report
        intends to dispatch — coupled with the test that the
        symbol resolves through ``jit_bridge``).

    **Important: a Graph-IR hash alone does NOT count.**  Every
    canonical synthesizes one whether or not anything dispatched,
    so leaning on ``ir_hashes`` would silently legitimize a pure
    numpy driver that claims native success.
    """
    report = program.run()
    if report.target == "cpu":
        return  # cpu reports are honest by construction
    if report.fallback_reason is not None:
        return  # explicit fallback acknowledged
    proof = _native_proof_strength(report)
    assert any(proof.values()), (
        f"{program.program_id} claims native execution "
        f"(target={report.target!r}, fallback_reason=None) but carries "
        f"NO valid proof: {proof}.  Either set "
        f"FallbackReason.REFERENCE_FORCED honestly, wire bridge "
        f"routes via dispatch_via_manifest, attach a plan_hash from "
        f"the compiled artifact, or name the tessera_* symbol in "
        f"target_decision."
    )


def test_no_silent_native_rejects_a_synthetic_ir_only_report() -> None:
    """Regression test for the reviewer follow-up: a hypothetical
    driver that ONLY populates ``ir_hashes`` (no routes, no
    plan_hash, no symbol in target_decision) must be rejected
    by the gate.  Catches a future regression that re-introduces
    the ``has_ir`` branch."""
    from tessera.compiler.compile_report import CompileReport
    bad = CompileReport(
        program_id="synthetic_bad",
        source="t",
        frontend="tessera.jit",
        value_kind="tensor",
        target="apple_gpu",     # claims native
        fallback_reason=None,    # claims success
        ir_hashes={"graph_ir": "abc123"},  # ONLY this — no routes, etc.
        target_decision={"apple_gpu": "would-be fused kernel"},
    )
    proof = _native_proof_strength(bad)
    assert not any(proof.values()), (
        f"the gate accepted ir-only proof: {proof}.  The reviewer "
        f"fix explicitly forbids ir_hashes alone as proof."
    )


def test_no_silent_native_accepts_plan_hash_only_report() -> None:
    """Conversely, a driver with a real ``plan_hash`` (e.g., from
    a @clifford_jit compiled artifact) is valid proof even if no
    routes were captured."""
    from tessera.compiler.compile_report import CompileReport
    good = CompileReport(
        program_id="synthetic_good",
        source="t",
        frontend="tessera.jit",
        value_kind="tensor",
        target="apple_gpu",
        fallback_reason=None,
        plan_hash="real_plan_hash_from_a_compiled_artifact",
        target_decision={"apple_gpu": "compiled artifact bound"},
    )
    proof = _native_proof_strength(good)
    assert proof["has_plan_hash"] and any(proof.values())


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


def test_stability_gate_enumerates_every_shipped_canonical() -> None:
    """The gate must cover every shipped program — parametrize
    silently dropping rows would be invisible.

    P2 reviewer fix (2026-05-18): the suite is now 6/6 shipped,
    so the gate asserts the full set rather than a lower bound.
    """
    from tessera.compiler import canonical
    shipped_ids = {p.program_id for p in _SHIPPED}
    registry_ids = {p.program_id for p in canonical.shipped_programs()}
    assert shipped_ids == registry_ids, (
        f"stability gate sees {sorted(shipped_ids)} but the registry "
        f"reports {sorted(registry_ids)} shipped — they must match"
    )
    # And the suite must be at its current 6/6 count.  Bump the
    # number when a new canonical lands; never lower it.
    assert len(_SHIPPED) == 6, (
        f"expected 6 shipped canonicals; got {len(_SHIPPED)}"
    )
