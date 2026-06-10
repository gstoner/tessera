"""Numerical-proof discipline gate for Apple GPU backend-manifest rows.

Audit 2026-06-10 follow-on. The conformance matrix already enforces "no
complete cell without a verified execute_compare_fixture" for its 7 curated
ops (test_conformance_complete_cells_proven.py). This gate extends the same
discipline to the *broader* Apple GPU manifest: an `apple_gpu` row that claims
a real kernel (`status in {fused, hardware_verified}`) must either declare an
`execute_compare_fixture` (numerical proof on disk) OR appear on an explicit,
reasoned allowlist of ops that genuinely have no dedicated GPU execute-compare
yet. This freezes the gap — a newly-added fused Apple op without proof fails
here, forcing either a fixture or a conscious allowlist entry.

`hardware_verified` is separately required (at BackendKernelEntry construction)
to carry a fixture; this gate also locks that invariant at the suite level.
"""

import pytest

from tessera.compiler import backend_manifest as bm

# Apple GPU `fused` ops that genuinely have NO dedicated GPU execute-compare
# test yet. Each must stay justified; shrink this list by writing a real
# execute-compare test and wiring its fixture, never by relaxing the gate.
_FUSED_WITHOUT_NUMERICAL_FIXTURE = {
    # No dedicated apple_gpu execute-compare: only referenced in an envelope
    # membership list in test_apple_gpu_ebm_lane.py, not numerically asserted.
    "ebm_self_verify",
    # runtime_symbol is None (no standalone GPU kernel — composed path); no
    # dedicated GPU numerical compare. The sharding-mock test is not an
    # execute-compare.
    "ebm_langevin_step",
    # Conformance op; the apple_gpu path executes but has no numerical
    # execute-compare fixture yet (tracked in MASTER_AUDIT as a software gap).
    "kv_cache_read",
}


def _apple_promotable_rows():
    out = []
    for op, entries in bm.all_manifests().items():
        for e in entries:
            if getattr(e, "target", "") == "apple_gpu" and e.status in (
                "fused", "hardware_verified"
            ):
                out.append((op, e))
    return out


def test_apple_gpu_fused_rows_have_proof_or_are_allowlisted():
    """Every Apple GPU fused/hardware_verified row has a numerical fixture, or
    is on the explicit no-execute-compare allowlist."""
    missing = sorted(
        op for op, e in _apple_promotable_rows()
        if not e.execute_compare_fixture
        and op not in _FUSED_WITHOUT_NUMERICAL_FIXTURE
    )
    assert not missing, (
        "Apple GPU rows claim a real kernel (fused/hardware_verified) but have "
        f"no execute_compare_fixture and are not allowlisted: {missing}. Wire a "
        "verified fixture into backend_manifest._NUMERICAL_FIXTURES, or add an "
        "explicit reason to _FUSED_WITHOUT_NUMERICAL_FIXTURE in this test.")


def test_allowlist_has_no_stale_entries():
    """An allowlisted op that has since gained a fixture must be removed from
    the allowlist (keeps the list honest and shrinking)."""
    have_fixture = {
        op for op, e in _apple_promotable_rows() if e.execute_compare_fixture
    }
    stale = sorted(_FUSED_WITHOUT_NUMERICAL_FIXTURE & have_fixture)
    assert not stale, (
        f"these ops now have a fixture — drop them from the allowlist: {stale}")


def test_hardware_verified_rows_carry_a_fixture():
    """Lock the construction invariant at the suite level: the strongest
    Apple GPU claim tier always has numerical proof on disk."""
    bad = sorted(
        op for op, e in _apple_promotable_rows()
        if e.status == "hardware_verified" and not e.execute_compare_fixture
    )
    assert not bad, f"hardware_verified rows missing a fixture: {bad}"
