"""M1.5 — canonical-program registry contract.

Locks the 6-program registry shape so the suite can't silently
look complete.  Per the plan: each program names an owner file
and has a status row; at least two programs run CPU-only in
default CI.
"""

from __future__ import annotations

import pytest

from tessera.compiler import canonical


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_has_six_programs() -> None:
    """The plan pins exactly six canonical programs.  Adding a
    seventh is a deliberate decision; this test catches drift."""
    assert len(canonical.CANONICAL_PROGRAMS) == 6


def test_program_ids_are_unique() -> None:
    ids = [p.program_id for p in canonical.CANONICAL_PROGRAMS]
    assert len(ids) == len(set(ids))


def test_every_program_names_an_owner_file() -> None:
    """The whole point of M1.5 — no vague wish-list entries."""
    for p in canonical.CANONICAL_PROGRAMS:
        assert p.owner_file, p.program_id


def test_every_program_has_a_valid_status() -> None:
    for p in canonical.CANONICAL_PROGRAMS:
        assert p.status in {"shipped", "planned"}, (p.program_id, p.status)


def test_shipped_programs_have_a_runner() -> None:
    """Shipped programs must be callable; planned ones must not."""
    for p in canonical.CANONICAL_PROGRAMS:
        if p.status == "shipped":
            assert p.run is not None, p.program_id
        else:
            assert p.run is None, p.program_id


def test_every_canonical_is_shipped() -> None:
    """M1.5 fully landed (P2 reviewer fix, 2026-05-18): all 6
    canonical programs are shipped.  The registry's
    ``shipped_programs()`` must equal the full ``CANONICAL_PROGRAMS``
    tuple — no remaining ``status="planned"`` entries.

    When a future canonical lands, both this test's count and
    ``len(CANONICAL_PROGRAMS)`` move together.  When one is
    intentionally promoted to ``planned`` again (e.g., gated on a
    runtime feature), update the test to use a lower bound."""
    shipped = canonical.shipped_programs()
    assert len(shipped) == len(canonical.CANONICAL_PROGRAMS), (
        f"{len(shipped)} of {len(canonical.CANONICAL_PROGRAMS)} canonical "
        "programs marked shipped — the suite should be at parity"
    )


def test_program_for_lookup_helper() -> None:
    p = canonical.program_for("rotor_sandwich_norm")
    assert p.status == "shipped"
    assert p.family == "geometric_algebra"
    with pytest.raises(KeyError):
        canonical.program_for("does_not_exist")


# ---------------------------------------------------------------------------
# Per-program runner sanity — every shipped program must produce a
# valid CompileReport without raising.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("program", canonical.shipped_programs(), ids=lambda p: p.program_id)
def test_shipped_program_runner_produces_compile_report(program) -> None:
    from tessera.compiler.compile_report import CompileReport
    report = program.run()
    assert isinstance(report, CompileReport)
    assert report.program_id == program.program_id


# ---------------------------------------------------------------------------
# Cross-check against the support-table audit row — the canonical
# programs section of the generated table should at least mention
# each shipped program once.
# ---------------------------------------------------------------------------

def test_planned_programs_say_they_are_planned() -> None:
    """Drift gate: planned program rows must include the word
    `planned` in their owner_file (we use the literal string
    ``"(planned — ...)"`` for clarity)."""
    for p in canonical.planned_programs():
        assert "planned" in p.owner_file, p.program_id
