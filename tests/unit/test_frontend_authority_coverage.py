"""E2E-REAL-6F (queue order 1) — the AST-oracle deletion gate becomes
evaluable.

MASTER_AUDIT §1 permits deleting the AST `_OpExtractor` and the
Graph-to-backend reconstruction only after differential execution covers each
migrated family. Measured 2026-08-25, **nothing enumerated that coverage**:
the certificates live in per-`JitFn` dicts (`_frontend_differential_certificates`,
`_frontend_nonreexecuting_certificates`) and die with the instance, and no
generated doc, registry, or test listed the families. The gate could not be
evaluated at all — not because the work was missing, but because the evidence
had nowhere to live.

These rows make it evaluable and keep it honest: every registered family must
declare a differential policy that a real certification entry point can
discharge, and a policy with no certifier must be reported as blocking rather
than assumed fine.
"""

from __future__ import annotations

import pytest

from tessera.compiler import frontend_authority_audit as audit

_CERTIFIERS = {"certify_frontends", "certify_frontends_non_reexecuting"}


@pytest.fixture(scope="module")
def rows():
    collected = audit.collect_rows()
    assert collected, "no registered families — the audit has nothing to gate"
    return collected


def test_every_family_declares_a_policy_with_a_real_certifier(rows):
    """The gate itself. A family whose policy no certifier can discharge has
    no path to proof, so the oracle cannot be deleted on its account."""
    blocking = [row.family for row in rows if row.blocks_oracle_deletion]
    assert not blocking, (
        f"families with no certification path: {blocking}. Either declare a "
        f"differential_policy a certifier handles, or record why the family "
        f"cannot be certified — never leave it unstated, which reads as "
        f"covered."
    )


def test_the_named_certifiers_exist(rows):
    """A certifier named in a declaration but absent from the code is exactly
    the unconsumed declaration Decision #29 rejects — and here it would make
    every row citing it read as proven."""
    from tessera.compiler import frontend_authority as certs

    for name in {row.certifier for row in rows}:
        assert name in _CERTIFIERS, f"unknown certifier {name!r}"
        assert hasattr(certs, name), (
            f"declaration names {name!r}, which does not exist — every family "
            f"under that policy would report a certification path it has not "
            f"got"
        )


def test_stateful_families_do_not_use_the_re_executing_certifier(rows):
    """The distinction with teeth. `certify_frontends` proves equality by
    RUNNING the source again; for a family that mutates state, the second run
    is a different program, so such a certificate would compare two things
    that were never meant to be equal — and would pass."""
    for row in rows:
        if row.differential_policy == "non_reexecuting_state_lineage":
            assert row.certifier == "certify_frontends_non_reexecuting", row


def test_every_family_declares_its_whole_spine(rows):
    """A family that names a Schedule consumer but no Tile consumer, or no
    target, is migrated only on paper."""
    for row in rows:
        assert row.schedule_consumer.startswith("schedule."), row
        assert row.tile_consumer.startswith("tile."), row
        assert row.targets, f"{row.family} declares no target consumer"
        assert row.ops, f"{row.family} owns no Graph op"
        assert row.execution_certificate_schema == (
            "tessera.native_vjp_execution.v1"
        )


def test_family_target_union_does_not_drop_variant_specific_owners(rows):
    optimizer = next(row for row in rows if row.family == "optimizer_vjp")
    assert set(optimizer.targets) == {"nvidia_sm120", "rocm", "x86"}


def test_exact_target_packets_cover_local_rows_and_leave_siblings_blocking(rows):
    from pathlib import Path

    target_rows = audit.collect_target_rows()
    declared = {(row.family, target) for row in rows for target in row.targets}
    assert {(row.family, row.target) for row in target_rows} == declared
    exact = {row.target for row in target_rows if not row.blocks_exact_coverage}
    blocking = {row.target for row in target_rows if row.blocks_exact_coverage}
    assert exact == {"rocm", "x86"}
    assert blocking == {"apple_gpu", "nvidia_sm120"}
    for row in target_rows:
        if not row.blocks_exact_coverage:
            assert Path(row.evidence_gate).is_file()


def test_the_dashboard_reports_what_the_registry_holds(rows):
    """Drift: the rendered doc must not disagree with the live registry."""
    text = audit.render_dashboard()
    assert f"families: **{len(rows)}**" in text
    for row in rows:
        assert f"`{row.family}`" in text
    blocking = sum(1 for row in rows if row.blocks_oracle_deletion)
    assert f"families with no certification path: {blocking}" in text


def test_the_gate_can_actually_fail():
    """A control. Without it, every row above passes on a registry that
    happens to be clean today and would keep passing if the check were
    vacuous."""
    unknown = audit.FamilyRow(
        family="fabricated", ops=("nope",), migration_state="canonical",
        differential_policy="a_policy_no_certifier_handles",
        certifier=audit._CERTIFIER.get("a_policy_no_certifier_handles", "none"),
        schedule_consumer="schedule.nope", tile_consumer="tile.nope",
        targets=("x86",),
    )
    assert unknown.blocks_oracle_deletion, (
        "a family with an unhandled policy must be reported as blocking; if it "
        "is not, the gate above proves nothing"
    )


def test_the_csv_and_dashboard_agree(rows):
    csv_text = audit.render_csv()
    assert csv_text.splitlines()[0].startswith("family,ops,")
    assert len(csv_text.strip().splitlines()) == len(rows) + 1
