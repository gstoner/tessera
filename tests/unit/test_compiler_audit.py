"""M0 / M0.5 — compiler audit drift gate + taxonomy contract.

These tests lock the eight-axis support taxonomy decided in
``docs/audit/compiler_improvement_milestone_plan_2026_05_18.md`` and
guarantee that the generated support table at
``docs/audit/generated/support_table.md`` stays in sync with the four
source modules:

- ``tessera.compiler.op_catalog``
- ``tessera.compiler.primitive_coverage``
- ``tessera.compiler.backend_manifest``
- ``tessera.compiler.capabilities``

Anyone modifying any of those sources must regenerate the table:

    python -m tessera.compiler.audit support_table

Otherwise ``test_generated_support_table_matches_audit`` will fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import audit


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TABLE_PATH = REPO_ROOT / "docs" / "audit" / "generated" / "support_table.md"


# ---------------------------------------------------------------------------
# Eight-axis taxonomy contract
# ---------------------------------------------------------------------------

def test_layer_axes_are_eight_in_canonical_order() -> None:
    """The plan pins exactly 8 axes in display order — any reorder or
    rename is a deliberate decision that must update this test."""
    assert audit.LAYER_AXES == (
        "api", "frontend", "graph_ir", "schedule_ir",
        "tile_ir", "target_ir", "runtime", "bench",
    )


def test_axis_value_glyphs_cover_every_status_used() -> None:
    """Every cell value the walkers emit must have a glyph — otherwise
    the compact summary view silently renders `?`."""
    rows = audit.all_support_rows()
    seen = {cell.status for row in rows for cell in row.cells.values()}
    missing = seen - set(audit.AXIS_VALUE_GLYPHS)
    assert not missing, (
        f"axis values {sorted(missing)} have no glyph in "
        f"AXIS_VALUE_GLYPHS — add them or remove them from a walker"
    )


def test_every_row_has_one_cell_per_axis() -> None:
    """The row contract: 8 cells, no more, no fewer."""
    rows = audit.all_support_rows()
    for row in rows:
        assert set(row.cells.keys()) == set(audit.LAYER_AXES), (
            f"{row.op_name}: cells={sorted(row.cells.keys())}"
        )


def test_axis_cells_carry_provenance() -> None:
    """Every cell must name its source module / metadata path so the
    table is auditable without re-running the walk."""
    rows = audit.all_support_rows()
    for row in rows:
        for axis, cell in row.cells.items():
            assert cell.source, f"{row.op_name}/{axis} missing provenance"


# ---------------------------------------------------------------------------
# Drift gate — the M0.5 deliverable
# ---------------------------------------------------------------------------

def test_generated_support_table_exists() -> None:
    """The checked-in artifact is the canonical view — without it the
    drift gate has nothing to compare against."""
    assert TABLE_PATH.exists(), (
        f"missing checked-in support table: {TABLE_PATH}\n"
        "Run: python -m tessera.compiler.audit support_table"
    )


def test_generated_support_table_matches_audit() -> None:
    """The drift gate.  Fails when any of the four sources changed
    without the checked-in table being regenerated.

    Fix: ``python -m tessera.compiler.audit support_table``.
    """
    expected = audit.render_markdown()
    actual = TABLE_PATH.read_text()
    if expected != actual:
        # Surface the first ~20 differing lines so the failure message
        # is actionable in CI.
        exp_lines = expected.splitlines()
        act_lines = actual.splitlines()
        diff_lines: list[str] = []
        for i, (e, a) in enumerate(zip(exp_lines, act_lines)):
            if e != a:
                diff_lines.append(f"line {i+1}:\n  expected: {e}\n  actual:   {a}")
                if len(diff_lines) >= 5:
                    break
        if len(exp_lines) != len(act_lines):
            diff_lines.append(
                f"length: expected {len(exp_lines)} lines, "
                f"actual {len(act_lines)} lines"
            )
        pytest.fail(
            "Support table drifted from the audit walker output.\n"
            "Fix: python -m tessera.compiler.audit support_table\n\n"
            + "\n".join(diff_lines)
        )


# ---------------------------------------------------------------------------
# Spot-checks — surface the known good signals so the audit can't
# silently regress for the families we care most about.
# ---------------------------------------------------------------------------

def test_native_ebm_ops_show_fused_at_target_ir_and_tile_ir() -> None:
    """9/9 native EBM ops surface as fused on both axes."""
    natives = (
        "ebm_inner_step", "ebm_refinement", "ebm_langevin_step",
        "ebm_decode_init", "ebm_bivector_langevin", "ebm_sphere_langevin",
        "ebm_self_verify", "ebm_energy", "ebm_partition_exact",
    )
    for name in natives:
        row = audit.support_row_for(name)
        assert row.cells["target_ir"].status == "fused", (
            f"{name}: target_ir={row.cells['target_ir'].status!r}, "
            "expected 'fused' (manifest fast path)"
        )
        assert row.cells["tile_ir"].status == "fused", (
            f"{name}: tile_ir={row.cells['tile_ir'].status!r}, "
            "expected 'fused' (manifest fast path)"
        )
        assert row.cells["bench"].status == "benchmarked", name


def test_native_ga_ops_show_fused_at_target_ir() -> None:
    """All 17 GA primitives surface as fused at target_ir."""
    from tessera.compiler import backend_manifest as bm
    for op in bm._CLIFFORD_APPLE_GPU_FUSED:
        row = audit.support_row_for(op)
        assert row.cells["target_ir"].status == "fused", op
        assert row.cells["bench"].status == "benchmarked", op


def test_matmul_is_public_and_fused_at_target_ir() -> None:
    """Tensor-side sanity: the canonical matmul row is public,
    Graph-IR-registered, and fused on the best target."""
    row = audit.support_row_for("matmul")
    assert row.cells["api"].status == "public"
    assert row.cells["frontend"].status == "public"
    assert row.cells["graph_ir"].status == "registered"
    assert row.cells["target_ir"].status == "fused"


def test_support_table_includes_canonical_program_section() -> None:
    """M1.5 acceptance: every canonical program (shipped or planned)
    has a row in the generated support table."""
    from tessera.compiler import canonical
    text = audit.render_markdown()
    assert "## Canonical end-to-end programs" in text
    for program in canonical.CANONICAL_PROGRAMS:
        assert program.program_id in text, program.program_id
        assert program.status in text


def test_runtime_axis_is_one_of_a_known_set() -> None:
    """The runtime axis can only emit values we have glyphs for."""
    rows = audit.all_support_rows()
    valid = set(audit.AXIS_VALUE_GLYPHS)
    for row in rows:
        assert row.cells["runtime"].status in valid, (
            row.op_name, row.cells["runtime"].status,
        )
