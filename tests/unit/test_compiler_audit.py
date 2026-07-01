"""M0 / M0.5 — compiler audit drift gate + taxonomy contract.

These tests lock the eight-axis support taxonomy decided in
``docs/audit/compiler/COMPILER_AUDIT.md`` and
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
# Shared expensive builds — done ONCE per module.
#
# audit.all_support_rows() walks the whole primitive-coverage registry across
# the 8 axes (~10s), and render_markdown() rebuilds the same table. Several
# tests below each called these independently — ~8×11s, a big chunk of the unit
# suite. These tests only READ the result, so a module-scoped build is safe and
# collapses the rebuilds to one each.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def support_rows():
    return audit.all_support_rows()


@pytest.fixture(scope="module")
def support_markdown():
    return audit.render_markdown()


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


def test_axis_value_glyphs_cover_every_status_used(support_rows) -> None:
    """Every cell value the walkers emit must have a glyph — otherwise
    the compact summary view silently renders `?`."""
    rows = support_rows
    seen = {cell.status for row in rows for cell in row.cells.values()}
    missing = seen - set(audit.AXIS_VALUE_GLYPHS)
    assert not missing, (
        f"axis values {sorted(missing)} have no glyph in "
        f"AXIS_VALUE_GLYPHS — add them or remove them from a walker"
    )


def test_every_row_has_one_cell_per_axis(support_rows) -> None:
    """The row contract: 8 cells, no more, no fewer."""
    rows = support_rows
    for row in rows:
        assert set(row.cells.keys()) == set(audit.LAYER_AXES), (
            f"{row.op_name}: cells={sorted(row.cells.keys())}"
        )


def test_axis_cells_carry_provenance(support_rows) -> None:
    """Every cell must name its source module / metadata path so the
    table is auditable without re-running the walk."""
    rows = support_rows
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


def test_generated_support_table_matches_audit(support_markdown) -> None:
    """The drift gate.  Fails when any of the four sources changed
    without the checked-in table being regenerated.

    Fix: ``python -m tessera.compiler.audit support_table``.
    """
    expected = support_markdown
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
    """All 17 GA primitives surface as fused at target_ir.

    Iterates the primitives (not `_CLIFFORD_APPLE_GPU_FUSED`, which also carries
    fused-chain ops like clifford_rotor_sandwich_norm that aren't benchmarked
    primitives — gap #6)."""
    from tessera.compiler import backend_manifest as bm
    for op in bm._CLIFFORD_PRIMITIVES:
        row = audit.support_row_for(op)
        assert row.cells["target_ir"].status == "fused", op
        assert row.cells["bench"].status == "benchmarked", op


def test_visual_complex_rows_match_public_api_and_backend_aliases(support_rows) -> None:
    """M7 rows use public ``tessera.complex`` names while honoring the
    prefixed backend-manifest symbols.

    ``mobius`` and ``stereographic`` are public API names, but their
    native Apple GPU entries live under ``complex_mobius`` and
    ``complex_stereographic`` in ``backend_manifest``.  The support
    table must show those public rows as fused.  Conversely,
    ``complex_add`` must stay absent because no public function with
    that name exists.

    The list of fused public ops is **derived** via
    :func:`audit.m7_fused_public_ops` rather than hardcoded — the
    parametrized check in
    ``tests/unit/test_m7_audit_visibility.py`` reads from the same
    helper, so a new fused complex kernel widens both guards
    automatically.  This test focuses on the per-row *shape*
    (``api`` / ``frontend`` / ``family`` + ``complex_add`` absence);
    the parametrized version focuses on tile/target axis status.
    """
    rows = {row.op_name: row for row in support_rows}
    assert "complex_add" not in rows

    fused_ops = audit.m7_fused_public_ops()
    assert fused_ops, (
        "m7_fused_public_ops() returned an empty set — at minimum "
        "complex_mul / complex_exp / mobius / stereographic must "
        "have fused backend entries"
    )
    for op in sorted(fused_ops):
        row = rows[op]
        assert row.family == "visual_complex", op
        assert row.cells["api"].status == "public", op
        assert row.cells["frontend"].status == "public", op
        assert row.cells["tile_ir"].status == "fused", op
        assert row.cells["target_ir"].status == "fused", op


def test_matmul_is_public_and_native_at_target_ir() -> None:
    """Tensor-side sanity: the canonical matmul row is public,
    Graph-IR-registered, and native on the best target."""
    row = audit.support_row_for("matmul")
    assert row.cells["api"].status == "public"
    assert row.cells["frontend"].status == "public"
    assert row.cells["graph_ir"].status == "registered"
    assert row.cells["tile_ir"].status == "fused"
    assert row.cells["target_ir"].status in {
        "fused", "compiled", "hardware_verified", "packaged",
    }


def test_native_target_ir_rows_are_not_left_tile_partial(support_rows) -> None:
    """A native backend manifest row is terminal Tile IR evidence.

    The primitive registry's backend-kernel axis is intentionally conservative,
    so the support table must not leave a row as `tile_ir=partial` once the
    backend manifest proves a native/compiled Target IR lane.
    """
    native_statuses = {"fused", "compiled", "hardware_verified", "packaged"}
    offenders = [
        row.op_name
        for row in support_rows
        if row.cells["target_ir"].status in native_statuses
        and row.cells["tile_ir"].status == "partial"
    ]
    assert not offenders


def test_x86_and_rocm_native_manifest_rows_close_tile_ir(support_rows) -> None:
    """x86/ROCm native lanes must not drift back to Tile IR partial."""
    from tessera.compiler import backend_manifest as bm

    native_statuses = {"fused", "compiled", "hardware_verified", "packaged"}
    native_x86_rocm = {
        op_name
        for op_name, entries in bm.all_manifests().items()
        if any(
            entry.target in {"x86", "rocm"}
            and entry.status in native_statuses
            for entry in entries
        )
    }
    assert native_x86_rocm

    offenders = [
        row.op_name
        for row in support_rows
        if row.op_name in native_x86_rocm
        and row.cells["tile_ir"].status == "partial"
    ]
    assert not offenders


def test_compiled_rocm_lanes_beat_reference_target_ir() -> None:
    """Compiled ROCm proof must not be hidden behind CPU reference entries."""
    for name in (
        "gated_attention",
        "spec_accept",
        "spec_accept_sample",
        "spec_accept_tree_sample",
    ):
        row = audit.support_row_for(name)
        assert row.cells["target_ir"].status == "compiled", name
        assert row.cells["tile_ir"].status == "fused", name


def test_backend_kernel_not_applicable_rows_do_not_need_target_ir() -> None:
    """Pure metadata/view ops should not look like Target IR reference debt."""
    for name in (
        "reshape", "view", "squeeze", "unsqueeze",
        "flatten", "expand", "broadcast",
    ):
        row = audit.support_row_for(name)
        assert row.cells["tile_ir"].status == "not_applicable", name
        assert row.cells["target_ir"].status == "not_applicable", name


def test_apple_gpu_structural_data_movers_close_tile_and_target_ir() -> None:
    for name in ("transpose", "gather", "slice"):
        row = audit.support_row_for(name)
        assert row.cells["tile_ir"].status == "fused", name
        assert row.cells["target_ir"].status == "fused", name


def test_acceptance_verification_rows_do_not_require_tile_ir() -> None:
    """Verifier/reference acceptance ops are not executable tile kernels."""
    for name in (
        "spec_accept",
        "spec_accept_sample",
        "spec_accept_tree_sample",
    ):
        row = audit.support_row_for(name)
        assert row.family == "acceptance_verification"
        assert row.cells["tile_ir"].status == "fused"

    target_verify = audit.support_row_for("target_verify")
    assert target_verify.family == "acceptance_verification"
    assert target_verify.cells["tile_ir"].status == "not_applicable"


def test_support_table_includes_canonical_program_section(support_markdown) -> None:
    """M1.5 acceptance: every canonical program (shipped or planned)
    has a row in the generated support table."""
    from tessera.compiler import canonical
    text = support_markdown
    assert "## Canonical end-to-end programs" in text
    for program in canonical.CANONICAL_PROGRAMS:
        assert program.program_id in text, program.program_id
        assert program.status in text


def test_runtime_axis_is_one_of_a_known_set(support_rows) -> None:
    """The runtime axis can only emit values we have glyphs for."""
    rows = support_rows
    valid = set(audit.AXIS_VALUE_GLYPHS)
    for row in rows:
        assert row.cells["runtime"].status in valid, (
            row.op_name, row.cells["runtime"].status,
        )
