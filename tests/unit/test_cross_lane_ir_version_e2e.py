"""Combined tests for Issues 1, 2, and 4 (2026-05-20).

  * **Issue 1** — cross-lane composition rules + detector.
  * **Issue 2** — IR versioning stub.
  * **Issue 4** — E2E coverage audit + dashboard drift gate.

(Issue 3 — downstream IR drift gate — extends
``test_optional_ir_metadata_contract.py`` so its tests live there.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import (
    CrossLaneViolation,
    E2ECoverageRow,
    E2EStatus,
    FrontendLane,
    GRAPH_IR_SCHEMA_VERSION,
    IR_VERSION_HISTORY,
    all_e2e_coverage_rows,
    allowed_nestings,
    detect_cross_lane_violation,
    e2e_coverage_row_for,
    e2e_status_counts,
    is_cross_lane_legal,
    migrate_ir_module,
)
from tessera.compiler.e2e_coverage import render_markdown


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DOC = REPO_ROOT / "docs" / "audit" / "generated" / "e2e_op_coverage.md"


# ─────────────────────────────────────────────────────────────────────
# Issue 1 — Cross-lane composition
# ─────────────────────────────────────────────────────────────────────


class TestCrossLaneRule:
    @pytest.mark.parametrize(
        "outer,inner",
        [
            (FrontendLane.TESSERA_JIT, FrontendLane.CLIFFORD_JIT),
            (FrontendLane.TESSERA_JIT, FrontendLane.COMPLEX_JIT),
            (FrontendLane.TESSERA_JIT, FrontendLane.ENERGY_JIT),
            (FrontendLane.TEXTUAL_DSL, FrontendLane.CLIFFORD_JIT),
            (FrontendLane.TEXTUAL_DSL, FrontendLane.COMPLEX_JIT),
            (FrontendLane.TEXTUAL_DSL, FrontendLane.ENERGY_JIT),
        ],
    )
    def test_general_to_constrained_is_legal(
        self, outer: FrontendLane, inner: FrontendLane,
    ) -> None:
        assert is_cross_lane_legal(outer, inner)
        assert detect_cross_lane_violation(outer, inner) is None

    @pytest.mark.parametrize(
        "outer,inner",
        [
            # Constrained → general — breaks the outer's invariant.
            (FrontendLane.CLIFFORD_JIT, FrontendLane.TESSERA_JIT),
            (FrontendLane.COMPLEX_JIT, FrontendLane.TESSERA_JIT),
            (FrontendLane.ENERGY_JIT, FrontendLane.TESSERA_JIT),
            # Constrained → different constrained — invariants don't
            # align across value kinds.
            (FrontendLane.CLIFFORD_JIT, FrontendLane.COMPLEX_JIT),
            (FrontendLane.COMPLEX_JIT, FrontendLane.CLIFFORD_JIT),
            (FrontendLane.ENERGY_JIT, FrontendLane.CLIFFORD_JIT),
        ],
    )
    def test_constrained_to_other_is_forbidden(
        self, outer: FrontendLane, inner: FrontendLane,
    ) -> None:
        violation = detect_cross_lane_violation(outer, inner)
        assert violation is not None
        assert isinstance(violation, CrossLaneViolation)
        assert violation.outer_lane is outer
        assert violation.inner_lane is inner
        assert "outer" in violation.reason or outer.value in violation.reason

    @pytest.mark.parametrize(
        "lane",
        [
            FrontendLane.TESSERA_JIT,
            FrontendLane.TEXTUAL_DSL,
            FrontendLane.CLIFFORD_JIT,
            FrontendLane.COMPLEX_JIT,
            FrontendLane.ENERGY_JIT,
        ],
    )
    def test_same_lane_is_always_legal(self, lane: FrontendLane) -> None:
        assert is_cross_lane_legal(lane, lane)

    def test_string_lane_accepted(self) -> None:
        assert is_cross_lane_legal("tessera_jit", "clifford_jit")
        assert not is_cross_lane_legal("clifford_jit", "tessera_jit")

    def test_unknown_lane_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_cross_lane_violation("bogus_lane", "tessera_jit")

    def test_violation_to_diagnostic(self) -> None:
        v = detect_cross_lane_violation(
            FrontendLane.CLIFFORD_JIT, FrontendLane.TESSERA_JIT
        )
        assert v is not None
        d = v.to_diagnostic()
        assert d.lane == "clifford_jit"
        assert "cross-lane" in d.message
        assert d.detail["outer_lane"] == "clifford_jit"
        assert d.detail["inner_lane"] == "tessera_jit"

    def test_allowed_nestings_returns_stable_order(self) -> None:
        result = allowed_nestings()
        # Stable ordering — alphabetical by (outer.value, inner.value)
        # Re-call must produce the same tuple.
        assert result == allowed_nestings()
        # Length matches the legal matrix: 6 general→constrained + 5 same-lane.
        assert len(result) == 11


# ─────────────────────────────────────────────────────────────────────
# Issue 2 — IR versioning
# ─────────────────────────────────────────────────────────────────────


class TestIRVersioning:
    def test_schema_version_is_string(self) -> None:
        assert isinstance(GRAPH_IR_SCHEMA_VERSION, str)
        assert GRAPH_IR_SCHEMA_VERSION

    def test_schema_version_matches_history_tail(self) -> None:
        """The version constant must equal the latest entry in
        the history table.  Drift means someone added a history
        row without bumping the constant."""

        assert (
            GRAPH_IR_SCHEMA_VERSION == IR_VERSION_HISTORY[-1].version
        ), (
            f"GRAPH_IR_SCHEMA_VERSION={GRAPH_IR_SCHEMA_VERSION!r} "
            f"does not match the latest history entry "
            f"({IR_VERSION_HISTORY[-1].version!r}).  Bump the "
            f"constant or remove the orphan history entry."
        )

    def test_version_history_is_non_empty(self) -> None:
        assert len(IR_VERSION_HISTORY) >= 1

    def test_migrate_is_identity_for_current_version(self) -> None:
        from tessera.compiler.graph_ir import GraphIRModule
        module = GraphIRModule()
        result = migrate_ir_module(
            module, from_version=GRAPH_IR_SCHEMA_VERSION,
        )
        assert result is module

    def test_migrate_unknown_version_raises(self) -> None:
        from tessera.compiler.graph_ir import GraphIRModule
        with pytest.raises(ValueError, match="unknown Graph IR schema version"):
            migrate_ir_module(
                GraphIRModule(),
                from_version="99.99",
            )


# ─────────────────────────────────────────────────────────────────────
# Issue 4 — E2E op coverage audit
# ─────────────────────────────────────────────────────────────────────


class TestE2ECoverageShape:
    def test_returns_rows_with_expected_shape(self) -> None:
        rows = all_e2e_coverage_rows()
        assert len(rows) > 0
        for row in rows:
            assert isinstance(row, E2ECoverageRow)
            assert isinstance(row.status, E2EStatus)
            # axis_status carries the 8 audit axes verbatim.
            for axis in (
                "api", "frontend", "graph_ir", "schedule_ir",
                "tile_ir", "target_ir", "runtime", "bench",
            ):
                assert axis in row.axis_status

    def test_status_counts_sum_to_row_count(self) -> None:
        rows = all_e2e_coverage_rows()
        counts = e2e_status_counts()
        assert sum(counts.values()) == len(rows)

    def test_every_op_in_op_specs_has_a_row(self) -> None:
        from tessera.compiler.op_catalog import OP_SPECS
        row_names = {r.op_name for r in all_e2e_coverage_rows()}
        # Every public op should be covered by the audit.
        missing = set(OP_SPECS) - row_names
        assert missing == set(), (
            f"OP_SPECS contains ops not in the E2E coverage view: "
            f"{sorted(missing)[:5]}"
        )


class TestClassifierBehavior:
    def test_matmul_is_runnable_reference_or_better(self) -> None:
        """matmul has a known fused Apple GPU kernel + cpu reference
        path; should not be PLANNED or PARTIAL."""

        row = e2e_coverage_row_for("matmul")
        assert row.status in (
            E2EStatus.COMPLETE,
            E2EStatus.RUNNABLE_REFERENCE,
        )

    def test_status_enum_has_five_tiers(self) -> None:
        # Locks the public taxonomy.  Adding tiers is a breaking
        # change to the dashboard schema.
        assert {s.value for s in E2EStatus} == {
            "complete",
            "runnable_reference",
            "artifact_only",
            "partial",
            "planned",
        }


class TestGeneratedDocDriftGate:
    def test_generated_doc_exists(self) -> None:
        assert GENERATED_DOC.is_file(), (
            f"missing {GENERATED_DOC.relative_to(REPO_ROOT)} — "
            f"run `python -m tessera.cli.e2e_coverage --render`"
        )

    def test_generated_doc_matches_render(self) -> None:
        on_disk = GENERATED_DOC.read_text(encoding="utf-8")
        rendered = render_markdown()
        if on_disk != rendered:
            pytest.fail(
                "e2e_op_coverage.md is out of date with the audit data.  "
                "Regenerate via "
                "`python -m tessera.cli.e2e_coverage --render`."
            )

    def test_generated_doc_lists_every_tier_header(self) -> None:
        on_disk = GENERATED_DOC.read_text(encoding="utf-8")
        counts = e2e_status_counts()
        # Each non-empty tier should have an `## tier_name (N)` header.
        for status_value, count in counts.items():
            if count > 0:
                assert f"## {status_value} ({count})" in on_disk


class TestCLI:
    def test_summary_runs(self) -> None:
        from tessera.cli.e2e_coverage import main
        rc = main(["--summary"])
        assert rc == 0

    def test_check_passes_when_doc_is_fresh(self) -> None:
        from tessera.cli.e2e_coverage import main
        rc = main(["--check"])
        assert rc == 0
