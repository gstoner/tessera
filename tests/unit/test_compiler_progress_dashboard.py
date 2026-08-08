"""Contract tests for the generated compiler-progress rollup."""
from __future__ import annotations

import csv
import io

from tessera.compiler import compiler_progress


def test_compiler_progress_csv_exposes_required_scopes() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    scopes = {r["scope"] for r in rows}
    assert {
        "overall",
        "phase",
        "primitive_axis",
        "integration",
        "codegen_pathway",
        "open_work",
        "dashboard_map",
    } <= scopes


def test_compiler_progress_tracks_required_codegen_pathways() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    pathways = {r["item"] for r in rows if r["scope"] == "codegen_pathway"}
    assert {
        "Portable CPU reference",
        "Apple CPU",
        "Apple GPU",
        "x86 / AVX-512",
        "ROCm / gfx1151",
        "NVIDIA SM90",
        "NVIDIA SM120",
    } <= pathways
    assert "x86 / CPU" not in pathways
    assert "CUDA / NVIDIA" not in pathways


def test_codegen_pathways_use_one_manifest_grain() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    pathways = [r for r in rows if r["scope"] == "codegen_pathway"]
    assert pathways
    for row in pathways:
        # Runtime paths are independent evidence, never additive inputs to the
        # op×target manifest denominator.
        assert "runtime_paths: executable=" in row["detail"]
        assert "manifest: exact_verified=" in row["detail"]
        assert int(row["ready"]) + int(row["open"]) == int(row["total"])


def test_terminal_no_kernel_required_closes_phase_rows() -> None:
    support = [{"tile_ir": "no_kernel_required", "target_ir": "no_kernel_required"}]
    rows = {row.item: row for row in compiler_progress._phase_rows(support)}
    assert rows["Tile IR"].status == "closed"
    assert rows["Target IR native/fused codegen"].status == "closed"


def test_open_work_omits_closed_rows() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    assert all(int(r["open"]) > 0 for r in rows if r["scope"] == "open_work")
    assert not any(
        r["item"] == "Verifier coverage" for r in rows if r["scope"] == "open_work"
    )


def test_compiler_progress_keeps_backend_axis_separate_from_all_up_status() -> None:
    md = compiler_progress.render_markdown()
    assert "A row is not marked incomplete merely because Apple, x86, ROCm, and CUDA are not all green." in md
    assert "`backend_kernel`" in md
    assert "## Open Work Summary" in md
    assert "## Dashboard Map" in md
    assert "counts dashboard checks, not operations or target capability" in md


def test_compiler_progress_csv_includes_next_action() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    assert "next_action" in rows[0]
    assert all(r["next_action"] for r in rows if r["scope"] == "open_work")
