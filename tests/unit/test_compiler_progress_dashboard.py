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
    } <= scopes


def test_compiler_progress_tracks_required_codegen_pathways() -> None:
    rows = list(csv.DictReader(io.StringIO(compiler_progress.render_csv())))
    pathways = {r["item"] for r in rows if r["scope"] == "codegen_pathway"}
    assert {
        "Apple CPU",
        "Apple GPU",
        "x86 / CPU",
        "ROCm / HIP",
        "CUDA / NVIDIA",
    } <= pathways


def test_compiler_progress_keeps_backend_axis_separate_from_all_up_status() -> None:
    md = compiler_progress.render_markdown()
    assert "A row is not marked incomplete merely because Apple, x86, ROCm, and CUDA are not all green." in md
    assert "`backend_kernel`" in md
    assert "## Open Work Summary" in md
