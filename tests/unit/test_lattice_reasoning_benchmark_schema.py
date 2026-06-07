"""Benchmark-row contract tests for lattice reasoning core."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.common import ExecutionKind, RuntimeStatus  # noqa: E402
from benchmarks.lattice_reasoning_core import (  # noqa: E402
    LDT_PRIMITIVE_GAPS,
    LatticeReasoningBenchmark,
    LatticeReasoningConfig,
    build_report,
)


def test_rows_distinguish_reference_and_artifact_only_execution() -> None:
    rows = LatticeReasoningBenchmark(warmup=0, reps=1).rows(
        LatticeReasoningConfig(H=4, W=4, V=4, K=3)
    )
    by_name = {row.operator.name: row for row in rows}
    assert by_name["lattice_reasoning_step"].execution_kind == ExecutionKind.REFERENCE
    artifact = by_name["lattice_reasoning_compiler_artifact"]
    assert artifact.execution_kind == ExecutionKind.ARTIFACT_ONLY
    assert artifact.runtime_status == RuntimeStatus.ARTIFACT_ONLY


def test_artifact_row_has_all_ir_levels_and_hash_inputs() -> None:
    rows = LatticeReasoningBenchmark(warmup=0, reps=1).rows(
        LatticeReasoningConfig(H=4, W=4, V=4, K=3)
    )
    artifact = next(row for row in rows if row.operator.name == "lattice_reasoning_compiler_artifact")
    assert artifact.artifact_levels.graph is True
    assert artifact.artifact_levels.schedule is True
    assert artifact.artifact_levels.tile is True
    assert artifact.artifact_levels.target is True
    assert artifact.artifact_levels.artifact_hash
    assert artifact.metrics["artifact_hash_inputs"] == [
        "graph_ir",
        "schedule_ir",
        "target_ir",
        "tile_ir",
    ]


def test_artifact_row_emits_declared_primitive_gaps() -> None:
    report = build_report(smoke=True, reps=1)
    rows = report["rows"]
    artifact = next(row for row in rows if row["operator"]["name"] == "lattice_reasoning_compiler_artifact")
    assert set(artifact["metrics"]["primitive_gaps"]) == set(LDT_PRIMITIVE_GAPS)
    assert artifact["execution_kind"] == "artifact_only"
    assert artifact["runtime_status"] == "artifact_only"


def test_report_has_no_unknown_execution_kind() -> None:
    report = build_report(smoke=True, reps=1)
    kinds = {row["execution_kind"] for row in report["rows"]}
    assert kinds == {"reference", "artifact_only"}
    assert "unknown" not in kinds


def test_cli_writes_smoke_json(tmp_path: Path) -> None:
    out = tmp_path / "lattice.json"
    subprocess.check_call([
        sys.executable,
        "benchmarks/lattice_reasoning_core/benchmark_lattice_reasoning.py",
        "--smoke",
        "--reps",
        "1",
        "--json",
        str(out),
    ], cwd=REPO_ROOT)
    payload = json.loads(out.read_text())
    assert payload["benchmark"] == "lattice_reasoning_core"
    assert payload["mode"] == "smoke"
    assert len(payload["rows"]) == 6
