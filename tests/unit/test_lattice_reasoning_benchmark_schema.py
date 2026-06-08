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
    APPLE_GPU_EXECUTABLE_MODEL_PRIMITIVES,
    LDT_PRIMITIVE_GAPS,
    LANDED_LDT_PRIMITIVES,
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
    assert set(artifact["metrics"]["landed_primitives"]) == set(LANDED_LDT_PRIMITIVES)
    assert set(artifact["metrics"]["remaining_integrated_step_work"]) == set(LDT_PRIMITIVE_GAPS)
    assert artifact["execution_kind"] == "artifact_only"
    assert artifact["runtime_status"] == "artifact_only"


def test_report_has_no_unknown_execution_kind() -> None:
    report = build_report(smoke=True, reps=1)
    kinds = {row["execution_kind"] for row in report["rows"]}
    assert {"reference", "artifact_only"} <= kinds
    assert kinds <= {"reference", "artifact_only", "optimized_native"}
    assert "unknown" not in kinds


def test_report_surfaces_current_apple_gpu_native_rows() -> None:
    report = build_report(smoke=True, reps=1)
    by_name = {row["operator"]["name"]: row for row in report["rows"]}
    if "apple_gpu_current_compiler_primitives" in by_name:
        row = by_name["apple_gpu_current_compiler_primitives"]
        assert row["runtime_status"] == "skipped"
        assert row["compiler_path"] == "runtime_unavailable"
        assert "apple_gpu runtime" in row["reason"]
        return
    for name in (
        "apple_gpu_mamba2_selective_ssm",
        "apple_gpu_grouped_gemm_fused",
    ):
        row = by_name[name]
        assert row["compiler_path"] == "tessera_jit_apple_gpu"
        assert row["execution_kind"] == "optimized_native"
        assert row["runtime_status"] == "executable"
        assert row["metrics"]["execution_mode"] == "metal_runtime"
        assert row["metrics"]["observed_native_execution"] is True


def test_apple_ldt_moe_rows_reach_metal_runtime() -> None:
    # 2026-06-07: all six LDT/MoE-aux primitives now have dedicated Metal kernels
    # (MSL + MPSGraph subgraphs) and are in the apple_gpu envelope, so — like the
    # selective_ssm / grouped_gemm rows above — they report execution_mode
    # "metal_runtime" (envelope-classified) with a numerically-correct result,
    # hence optimized_native. (Was artifact_only before the kernels landed.)
    report = build_report(smoke=True, reps=1)
    by_name = {row["operator"]["name"]: row for row in report["rows"]}
    if "apple_gpu_current_compiler_primitives" in by_name:
        row = by_name["apple_gpu_current_compiler_primitives"]
        assert row["runtime_status"] == "skipped"
        assert row["compiler_path"] == "runtime_unavailable"
        assert "apple_gpu runtime" in row["reason"]
        return
    for name in (
        "apple_gpu_ldt_count_nonzero",
        "apple_gpu_ldt_popcount",
        "apple_gpu_ldt_masked_categorical",
        "apple_gpu_ldt_asymmetric_bce",
        "apple_gpu_moe_z_loss",
        "apple_gpu_moe_load_balance_loss",
    ):
        row = by_name[name]
        assert row["compiler_path"] == "tessera_jit_apple_gpu"
        assert row["execution_kind"] == "optimized_native"
        assert row["runtime_status"] == "executable"
        assert row["metrics"]["execution_mode"] == "metal_runtime"
        assert row["metrics"]["observed_native_execution"] is True
    assert set(APPLE_GPU_EXECUTABLE_MODEL_PRIMITIVES) >= {
        "selective_ssm_scalar_A",
        "grouped_gemm_fused",
    }


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
    names = {row["operator"]["name"] for row in payload["rows"]}
    assert {
        "lattice_reasoning_step",
        "ldt_count_nonzero_tessera",
        "ldt_popcount_tessera",
        "ldt_masked_categorical_tessera",
        "ldt_asymmetric_bce_tessera",
        "mopd_policy_loss_core",
        "mamba2_ssd_core",
        "gqa_decode_core",
        "latent_moe_core",
        "lattice_reasoning_compiler_artifact",
    } <= names
    if "apple_gpu_current_compiler_primitives" in names:
        assert len(payload["rows"]) == 11
    else:
        assert len(payload["rows"]) >= 18
