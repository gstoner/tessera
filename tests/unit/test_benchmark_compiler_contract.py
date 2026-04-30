from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from benchmarks.common import (
    ArtifactLevels,
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    RuntimeStatus,
    compiler_flash_attention_ir,
    compiler_matmul_relu,
    compiler_spectral_ir,
    correctness_report,
)
from benchmarks.compiler_support import compiler_matmul_relu as compat_compiler_matmul_relu


def test_benchmark_row_serializes_standard_statuses():
    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", "f32", "16x16x16"),
        compiler_path=CompilerPath.GRAPH_IR_ONLY,
        runtime_status=RuntimeStatus.SKIPPED,
        artifact_levels=ArtifactLevels(graph=True, artifact_hash="abc"),
    )
    data = row.to_dict()
    assert data["compiler_path"] == "graph_ir_only"
    assert data["runtime_status"] == "skipped"
    assert row.flat_dict()["artifact_graph"] is True


def test_correctness_report_marks_pass():
    report = correctness_report(np.array([1.0, 2.0]), np.array([1.0, 2.001]), 0.01)
    assert report.passed is True
    assert report.max_error > 0


def test_compatibility_shim_exports_compiler_path():
    assert compat_compiler_matmul_relu is compiler_matmul_relu


def test_gemm_compiler_contract_runs_or_reports_none():
    a = np.ones((8, 8), dtype=np.float32)
    b = np.ones((8, 8), dtype=np.float32)
    run = compiler_matmul_relu(a, b, (8, 8, 8))
    assert run is None or run.graph_ir is not None
    if run is not None:
        assert run.artifact_hash


def test_artifact_only_contracts_emit_graph_ir():
    flash = compiler_flash_attention_ir()
    spectral = compiler_spectral_ir("fft1d")
    assert flash["available"] is True
    assert "tessera.flash_attn" in flash["graph_ir"]
    assert spectral["available"] is True
    assert "tessera.fft" in spectral["graph_ir"]


def test_superbench_gemm_module_emits_shared_row():
    cmd = [
        sys.executable,
        "benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera.py",
        "--m=8",
        "--n=8",
        "--k=8",
        "--repeat=1",
    ]
    out = subprocess.check_output(cmd, text=True)
    row = json.loads(out.splitlines()[-1])
    assert row["operator"] == "gemm"
    assert row["compiler_path"] in {"tessera_jit_cpu", "reference"}
    assert row["runtime_status"] == "executable"
