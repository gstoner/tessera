from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OPBENCH = ROOT / "benchmarks" / "Tessera_Operator_Benchmarks"
REGISTERED_OPS = {
    "matmul",
    "conv2d",
    "flash_attention",
    "reduce",
    "elementwise",
    "softmax_layernorm",
    "transpose_gather",
}
ARTIFACT_FILES = {
    "matmul": "MatmulOp.mlir",
    "conv2d": "Conv2dNHWC.mlir",
    "flash_attention": "FlashAttention.mlir",
    "reduce": "Reduce.mlir",
    "elementwise": "Elementwise.mlir",
    "softmax_layernorm": "SoftmaxLayerNorm.mlir",
    "transpose_gather": "TransposeGather.mlir",
}


def test_operator_quick_sweep_covers_every_registered_operator():
    cfg = yaml.safe_load((OPBENCH / "scripts" / "configs" / "quick_sweep.yaml").read_text())
    ops = {run["op"] for run in cfg["runs"]}
    assert ops == REGISTERED_OPS


def test_operator_artifact_samples_cover_every_registered_operator():
    sample_dir = OPBENCH / "mlir" / "tessera_ir_samples"
    for op, filename in ARTIFACT_FILES.items():
        text = (sample_dir / filename).read_text()
        assert "tessera.ir.level" in text
        assert "module attributes" in text
        assert op.split("_")[0] in text or "tessera." in text


def test_operator_harness_source_reports_current_backend_statuses():
    source = (OPBENCH / "harness" / "opbench_main.cpp").read_text()
    for term in (
        "tessera.telemetry.v1",
        '"artifact_only"',
        '"backend_unavailable"',
        "--artifact-root",
        "tessera-runtime",
    ):
        assert term in source


def test_operator_runner_accepts_json_rows_and_writes_telemetry_summary(tmp_path):
    fake_bin = tmp_path / "fake_opbench.py"
    fake_bin.write_text(
        """#!/usr/bin/env python3
import json, sys
op = sys.argv[sys.argv.index("--op") + 1]
backend = sys.argv[sys.argv.index("--backend") + 1]
status = "artifact_only" if backend == "artifact" else "executable"
print(json.dumps({
  "operator": {"name": op, "dtype": "f32", "shape": "cli", "target": "cpu"},
  "compiler_path": "artifact_only" if backend == "artifact" else "reference",
  "runtime_status": status,
  "artifact_levels": {"graph": backend == "artifact", "schedule": False, "tile": False, "target": False, "artifact_hash": None},
  "correctness": {"max_error": 0, "relative_error": None, "tolerance": None, "passed": None},
  "profile": {"cpu_wall_ms": 0.1, "kernel_elapsed_ms": None, "memory_bytes": None, "launch_overhead_ms": None},
  "metrics": {"backend": backend, "gflops": 1.0, "gbps": 0.0},
  "telemetry": {"schema": "tessera.telemetry.v1", "name": op, "source": "test", "op": op, "status": "ok", "bottleneck": "unknown"},
  "reason": ""
}))
""",
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | 0o111)
    out = tmp_path / "out"
    subprocess.check_call(
        [
            sys.executable,
            str(OPBENCH / "scripts" / "opbench.py"),
            "--config",
            str(OPBENCH / "scripts" / "configs" / "quick_sweep.yaml"),
            "--bin",
            str(fake_bin),
            "--backend",
            "artifact",
            "--out",
            str(out),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )
    payload = json.loads((out / "results.json").read_text())
    rows = list(csv.DictReader((out / "results.csv").open()))
    assert payload["schema"] == "tessera.operator_bench.v1"
    assert payload["telemetry_summary"]["schema"] == "tessera.telemetry.v1"
    assert payload["telemetry_summary"]["event_count"] == len(REGISTERED_OPS)
    assert {row["op"] for row in rows} == REGISTERED_OPS
