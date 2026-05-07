from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
OPBENCH = ROOT / "benchmarks" / "Tessera_Operator_Benchmarks"
BRIDGE = OPBENCH / "scripts" / "opbench_bridge.py"
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
        "Native C ABI operator launch is pending",
        "run_python_bridge",
        "--runtime",
        "--artifact-root",
        "tessera-runtime",
    ):
        assert term in source


def _bridge_row(op: str, mode: str = "runtime", *extra: str) -> dict:
    cmd = [
        sys.executable,
        str(BRIDGE),
        "--mode",
        mode,
        "--op",
        op,
        "--m",
        "8",
        "--n",
        "8",
        "--k",
        "4",
        "--Nn",
        "1",
        "--H",
        "5",
        "--W",
        "5",
        "--C",
        "2",
        "--Kc",
        "2",
        "--R",
        "3",
        "--S",
        "3",
        "--pad_h",
        "1",
        "--pad_w",
        "1",
        "--B",
        "1",
        "--heads",
        "1",
        "--seq",
        "4",
        "--dim",
        "4",
        *extra,
    ]
    out = subprocess.check_output(
        cmd,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )
    return json.loads(out)


def test_operator_bridge_maps_every_registered_operator_to_runtime_artifacts():
    for op in REGISTERED_OPS:
        row = _bridge_row(op)
        assert row["runtime_status"] == "success", (op, row.get("reason"))
        assert row["compiler_path"] == "jit_cpu_numpy"
        assert row["correctness"]["passed"] is True
        assert row["telemetry"]["schema"] == "tessera.telemetry.v1"
        assert row["telemetry"]["op"] == op
        levels = row["artifact_levels"]
        assert levels["graph"] is True
        assert levels["schedule"] is True
        assert levels["tile"] is True
        assert levels["target"] is True
        assert levels["artifact_hash"]
        assert levels["graph_hash"]
        assert levels["schedule_hash"]
        assert levels["tile_hash"]
        assert levels["target_hash"]


def test_operator_bridge_artifact_mode_validates_full_spine_bundles():
    for op in REGISTERED_OPS:
        row = _bridge_row(op, mode="artifact")
        assert row["runtime_status"] == "artifact_only", (op, row.get("reason"))
        assert row["correctness"]["passed"] is None
        assert row["artifact_levels"]["graph"] is True
        assert row["artifact_levels"]["schedule"] is True
        assert row["artifact_levels"]["tile"] is True
        assert row["artifact_levels"]["target"] is True


def test_operator_bridge_native_runtime_milestone_is_explicitly_pending():
    row = _bridge_row("matmul", "runtime", "--runtime", "native")
    assert row["runtime_status"] == "backend_unavailable"
    assert "Native C ABI" in row["reason"]


def test_operator_bridge_reports_structured_invalid_cases():
    cases = [
        ("matmul", ("--k2", "5"), "invalid_argument"),
        ("conv2d", ("--stride_h", "0"), "invalid_argument"),
        ("flash_attention", ("--dim", "0"), "invalid_argument"),
        ("reduce", ("--axis", "9"), "invalid_argument"),
        ("matmul", ("--dtype", "i32"), "invalid_argument"),
    ]
    for op, extra, status in cases:
        row = _bridge_row(op, "runtime", *extra)
        assert row["runtime_status"] == status
        assert row["telemetry"]["schema"] == "tessera.telemetry.v1"
        assert row["reason"]


def test_reduce_and_sum_public_ops_lower_and_execute():
    import tessera as ts

    @ts.jit(source='def reduce_kernel(x):\n    return ts.ops.reduce(x, op="sum", axis=1)\n')
    def reduce_kernel(x):
        return ts.ops.reduce(x, op="sum", axis=1)

    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    artifact = reduce_kernel.runtime_artifact()
    assert artifact.graph_ir
    assert artifact.schedule_ir
    assert artifact.tile_ir
    assert artifact.target_ir
    result = ts.launch(artifact, (x,))
    assert result["runtime_status"] == "success"
    np.testing.assert_allclose(result["output"], np.sum(x, axis=1))
    np.testing.assert_allclose(ts.ops.sum(x, axis=0), np.sum(x, axis=0))


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
