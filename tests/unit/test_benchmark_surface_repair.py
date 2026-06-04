"""Focused guards for the benchmark-surface repair work."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tessera.compiler import benchmarks_manifest
from tessera.compiler.operator_benchmarks_coverage import COVERAGE_ROWS, render_markdown


ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}


def test_deepscholar_smoke_writes_current_api_json(tmp_path: Path) -> None:
    out = tmp_path / "deepscholar.json"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py"),
            "--output",
            str(out),
        ],
        env=_env(),
    )
    payload = json.loads(out.read_text())
    assert payload["schema"] == "tessera.deepscholar_smoke.v1"
    assert payload["correctness"]["passed"] is True
    assert payload["compiler"]["compiler_path"] == "jit_cpu_numpy"
    assert payload["compiler"]["runtime_status"] == "ready"
    assert payload["compiler"]["execution_kind"] == "reference_cpu"
    assert payload["compiler"]["artifact_levels"] == {
        "graph": True,
        "schedule": True,
        "tile": True,
        "target": True,
    }
    assert payload["operator_chain"] == [
        "matmul",
        "layer_norm",
        "matmul",
        "layer_norm",
        "matmul",
        "softmax",
    ]


def test_deepscholar_lotus_adapter_imports_without_research_extras() -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, 'benchmarks/DeepScholar-Bench'); "
            "import tessera_lotus_deepscholar as m; "
            "print(m.OptionalDependencyError.__name__)"
        ),
    ]
    out = subprocess.check_output(cmd, cwd=ROOT, env=_env(), text=True)
    assert "OptionalDependencyError" in out


def test_benchmark_manifest_promotes_deepscholar_and_adds_fusion_row() -> None:
    deepscholar = benchmarks_manifest.find_by_directory("benchmarks/DeepScholar-Bench")
    assert deepscholar is not None
    assert deepscholar.status == "runnable"
    entries = benchmarks_manifest.all_entries()
    assert any(e.entry_point.endswith("benchmark_fusion.py") for e in entries)
    fusion = next(e for e in entries if e.entry_point.endswith("benchmark_fusion.py"))
    assert fusion.status == "runnable"
    assert "--output /tmp/tessera_apple_gpu_fusion_smoke.json" in (fusion.command or "")


def test_spectral_artifact_mode_carries_compiler_proof_fields(tmp_path: Path) -> None:
    out = tmp_path / "spectral.csv"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "benchmarks/spectral/spectral_bench.py"),
            "--backend",
            "tessera-artifact",
            "--ops",
            "fft1d",
            "--sizes",
            "16",
            "--device",
            "cpu",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--outcsv",
            str(out),
        ],
        env=_env(),
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["compiler_path"] == "artifact_only"
    assert row["runtime_status"] == "skipped"
    assert row["execution_kind"] == "artifact_only"
    assert row["artifact_hash"]


def test_spectral_bench_default_replaces_output_not_appends(tmp_path: Path) -> None:
    out = tmp_path / "spectral.csv"
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks/spectral/spectral_bench.py"),
        "--backend",
        "tessera-artifact",
        "--ops",
        "fft1d",
        "--sizes",
        "16",
        "--device",
        "cpu",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--outcsv",
        str(out),
    ]
    subprocess.check_call(cmd, env=_env())
    subprocess.check_call(cmd, env=_env())
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["execution_kind"] == "artifact_only"


def test_spectral_bench_append_mode_is_explicit(tmp_path: Path) -> None:
    out = tmp_path / "spectral.csv"
    base = [
        sys.executable,
        str(ROOT / "benchmarks/spectral/spectral_bench.py"),
        "--backend",
        "tessera-artifact",
        "--ops",
        "fft1d",
        "--sizes",
        "16",
        "--device",
        "cpu",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--outcsv",
        str(out),
    ]
    subprocess.check_call(base, env=_env())
    subprocess.check_call([*base, "--append"], env=_env())
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert {row["runtime_status"] for row in rows} == {"skipped"}


def test_spectral_report_degrades_without_chart_requirement(tmp_path: Path) -> None:
    csv_path = tmp_path / "spectral.csv"
    csv_path.write_text(
        "timestamp,op,device,backend,dtype,shape,batch,repeats,time_ms,gflops,gbs,ai,bytes,flops,err_rel,compiler_path,runtime_status,execution_kind,artifact_hash,reason\n"
        "now,fft1d,cpu,tessera-artifact,float32,16,1,1,0,0,0,0,64,0,nan,artifact_only,skipped,artifact_only,abc,artifact only\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "report"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "benchmarks/spectral/spectral_report.py"),
            "--results",
            str(csv_path),
            "--outdir",
            str(outdir),
        ],
        env={**_env(), "MPLCONFIGDIR": str(tmp_path / "mplconfig")},
    )
    html = (outdir / "index.html").read_text()
    assert "runtime_status" in html
    assert "artifact_only" in html


def test_operator_benchmark_coverage_dashboard_is_current() -> None:
    groups = {row.opbench_group for row in COVERAGE_ROWS}
    assert {
        "matmul",
        "conv2d",
        "flash_attention",
        "reduce",
        "elementwise",
        "softmax_layernorm",
        "transpose_gather",
    }.issubset(groups)
    # operator-benchmark coverage was folded (2026-06-04) into the
    # registry-managed consolidated ``surface_status`` dashboard; verify
    # its rendered content is faithfully included there.
    from tessera.compiler import generated_docs as gd

    surface_md = gd.get("surface_status").render_md()
    assert "## Operator Benchmark Coverage" in surface_md
    # The operator-benchmark table rows must survive the fold verbatim.
    obc_table = [
        ln for ln in render_markdown().splitlines() if ln.startswith("| ")
    ]
    assert obc_table and all(row in surface_md for row in obc_table)


def test_apple_fusion_smoke_writes_output_or_skip_reason(tmp_path: Path) -> None:
    out = tmp_path / "fusion.json"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "benchmarks/apple_gpu/benchmark_fusion.py"),
            "--shapes",
            "4x4x4",
            "--swiglu-shapes",
            "1x4x4x4",
            "--reps",
            "2",
            "--output",
            str(out),
        ],
        env=_env(),
    )
    payload = json.loads(out.read_text())
    assert "runs" in payload
    if payload["runs"]:
        assert {row["op"] for row in payload["runs"]} == {"matmul_softmax", "swiglu"}
        assert {row["mode"] for row in payload["runs"]} == {"fused", "sequential"}
        assert all(row["backend"] == "apple_gpu" for row in payload["runs"])
    else:
        assert payload["skipped_apple_gpu"]


def test_benchmark_rows_have_execution_runtime_and_compiler_fields() -> None:
    from benchmarks.run_all import run_all_benchmarks

    suite = run_all_benchmarks(
        gemm_sizes=[(8, 8, 8)],
        attn_configs=[(1, 1, 8, 4)],
        collective_ranks=[2],
        collective_sizes=[128],
        verbose=False,
    )
    payload = suite.to_dict()
    rows = payload["gemm"] + payload["attention"] + payload["collective"]
    for row in rows:
        assert row.get("compiler_path")
        assert row.get("runtime_status")
        assert row.get("execution_kind")
