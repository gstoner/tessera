"""Regression coverage for active ``tools/`` subprojects."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOFLINE_ROOT = REPO_ROOT / "tools" / "roofline_tools" / "tools" / "roofline"
ROOFLINE_MODEL = ROOFLINE_ROOT / "tprof_roofline" / "model.py"


def _load_roofline_model():
    spec = importlib.util.spec_from_file_location(
        "tessera_roofline_model_under_test",
        ROOFLINE_MODEL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_roofline_analyze_classifies_kernel_samples() -> None:
    model = _load_roofline_model()
    device = model.DevicePeaks(
        name="unit-device",
        hbm_bw_GBps=1000.0,
        compute_peaks_GFLOPs={"fp32": 10000.0},
    )
    samples = [
        model.KernelSample(
            name="memoryish",
            flop_count=1.0e9,
            dram_bytes=1.0e9,
            time_ms=1.0,
        ),
        model.KernelSample(
            name="computeish",
            flop_count=100.0e9,
            dram_bytes=1.0e9,
            time_ms=10.0,
        ),
    ]

    result = model.analyze(samples, device, dtype_key="fp32")

    assert result.device is device
    assert result.samples == samples
    assert result.compute_peak_GFLOPs == 10000.0
    assert result.mem_bw_GBps == 1000.0
    assert result.points == [(1.0, 1000.0), (100.0, 10000.0)]
    assert result.classify() == ["memory-bound", "compute-bound"]


def test_roofline_analyze_rejects_unknown_dtype_key() -> None:
    model = _load_roofline_model()
    device = model.DevicePeaks(
        name="unit-device",
        hbm_bw_GBps=1000.0,
        compute_peaks_GFLOPs={"fp32": 10000.0, "bf16_tensor": 20000.0},
    )

    with pytest.raises(ValueError, match="bf16_tensor, fp32"):
        model.analyze([], device, dtype_key="tf32")


def test_roofline_cli_runs_bundled_nsight_example(tmp_path: Path) -> None:
    outdir = tmp_path / "roofline"
    json_path = outdir / "classification.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOFLINE_ROOT / "cli_v2.py"),
            "one",
            "--peaks",
            str(ROOFLINE_ROOT / "peaks" / "sm90_with_links.yaml"),
            "--input",
            str(ROOFLINE_ROOT / "examples" / "nsight_min.csv"),
            "--fmt",
            "nsight",
            "--dtype",
            "fp32",
            "--outdir",
            str(outdir),
            "--export-json",
            str(json_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert (outdir / "roofline_report.html").is_file()
    assert (outdir / "roofline_comm.png").is_file()
    assert json_path.is_file()


def test_profiler_standalone_build_and_demo_smoke(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not available")

    build_dir = tmp_path / "tprof-build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "tools" / "profiler"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert configure.returncode == 0, configure.stderr

    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "tprof"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stderr

    help_proc = subprocess.run(
        [str(build_dir / "tprof"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert help_proc.returncode == 0
    assert "tprof peaks print" in help_proc.stdout

    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    demo = subprocess.run(
        [
            str(build_dir / "tprof"),
            "--demo-out",
            str(demo_dir / "demo.trace.json"),
            "--perfetto-out",
            str(demo_dir / "demo.perfetto.json"),
            "--report-out",
            str(demo_dir / "demo.report.html"),
            "--peaks",
            str(REPO_ROOT / "tools" / "profiler" / "scripts" / "peaks_sample.yaml"),
            "--arch",
            "sm90",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert demo.returncode == 0, demo.stderr
    assert (demo_dir / "demo.trace.json").is_file()
    assert (demo_dir / "demo.perfetto.json").is_file()
    assert (demo_dir / "demo.report.html").is_file()
    trace = json.loads((demo_dir / "demo.trace.json").read_text())
    categories = {
        event.get("cat")
        for event in trace["traceEvents"]
        if event.get("cat") is not None
    }
    assert {"runtime_api", "device_activity", "intra_kernel"} <= categories
