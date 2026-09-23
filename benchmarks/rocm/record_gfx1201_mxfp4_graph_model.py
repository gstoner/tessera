#!/usr/bin/env python3
"""Record model-owned graph lifetime proof and a matched producer ablation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from benchmarks.rocm.benchmark_gfx1201_mxfp4_graph_producer import measure
from benchmarks.rocm.record_gfx1201_scheduled_closure import (
    _loaded_hip_runtime_path, _require_toolkit_runtime, _rocm_release,
    _selected_hip_device, _selected_rocm_toolkit,
)
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_graph_selection import assess_graph_pipeline_admission


ROOT = Path(__file__).resolve().parents[2]
SYNC_KEY = "GFX1201-MXFP4-GRAPH-MODEL-PRODUCER-2026-09-23"
SOURCES = (
    "python/tessera/compiler/rocm_mxfp4_graph_pipeline.py",
    "python/tessera/compiler/rocm_mxfp4_graph_tensor.py",
    "python/tessera/compiler/rocm_mxfp4_graph_selection.py",
    "benchmarks/rocm/benchmark_gfx1201_mxfp4_graph_producer.py",
    "benchmarks/rocm/record_gfx1201_mxfp4_graph_model.py",
    "tests/device/rocm/test_mxfp4_graph_pipeline.py",
    "tests/unit/test_rocm_mxfp4_resident.py",
)


def _tests() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="tessera-gfx1201-graph-proof-") as directory:
        report = Path(directory) / "pytest.xml"
        env = dict(os.environ, TESSERA_GFX1201_DEVICE_PROOF="1")
        command = (
            sys.executable, "-m", "pytest", "-q",
            "tests/device/rocm/test_mxfp4_graph_pipeline.py",
            "tests/unit/test_rocm_mxfp4_resident.py",
            f"--junitxml={report}",
        )
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError("graph model proof failed:\n" + result.stdout + result.stderr)
        suite = ET.parse(report).getroot().find("testsuite")
        if suite is None:
            raise RuntimeError("graph model proof lacks a JUnit testsuite")
        counts = {
            key: int(suite.attrib[key])
            for key in ("tests", "failures", "errors", "skipped")
        }
        if counts != {"tests": 18, "failures": 0, "errors": 0, "skipped": 0}:
            raise RuntimeError(f"graph model proof inventory drift: {counts}")
        return {"counts": counts, "summary": result.stdout.strip().splitlines()[-1]}


def record() -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201" or rt._rocm_chip() != "gfx1201":
        raise RuntimeError("graph model proof requires selected exact gfx1201")
    device, hip = _selected_hip_device()
    if device != "AMD Radeon RX 9070 XT":
        raise RuntimeError(f"Tajasarus proof requires RX 9070 XT, got {device}")
    toolkit, hipcc = _selected_rocm_toolkit()
    hip_runtime = _require_toolkit_runtime(toolkit, _loaded_hip_runtime_path(hip))
    hipcc_version = subprocess.check_output((str(hipcc), "--version"), text=True)
    if "HIP version: 7.15." not in hipcc_version:
        raise RuntimeError("graph model proof requires HIP 7.15")
    if not _rocm_release(toolkit).startswith("10.0"):
        raise RuntimeError("graph model proof requires selected ROCm 10.0")
    tests = _tests()
    rows = [measure(*shape) for shape in ((256, 5120, 8704), (1024, 17408, 5120))]
    packet = {
        "schema_version": 1,
        "sync_key": SYNC_KEY,
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "target": "rocm_gfx1201",
        "device_name": device,
        "rocm_toolkit": str(toolkit),
        "rocm_release": _rocm_release(toolkit),
        "hipcc": hipcc_version.splitlines()[0],
        "hip_runtime": str(hip_runtime),
        "tests": tests,
        "benchmarks": rows,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "model_frontend_route": False,
        "matched_radiance_parity": False,
        "automatic_selection": False,
    }
    packet["admission"] = assess_graph_pipeline_admission(packet)
    return packet


if __name__ == "__main__":
    print(json.dumps(record(), indent=2, sort_keys=True))
