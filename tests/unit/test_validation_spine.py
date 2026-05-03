from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks import perf_gate


ROOT = Path(__file__).resolve().parents[2]


def test_version_declarations_are_consistent():
    result = subprocess.run(
        [sys.executable, "scripts/check_versions.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "aligned" in result.stdout


def test_perf_gate_passes_cpu_smoke_event(tmp_path):
    report = tmp_path / "report.json"
    baseline = ROOT / "benchmarks/baselines/cpu_smoke.json"
    report.write_text(
        json.dumps(
            {
                "telemetry_events": [
                    {
                        "schema": "tessera.telemetry.v1",
                        "name": "runtime.init",
                        "source": "runtime",
                        "status": "ok",
                        "latency_ms": 0.1,
                    }
                ]
            }
        )
    )

    assert perf_gate.main([str(report), "--baseline", str(baseline)]) == 0


def test_perf_gate_fails_bad_status(tmp_path):
    report = tmp_path / "report.json"
    baseline = ROOT / "benchmarks/baselines/cpu_smoke.json"
    report.write_text(
        json.dumps(
            {
                "telemetry_events": [
                    {
                        "schema": "tessera.telemetry.v1",
                        "name": "runtime.launch",
                        "source": "runtime",
                        "status": "unimplemented",
                    }
                ]
            }
        )
    )

    assert perf_gate.main([str(report), "--baseline", str(baseline)]) == 1


def test_validation_script_documents_cpu_spine_steps():
    text = (ROOT / "scripts/validate.sh").read_text()

    for needle in [
        "scripts/check_versions.py",
        "pytest tests/unit",
        "tessera.cli.runtime",
        "benchmarks/perf_gate.py",
        "cmake -S src/runtime",
        "cmake -S tools/profiler",
        "Collectives runtime compile check",
    ]:
        assert needle in text
