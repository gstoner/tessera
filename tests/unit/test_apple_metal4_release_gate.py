"""Portable contracts for the APPLE-CI-1 local Metal 4 proof gate."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.capture_apple_metal4_environment import capture_environment
from scripts.validate_apple_metal4_evidence import (
    seal_bundle,
    validate_bundle,
    validate_committed_proof,
    validate_environment,
    validate_junit,
    validate_ledger,
    validate_route_report,
)


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/run_apple_metal4_release_gate.sh"
EVIDENCE_ROOT = ROOT / "docs/audit/evidence/apple/metal4"


def test_metal4_gate_is_local_and_never_registers_a_github_runner() -> None:
    text = GATE.read_text(encoding="utf-8")
    assert "local exact-device proof gate" in text
    assert "does not register or use a" in text
    assert "GitHub self-hosted runner" in text
    assert "--publish-dir" in text
    assert "TESSERA_APPLE_LOCK_DIR" in text
    assert not (ROOT / ".github/workflows/apple-metal4-release.yml").exists()
    validate = (ROOT / ".github/workflows/validate.yml").read_text(encoding="utf-8")
    assert "apple-metal4-release" not in validate


def test_metal4_gate_owns_fresh_llvm23_correctness_and_paired_performance() -> None:
    text = GATE.read_text(encoding="utf-8")
    for phrase in (
        "/opt/homebrew/llvm-23.1.0-rc1",
        "python3 -m venv",
        "tessera-opt tessera-translate-mlir tessera_jit TesseraAppleRuntime",
        "capture_apple_metal4_environment.py",
        'metal4 and hardware_apple_gpu and not performance',
        "metal4-correctness-$run.xml",
        "benchmark_route_characterization.py",
        "metal4-routes-$run.json",
        "select_stable_gemm_routes.py",
        "metal4-stable-ledger.json",
        "validate_apple_metal4_evidence.py seal",
    ):
        assert phrase in text


def _write_junit(path: Path, *, skipped: bool = False) -> None:
    suite = ET.Element("testsuite")
    case = ET.SubElement(suite, "testcase")
    if skipped:
        ET.SubElement(case, "skipped")
    ET.ElementTree(suite).write(path, encoding="unicode")


def _environment_payload(*, unavailable_reason: str = "no privileged sampler") -> dict:
    return {
        "schema": "tessera.apple.metal4-environment.v1",
        "captured_at": "2026-07-18T18:00:00+00:00",
        "host": {"system": "Darwin", "machine": "arm64", "platform": "macOS"},
        "probes": {
            "power_mode": {"status": "available", "command": ["pmset", "-g"], "stdout": "AC"},
            "thermal_state": {
                "status": "available", "command": ["pmset", "-g", "therm"], "stdout": "nominal"
            },
            "gpu_contention": {
                "status": "unavailable",
                "command": ["sudo", "powermetrics"],
                "reason": unavailable_reason,
            },
        },
    }


def _route_payload(*, device_time: int | None = 100) -> dict:
    return {
        "schema_version": 1,
        "device": "apple7",
        "runs": [{
            "route": "cooperative_tensor",
            "native_dispatched": True,
            "numerically_validated": True,
            "telemetry": {
                "end_to_end_median_ns": 200,
                "device_time_median_ns": device_time,
                "device_time_coverage": 1.0,
            },
        }],
    }


def _write_bundle(path: Path) -> None:
    sections = (
        "[sw_vers]",
        "[system_profiler SPDisplaysDataType]",
        "[xcodebuild]",
        "[macOS SDK]",
        "[metal compiler]",
        "[llvm-config]",
        "[python]",
        "[commit]",
    )
    (path / "machine-identity.txt").write_text("\n".join(sections), encoding="utf-8")
    (path / "metal4-environment.json").write_text(
        json.dumps(_environment_payload()), encoding="utf-8"
    )
    (path / "metal4-capabilities.json").write_text(
        json.dumps({"device": "apple7", "metal4": {"available": True}}), encoding="utf-8"
    )
    for run in (1, 2):
        _write_junit(path / f"metal4-correctness-{run}.xml")
        (path / f"metal4-routes-{run}.json").write_text(
            json.dumps(_route_payload()), encoding="utf-8"
        )
    (path / "metal4-stable-ledger.json").write_text(
        json.dumps({"decisions": [
            {"timing_domain": "end_to_end", "selected_route": "mps"},
            {"timing_domain": "device", "selected_route": "cooperative_tensor"},
        ]}),
        encoding="utf-8",
    )
    (path / "apple-cmake-cache.txt").write_text(
        "LLVM_DIR:PATH=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/llvm\n"
        "MLIR_DIR:UNINITIALIZED=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/mlir\n"
        "TESSERA_BUILD_APPLE_BACKEND:BOOL=ON\n",
        encoding="utf-8",
    )
    (path / "ninja-log.txt").write_text("# ninja log v5\n", encoding="utf-8")
    commit = "a" * 40
    (path / "source-commit.txt").write_text(commit + "\n", encoding="utf-8")
    (path / "status.txt").write_text(
        f"status=success\ncommit={commit}\nexit_code=0\n", encoding="utf-8"
    )


def test_metal4_environment_records_unavailable_probes_explicitly(tmp_path: Path) -> None:
    portable = capture_environment(system_name="Linux")
    assert {row["status"] for row in portable["probes"].values()} == {"unavailable"}
    report = tmp_path / "environment.json"
    report.write_text(json.dumps(_environment_payload()), encoding="utf-8")
    assert validate_environment(report) == {"available": 2, "unavailable": 1}
    report.write_text(json.dumps(_environment_payload(unavailable_reason="")), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a reason"):
        validate_environment(report)


def test_metal4_validators_reject_skips_and_missing_device_timing(tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    _write_junit(junit)
    assert validate_junit(junit)["tests"] == 1
    _write_junit(junit, skipped=True)
    with pytest.raises(ValueError, match="not a clean native pass"):
        validate_junit(junit)

    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps(_route_payload(device_time=None)), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks device timing"):
        validate_route_report(routes)


def test_metal4_packet_is_hash_sealed_and_tamper_evident(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    assert validate_bundle(tmp_path)["files"] == 12
    assert seal_bundle(tmp_path)["files"] == 12
    assert validate_committed_proof(tmp_path)["tested_commit"] == "a" * 40
    (tmp_path / "ninja-log.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_committed_proof(tmp_path)


def test_metal4_ledger_validator_requires_both_domains(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"decisions": [
        {"timing_domain": "end_to_end", "selected_route": "mps"},
        {"timing_domain": "device", "selected_route": "cooperative_tensor"},
    ]}), encoding="utf-8")
    assert validate_ledger(ledger)["timing_domains"] == 2


def test_committed_metal4_proof_packets_are_complete() -> None:
    packets = sorted(path.parent for path in EVIDENCE_ROOT.glob("*/proof-manifest.json"))
    assert packets, "APPLE-CI-1 requires a pushed local Metal 4 proof packet"
    for packet in packets:
        validate_committed_proof(packet)
