"""Static and portable contracts for the APPLE-CI-1 Metal 4 release lane."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.capture_apple_metal4_environment import capture_environment
from scripts.validate_apple_metal4_evidence import (
    validate_bundle,
    validate_environment,
    validate_junit,
    validate_ledger,
    validate_route_report,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/apple-metal4-release.yml"
VALIDATE_WORKFLOW = ROOT / ".github/workflows/validate.yml"


def test_metal4_workflow_owns_one_exact_device_and_fresh_llvm23_build() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for phrase in (
        "runs-on: [self-hosted, macOS, ARM64, tessera-metal4]",
        "group: apple-metal4-exact-device",
        "cancel-in-progress: false",
        "/opt/homebrew/llvm-23.1.0-rc1",
        "scripts/install_test_deps.sh",
        "tessera_jit TesseraAppleRuntime",
        'echo "TESSERA_CACHE_DIR=$RUN_ROOT/cache" >> "$GITHUB_ENV"',
        'RUN_ROOT="$RUNNER_TEMP/tessera-metal4-$GITHUB_RUN_ID"',
        "assert r.DeviceTensor.is_metal()",
        'assert caps.get("available")',
        "capture_apple_metal4_environment.py",
        "workflow_call:",
    ):
        assert phrase in text


def test_metal4_workflow_retains_two_clean_runs_and_two_timing_domains() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for phrase in (
        'metal4 and hardware_apple_gpu and not performance',
        "metal4-correctness-$run.xml",
        "benchmark_route_characterization.py",
        "metal4-routes-$run.json",
        "select_stable_gemm_routes.py",
        "metal4-stable-ledger.json",
        "validate_apple_metal4_evidence.py junit",
        "validate_apple_metal4_evidence.py route-report",
        "validate_apple_metal4_evidence.py ledger",
        "validate_apple_metal4_evidence.py bundle",
        "bundle-validation.json",
        "if: always()",
        "retention-days: 30",
    ):
        assert phrase in text
    assert "allow-unavailable" not in text


def test_required_validate_fan_in_enforces_labeled_metal4_promotion() -> None:
    text = VALIDATE_WORKFLOW.read_text(encoding="utf-8")
    for phrase in (
        "types: [opened, reopened, synchronize, labeled, unlabeled, ready_for_review]",
        "uses: ./.github/workflows/apple-metal4-release.yml",
        "needs: [lint, unit, audit, build, apple-metal4-release]",
        "contains(github.event.pull_request.labels.*.name, 'apple-metal4-release')",
        'metal4_result="${{ needs.apple-metal4-release.result }}"',
        'if [ "$metal4_required" = "true" ] && [ "$metal4_result" != "success" ]',
    ):
        assert phrase in text


def _write_junit(path: Path, *, skipped: bool = False) -> None:
    suite = ET.Element("testsuite")
    case = ET.SubElement(suite, "testcase")
    if skipped:
        ET.SubElement(case, "skipped")
    ET.ElementTree(suite).write(path, encoding="unicode")


def test_metal4_junit_validator_rejects_skip(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _write_junit(report)
    assert validate_junit(report) == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    _write_junit(report, skipped=True)
    with pytest.raises(ValueError, match="not a clean native pass"):
        validate_junit(report)


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


def test_metal4_route_validator_requires_device_timing(tmp_path: Path) -> None:
    report = tmp_path / "routes.json"
    report.write_text(json.dumps(_route_payload()), encoding="utf-8")
    assert validate_route_report(report)["metal4_rows"] == 1
    report.write_text(json.dumps(_route_payload(device_time=None)), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks device timing"):
        validate_route_report(report)


def test_metal4_ledger_validator_requires_both_domains(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"decisions": [
        {"timing_domain": "end_to_end", "selected_route": "mps"},
        {"timing_domain": "device", "selected_route": "cooperative_tensor"},
    ]}), encoding="utf-8")
    assert validate_ledger(ledger)["timing_domains"] == 2


def _environment_payload(*, unavailable_reason: str = "no privileged sampler") -> dict:
    return {
        "schema": "tessera.apple.metal4-environment.v1",
        "captured_at": "2026-07-18T18:00:00+00:00",
        "host": {"system": "Darwin", "machine": "arm64", "platform": "macOS"},
        "probes": {
            "power_mode": {"status": "available", "command": ["pmset", "-g"], "stdout": "AC Power"},
            "thermal_state": {"status": "available", "command": ["swift", "-e"], "stdout": "0"},
            "gpu_contention": {
                "status": "unavailable",
                "command": ["sudo", "powermetrics"],
                "reason": unavailable_reason,
            },
        },
    }


def test_metal4_environment_records_unavailable_probes_explicitly(tmp_path: Path) -> None:
    portable = capture_environment(system_name="Linux")
    assert {row["status"] for row in portable["probes"].values()} == {"unavailable"}

    report = tmp_path / "environment.json"
    report.write_text(json.dumps(_environment_payload()), encoding="utf-8")
    assert validate_environment(report) == {"available": 2, "unavailable": 1}

    payload = _environment_payload(unavailable_reason="")
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a reason"):
        validate_environment(report)


def test_metal4_bundle_validator_requires_every_proof_surface(tmp_path: Path) -> None:
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
    (tmp_path / "machine-identity.txt").write_text("\n".join(sections), encoding="utf-8")
    (tmp_path / "metal4-environment.json").write_text(
        json.dumps(_environment_payload()), encoding="utf-8"
    )
    (tmp_path / "metal4-capabilities.json").write_text(
        json.dumps({"device": "apple7", "metal4": {"available": True}}), encoding="utf-8"
    )
    for run in (1, 2):
        _write_junit(tmp_path / f"metal4-correctness-{run}.xml")
        (tmp_path / f"metal4-routes-{run}.json").write_text(
            json.dumps(_route_payload()), encoding="utf-8"
        )
    (tmp_path / "metal4-stable-ledger.json").write_text(
        json.dumps({"decisions": [
            {"timing_domain": "end_to_end", "selected_route": "mps"},
            {"timing_domain": "device", "selected_route": "cooperative_tensor"},
        ]}),
        encoding="utf-8",
    )
    (tmp_path / "CMakeCache.txt").write_text(
        "LLVM_DIR:PATH=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/llvm\n"
        "MLIR_DIR:UNINITIALIZED=/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/mlir\n"
        "TESSERA_BUILD_APPLE_BACKEND:BOOL=ON\n",
        encoding="utf-8",
    )
    (tmp_path / "ninja-log.txt").write_text("# ninja log v5\n", encoding="utf-8")

    assert validate_bundle(tmp_path)["files"] == 10
    (tmp_path / "ninja-log.txt").unlink()
    with pytest.raises(ValueError, match="proof bundle incomplete"):
        validate_bundle(tmp_path)
