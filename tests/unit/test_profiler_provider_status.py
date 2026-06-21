from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tessera.compiler.profiler_provider_status import (
    PROFILER_PROVIDER_STATUS_SCHEMA_VERSION,
    collect_provider_status,
    validate_provider_status_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "profiler"


@pytest.mark.parametrize("provider,target", [
    ("apple", "apple_gpu"),
    ("rocm", "rocm"),
    ("nvidia", "nvidia"),
    ("cpu", "cpu"),
])
def test_collect_provider_status_emits_stable_artifact(provider: str, target: str) -> None:
    payload = collect_provider_status(provider)

    assert payload["schema"] == PROFILER_PROVIDER_STATUS_SCHEMA_VERSION
    assert payload["provider"] == provider
    assert payload["target"] == target
    assert payload["status"] in {
        "planned",
        "compiled_shell",
        "native_available",
        "native_failed",
        "unavailable",
    }
    assert isinstance(payload["diagnostics"], dict)
    validate_provider_status_artifact(payload)


def test_apple_status_stays_shell_until_fresh_process_metal_proof() -> None:
    no_proof = collect_provider_status("apple")
    assert no_proof["status"] in {"planned", "compiled_shell"}
    assert "availability_rule" in no_proof["diagnostics"]

    failed_proof = collect_provider_status(
        "apple",
        native_proof={"fresh_process": True, "metal_visible": False},
    )
    assert failed_proof["status"] in {"planned", "compiled_shell"}
    assert failed_proof["diagnostics"]["native_proof"]["metal_visible"] is False

    passed_proof = collect_provider_status(
        "apple",
        native_proof={"fresh_process": True, "metal_visible": True},
    )
    assert passed_proof["status"] == "native_available"
    validate_provider_status_artifact(passed_proof)


def test_tprof_provider_status_cli_prints_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_status.py"),
            "--provider",
            "nvidia",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == PROFILER_PROVIDER_STATUS_SCHEMA_VERSION
    assert payload["provider"] == "nvidia"
    validate_provider_status_artifact(payload)


def test_tprof_apple_metal_smoke_ci_mode_writes_status_json(tmp_path: Path) -> None:
    out = tmp_path / "apple_status.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_apple_metal_smoke.py"),
            "--allow-unavailable",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout == ""
    payload = json.loads(out.read_text())
    assert payload["schema"] == PROFILER_PROVIDER_STATUS_SCHEMA_VERSION
    assert payload["provider"] == "apple"
    assert payload["status"] in {"planned", "compiled_shell", "native_available"}
    assert payload["diagnostics"]["smoke_script"] == "tprof-apple-metal-smoke"
    assert "native_proof" in payload["diagnostics"]
    validate_provider_status_artifact(payload)


def test_tprof_apple_metal_smoke_counter_probe_is_ci_safe(tmp_path: Path) -> None:
    out = tmp_path / "apple_counter_status.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_apple_metal_smoke.py"),
            "--allow-unavailable",
            "--prove-counters",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    proof = payload["diagnostics"]["native_proof"]
    if proof.get("metal_visible"):
        assert "counter_discovery" in proof
        assert proof["counter_discovery"]["proof_api"] == "MTLDevice.counterSets"
    validate_provider_status_artifact(payload)


@pytest.mark.parametrize("name", [
    "provider_status_apple_compiled_shell.json",
    "provider_status_rocm_planned.json",
    "provider_status_nvidia_planned.json",
    "provider_status_cpu_native_available.json",
])
def test_provider_status_fixtures_validate(name: str) -> None:
    payload = json.loads((FIXTURES / name).read_text())
    validate_provider_status_artifact(payload)
