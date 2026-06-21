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


@pytest.mark.parametrize("name", [
    "provider_status_apple_compiled_shell.json",
    "provider_status_rocm_planned.json",
    "provider_status_nvidia_planned.json",
])
def test_provider_status_fixtures_validate(name: str) -> None:
    payload = json.loads((FIXTURES / name).read_text())
    validate_provider_status_artifact(payload)
