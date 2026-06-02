"""Glass-jaw #12 (2026-06-01) — pre-commit drift-gate guard.

Pins the generated-doc drift infrastructure so it can't silently rot:

* ``scripts/check_generated_docs.sh`` exists + is executable.
* ``.pre-commit-config.yaml`` exists, is valid YAML, and wires the
  drift script as a local hook.
* Every doc-CLI the script invokes actually supports ``--check``.
* The script currently passes (all generated docs are in sync) — this
  doubles as a fast drift gate in the unit suite, so CI catches drift
  even on a machine without pre-commit installed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_generated_docs.sh"
_PRECOMMIT = _REPO / ".pre-commit-config.yaml"


def test_drift_script_exists_and_executable():
    assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
    assert os.access(_SCRIPT, os.X_OK), (
        f"{_SCRIPT} is not executable — `chmod +x` it")


def test_precommit_config_exists_and_references_script():
    assert _PRECOMMIT.is_file(), f"missing {_PRECOMMIT}"
    text = _PRECOMMIT.read_text()
    assert "scripts/check_generated_docs.sh" in text, (
        "pre-commit config must wire the drift script as a hook")
    assert "generated-doc-drift" in text


def test_precommit_config_is_valid_yaml():
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("pyyaml not installed")
    data = yaml.safe_load(_PRECOMMIT.read_text())
    assert "repos" in data and data["repos"], "no repos in pre-commit config"
    hook_ids = {
        h["id"]
        for repo in data["repos"]
        for h in repo.get("hooks", [])
    }
    assert "generated-doc-drift" in hook_ids


def test_every_check_in_script_supports_check_flag():
    """Each module the script invokes with --check must actually
    accept --check (catches a renamed CLI that would make the hook
    silently no-op or hard-error)."""
    text = _SCRIPT.read_text()
    # Pull the "-m <module> ... --check" invocations out of the CHECKS array.
    invocations = re.findall(r"-m (tessera\.[\w.]+)((?: \w+)*) --check", text)
    assert invocations, "no '--check' invocations found in the script"
    env = {**os.environ, "PYTHONPATH": str(_REPO / "python")}
    for module, subcmd in invocations:
        # `--help` must succeed and mention --check.
        args = [sys.executable, "-m", module]
        if subcmd.strip():
            args += subcmd.split()
        proc = subprocess.run(
            args + ["--help"], capture_output=True, text=True,
            cwd=str(_REPO), env=env, timeout=60)
        combined = proc.stdout + proc.stderr
        assert "--check" in combined, (
            f"{module}{subcmd} --help does not mention --check; "
            f"the drift script references a stale CLI")


def test_drift_script_passes_today():
    """All generated docs are currently in sync. This is the actual
    drift gate — runs the script and asserts exit 0. If a manifest
    edit drifted a dashboard, this fails with the regen hint."""
    proc = subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True,
        cwd=str(_REPO), timeout=300)
    assert proc.returncode == 0, (
        f"generated-doc drift detected:\n{proc.stdout}\n{proc.stderr}")
