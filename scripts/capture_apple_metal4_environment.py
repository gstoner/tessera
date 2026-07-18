#!/usr/bin/env python3
"""Capture reproducible Metal 4 power, thermal, and contention metadata."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence


_MAX_OUTPUT_CHARS = 65_536


def _unavailable(command: Sequence[str], reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "command": list(command),
        "reason": reason,
    }


def _run_probe(command: Sequence[str], *, timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return _unavailable(command, f"probe executable not found: {command[0]}")
    except subprocess.TimeoutExpired:
        return _unavailable(command, f"probe exceeded {timeout_s:g} second timeout")

    stdout = completed.stdout[-_MAX_OUTPUT_CHARS:]
    stderr = completed.stderr[-_MAX_OUTPUT_CHARS:]
    if completed.returncode != 0:
        result = _unavailable(command, f"probe exited with status {completed.returncode}")
        result.update({"returncode": completed.returncode, "stdout": stdout, "stderr": stderr})
        return result
    return {
        "status": "available",
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def capture_environment(*, system_name: str | None = None) -> dict[str, Any]:
    system = system_name or platform.system()
    commands = {
        "power_mode": ("/usr/bin/pmset", "-g"),
        "thermal_state": ("/usr/bin/pmset", "-g", "therm"),
        "gpu_contention": (
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/powermetrics",
            "--samplers",
            "gpu_power,tasks",
            "--show-process-gpu",
            "--sample-rate",
            "250",
            "--sample-count",
            "1",
        ),
    }
    if system != "Darwin":
        probes = {
            name: _unavailable(command, f"probe requires Darwin, found {system}")
            for name, command in commands.items()
        }
    else:
        probes = {
            "power_mode": _run_probe(commands["power_mode"]),
            "thermal_state": _run_probe(commands["thermal_state"]),
            # powermetrics requires passwordless sudo on most self-hosted Macs.
            # Failure is retained as explicit unavailable evidence, never hidden.
            "gpu_contention": _run_probe(commands["gpu_contention"], timeout_s=15.0),
        }
    return {
        "schema": "tessera.apple.metal4-environment.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": system,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "probes": probes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = capture_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: row["status"] for name, row in payload["probes"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
