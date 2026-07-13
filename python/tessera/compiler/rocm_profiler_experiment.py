"""Native rocprofv3 counter collection for ROCM-6 comparative experiments."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROCM6_EXPERIMENTS = frozenset({"G6-A", "G6-B", "G6-C"})


@dataclass(frozen=True)
class ROCmCounterRun:
    experiment: str
    variant: str
    command: tuple[str, ...]
    counters: tuple[str, ...]
    output_directory: str
    returncode: int | None
    status: str
    reason: str | None = None

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "schema": "tessera.rocm6.native-counters.v1",
            "experiment": self.experiment,
            "variant": self.variant,
            "command": list(self.command),
            "counters": list(self.counters),
            "output_directory": self.output_directory,
            "returncode": self.returncode,
            "provider": "rocprofv3",
            "native": self.status == "collected",
            "status": self.status,
            "reason": self.reason,
        }


def is_wsl() -> bool:
    """Return whether this process is running under Microsoft's WSL kernel."""
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release


def collect_native_counters(
    experiment: str,
    variant: str,
    application: Sequence[str],
    *,
    counters: Sequence[str],
    output_directory: str | Path,
    rocprofv3: str | None = None,
    enabled: bool = False,
) -> ROCmCounterRun:
    """Run one retained-production/candidate command under native rocprofv3.

    Native PMC collection is an explicit switch and currently a bare-metal-only
    capability. WSL exposes gfx1151 execution but not a usable rocprofiler PMC
    enumeration path, so an enabled WSL run fails before spawning rocprofv3.
    """
    if experiment not in ROCM6_EXPERIMENTS:
        raise ValueError(f"unknown ROCM-6 experiment {experiment!r}")
    if variant not in {"production", "candidate"}:
        raise ValueError("variant must be production or candidate")
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    if not enabled:
        run = ROCmCounterRun(
            experiment, variant, tuple(application), tuple(counters), str(out),
            None, "disabled", "native counters were not enabled",
        )
        (out / "tessera_rocm6_run.json").write_text(
            json.dumps(run.as_metadata_dict(), indent=2) + "\n", encoding="utf-8")
        return run
    if is_wsl():
        raise RuntimeError(
            "native ROCm performance counters are unsupported under WSL; "
            "run this switch on a bare-metal ROCm system")
    if not application:
        raise ValueError("application command must not be empty")
    if not counters:
        raise ValueError("at least one native counter is required")
    profiler = rocprofv3 or shutil.which("rocprofv3")
    if profiler is None:
        raise RuntimeError("rocprofv3 is not installed or not on PATH")
    cmd = (
        profiler, "--kernel-trace", "--stats", "--pmc", *counters,
        "--output-format", "csv", "json", "--output-directory", str(out),
        "--", *application,
    )
    completed = subprocess.run(cmd, check=False)
    run = ROCmCounterRun(
        experiment, variant, tuple(application), tuple(counters), str(out),
        completed.returncode, "collected",
    )
    (out / "tessera_rocm6_run.json").write_text(
        json.dumps(run.as_metadata_dict(), indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"rocprofv3 failed for {experiment}/{variant}: rc={completed.returncode}")
    return run


def read_counter_csv(path: str | Path) -> list[dict[str, str]]:
    """Normalize a native rocprof CSV without guessing counter semantics."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


__all__ = [
    "ROCM6_EXPERIMENTS", "ROCmCounterRun", "collect_native_counters",
    "is_wsl", "read_counter_csv",
]
