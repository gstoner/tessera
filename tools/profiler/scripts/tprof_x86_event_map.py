#!/usr/bin/env python3
"""Record the exact Linux PMU catalog and stable digest for a Zen 5 host."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _cpu() -> dict[str, object]:
    fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    return {
        "vendor_id": fields.get("vendor_id", "unknown"),
        "cpu_family": int(fields["cpu family"]) if fields.get("cpu family", "").isdigit() else None,
        "model": int(fields["model"]) if fields.get("model", "").isdigit() else None,
        "stepping": int(fields["stepping"]) if fields.get("stepping", "").isdigit() else None,
        "model_name": fields.get("model name", "unknown"),
        "microcode": fields.get("microcode", "unknown"),
        "flags": sorted(fields.get("flags", "").split()),
    }


def _event_sources() -> dict[str, object]:
    root = Path("/sys/bus/event_source/devices")
    sources: dict[str, object] = {}
    if not root.is_dir():
        return sources
    for source in sorted(root.iterdir()):
        entry: dict[str, object] = {}
        for field in ("type", "cpumask"):
            path = source / field
            if path.is_file():
                entry[field] = path.read_text(encoding="utf-8").strip()
        for directory in ("format", "events", "caps"):
            path = source / directory
            if path.is_dir():
                entry[directory] = {
                    child.name: child.read_text(encoding="utf-8").strip()
                    for child in sorted(path.iterdir()) if child.is_file()
                }
        sources[source.name] = entry
    return sources


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "python"))
    from tessera.compiler.profiler_x86_event_map import build_x86_event_map

    parser = argparse.ArgumentParser(prog="tprof-x86-event-map")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unavailable", action="store_true")
    args = parser.parse_args(argv)
    perf_path = shutil.which("perf")
    version = _run([perf_path, "--version"]) if perf_path else None
    listing = _run([perf_path, "list", "--json"]) if perf_path else None
    listing_format = "json"
    if listing is not None and listing.returncode != 0:
        listing = _run([perf_path, "list"])
        listing_format = "text"
    release = platform.release().lower()
    artifact = build_x86_event_map(
        cpu=_cpu(),
        environment={
            "platform": platform.platform(),
            "kernel": platform.release(),
            "virtualized": "microsoft" in release or "wsl" in release,
            "wsl": "microsoft" in release or "wsl" in release,
        },
        event_sources=_event_sources(),
        perf={
            "available": perf_path is not None,
            "path": perf_path,
            "version": version.stdout.strip() if version else None,
            "version_returncode": version.returncode if version else None,
            "catalog_text": listing.stdout if listing and listing.returncode == 0 else None,
            "catalog_format": listing_format if listing and listing.returncode == 0 else None,
            "list_returncode": listing.returncode if listing else None,
            "list_stderr": listing.stderr.strip() if listing else "perf executable not found",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["eligible_for_collection"] or args.allow_unavailable:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
