#!/usr/bin/env python3
"""Collect ASLR-safe, symbol-correlated Linux perf or AMD IBS samples."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _cpu_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() not in fields:
                fields[key.strip()] = value.strip()
    return {
        "vendor_id": fields.get("vendor_id", "unknown"),
        "cpu_family": int(fields["cpu family"]) if fields.get("cpu family", "").isdigit() else None,
        "model": int(fields["model"]) if fields.get("model", "").isdigit() else None,
        "model_name": fields.get("model name", "unknown"),
        "microcode": fields.get("microcode", "unknown"),
        "flags": sorted(fields.get("flags", "").split()),
    }


def _symbols(binary: Path) -> str:
    for command in (
        ["nm", "-n", "-S", "--defined-only", str(binary)],
        ["nm", "-D", "-n", "-S", "--defined-only", str(binary)],
    ):
        result = _run(command)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    raise RuntimeError(f"nm could not read symbols from {binary}")


def _build_id(binary: Path) -> str:
    result = _run(["readelf", "-n", str(binary)])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "Build ID:" in line:
                return line.partition("Build ID:")[2].strip()
    return "not-present"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_symbol_sampling import (
        build_symbol_sampling_artifact,
        ibs_capability,
        parse_nm_symbols,
        parse_perf_script,
        select_symbol,
    )

    parser = argparse.ArgumentParser(prog="tprof-x86-sample")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--event", default="cycles:u")
    parser.add_argument("--frequency", type=int, default=997)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-map", type=Path)
    parser.add_argument("--cpu", type=int, help="Pin the sampled command to one logical CPU.")
    parser.add_argument("--call-graph", choices=("none", "fp", "dwarf", "lbr"), default="fp")
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    cpu = _cpu_identity()
    event_map = None
    if args.event_map is not None:
        from tessera.compiler.profiler_x86_event_map import validate_x86_event_map

        event_map = json.loads(args.event_map.read_text(encoding="utf-8"))
        validate_x86_event_map(event_map)
    perf = shutil.which("perf")
    perf_list = _run([perf, "list"]).stdout if perf is not None else ""
    ibs = ibs_capability(cpu=cpu, perf_list_text=perf_list)
    if args.event.startswith("ibs_") and not ibs["eligible"]:
        parser.error(str(ibs["reason"]))
    symbol = select_symbol(parse_nm_symbols(_symbols(binary)), args.symbol)
    perf_version = _run([perf, "--version"]).stdout.strip() if perf is not None else "unavailable"
    perf_returncode = 127
    perf_command: list[str] = []
    perf_script = ""
    stderr = "perf executable not found"
    with tempfile.TemporaryDirectory(prefix="tprof-perf-") as directory:
        data = Path(directory) / "perf.data"
        if perf is not None:
            perf_command = [
                perf,
                "record",
                "-q",
                "-e",
                args.event,
                "-F",
                str(args.frequency),
                "-o",
                str(data),
            ]
            if args.call_graph != "none":
                perf_command.extend(["--call-graph", args.call_graph])
            sampled_command = command
            if args.cpu is not None:
                if args.cpu < 0:
                    parser.error("--cpu must be non-negative")
                taskset = shutil.which("taskset")
                if taskset is None:
                    parser.error("--cpu requires taskset")
                sampled_command = [taskset, "-c", str(args.cpu), *command]
            perf_command.extend([
                "--",
                *sampled_command,
            ])
            recorded = _run(perf_command)
            perf_returncode = recorded.returncode
            stderr = recorded.stderr.strip()
            if perf_returncode == 0:
                scripted = _run(
                    [
                        perf,
                        "script",
                        "-i",
                        str(data),
                        "-F",
                        "event,ip,sym,dso,period",
                    ]
                )
                if scripted.returncode == 0:
                    perf_script = scripted.stdout
                else:
                    perf_returncode = scripted.returncode
                    stderr = scripted.stderr.strip()
    samples, rejected = parse_perf_script(perf_script)
    artifact = build_symbol_sampling_artifact(
        binary=binary,
        build_id=_build_id(binary),
        target_symbol=symbol,
        event=args.event,
        samples=samples,
        rejected_lines=rejected,
        cpu=cpu,
        ibs=ibs,
        perf_version=perf_version,
        perf_record_command=perf_command,
        perf_returncode=perf_returncode,
        event_map=event_map,
        affinity={
            "requested_logical_cpu": args.cpu,
            "pinned": args.cpu is not None,
            "mechanism": "taskset" if args.cpu is not None else "inherited",
        },
    )
    artifact["perf"]["stderr"] = stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["eligible_for_regression"] or args.allow_unavailable:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
