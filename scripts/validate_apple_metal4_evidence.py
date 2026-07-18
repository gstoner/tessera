#!/usr/bin/env python3
"""Fail closed when an Apple Metal 4 release artifact lacks native evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any, Mapping


_METAL4_ROUTES = {"cooperative_tensor", "simdgroup_matrix"}
_ENVIRONMENT_PROBES = {"power_mode", "thermal_state", "gpu_contention"}
_REQUIRED_BUNDLE_FILES = frozenset({
    "machine-identity.txt",
    "metal4-environment.json",
    "metal4-capabilities.json",
    "metal4-correctness-1.xml",
    "metal4-correctness-2.xml",
    "metal4-routes-1.json",
    "metal4-routes-2.json",
    "metal4-stable-ledger.json",
    "CMakeCache.txt",
    "ninja-log.txt",
    "source-commit.txt",
    "status.txt",
})


def validate_environment(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tessera.apple.metal4-environment.v1":
        raise ValueError(f"unexpected Metal 4 environment schema: {path}")
    captured_at = payload.get("captured_at")
    if not isinstance(captured_at, str):
        raise ValueError("Metal 4 environment lacks captured_at")
    try:
        if datetime.fromisoformat(captured_at).tzinfo is None:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Metal 4 environment has invalid captured_at: {captured_at!r}") from exc
    probes = payload.get("probes")
    if not isinstance(probes, Mapping) or set(probes) != _ENVIRONMENT_PROBES:
        raise ValueError(f"Metal 4 environment probes must be {_ENVIRONMENT_PROBES}")
    available = 0
    unavailable = 0
    for name, row in probes.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"Metal 4 environment probe is malformed: {name}")
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError(f"Metal 4 environment probe lacks its command: {name}")
        status = row.get("status")
        if status == "available":
            if not isinstance(row.get("stdout"), str):
                raise ValueError(f"available Metal 4 environment probe lacks output: {name}")
            available += 1
        elif status == "unavailable":
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                raise ValueError(f"unavailable Metal 4 environment probe lacks a reason: {name}")
            unavailable += 1
        else:
            raise ValueError(f"Metal 4 environment probe has invalid status: {name}={status!r}")
    return {"available": available, "unavailable": unavailable}


def validate_capabilities(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    device = payload.get("device")
    metal4 = payload.get("metal4")
    if not isinstance(device, str) or not device.startswith("apple") or "unknown" in device:
        raise ValueError(f"Metal 4 capabilities lack an exact Apple GPU family: {device!r}")
    if not isinstance(metal4, Mapping) or not metal4.get("available"):
        raise ValueError("Metal 4 capabilities do not prove an available native device")
    return {"device": device, "metal4": True}


def validate_junit(path: Path) -> dict[str, int]:
    # The local gate parses only the JUnit file emitted by its immediately
    # preceding pytest process; a pushed packet is hash-sealed afterward.
    root = ET.parse(path).getroot()  # noqa: S314
    cases = root.findall(".//testcase")
    counts = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    if counts["tests"] == 0:
        raise ValueError(f"Metal 4 JUnit report selected no tests: {path}")
    if any(counts[name] for name in ("failures", "errors", "skipped")):
        raise ValueError(f"Metal 4 JUnit report is not a clean native pass: {counts}")
    return counts


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_route_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unexpected Apple route-report schema: {path}")
    if payload.get("skipped_apple_gpu"):
        raise ValueError(f"Metal 4 route report used a device skip: {payload['skipped_apple_gpu']}")
    device = payload.get("device")
    if not isinstance(device, str) or not device.startswith("apple") or "unknown" in device:
        raise ValueError(f"Metal 4 report lacks an exact Apple GPU family: {device!r}")
    rows = payload.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Metal 4 route report contains no measured rows: {path}")
    metal4_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Metal 4 route report contains a malformed row")
        if not row.get("native_dispatched") or not row.get("numerically_validated"):
            raise ValueError(f"unproven route row cannot enter Metal 4 evidence: {row.get('route')}")
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, Mapping) or not _positive_int(
            telemetry.get("end_to_end_median_ns")
        ):
            raise ValueError(f"route row lacks end-to-end timing: {row.get('route')}")
        if row.get("route") in _METAL4_ROUTES:
            metal4_rows.append(row)
            if not _positive_int(telemetry.get("device_time_median_ns")):
                raise ValueError(f"Metal 4 route lacks device timing: {row.get('route')}")
            if float(telemetry.get("device_time_coverage", 0.0)) < 0.9:
                raise ValueError(f"Metal 4 route has incomplete device timing: {row.get('route')}")
    if not metal4_rows:
        raise ValueError("route report contains no native Metal 4 candidate")
    return {"device": device, "rows": len(rows), "metal4_rows": len(metal4_rows)}


def validate_ledger(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"Metal 4 stable ledger contains no decisions: {path}")
    domains = {row.get("timing_domain") for row in decisions if isinstance(row, Mapping)}
    if not {"device", "end_to_end"}.issubset(domains):
        raise ValueError(f"Metal 4 ledger does not retain both timing domains: {domains}")
    if any(row.get("selected_route") is None for row in decisions if isinstance(row, Mapping)):
        raise ValueError("Metal 4 ledger contains a decision without eligible incumbent evidence")
    return {"decisions": len(decisions), "timing_domains": len(domains)}


def validate_bundle(path: Path) -> dict[str, Any]:
    missing = sorted(name for name in _REQUIRED_BUNDLE_FILES if not (path / name).is_file())
    empty = sorted(
        name for name in _REQUIRED_BUNDLE_FILES
        if (path / name).is_file() and (path / name).stat().st_size == 0
    )
    if missing or empty:
        raise ValueError(f"Metal 4 proof bundle incomplete: missing={missing}, empty={empty}")

    machine = (path / "machine-identity.txt").read_text(encoding="utf-8")
    identity_sections = {
        "[sw_vers]",
        "[system_profiler SPDisplaysDataType]",
        "[xcodebuild]",
        "[macOS SDK]",
        "[metal compiler]",
        "[llvm-config]",
        "[python]",
        "[commit]",
    }
    missing_sections = sorted(identity_sections - set(machine.splitlines()))
    if missing_sections:
        raise ValueError(f"Metal 4 machine identity lacks sections: {missing_sections}")

    source_commit = (path / "source-commit.txt").read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError(f"Metal 4 proof has invalid source commit: {source_commit!r}")
    status = (path / "status.txt").read_text(encoding="utf-8")
    if "status=success\n" not in status or f"commit={source_commit}\n" not in status:
        raise ValueError("Metal 4 proof status is not a success for its source commit")

    cmake_cache = (path / "CMakeCache.txt").read_text(encoding="utf-8")
    cache_paths = {
        "LLVM_DIR": "/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/llvm",
        "MLIR_DIR": "/opt/homebrew/llvm-23.1.0-rc1/lib/cmake/mlir",
    }
    for key, value in cache_paths.items():
        if re.search(rf"^{key}:[^=]+={re.escape(value)}$", cmake_cache, re.MULTILINE) is None:
            raise ValueError(f"Metal 4 CMake cache lacks required setting: {key}={value}")
    if "TESSERA_BUILD_APPLE_BACKEND:BOOL=ON" not in cmake_cache:
        raise ValueError("Metal 4 CMake cache does not enable the Apple backend")

    return {
        "files": len(_REQUIRED_BUNDLE_FILES),
        "tested_commit": source_commit,
        "environment": validate_environment(path / "metal4-environment.json"),
        "capabilities": validate_capabilities(path / "metal4-capabilities.json"),
        "correctness": [
            validate_junit(path / "metal4-correctness-1.xml"),
            validate_junit(path / "metal4-correctness-2.xml"),
        ],
        "routes": [
            validate_route_report(path / "metal4-routes-1.json"),
            validate_route_report(path / "metal4-routes-2.json"),
        ],
        "ledger": validate_ledger(path / "metal4-stable-ledger.json"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_bundle(path: Path) -> dict[str, Any]:
    validation = validate_bundle(path)
    manifest = {
        "schema": "tessera.apple.metal4-proof.v1",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "tested_commit": validation["tested_commit"],
        "files": {
            name: {"sha256": _sha256(path / name), "bytes": (path / name).stat().st_size}
            for name in sorted(_REQUIRED_BUNDLE_FILES)
        },
        "validation": validation,
    }
    (path / "proof-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"files": len(_REQUIRED_BUNDLE_FILES), "tested_commit": validation["tested_commit"]}


def validate_committed_proof(path: Path) -> dict[str, Any]:
    manifest_path = path / "proof-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"committed Metal 4 proof lacks proof-manifest.json: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tessera.apple.metal4-proof.v1":
        raise ValueError("unexpected committed Metal 4 proof schema")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != _REQUIRED_BUNDLE_FILES:
        raise ValueError("committed Metal 4 proof manifest has the wrong file inventory")
    for name, record in files.items():
        if not isinstance(record, Mapping) or record.get("sha256") != _sha256(path / name):
            raise ValueError(f"committed Metal 4 proof hash mismatch: {name}")
        if record.get("bytes") != (path / name).stat().st_size:
            raise ValueError(f"committed Metal 4 proof size mismatch: {name}")
    validation = validate_bundle(path)
    if manifest.get("tested_commit") != validation["tested_commit"]:
        raise ValueError("committed Metal 4 proof tested-commit mismatch")
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=(
            "environment", "capabilities", "junit", "route-report", "ledger", "bundle",
            "seal", "committed-proof",
        ),
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    result = {
        "environment": validate_environment,
        "capabilities": validate_capabilities,
        "junit": validate_junit,
        "route-report": validate_route_report,
        "ledger": validate_ledger,
        "bundle": validate_bundle,
        "seal": seal_bundle,
        "committed-proof": validate_committed_proof,
    }[args.kind](args.path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
