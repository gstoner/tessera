#!/usr/bin/env python3
"""ROCM-TEST-1 host-free compiler ownership gate.

The ROCm compiler build is intentionally valid without Apple, CUDA, or an
active GPU. This gate records the declared CMake capability set, probes the
ROCm pipeline and foreign pipeline absence, collects the owned pytest nodes,
and runs only the host-free compiler proofs that the build can satisfy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._support.compiler_ownership import (  # noqa: E402
    COMPILER_TEST_PLATFORM_ENV,
    CompilerBuildCapabilities,
    rocm_host_free_compiler_expression,
)


_RECORDED_CMAKE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "LLVM_DIR",
    "MLIR_DIR",
    "TESSERA_BUILD_APPLE_BACKEND",
    "TESSERA_BUILD_NVIDIA_BACKEND",
    "TESSERA_BUILD_ROCM_BACKEND",
    "TESSERA_ENABLE_CUDA",
    "TESSERA_ENABLE_HIP",
)


def _cache_values(build_dir: Path) -> dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        name_and_type, separator, value = line.partition("=")
        if not separator:
            continue
        name = name_and_type.partition(":")[0]
        if name in _RECORDED_CMAKE_KEYS:
            values[name] = value
    return values


def _probe(tool: Path, pipeline: str) -> dict[str, object]:
    command = [str(tool), f"--pass-pipeline=builtin.module({pipeline})", "-"]
    result = subprocess.run(
        command, input="module {}\n", text=True, capture_output=True, check=False,
    )
    return {
        "command": command,
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "diagnostic": result.stderr.strip(),
    }


def _collect_nodes(command: list[str], env: dict[str, str]) -> tuple[list[str], str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "--collect-only", "-q",
         "--no-header", "-m", command[-1]],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    nodes = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return nodes, result.stderr.strip()


def _llvm_runner_utils(build_dir: Path) -> Path:
    """Resolve the Linux MLIR runner-utils library from this build's LLVM_DIR."""

    values = _cache_values(build_dir)
    raw = values.get("LLVM_DIR")
    if not raw:
        raise ValueError(f"CMake cache does not declare LLVM_DIR: {build_dir / 'CMakeCache.txt'}")
    llvm_dir = Path(raw)
    candidates = (
        llvm_dir.parents[2] / "lib" / "libmlir_c_runner_utils.so",
        llvm_dir.parents[2] / "lib" / "libmlir_runner_utils.so",
    )
    runner_utils = next((path for path in candidates if path.is_file()), None)
    if runner_utils is None:
        raise FileNotFoundError(
            f"MLIR runner-utils library not found for LLVM_DIR {llvm_dir}: "
            + ", ".join(str(path) for path in candidates)
        )
    return runner_utils


def _tool_version(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "output": "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    capabilities = CompilerBuildCapabilities.from_cmake_cache(args.build_dir)
    if not capabilities.rocm:
        parser.error("ROCM-TEST-1 requires TESSERA_BUILD_ROCM_BACKEND=ON")
    if not args.tool.is_file():
        parser.error(f"tessera-opt not found: {args.tool}")

    probes = {
        "rocm": _probe(args.tool, "lower-tile-to-rocm"),
        "nvidia": _probe(args.tool, "lower-tile-to-nvidia"),
        "apple": _probe(args.tool, "tessera-lower-to-apple_gpu"),
    }
    if not probes["rocm"]["available"]:
        print(json.dumps({"capabilities": capabilities.as_dict(), "probes": probes}, indent=2))
        return 1

    try:
        runner_utils = _llvm_runner_utils(args.build_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    marker = rocm_host_free_compiler_expression()
    command = [sys.executable, "-m", "pytest", "tests/unit", "-q", "-m", marker]
    env = os.environ.copy()
    env[COMPILER_TEST_PLATFORM_ENV] = "rocm"
    env["TESSERA_OPT"] = str(args.tool.resolve())
    env["TESSERA_OPT_BIN"] = str(args.tool.resolve())
    env["TESSERA_MLIR_C_RUNNER_UTILS"] = str(runner_utils)
    env["PATH"] = f"{args.tool.parent.resolve()}{os.pathsep}{env.get('PATH', '')}"
    try:
        nodes, collection_diagnostic = _collect_nodes(command, env)
    except RuntimeError as error:
        print(json.dumps({
            "capabilities": capabilities.as_dict(), "probes": probes,
            "collection_error": str(error),
        }, indent=2))
        return 1

    report = {
        "schema": "tessera.rocm.host-free-compiler-ownership.v1",
        "work_item": "ROCM-TEST-1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "build_dir": str(args.build_dir.resolve()),
        "build_command": [
            "cmake", "--build", str(args.build_dir.resolve()),
            "--target", "tessera-opt", "tessera-rocm-opt",
        ],
        "tool": str(args.tool.resolve()),
        "mlir_c_runner_utils": str(runner_utils),
        "cmake_cache": _cache_values(args.build_dir),
        "capabilities": capabilities.as_dict(),
        "probes": probes,
        "tool_versions": {
            "tessera_opt": _tool_version([str(args.tool), "--version"]),
            "mlir_opt": _tool_version(["/usr/lib/llvm-23/bin/mlir-opt", "--version"]),
        },
        "marker_expression": marker,
        "selected_platform": "ROCm",
        "foreign_owner_behavior": (
            "foreign compiler tests are skipped with their required system in "
            "the pytest terminal summary"
        ),
        "pytest_command": command,
        "collected_node_ids": nodes,
        "collection_diagnostic": collection_diagnostic,
    }
    if args.dry_run:
        report["execution"] = {
            "status": "not_run",
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    report["execution"] = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
