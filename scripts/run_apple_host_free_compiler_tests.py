#!/usr/bin/env python3
"""APPLE-CI-2 host-free compiler ownership gate.

The Apple compiler build deliberately need not contain ROCm or NVIDIA lowering
passes.  This gate records that build contract, probes the Apple pipeline, and
runs only compiler tests owned by the capability set declared in CMake.
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
    CompilerBuildCapabilities,
    apple_host_free_compiler_expression,
)


def _probe(tool: Path, pipeline: str) -> dict[str, object]:
    command = [str(tool), f"--pass-pipeline=builtin.module({pipeline})", "-"]
    result = subprocess.run(
        command,
        input="module {}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "available": result.returncode == 0,
        "diagnostic": result.stderr.strip(),
    }


def _collect_nodes(command: list[str], env: dict[str, str]) -> tuple[list[str], str]:
    result = subprocess.run(
        [*command[:3], "tests/unit", "--collect-only", "-q", "--no-header", "-m", command[-1]],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    nodes = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return nodes, result.stderr.strip()


def _llvm_runner_utils(build_dir: Path) -> Path:
    """Return and validate the runner-utils dylib from this build's LLVM prefix."""

    cache = build_dir / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.startswith("LLVM_DIR:"):
            continue
        _, separator, value = line.partition("=")
        if not separator or not value.strip():
            raise ValueError(f"CMake cache has an invalid LLVM_DIR entry: {line!r}")
        llvm_dir = Path(value.strip())
        runner_utils = llvm_dir.parents[2] / "lib" / "libmlir_c_runner_utils.dylib"
        if not runner_utils.is_file():
            raise FileNotFoundError(
                f"MLIR runner-utils dylib not found for LLVM_DIR {llvm_dir}: "
                f"{runner_utils}"
            )
        return runner_utils
    raise ValueError(f"CMake cache does not declare LLVM_DIR: {cache}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path,
        help="write the ownership, probe, and selected-node record as JSON",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    capabilities = CompilerBuildCapabilities.from_cmake_cache(args.build_dir)
    if not capabilities.apple:
        parser.error("APPLE-CI-2 requires TESSERA_BUILD_APPLE_BACKEND=ON")
    if not args.tool.is_file():
        parser.error(f"tessera-opt not found: {args.tool}")

    probes = {
        "apple": _probe(args.tool, "tessera-lower-to-apple_gpu"),
        "nvidia": _probe(args.tool, "lower-tile-to-nvidia"),
        "rocm": _probe(args.tool, "lower-tile-to-rocm"),
    }
    if not probes["apple"]["available"]:
        print(json.dumps({"capabilities": capabilities.as_dict(), "probes": probes}, indent=2))
        return 1

    marker = apple_host_free_compiler_expression()
    command = [sys.executable, "-m", "pytest", "tests/unit", "-q", "-m", marker]
    env = os.environ.copy()
    env["TESSERA_OPT"] = str(args.tool)
    env["TESSERA_OPT_BIN"] = str(args.tool)
    env["PATH"] = f"{args.tool.parent}{os.pathsep}{env.get('PATH', '')}"
    jit_library = args.build_dir / "tools" / "tessera-jit" / "libtessera_jit.dylib"
    if jit_library.is_file():
        env["TESSERA_JIT_LIB"] = str(jit_library)
    try:
        runner_utils = _llvm_runner_utils(args.build_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    env["TESSERA_MLIR_C_RUNNER_UTILS"] = str(runner_utils)
    try:
        nodes, collection_diagnostic = _collect_nodes(command, env)
    except RuntimeError as error:
        print(json.dumps({"capabilities": capabilities.as_dict(), "probes": probes, "collection_error": str(error)}, indent=2))
        return 1
    report = {
        "build_dir": str(args.build_dir),
        "tool": str(args.tool),
        "mlir_c_runner_utils": str(runner_utils),
        "capabilities": capabilities.as_dict(),
        "probes": probes,
        "marker_expression": marker,
        "pytest_command": command,
        "collected_node_ids": nodes,
        "collection_diagnostic": collection_diagnostic,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
