"""Runtime-library optimization record (RUNTIME-LIB-OPT-1).

Configure writes ``<build>/runtime_library_build.json`` naming each runtime
kernel library's effective optimization (see
``cmake/TesseraRuntimeLibraryOptimization.cmake``). Benchmark recorders stamp
it into evidence beside ``route`` (Decisions #11/#12): a latency measured from
an unoptimized library is not comparable to one from an optimized library, so
a packet must say which it measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "tessera.runtime_library_build.v1"


class RuntimeLibraryBuildError(ValueError):
    """The record is missing, malformed, or does not describe a library."""


def runtime_library_build(build_dir: str | Path, *, require: tuple[str, ...] = ()) -> dict[str, Any]:
    """Read and validate the record; ``require`` names libraries it must list.

    Fails rather than returning an empty record: evidence stamped with "no
    information" would read as comparable when it is not.
    """
    path = Path(build_dir) / "runtime_library_build.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeLibraryBuildError(
            f"{path} is missing; reconfigure this tree (RUNTIME-LIB-OPT-1) before "
            "recording evidence") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeLibraryBuildError(f"{path} is not valid JSON: {exc}") from exc
    if record.get("schema") != SCHEMA or not isinstance(record.get("libraries"), dict):
        raise RuntimeLibraryBuildError(f"{path} is not a {SCHEMA} record")
    missing = [name for name in require if name not in record["libraries"]]
    if missing:
        raise RuntimeLibraryBuildError(f"{path} does not describe {missing}")
    return record


def record_for_library(library: str | Path, target: str) -> dict[str, Any]:
    """The build record describing one loaded runtime library.

    Walks up from the library to the build tree that holds the record and
    requires that tree to name ``target`` (the CMake target, not the file
    name). A library loaded from outside a configured tree -- an override
    path, a hand-rebuilt copy -- has no record and is refused: its
    optimization level is exactly what cannot be vouched for.
    """
    path = Path(library).resolve()
    for parent in path.parents:
        if (parent / "runtime_library_build.json").is_file():
            record = runtime_library_build(parent, require=(target,))
            return {"library": str(path), "target": target,
                    "level": record["libraries"][target],
                    "optimized": is_optimized(record["libraries"][target]),
                    "cmake_build_type": record.get("cmake_build_type", ""),
                    "cxx_compiler": record.get("cxx_compiler", "")}
    raise RuntimeLibraryBuildError(
        f"{path} is not inside a build tree with runtime_library_build.json; "
        "its optimization level cannot be recorded")


def is_optimized(level: str) -> bool:
    """True unless the recorded level is an unoptimized build type."""
    text = level.strip()
    if text.startswith("O2"):
        return True
    build_type = text.split(":", 1)[0].strip().lower()
    return build_type in {"release", "relwithdebinfo", "minsizerel"}


__all__ = ["RuntimeLibraryBuildError", "SCHEMA", "is_optimized", "record_for_library",
           "runtime_library_build"]
