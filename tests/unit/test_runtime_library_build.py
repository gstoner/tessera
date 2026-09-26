"""RUNTIME-LIB-OPT-1: the runtime-library optimization record."""

from __future__ import annotations

import json

import pytest

from tessera.compiler.runtime_library_build import (
    RuntimeLibraryBuildError, is_optimized, runtime_library_build)


def _write(tmp_path, record):
    (tmp_path / "runtime_library_build.json").write_text(json.dumps(record))


def test_reads_the_record_and_requires_named_libraries(tmp_path) -> None:
    _write(tmp_path, {"schema": "tessera.runtime_library_build.v1", "cmake_build_type": "",
                      "libraries": {"tessera_x86_elementwise": "O2 (runtime-library default: tree has no build type)"}})
    rec = runtime_library_build(tmp_path, require=("tessera_x86_elementwise",))
    assert is_optimized(rec["libraries"]["tessera_x86_elementwise"])
    with pytest.raises(RuntimeLibraryBuildError, match="does not describe"):
        runtime_library_build(tmp_path, require=("TesseraSpectralHIP",))


def test_a_missing_or_malformed_record_fails_loudly(tmp_path) -> None:
    with pytest.raises(RuntimeLibraryBuildError, match="missing"):
        runtime_library_build(tmp_path)
    (tmp_path / "runtime_library_build.json").write_text('{"libraries": {"a": "O2 (x, ')
    with pytest.raises(RuntimeLibraryBuildError, match="not valid JSON"):
        runtime_library_build(tmp_path)


@pytest.mark.parametrize("level, optimized", [
    ("O2 (runtime-library default: tree has no build type)", True),
    ("Release: -O3 -DNDEBUG", True),
    ("RelWithDebInfo: -O2 -g -DNDEBUG", True),
    ("Debug: -g", False),
])
def test_is_optimized(level, optimized) -> None:
    assert is_optimized(level) is optimized
