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


def test_record_for_library_walks_to_its_build_tree(tmp_path) -> None:
    from tessera.compiler.runtime_library_build import record_for_library
    _write(tmp_path, {"schema": "tessera.runtime_library_build.v1", "cmake_build_type": "",
                      "cxx_compiler": "GNU 15",
                      "libraries": {"tessera_x86_elementwise": "O2 (runtime-library default: tree has no build type)"}})
    library = tmp_path / "src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"")
    stamp = record_for_library(library, "tessera_x86_elementwise")
    assert stamp["optimized"] is True and stamp["target"] == "tessera_x86_elementwise"
    with pytest.raises(RuntimeLibraryBuildError, match="does not describe"):
        record_for_library(library, "tessera_x86_base")
    stray = tmp_path.parent / f"{tmp_path.name}-override.so"
    stray.write_bytes(b"")
    with pytest.raises(RuntimeLibraryBuildError, match="cannot be recorded"):
        record_for_library(stray, "tessera_x86_elementwise")


@pytest.mark.parametrize("level, optimized", [
    ("O2 (runtime-library default: tree has no build type)", True),
    ("Release: -O3 -DNDEBUG", True),
    ("RelWithDebInfo: -O2 -g", True),
    ("CMAKE_CXX_FLAGS: -O3 -march=native", True),
    ("Custom: -Os", True),
    ("X: -Ofast", True),
    ("Debug: -O2", True),
    # A build type's name is not evidence of what it compiled with.
    ("Release: ", False),
    ("RelWithDebInfo: -O0 -g", False),
    ("Debug: -g", False),
    ("X: -O2 -O0", False),
    ("X: -Og", False),
    ("O2x", False),
    ("multi-config generator: per-configuration flags", False),
])
def test_is_optimized_reads_the_flags_not_the_build_type_name(level, optimized) -> None:
    assert is_optimized(level) is optimized
