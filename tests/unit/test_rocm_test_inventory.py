"""ROCM-TEST-1 compiler ownership and runner ratchets."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_rocm_host_free_compiler_tests import (
    _llvm_runner_utils,
    _pipeline_probe_error,
    _rocm_only_capability_error,
)
from tests._support.compiler_ownership import (
    CompilerBuildCapabilities,
    rocm_host_free_compiler_expression,
)


def test_rocm_host_free_compiler_selection_excludes_foreign_owners() -> None:
    expression = rocm_host_free_compiler_expression()
    assert "compiler_tool" in expression
    assert "not compiler_apple" not in expression
    assert "not compiler_cpu" not in expression
    assert "not compiler_nvidia" not in expression
    assert "not hardware_rocm" in expression
    assert "not performance" in expression
    assert "not compiler_rocm" not in expression


@pytest.mark.parametrize("cache_type", ("PATH", "UNINITIALIZED", "FILEPATH"))
def test_rocm_test1_resolves_linux_runner_utils_for_any_cache_type(
    tmp_path: Path, cache_type: str,
) -> None:
    prefix = tmp_path / "llvm-23"
    llvm_dir = prefix / "lib" / "cmake" / "llvm"
    llvm_dir.mkdir(parents=True)
    runner_utils = prefix / "lib" / "libmlir_c_runner_utils.so"
    runner_utils.touch()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"LLVM_DIR:{cache_type}={llvm_dir}\n", encoding="utf-8",
    )

    assert _llvm_runner_utils(build_dir) == runner_utils


def test_rocm_test1_rejects_missing_runner_utils(tmp_path: Path) -> None:
    llvm_dir = tmp_path / "llvm-23" / "lib" / "cmake" / "llvm"
    llvm_dir.mkdir(parents=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"LLVM_DIR:UNINITIALIZED={llvm_dir}\n", encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="runner-utils library not found"):
        _llvm_runner_utils(build_dir)


def test_rocm_test1_requires_llvm_dir(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")

    with pytest.raises(ValueError, match="does not declare LLVM_DIR"):
        _llvm_runner_utils(tmp_path)


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    (
        (CompilerBuildCapabilities(False, False, True), None),
        (
            CompilerBuildCapabilities(True, False, True),
            "disable Apple backend support",
        ),
        (
            CompilerBuildCapabilities(False, True, True),
            "disable NVIDIA backend support",
        ),
        (
            CompilerBuildCapabilities(False, False, False),
            "requires TESSERA_BUILD_ROCM_BACKEND=ON",
        ),
    ),
)
def test_rocm_test1_requires_rocm_only_capabilities(
    capabilities: CompilerBuildCapabilities, expected: str | None,
) -> None:
    error = _rocm_only_capability_error(capabilities)
    if expected is None:
        assert error is None
    else:
        assert error is not None
        assert expected in error


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        ({"rocm": True, "nvidia": False, "apple": False}, None),
        ({"rocm": False, "nvidia": False, "apple": False}, "ROCm pipeline"),
        ({"rocm": True, "nvidia": True, "apple": False}, "nvidia"),
        ({"rocm": True, "nvidia": False, "apple": True}, "apple"),
    ),
)
def test_rocm_test1_requires_rocm_only_pipeline_registration(
    available: dict[str, bool], expected: str | None,
) -> None:
    probes = {
        name: {"available": value} for name, value in available.items()
    }
    error = _pipeline_probe_error(probes)
    if expected is None:
        assert error is None
    else:
        assert error is not None
        assert expected in error
