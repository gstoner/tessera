"""APPLE-TEST-1 inventory and APPLE-CI-2 ownership ratchets."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.run_apple_host_free_compiler_tests import _llvm_runner_utils

from tests._support.apple_inventory import (
    METAL_COMPILER_TESTS,
    NATIVE_RESIDENCY_TESTS,
    NATIVE_RUNTIME_TESTS,
    inline_apple_capability_gates,
)
from tests._support.compiler_ownership import (
    CompilerBuildCapabilities,
    apple_host_free_compiler_expression,
)


ROOT = Path(__file__).resolve().parents[2]


def test_inline_apple_capability_gates_are_globally_inventoryed() -> None:
    gates = inline_apple_capability_gates(ROOT)
    assert gates == ()


def test_hardware_apple_gpu_boundary_is_centralized() -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "item.get_closest_marker(\"hardware_apple_gpu\")" in conftest
    assert "require_apple_metal()" in conftest


def test_native_residency_cohort_uses_the_hardware_boundary_and_jit_provenance() -> None:
    for node_id in NATIVE_RESIDENCY_TESTS:
        relative, name = node_id.split("::", 1)
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        decorators = [ast.unparse(decorator) for decorator in function.decorator_list]
        assert "pytest" + ".mark.hardware_apple_gpu" in decorators, node_id
        source = ast.get_source_segment(text, function) or ""
        assert "assert_native_apple_jit(" in source, node_id
        assert "pytest" + ".mark.skipif" not in "\n".join(decorators), node_id


def test_native_runtime_cohort_uses_the_hardware_boundary() -> None:
    for node_id in NATIVE_RUNTIME_TESTS:
        relative, name = node_id.split("::", 1)
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        decorators = [ast.unparse(decorator) for decorator in function.decorator_list]
        assert "pytest" + ".mark.hardware_apple_gpu" in decorators, node_id
        assert "pytest" + ".mark.skipif" not in "\n".join(decorators), node_id


def test_offline_metal_compiler_cohort_uses_the_compiler_tool_boundary() -> None:
    for node_id in METAL_COMPILER_TESTS:
        relative, name = node_id.split("::", 1)
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        decorators = [ast.unparse(decorator) for decorator in function.decorator_list]
        assert "pytest" + ".mark.compiler_tool" in decorators, node_id
        source = ast.get_source_segment(text, function) or ""
        assert "require_metal_compiler()" in source, node_id
        assert "pytest" + ".mark.skipif" not in "\n".join(decorators), node_id


def test_apple_host_free_compiler_selection_excludes_foreign_owners() -> None:
    expression = apple_host_free_compiler_expression()
    assert "compiler_tool" in expression
    assert "not compiler_nvidia" in expression
    assert "not compiler_rocm" in expression
    assert "not hardware_apple_gpu" in expression


def test_compiler_owner_markers_cover_the_apple_lane_and_known_foreign_proofs() -> None:
    expected = {
        "tests/unit/test_apple_compiler_artifacts.py": "compiler_apple",
        "tests/unit/test_msl_gemm_emit.py": "compiler_apple",
        "tests/unit/test_nvidia_compiler_artifacts.py": "compiler_nvidia",
        "tests/unit/test_rocm_arch_fragment_compiler.py": "compiler_rocm",
        "tests/unit/test_rocm_dequant_gemm_compiled.py": "compiler_rocm",
        "tests/unit/test_rocm_mla_decode_step_compiled.py": "compiler_rocm",
        "tests/unit/test_rocm_sparse_attn_compiled.py": "compiler_rocm",
    }
    for relative, marker in expected.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert f"mark.{marker}" in text, relative


def test_cmake_capabilities_require_all_backend_declarations(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text(
        "\n".join((
            "TESSERA_BUILD_APPLE_BACKEND:BOOL=ON",
            "TESSERA_BUILD_NVIDIA_BACKEND:BOOL=OFF",
            "TESSERA_BUILD_ROCM_BACKEND:BOOL=OFF",
        )),
        encoding="utf-8",
    )
    assert CompilerBuildCapabilities.from_cmake_cache(tmp_path).as_dict() == {
        "apple": True,
        "nvidia": False,
        "rocm": False,
    }


@pytest.mark.parametrize("cache_type", ("PATH", "UNINITIALIZED", "FILEPATH"))
def test_apple_ci2_resolves_runner_utils_for_any_cmake_cache_type(
    tmp_path: Path, cache_type: str,
) -> None:
    prefix = tmp_path / "llvm-23"
    llvm_dir = prefix / "lib" / "cmake" / "llvm"
    llvm_dir.mkdir(parents=True)
    runner_utils = prefix / "lib" / "libmlir_c_runner_utils.dylib"
    runner_utils.touch()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"LLVM_DIR:{cache_type}={llvm_dir}\n", encoding="utf-8"
    )

    assert _llvm_runner_utils(build_dir) == runner_utils


def test_apple_ci2_rejects_missing_runner_utils(tmp_path: Path) -> None:
    llvm_dir = tmp_path / "llvm-23" / "lib" / "cmake" / "llvm"
    llvm_dir.mkdir(parents=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"LLVM_DIR:UNINITIALIZED={llvm_dir}\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="runner-utils dylib not found"):
        _llvm_runner_utils(build_dir)


def test_apple_ci2_requires_llvm_dir_in_cmake_cache(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")

    with pytest.raises(ValueError, match="does not declare LLVM_DIR"):
        _llvm_runner_utils(tmp_path)
