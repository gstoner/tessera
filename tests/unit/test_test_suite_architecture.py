"""Ratchets for the compiler test-suite layering rules."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from tests._support.environment import (
    CompilerToolchain,
    PYTHON_ROOT,
    python_subprocess_environment,
)
from tests._support.policy import MARKERS, PR_MARKER_EXPRESSION


ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "tests/unit"


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        current = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        names.add(".".join(reversed(parts)))
    return names


def _uses_wall_clock(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr in {"perf_counter", "perf_counter_ns"}:
            return True
    return False


def test_every_direct_wall_clock_test_is_performance_marked():
    """Measured timing must never leak into the hermetic CPU PR state."""

    violations: list[str] = []
    for path in UNIT.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_") or not _uses_wall_clock(node):
                continue
            marked = "pytest.mark.performance" in _decorator_names(node)
            if not marked:
                violations.append(f"{path.relative_to(ROOT)}::{node.name}")
    assert not violations, (
        "wall-clock tests require @pytest.mark.performance: "
        + ", ".join(violations)
    )


def test_required_pr_entry_points_share_the_canonical_marker_expression():
    owners = (
        ROOT / ".github/workflows/validate.yml",
        ROOT / "scripts/validate.sh",
        ROOT / "scripts/setup_ubuntu.sh",
        ROOT / "python/tessera/compiler/tests_manifest.py",
    )
    for owner in owners:
        text = owner.read_text(encoding="utf-8")
        for term in PR_MARKER_EXPRESSION.split(" and "):
            assert term in text, f"{owner.relative_to(ROOT)} lost PR state {term!r}"


def test_marker_policy_is_registered_in_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for marker in MARKERS:
        assert f'"{marker}:' in text


def test_marker_bearing_modules_import_pytest_explicitly():
    violations = []
    for path in UNIT.glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        if "pytest.mark." not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        imports_pytest = any(
            isinstance(node, ast.Import)
            and any(alias.name == "pytest" for alias in node.names)
            for node in tree.body
        )
        if not imports_pytest:
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, "marker-bearing modules must import pytest: " + ", ".join(
        violations
    )


def test_live_nvidia_tests_declare_the_cuda_hardware_state():
    violations = []
    for path in UNIT.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_marked = "pytestmark = pytest.mark.hardware_nvidia" in path.read_text(
            encoding="utf-8"
        )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_live_nvidia"):
                continue
            marked = "pytest.mark.hardware_nvidia" in _decorator_names(node)
            if not (marked or module_marked):
                violations.append(f"{path.relative_to(ROOT)}::{node.name}")
    assert not violations, (
        "live CUDA tests require @pytest.mark.hardware_nvidia: "
        + ", ".join(violations)
    )


def test_compiler_toolchain_missing_state_is_a_canonical_skip():
    tools = CompilerToolchain(tessera_opt=None, mlir_opt=None)
    with pytest.raises(pytest.skip.Exception, match="requires tessera-opt"):
        tools.require_tessera_opt()
    with pytest.raises(pytest.skip.Exception, match="requires MLIR 23 mlir-opt"):
        tools.require_mlir_opt()
    with pytest.raises(pytest.skip.Exception, match="requires tessera-nvidia-opt"):
        tools.require_nvidia_opt()


def test_python_subprocess_environment_preserves_and_prepends_import_roots(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    env = python_subprocess_environment({"TESSERA_TEST_SENTINEL": "1"})
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [str(PYTHON_ROOT), str(ROOT)]
    assert entries[-1] == "/existing/path"
    assert env["TESSERA_TEST_SENTINEL"] == "1"


def test_migrated_compiler_fixture_has_no_private_tool_probe():
    text = (UNIT / "test_rocm_arch_fragment_compiler.py").read_text(
        encoding="utf-8"
    )
    assert "pytest.mark.compiler_tool" in text
    assert 'REPO / "build/tools/tessera-opt/tessera-opt"' not in text
    assert "compiler_toolchain.require_tessera_opt()" in text


@pytest.mark.parametrize(
    "filename",
    ["test_nvidia_plugin.py", "test_rocm_plugin.py", "test_x86_plugin.py"],
)
def test_migrated_child_process_guards_use_shared_import_state(filename):
    path = (
        ROOT / "tests/_support/nvidia_plugin_cases.py"
        if filename == "test_nvidia_plugin.py"
        else UNIT / filename
    )
    text = path.read_text(encoding="utf-8")
    assert "python_subprocess_env" in text
    assert "env=python_subprocess_env" in text
