"""Configured native LLVM tools must outrank unrelated vendor fallbacks."""
from pathlib import Path

import pytest

from tests._support import compiler_tool, environment


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    for name in ("MLIR_OPT", "TESSERA_LLVM_BIN", "LLVM_BIN"):
        monkeypatch.delenv(name, raising=False)
    fallback = tmp_path / "vendor" / "mlir-opt"
    fallback.parent.mkdir()
    fallback.touch()
    original = environment._tool_path

    def resolve(name, *candidates):
        if name != "MLIR_OPT":
            return None
        # Make the legacy host fallback deterministic on every architecture.
        return original(name, fallback)

    monkeypatch.setattr(environment, "_tool_path", resolve)
    monkeypatch.setattr(compiler_tool, "tessera_opt_path", lambda *args: None)
    monkeypatch.setattr(compiler_tool, "is_runnable", lambda path: True)
    return environment.CompilerToolchain.discover, fallback


def executable(root: Path) -> Path:
    root.mkdir()
    path = root / "mlir-opt"
    path.touch()
    return path


@pytest.mark.parametrize("name", ["TESSERA_LLVM_BIN", "LLVM_BIN"])
def test_configured_llvm_directory_precedes_vendor_fallback(discovery, tmp_path, monkeypatch, name):
    discover, fallback = discovery
    selected = executable(tmp_path / "matching")
    monkeypatch.setenv(name, str(selected.parent))
    assert discover().mlir_opt == selected
    assert selected != fallback


def test_tessera_directory_precedes_generic_llvm_directory(discovery, tmp_path, monkeypatch):
    discover, _ = discovery
    selected = executable(tmp_path / "matching")
    other = executable(tmp_path / "other")
    monkeypatch.setenv("TESSERA_LLVM_BIN", str(selected.parent))
    monkeypatch.setenv("LLVM_BIN", str(other.parent))
    assert discover().mlir_opt == selected


@pytest.mark.parametrize("present", [False, True])
def test_explicit_mlir_selector_remains_final(discovery, tmp_path, monkeypatch, present):
    discover, _ = discovery
    selected = executable(tmp_path / "matching")
    explicit = executable(tmp_path / "explicit") if present else tmp_path / "missing"
    monkeypatch.setenv("TESSERA_LLVM_BIN", str(selected.parent))
    monkeypatch.setenv("MLIR_OPT", str(explicit))
    assert discover().mlir_opt == (explicit if present else None)


@pytest.mark.parametrize("present", [False, True])
def test_broken_configured_directory_cannot_fall_back_silently(discovery, tmp_path, monkeypatch, present):
    discover, _ = discovery
    selected = executable(tmp_path / "broken") if present else tmp_path / "missing" / "mlir-opt"
    monkeypatch.setenv("TESSERA_LLVM_BIN", str(selected.parent))
    if present:
        monkeypatch.setattr(compiler_tool, "is_runnable", lambda path: path != selected)
    assert discover().mlir_opt is None


def test_without_selectors_existing_fallback_remains(discovery):
    discover, fallback = discovery
    assert discover().mlir_opt == fallback
