"""Static drift gates for Tessera's single LLVM/MLIR 23 build backbone."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Build entry points and environment documentation must never suggest that a
# different LLVM/MLIR major is a supported fallback.
BUILD_ENVIRONMENT_FILES = (
    REPO_ROOT / "CMakeLists.txt",
    REPO_ROOT / "scripts" / "build.sh",
    REPO_ROOT / "scripts" / "run_sanitizers.sh",
    REPO_ROOT / "scripts" / "install_test_deps.sh",
    REPO_ROOT / ".github" / "workflows" / "validate.yml",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "CLAUDE.md",
)
NON_23_TOOLCHAIN = re.compile(
    r"(?:LLVM(?:/MLIR)?|MLIR|llvm|mlir)(?:[-/@ ]+)(?:1\d|2[0-24-9])(?:\D|$)"
)


def test_build_environment_names_only_llvm_mlir_23() -> None:
    violations: list[str] = []
    for path in BUILD_ENVIRONMENT_FILES:
        text = path.read_text(encoding="utf-8")
        if match := NON_23_TOOLCHAIN.search(text):
            violations.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)!r}")
    assert not violations, "non-23 LLVM/MLIR build references:\n" + "\n".join(violations)


def test_every_standalone_cmake_package_gate_requires_major_23() -> None:
    violations: list[str] = []
    for path in REPO_ROOT.rglob("CMakeLists.txt"):
        parts = path.relative_to(REPO_ROOT).parts
        if "archive" in parts or any(part.startswith("build") for part in parts):
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*find_package\(LLVM", text, re.MULTILINE) and (
            "LLVM_VERSION_MAJOR EQUAL 23" not in text
        ):
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing exact LLVM 23 gate")
        if re.search(r"^\s*find_package\(MLIR", text, re.MULTILINE) and (
            "MLIR_VERSION_MAJOR EQUAL 23" not in text
        ):
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing exact MLIR 23 gate")
    assert not violations, "\n".join(violations)


def test_canonical_build_discovery_has_no_cross_major_fallback() -> None:
    text = (REPO_ROOT / "scripts" / "build.sh").read_text(encoding="utf-8")
    assert "/usr/lib/llvm-23" in text
    assert "brew --prefix llvm@23" in text
    assert "/usr/lib/llvm-24" not in text
    assert "/usr/lib/llvm-22" not in text
    assert "brew --prefix llvm 2>" not in text


def test_linux_tsan_uses_llvm23_clang_non_pie_runtime() -> None:
    text = (REPO_ROOT / "scripts" / "run_sanitizers.sh").read_text(encoding="utf-8")
    assert 'llvm_prefix="/usr/lib/llvm-23"' in text
    assert '"$label" == "tsan"' in text
    assert "-DCMAKE_CXX_COMPILER=\"$llvm_prefix/bin/clang++\"" in text
    assert "-DCMAKE_CXX_FLAGS=-fno-pie" in text
    assert "-DCMAKE_EXE_LINKER_FLAGS=-no-pie" in text
    assert "cmake_fresh=(--fresh)" in text
