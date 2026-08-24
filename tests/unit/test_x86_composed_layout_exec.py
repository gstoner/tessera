"""Exact CPU proof for the x86 composed-layout Target consumer."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/tessera-ir/phase2/x86_composed_layout_exec.mlir"
LLVM_BIN = Path("/usr/lib/llvm-23/bin")


def _tool(name: str, *, repo_candidate: Path | None = None) -> str:
    if repo_candidate is not None and repo_candidate.is_file():
        return str(repo_candidate)
    resolved = shutil.which(name)
    if resolved is None:
        pytest.skip(f"{name} is required for x86 composed-layout execution")
    return resolved


def _execute(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    build_dir = Path(os.environ.get("TESSERA_BUILD_DIR", ROOT / "build"))
    tessera_opt = _tool(
        "tessera-opt", repo_candidate=build_dir / "tools/tessera-opt/tessera-opt"
    )
    mlir_opt = _tool("mlir-opt", repo_candidate=LLVM_BIN / "mlir-opt")
    mlir_translate = _tool(
        "mlir-translate", repo_candidate=LLVM_BIN / "mlir-translate"
    )
    lli = _tool("lli", repo_candidate=LLVM_BIN / "lli")

    input_path = tmp_path / "input.mlir"
    lowered_path = tmp_path / "lowered.mlir"
    llvm_mlir_path = tmp_path / "llvm.mlir"
    llvm_ir_path = tmp_path / "layout.ll"
    input_path.write_text(source, encoding="utf-8")

    subprocess.run(
        [
            tessera_opt,
            "--tessera-tile-to-x86=architecture=base prefer-amx=false",
            str(input_path),
            "-o",
            str(lowered_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            mlir_opt,
            str(lowered_path),
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(llvm_mlir_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            mlir_translate,
            "--mlir-to-llvmir",
            str(llvm_mlir_path),
            "-o",
            str(llvm_ir_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return subprocess.run([lli, str(llvm_ir_path)], text=True, capture_output=True)


def test_x86_composed_layout_executes_exact_dynamic_nested_and_tuple_maps(
    tmp_path: Path,
) -> None:
    result = _execute(FIXTURE.read_text(encoding="utf-8"), tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("%row = arith.constant 3 : i64", "%row = arith.constant -1 : i64"),
        ("%row = arith.constant 3 : i64", "%row = arith.constant 18 : i64"),
        ("%m = arith.constant 17 : i64", "%m = arith.constant 0 : i64"),
        ("%lda = arith.constant 29 : i64", "%lda = arith.constant -1 : i64"),
    ],
)
def test_x86_composed_layout_runtime_guards_fail_closed(
    tmp_path: Path, old: str, new: str
) -> None:
    source = FIXTURE.read_text(encoding="utf-8")
    assert old in source
    result = _execute(source.replace(old, new), tmp_path)
    assert result.returncode != 0
