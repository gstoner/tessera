"""IRContractLegalityPass — dtype / aliasing / buffer-binding contracts (#13).

LayoutLegalityPass's sibling for the three remaining contract families in
COMPILER_AUDIT's "Layout and binding contracts are uneven" item. dtype rules
enforce CANONICAL_API Decision #15a (storage/accum coupling; TF32 is a math_mode,
not a storage dtype). Lit coverage: tests/tessera-ir/phase2/ir_contract_legality.mlir.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PASS_SRC = REPO / "src" / "transforms" / "lib" / "IRContractLegalityPass.cpp"
_CANDIDATES = (
    REPO / "build" / "tools" / "tessera-opt" / "tessera-opt",
    REPO / "build-llvm23" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_opt():
    if explicit := os.environ.get("TESSERA_OPT_PATH"):
        if Path(explicit).is_file():
            return explicit
    for c in _CANDIDATES:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
_needs_opt = pytest.mark.skipif(_OPT is None, reason="tessera-opt not built")


def _run(mlir: str, tmp_path):
    f = tmp_path / "c.mlir"
    f.write_text(mlir)
    return subprocess.run(
        [_OPT, str(f), "-tessera-ir-contracts", "--allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60)


# ── source-level guards (run even without the binary) ────────────────────────


def test_pass_source_covers_three_contract_families():
    src = PASS_SRC.read_text()
    # dtype (Decision #15a)
    assert "DTYPE_LEGALITY_TF32_AS_STORAGE" in src
    assert "DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM" in src
    assert "DTYPE_LEGALITY_UNKNOWN_STORAGE" in src
    # aliasing
    assert "ALIAS_LEGALITY_MISSING_ALIASES" in src
    assert "ALIAS_LEGALITY_OPERAND_OOB" in src
    # buffer-binding
    assert "BUFFER_BINDING_UNKNOWN_ROLE" in src
    assert "BUFFER_BINDING_CONFLICT" in src


def test_pass_registered_in_passes_cpp():
    reg = (REPO / "src" / "transforms" / "lib" / "Passes.cpp").read_text()
    assert "createIRContractLegalityPass" in reg
    cmake = (REPO / "src" / "transforms" / "lib" / "CMakeLists.txt").read_text()
    assert "IRContractLegalityPass.cpp" in cmake


# ── dtype contract (Decision #15a) ────────────────────────────────────────────

_MM = ('"tessera.matmul"(%a, %b) {{numeric_policy = {{{policy}}}}} '
       ': (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>')


def _matmul_fn(policy: str) -> str:
    return (f"func.func @f(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> "
            f"tensor<8x8xf32> {{\n  %0 = " + _MM.format(policy=policy) +
            "\n  return %0 : tensor<8x8xf32>\n}\n")


@_needs_opt
def test_tf32_as_storage_rejected(tmp_path):
    out = _run(_matmul_fn('storage = "tf32", accum = "fp32"'), tmp_path)
    assert out.returncode != 0
    assert "DTYPE_LEGALITY_TF32_AS_STORAGE" in out.stderr


@_needs_opt
def test_lowp_without_accum_rejected(tmp_path):
    out = _run(_matmul_fn('storage = "fp8_e4m3"'), tmp_path)
    assert "DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM" in out.stderr


@_needs_opt
def test_lowp_narrow_accum_rejected(tmp_path):
    out = _run(_matmul_fn('storage = "int4", accum = "int8"'), tmp_path)
    assert "DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM" in out.stderr


@_needs_opt
def test_unknown_storage_rejected(tmp_path):
    out = _run(_matmul_fn('storage = "float7", accum = "fp32"'), tmp_path)
    assert "DTYPE_LEGALITY_UNKNOWN_STORAGE" in out.stderr


@_needs_opt
def test_valid_fp8_policy_passes(tmp_path):
    out = _run(_matmul_fn('storage = "fp8_e4m3", accum = "fp32"'), tmp_path)
    assert out.returncode == 0, out.stderr


@_needs_opt
def test_valid_bf16_policy_passes(tmp_path):
    out = _run(_matmul_fn('storage = "bf16", accum = "fp32"'), tmp_path)
    assert out.returncode == 0, out.stderr


# ── aliasing contract ─────────────────────────────────────────────────────────


@_needs_opt
def test_inplace_without_aliases_rejected(tmp_path):
    fn = ('func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> {\n'
          '  %0 = "tessera.relu"(%a) {tessera.inplace = true} '
          ': (tensor<8xf32>) -> tensor<8xf32>\n  return %0 : tensor<8xf32>\n}\n')
    out = _run(fn, tmp_path)
    assert "ALIAS_LEGALITY_MISSING_ALIASES" in out.stderr


@_needs_opt
def test_aliases_in_range_passes(tmp_path):
    fn = ('func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> {\n'
          '  %0 = "tessera.relu"(%a) {tessera.inplace = true, tessera.aliases = 0 : i64} '
          ': (tensor<8xf32>) -> tensor<8xf32>\n  return %0 : tensor<8xf32>\n}\n')
    out = _run(fn, tmp_path)
    assert out.returncode == 0, out.stderr


# ── buffer-binding contract ───────────────────────────────────────────────────


@_needs_opt
def test_unknown_buffer_role_rejected(tmp_path):
    fn = ('func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> {\n'
          '  %0 = "tessera.relu"(%a) {tessera.buffer_role = "bogus"} '
          ': (tensor<8xf32>) -> tensor<8xf32>\n  return %0 : tensor<8xf32>\n}\n')
    out = _run(fn, tmp_path)
    assert "BUFFER_BINDING_UNKNOWN_ROLE" in out.stderr


@_needs_opt
def test_binding_role_conflict_rejected(tmp_path):
    fn = ('func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> {\n'
          '  %0 = "tessera.relu"(%a) {tessera.binding = "buf0", tessera.buffer_role = "input"} '
          ': (tensor<8xf32>) -> tensor<8xf32>\n'
          '  %1 = "tessera.relu"(%0) {tessera.binding = "buf0", tessera.buffer_role = "scratch"} '
          ': (tensor<8xf32>) -> tensor<8xf32>\n  return %1 : tensor<8xf32>\n}\n')
    out = _run(fn, tmp_path)
    assert "BUFFER_BINDING_CONFLICT" in out.stderr
