"""Sub-1 — Graph IR op for ``tessera.attn_local_window_2d``.

Structural + behavioral guards for the new ODS op:
* ODS defines the op in TesseraOps.td with rank-5 operands + window=[rh, rw].
* Verifier in TesseraOps.cpp enforces rank-5 + matching dtypes + spatial
  agreement + non-negative window.
* Round-trip lit fixture proves parse/print/parse stability.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_TD = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.td"
OPS_CPP = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.cpp"
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7"
    / "attn_local_window_2d_graph_ir.mlir"
)


# --------------------------------------------------------------------------- #
# ODS / verifier source
# --------------------------------------------------------------------------- #


def test_ods_op_defined() -> None:
    text = OPS_TD.read_text()
    assert "Tessera_AttnLocalWindow2DOp" in text
    assert '"attn_local_window_2d"' in text
    # ODS records the window attribute as an I64ArrayAttr.
    assert "I64ArrayAttr:$window" in text


def test_ods_op_has_verifier() -> None:
    text = OPS_TD.read_text()
    assert "Tessera_AttnLocalWindow2DOp" in text
    # The verifier is declared in ODS via `let hasVerifier = 1;`
    # — confirm by locating the block.
    block_start = text.find("Tessera_AttnLocalWindow2DOp")
    block_end = text.find("def Tessera_DeepSeekSparseAttentionOp",
                          block_start)
    assert block_start != -1 and block_end != -1
    block = text[block_start:block_end]
    assert "hasVerifier = 1" in block


def test_verifier_enforces_rank5_and_matching_dtypes() -> None:
    text = OPS_CPP.read_text()
    assert "AttnLocalWindow2DOp::verify" in text
    # Six checks the verifier performs:
    assert "rank-5" in text
    assert "element types must match" in text
    assert "window must be [rh, rw]" in text
    assert "window half-widths must be non-negative" in text
    assert "K and V must agree on spatial axis" in text
    assert "Q and K must share spatial layout on axis" in text


def test_lit_fixture_has_three_dtype_variants() -> None:
    text = LIT_FIXTURE.read_text()
    assert "local_window_2d_static" in text
    assert "local_window_2d_asymmetric_window" in text
    assert "local_window_2d_bf16" in text
    # The fixture has two RUN lines — one bare parse, one parse|print|parse.
    assert text.count("RUN:") >= 2


# --------------------------------------------------------------------------- #
# Behavioral — round-trip
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for candidate in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
        str(REPO_ROOT / "build" / "bin" / "tessera-opt"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def test_roundtrip_through_tessera_opt() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built — skipping behavioral contract")
    # First pass — must parse and print.
    r1 = subprocess.run([binary, str(LIT_FIXTURE)],
                         capture_output=True, text=True, timeout=30)
    if (
        r1.returncode != 0
        and "Did you mean" in r1.stderr
        and "attn_local_window_2d" in r1.stderr
    ):
        pytest.skip("tessera-opt predates the new op — rebuild required")
    assert r1.returncode == 0, f"first parse failed: {r1.stderr}"
    # Second pass — feed the printed IR back in.
    r2 = subprocess.run([binary], input=r1.stdout,
                         capture_output=True, text=True, timeout=30)
    assert r2.returncode == 0, f"roundtrip failed: {r2.stderr}"
    # Stable output: should contain the op three times.
    assert r2.stdout.count("tessera.attn_local_window_2d") == 3


_VERIFIER_INPUT_BAD_RANK = """\
func.func @bad_rank(%q: tensor<2x4x8x16xf32>, %k: tensor<2x4x8x16xf32>, %v: tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32>
  return %o : tensor<2x4x8x16xf32>
}
"""


def test_verifier_rejects_rank4_input() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built")
    r = subprocess.run([binary], input=_VERIFIER_INPUT_BAD_RANK,
                         capture_output=True, text=True, timeout=30)
    if (
        r.returncode != 0
        and "Did you mean" in r.stderr
        and "attn_local_window_2d" in r.stderr
    ):
        pytest.skip("tessera-opt predates the new op — rebuild required")
    # Bad rank must fail with a named error.
    assert r.returncode != 0
    assert "rank-5" in r.stderr
