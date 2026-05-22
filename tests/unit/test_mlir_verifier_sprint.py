"""MLIR Verifier Sprint V1+V2+V3 (2026-05-22) — structural guard.

The verifier sprint adds C++ ODS verifiers + a new LayoutLegalityPass +
a target-aware FlashAttnOp extension.  This file pins the source-level
content so a future edit that accidentally removes a `let hasVerifier
= 1;` clause or a `verify()` body fails this test rather than waiting
for the next rebuild.

It does NOT re-run tessera-opt — that's covered by the lit fixtures
under `tests/tessera-ir/phase{2,3}/sprint_v*.mlir`.  The Python guard
exists because the C++ build is a slow signal; the structural check is
fast and runs on every `pytest tests/unit` invocation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_TD = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.td"
OPS_CPP = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.cpp"
PASSES_H = (
    REPO_ROOT / "src" / "transforms" / "include" / "Tessera"
    / "Transforms" / "Passes.h"
)
PASSES_CPP = REPO_ROOT / "src" / "transforms" / "lib" / "Passes.cpp"
LAYOUT_PASS = REPO_ROOT / "src" / "transforms" / "lib" / "LayoutLegalityPass.cpp"
PASSES_CMAKE = REPO_ROOT / "src" / "transforms" / "lib" / "CMakeLists.txt"
V1_LIT = REPO_ROOT / "tests" / "tessera-ir" / "phase2" / "sprint_v1_verifiers.mlir"
V2_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v2_layout_legality.mlir"
)
V3_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase3"
    / "sprint_v3_flash_attn_target_aware.mlir"
)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V1 — 3 new hasVerifier ops
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "op_def_marker",
    [
        # TransposeOp
        ('def Tessera_TransposeOp', 'transpose'),
        # LayerNormOp
        ('def Tessera_LayerNormOp', 'layer_norm'),
        # MoeDispatchOp
        ('def Tessera_MoeDispatchOp', 'moe_dispatch'),
        # Sprint V6a — ReshapeOp registered as ODS op
        ('def Tessera_ReshapeOp', 'reshape'),
    ],
)
def test_sprint_v1_op_has_hasverifier(op_def_marker) -> None:
    """Each of the 3 Sprint V1 ops must declare `let hasVerifier = 1;`
    in TesseraOps.td.  If this regresses, the ODS verifier is no
    longer generated."""
    op_def, _name = op_def_marker
    td = OPS_TD.read_text()
    idx = td.find(op_def)
    assert idx >= 0, f"op definition not found: {op_def}"
    # Look at the next 600 chars (size of one op block) for the marker.
    window = td[idx: idx + 800]
    assert "let hasVerifier = 1;" in window, (
        f"{op_def} is missing `let hasVerifier = 1;` — Sprint V1 "
        f"closure regressed.  Open SHAPE_SYSTEM.md §11 and "
        f"docs/audit/compiler_spec_gap_audit.md."
    )


_V1_VERIFIER_SIGNATURES = (
    "LogicalResult TransposeOp::verify()",
    "LogicalResult LayerNormOp::verify()",
    "LogicalResult MoeDispatchOp::verify()",
    # Sprint V6a (2026-05-22) — tessera.reshape registered as ODS op
    # so V5's SymbolicDimEqualityPass can exercise the reshape branch
    # end-to-end without --allow-unregistered-dialect.
    "LogicalResult ReshapeOp::verify()",
)


@pytest.mark.parametrize("sig", _V1_VERIFIER_SIGNATURES)
def test_sprint_v1_verifier_impl_present(sig: str) -> None:
    """Each Sprint V1 op must have a verify() body in TesseraOps.cpp."""
    cpp = OPS_CPP.read_text()
    assert sig in cpp, (
        f"verifier impl missing: {sig!r} — Sprint V1 closure regressed"
    )


_V1_DIAGNOSTIC_PHRASES = (
    # TransposeOp
    "transpose must preserve rank",
    "transpose must preserve element type",
    "output static dims must be a permutation",
    # LayerNormOp
    "layer_norm must preserve rank",
    "layer_norm must preserve dim",
    "eps must be positive for stable rsqrt",
    # MoeDispatchOp
    "token count mismatch",
    # Sprint V6a — ReshapeOp
    "reshape must preserve element type",
    "reshape must preserve element count",
)


@pytest.mark.parametrize("phrase", _V1_DIAGNOSTIC_PHRASES)
def test_sprint_v1_diagnostic_phrases_present(phrase: str) -> None:
    """Each Sprint V1 verifier emits a specific diagnostic phrase that
    the lit fixture relies on via `expected-error`.  Pin the phrase
    here so a refactor that softens the wording fails."""
    cpp = OPS_CPP.read_text()
    assert phrase in cpp, (
        f"diagnostic phrase missing: {phrase!r} — lit fixture "
        f"`expected-error` will silently miss after this regression"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V2 — LayoutLegalityPass
# ─────────────────────────────────────────────────────────────────────────────


def test_sprint_v2_layout_pass_file_exists() -> None:
    assert LAYOUT_PASS.exists(), (
        "src/transforms/lib/LayoutLegalityPass.cpp missing — Sprint V2 "
        "closure regressed"
    )


def test_sprint_v2_layout_pass_canonical_layouts() -> None:
    """The pass must list the canonical layout accept-set documented
    in SHAPE_SYSTEM.md §2.1."""
    text = LAYOUT_PASS.read_text()
    for layout in (
        "row_major", "col_major", "nhwc", "nchw",
        "bhsd", "tile", "bsr", "packed",
    ):
        assert f'"{layout}"' in text, (
            f"LayoutLegalityPass.cpp missing canonical layout name "
            f"{layout!r}"
        )


def test_sprint_v2_layout_pass_diagnostic_code() -> None:
    """The stable diagnostic code that lit fixtures match on."""
    text = LAYOUT_PASS.read_text()
    assert "LAYOUT_LEGALITY_UNKNOWN_LAYOUT" in text, (
        "LayoutLegalityPass.cpp must emit LAYOUT_LEGALITY_UNKNOWN_LAYOUT "
        "as the stable diagnostic code for SHAPE_SYSTEM.md §11 "
        "cross-linking"
    )


def test_sprint_v2_pass_registered_in_passes_cpp() -> None:
    """The pass must be wired through `registerTesseraPasses` so
    `tessera-opt --tessera-layout-legality` resolves."""
    cpp = PASSES_CPP.read_text()
    assert "createLayoutLegalityPass()" in cpp, (
        "Passes.cpp must call createLayoutLegalityPass() inside "
        "registerTesseraPasses()"
    )


def test_sprint_v2_pass_declared_in_passes_h() -> None:
    """Prototype must be in the header so the registration call type-checks."""
    h = PASSES_H.read_text()
    assert "createLayoutLegalityPass" in h, (
        "Passes.h must declare createLayoutLegalityPass()"
    )


def test_sprint_v2_pass_in_cmake_sources() -> None:
    """Must be in the TesseraPasses target's source list."""
    cmake = PASSES_CMAKE.read_text()
    assert "LayoutLegalityPass.cpp" in cmake, (
        "src/transforms/lib/CMakeLists.txt must include "
        "LayoutLegalityPass.cpp in the TesseraPasses target"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V3 — target-aware FlashAttnOp verifier extension
# ─────────────────────────────────────────────────────────────────────────────


def test_sprint_v3_per_sm_head_dim_table_present() -> None:
    """The per-SM head_dim ceiling table must be in TesseraOps.cpp."""
    cpp = OPS_CPP.read_text()
    assert "maxHeadDimForTargetSm" in cpp, (
        "FlashAttnOp::verify() must consult maxHeadDimForTargetSm() — "
        "Sprint V3 closure regressed"
    )
    # The 6 SM variants from Sprint G-1's CUDA 13.2 U1 matrix.
    for sm in ("sm_80", "sm_90", "sm_90a", "sm_100", "sm_120", "sm_120a"):
        assert f'"{sm}"' in cpp, (
            f"head_dim table is missing SM variant {sm!r} — keep "
            f"docs/nvidia_cuda13_kernel_inventory.md in sync"
        )


def test_sprint_v3_flash_attn_consults_parent_target_sm() -> None:
    """The verifier must walk parents to find tessera.target_sm."""
    cpp = OPS_CPP.read_text()
    assert "tessera.target_sm" in cpp, (
        "FlashAttnOp::verify() must consult the parent function's "
        "tessera.target_sm attribute"
    )
    assert "exceeds the SM" in cpp, (
        "diagnostic phrase 'exceeds the SM' must be in FlashAttnOp "
        "verifier — lit fixture relies on it"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V6c (2026-05-22) — target-aware verifier on ScaledDotProductOp
# (FA-4 Tile IR).  The verifier code lives in a different file (FA-4
# Tile IR has its own dialect lib) so pin its source explicitly.
# ─────────────────────────────────────────────────────────────────────────────

_ATTN_OPS_CPP = (
    REPO_ROOT / "src" / "compiler" / "tile_opt_fa4" / "lib" / "Dialect"
    / "Attn" / "AttnOps.cpp"
)
_V6C_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase3"
    / "sprint_v6c_scaled_dot_product_target_aware.mlir"
)


def test_sprint_v6c_scaled_dot_product_tile_size_table_present() -> None:
    """The per-SM (tile_q, tile_kv) table must be in AttnOps.cpp."""
    cpp = _ATTN_OPS_CPP.read_text()
    assert "maxTileSizesForTargetSm" in cpp, (
        "ScaledDotProductOp::verify() must consult "
        "maxTileSizesForTargetSm() — Sprint V6c closure regressed"
    )
    for sm in ("sm_80", "sm_90", "sm_90a", "sm_100", "sm_120", "sm_120a"):
        assert f'"{sm}"' in cpp, (
            f"V6c tile-size table is missing SM variant {sm!r}"
        )


def test_sprint_v6c_diagnostic_phrases_present() -> None:
    """Stable diagnostic phrases V6c lit fixture matches on."""
    cpp = _ATTN_OPS_CPP.read_text()
    for phrase in (
        "tile_q=",
        "tile_kv=",
        "ScaledDotProduct kernel limit of",
        "Sprint V6c FA-4 tile size table",
    ):
        assert phrase in cpp, (
            f"V6c diagnostic phrase missing: {phrase!r}"
        )


def test_sprint_v6c_scaled_dot_product_consults_parent_target_sm() -> None:
    """Mirrors Sprint V3 FlashAttnOp pattern — walk parents for
    tessera.target_sm."""
    cpp = _ATTN_OPS_CPP.read_text()
    assert "tessera.target_sm" in cpp, (
        "ScaledDotProductOp::verify() must consult parent's "
        "tessera.target_sm attribute (Sprint V6c)"
    )


def test_sprint_v6c_lit_fixture_deferred_to_v7() -> None:
    """V6c verifier code is implemented + compiled, but its end-to-end
    lit exercise is deferred to Sprint V7 because the `tessera.attn`
    dialect is not yet registered in tessera-opt (see the two existing
    XFAIL'd scaled_dot_product fixtures: flash_attn_full.mlir,
    tile_ir_lowering.mlir).

    This test confirms NO orphan V6c lit fixture exists.  When V7
    registers the Attn dialect AND adds a V6c lit fixture, this test
    will need to be updated to assert the new fixture's content."""
    orphan_path = (
        REPO_ROOT / "tests" / "tessera-ir" / "phase3"
        / "sprint_v6c_scaled_dot_product_target_aware.mlir"
    )
    if orphan_path.exists():
        text = orphan_path.read_text()
        # Allow the fixture to exist ONLY if it's marked as
        # explicitly deferred (REQUIRES feature gate that lit.cfg
        # doesn't yet define) — otherwise it would cause lit
        # `Unresolved` results.
        assert "REQUIRES: tessera-attn-dialect-registered" in text, (
            f"V6c lit fixture {orphan_path.relative_to(REPO_ROOT)} "
            f"must either be deleted OR carry "
            f"`REQUIRES: tessera-attn-dialect-registered` so lit "
            f"skips it cleanly (the dialect isn't registered in "
            f"tessera-opt until Sprint V7 lands)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lit fixture presence (the C++ build verifies behaviour; lit existence
# is the structural guard)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [V1_LIT, V2_LIT, V3_LIT])
def test_sprint_lit_fixtures_present(path: Path) -> None:
    assert path.exists(), (
        f"lit fixture missing: {path.relative_to(REPO_ROOT)} — "
        f"Sprint V{path.stem.split('_')[1]} closure regressed"
    )
    text = path.read_text()
    assert "RUN: tessera-opt" in text, (
        f"{path.name} must start with `// RUN: tessera-opt ...`"
    )
    assert "expected-error" in text, (
        f"{path.name} must include at least one `expected-error` "
        "negative case so the verifier is actually exercised"
    )
