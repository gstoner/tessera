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
V3A_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v3a_affine_bindings.mlir"
)
V4A_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v4a_layout_producer_consumer.mlir"
)
V4B_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v4b_per_op_verifiers.mlir"
)
SYMDIM_PASS = (
    REPO_ROOT / "src" / "transforms" / "lib" / "SymbolicDimEqualityPass.cpp"
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


def test_sprint_v6c_lit_fixture_runs_via_v7b() -> None:
    """Sprint V7b (2026-05-22) added a DialectExtension on
    `TesseraAttnDialect` that eager-loads the dialect when the
    `tessera` Graph IR dialect attaches to a context.  This unblocked
    the V6c lit fixture, which previously could only run inside a
    pipeline that referenced the dialect.

    The V6c fixture now exists as a normal lit fixture (no XFAIL /
    REQUIRES gate) and passes end-to-end through the real C++
    binary.  Pin its content here so a regression on V7b's
    extension mechanism — which would re-break the parser lookup —
    fails this fast Python sweep."""
    v6c = (
        REPO_ROOT / "tests" / "tessera-ir" / "phase3"
        / "sprint_v6c_scaled_dot_product_target_aware.mlir"
    )
    assert v6c.exists(), (
        f"V6c lit fixture missing: {v6c.relative_to(REPO_ROOT)} "
        "(V7b unblocked it; recreate the file if a refactor "
        "deleted it)."
    )
    text = v6c.read_text()
    assert "XFAIL" not in text, (
        "V6c lit fixture must NOT carry XFAIL — V7b's eager-load "
        "extension makes it runnable end-to-end."
    )
    assert "REQUIRES:" not in text, (
        "V6c lit fixture must NOT carry a REQUIRES gate — V7b "
        "makes it runnable on the default lit setup."
    )
    assert "Sprint V7b" in text, (
        "V6c lit fixture should reference Sprint V7b in its header "
        "comment so the future reader knows what unblocked it."
    )
    # 2 @+1-anchored negative cases (tile_q + tile_kv overflows).
    expected_errors = text.count("// expected-error @+1")
    assert expected_errors == 2, (
        f"V6c lit fixture must have 2 @+1-anchored expected-error "
        f"directives; got {expected_errors}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V3a — affine non-product (sum-of-products) bindings
# ─────────────────────────────────────────────────────────────────────────────


def test_sprint_v3a_pass_supports_sum_of_products() -> None:
    """V3a extended SymbolicDimEqualityPass's binding parser to accept
    `D = H * Dh + K` (sum-of-products) instead of only `D = H * Dh`.
    The pass body must hold a sum-vector member + a per-term parser."""
    cpp = SYMDIM_PASS.read_text()
    # Sentinel: the Binding struct holds a vector of product-term
    # vectors (sum-of-products), not a single flat product vector.
    assert "SmallVector<SmallVector<" in cpp or "vector<vector<" in cpp.lower(), (
        "V3a regression: Binding must carry sum-of-products terms"
    )
    # Sentinel: the parser splits on '+' (sum) and '*' (product).
    assert "splitOn" in cpp, (
        "V3a regression: SymbolicDimEqualityPass must use splitOn() "
        "helper to break sum-of-products bindings"
    )
    # Sentinel: V3a diagnostic split — multi-term renders as
    # "value of RHS (sum of products) = ..." while V5 single-term
    # keeps "product of RHS = ..." for backward compat.
    assert "sum of products" in cpp, (
        "V3a regression: multi-term diagnostic phrase missing"
    )
    assert "product of RHS" in cpp, (
        "V5 regression: single-product diagnostic phrase removed"
    )


def test_sprint_v3a_lit_fixture_shape() -> None:
    """The V3a lit fixture pins the diagnostic wording and covers both
    multi-term and single-term (backward-compat) paths."""
    assert V3A_LIT.exists(), "V3a lit fixture missing"
    text = V3A_LIT.read_text()
    assert "value of RHS (sum of products)" in text, (
        "V3a lit fixture must lock the multi-term diagnostic wording"
    )
    # 1 negative case (sum binding violated).
    assert text.count("// expected-error @+1") == 1, (
        "V3a lit fixture should have exactly 1 expected-error case"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V4a — LayoutLegality producer/consumer rule
# ─────────────────────────────────────────────────────────────────────────────


def test_sprint_v4a_layout_pass_has_producer_consumer_rule() -> None:
    """V4a extended LayoutLegalityPass with a producer→matmul accept-set
    check on the def-use chain.  The pass body must hold a matmul
    accept-set + a `checkMatmulOperandLayouts` walker."""
    cpp = LAYOUT_PASS.read_text()
    assert "matmulAcceptSet" in cpp, (
        "V4a regression: LayoutLegalityPass must define matmulAcceptSet()"
    )
    assert "checkMatmulOperandLayouts" in cpp, (
        "V4a regression: LayoutLegalityPass must define "
        "checkMatmulOperandLayouts() walker"
    )
    assert "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH" in cpp, (
        "V4a regression: stable diagnostic code missing"
    )
    # Sentinel: accept-set is {row_major, col_major} for matmul.
    assert '"row_major"' in cpp and '"col_major"' in cpp, (
        "V4a regression: matmul accept-set must include row_major/col_major"
    )


def test_sprint_v4a_lit_fixture_shape() -> None:
    assert V4A_LIT.exists(), "V4a lit fixture missing"
    text = V4A_LIT.read_text()
    assert "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH" in text, (
        "V4a lit fixture must lock the diagnostic code"
    )
    # 2 negative cases (lhs bsr, rhs packed).
    assert text.count("// expected-error @+1") == 2, (
        "V4a lit fixture should have 2 expected-error cases"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V4b — long-tail per-op verifiers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op_def,name",
    [
        ("def Tessera_CastOp", "cast"),
        ("def Tessera_SoftmaxOp", "softmax"),
        ("def Tessera_RopeOp", "rope"),
        ("def Tessera_DropoutOp", "dropout"),
    ],
)
def test_sprint_v4b_op_has_hasverifier(op_def: str, name: str) -> None:
    """V4b adds `let hasVerifier = 1;` to cast/softmax/rope and
    upgrades dropout's existing trivial verifier to a real one."""
    td = OPS_TD.read_text()
    idx = td.find(op_def)
    assert idx >= 0, f"op definition not found: {op_def} (V4b)"
    window = td[idx: idx + 1000]
    assert "let hasVerifier = 1;" in window, (
        f"{op_def} ({name}) missing `let hasVerifier = 1;` — "
        "Sprint V4b closure regressed"
    )


_V4B_VERIFIER_SIGNATURES = (
    "LogicalResult CastOp::verify()",
    "LogicalResult SoftmaxOp::verify()",
    "LogicalResult RopeOp::verify()",
    "LogicalResult DropoutOp::verify()",
)


@pytest.mark.parametrize("sig", _V4B_VERIFIER_SIGNATURES)
def test_sprint_v4b_verifier_impl_present(sig: str) -> None:
    """Each V4b op must have a real (non-trivial) verify() body."""
    cpp = OPS_CPP.read_text()
    assert sig in cpp, (
        f"verifier impl missing: {sig!r} — Sprint V4b closure regressed"
    )
    # Specifically reject the old trivial-stub form for DropoutOp.
    if "DropoutOp" in sig:
        assert "LogicalResult DropoutOp::verify() { return success(); }" not in cpp, (
            "V4b regression: DropoutOp::verify() is still the trivial stub"
        )


_V4B_DIAGNOSTIC_PHRASES = (
    # CastOp
    "cast must preserve rank",
    "cast must preserve dim",
    # SoftmaxOp
    "softmax must preserve rank",
    "softmax must preserve dim",
    "axis out of range",
    # RopeOp
    "rope must preserve rank",
    "rope must preserve element type",
    "rope must preserve dim",
    # DropoutOp
    "dropout probability must satisfy 0.0 <= p < 1.0",
    "dropout must preserve rank",
)


@pytest.mark.parametrize("phrase", _V4B_DIAGNOSTIC_PHRASES)
def test_sprint_v4b_diagnostic_phrases_present(phrase: str) -> None:
    cpp = OPS_CPP.read_text()
    assert phrase in cpp, (
        f"V4b diagnostic phrase missing: {phrase!r} — lit fixture "
        f"`expected-error` will silently miss after this regression"
    )


def test_sprint_v4b_lit_fixture_shape() -> None:
    assert V4B_LIT.exists(), "V4b lit fixture missing"
    text = V4B_LIT.read_text()
    # 4 positive cases (one per op) + 7 negative cases.
    pos = text.count("CHECK-LABEL:")
    neg = text.count("// expected-error @+1")
    assert pos == 4, (
        f"V4b lit fixture should have 4 CHECK-LABEL positives; got {pos}"
    )
    assert neg == 7, (
        f"V4b lit fixture should have 7 expected-error negatives; got {neg}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lit fixture presence (the C++ build verifies behaviour; lit existence
# is the structural guard)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [V1_LIT, V2_LIT, V3_LIT, V3A_LIT, V4A_LIT, V4B_LIT])
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
