"""Sprint V5 (2026-05-22) — SymbolicDimEqualityPass structural guard.

Closes the 4th MLIR-verifier gap from SHAPE_SYSTEM.md §11.2.  This
fast Python test pins the .cpp / .h / Passes.cpp / CMakeLists / lit
fixture content so a future regression (diagnostic-wording softening,
file deletion, pass-registration removal) fails immediately rather
than during the next C++ build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_CPP = (
    REPO_ROOT / "src" / "transforms" / "lib" / "SymbolicDimEqualityPass.cpp"
)
PASSES_H = (
    REPO_ROOT / "src" / "transforms" / "include" / "Tessera"
    / "Transforms" / "Passes.h"
)
PASSES_CPP = REPO_ROOT / "src" / "transforms" / "lib" / "Passes.cpp"
PASSES_CMAKE = REPO_ROOT / "src" / "transforms" / "lib" / "CMakeLists.txt"
LIT_FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v5_symdim_equality.mlir"
)


def test_pass_source_file_exists() -> None:
    assert PASS_CPP.exists(), (
        "src/transforms/lib/SymbolicDimEqualityPass.cpp missing — "
        "Sprint V5 closure regressed"
    )


@pytest.mark.parametrize("code", [
    "SYMDIM_BINDING_VIOLATION",
    "SYMDIM_RESHAPE_VIOLATION",
    "SYMDIM_TRANSPOSE_VIOLATION",
    "SYMDIM_MATMUL_CONTRACT_VIOLATION",
])
def test_stable_diagnostic_codes_present(code: str) -> None:
    """Every diagnostic code SHAPE_SYSTEM.md §11 cross-links must
    actually appear in the pass source — softening a wording silently
    would be a real footgun."""
    text = PASS_CPP.read_text()
    assert code in text, (
        f"SymbolicDimEqualityPass.cpp is missing stable diagnostic "
        f"code {code!r}"
    )


@pytest.mark.parametrize("attr", [
    "tessera.dim_bindings",
    "tessera.dim_sizes",
    "tessera.dim_names_in",
    "tessera.dim_names_out",
    "tessera.dim_names_lhs",
    "tessera.dim_names_rhs",
])
def test_attribute_names_documented(attr: str) -> None:
    """Each function-level / op-level attribute the pass reads must be
    referenced in the source so a future renamer can find every site."""
    text = PASS_CPP.read_text()
    assert attr in text, (
        f"attribute name {attr!r} missing from pass source — keep the "
        f"name surface single-sourced"
    )


def test_pass_reads_function_level_bindings() -> None:
    """The function-level `tessera.dim_bindings` ArrayAttr<StringAttr>
    must be read; helper name is part of the contract."""
    text = PASS_CPP.read_text()
    assert "readBindings" in text
    assert "readDimSizes" in text
    assert "parseBinding" in text


def test_pass_handles_three_op_kinds() -> None:
    """V1 dispatches on tessera.reshape, tessera.transpose,
    tessera.matmul.  V2 adds more — but V1's promise is exactly these
    three."""
    text = PASS_CPP.read_text()
    for op in ("tessera.reshape", "tessera.transpose", "tessera.matmul"):
        assert f'"{op}"' in text, (
            f"V1 promised op {op!r} is not dispatched in the pass walk"
        )


def test_pass_declared_in_passes_h() -> None:
    h = PASSES_H.read_text()
    assert "createSymbolicDimEqualityPass" in h


def test_pass_registered_in_passes_cpp() -> None:
    cpp = PASSES_CPP.read_text()
    assert "createSymbolicDimEqualityPass()" in cpp, (
        "Passes.cpp must call createSymbolicDimEqualityPass() inside "
        "registerTesseraPasses()"
    )


def test_pass_in_cmake_sources() -> None:
    cmake = PASSES_CMAKE.read_text()
    assert "SymbolicDimEqualityPass.cpp" in cmake, (
        "src/transforms/lib/CMakeLists.txt must include "
        "SymbolicDimEqualityPass.cpp in TesseraPasses"
    )


def test_lit_fixture_present() -> None:
    assert LIT_FIXTURE.exists(), (
        f"lit fixture missing: {LIT_FIXTURE.relative_to(REPO_ROOT)} — "
        "Sprint V5 closure regressed"
    )
    text = LIT_FIXTURE.read_text()
    assert "RUN: tessera-opt --tessera-symdim-equality" in text
    assert "expected-error" in text, (
        "lit fixture must include `expected-error` so the verifier is "
        "actually exercised"
    )


def test_lit_fixture_covers_one_positive_three_negative() -> None:
    """V5 scope was 1 positive + 2 negative; V6a (2026-05-22)
    registered `tessera.reshape` as an ODS op, so the lit fixture now
    exercises the reshape branch end-to-end too — 1 positive + 3
    negative.  The three negative codes lit-exercised through the
    real C++ binary are:

      SYMDIM_BINDING_VIOLATION
      SYMDIM_RESHAPE_VIOLATION            (added in V6a)
      SYMDIM_MATMUL_CONTRACT_VIOLATION

    Pin the count and the positive function name so a future V6.1
    addition (e.g., transpose violation case) fails this test and
    forces the count update."""
    text = LIT_FIXTURE.read_text()
    expected_errors = text.count("// expected-error @+1")
    assert expected_errors == 3, (
        f"V5+V6a lit fixture must contain exactly 3 @+1-anchored "
        f"`expected-error` directives (binding + reshape + matmul "
        f"contract); got {expected_errors}.  If V6.1 added new "
        f"cases, update this test to match."
    )
    # The positive case is the func.func without expected-error.
    assert "func.func @symdim_transpose_ok" in text, (
        "positive lit case 'symdim_transpose_ok' missing"
    )
    # Confirm each of the three negative case function names is
    # present.  Pin them so a refactor that renames one fails here.
    for name in (
        "@symdim_binding_broken",
        "@symdim_reshape_product_broken",
        "@symdim_matmul_contract_broken",
    ):
        assert name in text, f"V5+V6a lit case {name!r} missing"


def test_pass_argument_name_stable() -> None:
    """The pass's CLI argument must be `--tessera-symdim-equality`.
    Changing this silently would break lit fixtures + future
    pipeline composition."""
    text = PASS_CPP.read_text()
    assert '"tessera-symdim-equality"' in text, (
        "pass argument name must be tessera-symdim-equality"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V6b (2026-05-22) — pipeline integration guards.  The pass must be
# inserted into the canonical named lowering pipelines after
# DistributionLoweringPass so broken `where D = H * Dh` clauses are caught
# automatically (not just when the user explicitly opts in via
# --tessera-symdim-equality standalone).
# ─────────────────────────────────────────────────────────────────────────────


_V6B_PIPELINE_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v6b_symdim_in_pipeline.mlir"
)


def test_v6b_pipeline_integration_lit_fixture_present() -> None:
    """Lit fixture must run via the named pipeline (not the standalone
    pass) so the integration is exercised end-to-end."""
    assert _V6B_PIPELINE_LIT.exists(), (
        f"V6b pipeline integration lit fixture missing: "
        f"{_V6B_PIPELINE_LIT.relative_to(REPO_ROOT)}"
    )
    text = _V6B_PIPELINE_LIT.read_text()
    assert "RUN: tessera-opt --tessera-lower-to-x86" in text, (
        "V6b lit fixture must invoke the named pipeline "
        "`--tessera-lower-to-x86`, not the standalone pass"
    )
    assert "SYMDIM_BINDING_VIOLATION" in text, (
        "V6b lit fixture must exercise the same stable diagnostic "
        "the standalone pass emits"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint V2-flow (2026-05-22) — SSA-value dim-name propagation.
# Lifts the V1 pass from "best-effort verifier of explicit per-op
# annotations" to "actually infers dim-names through op chains and
# cross-checks them against any explicit per-op declarations".
# ─────────────────────────────────────────────────────────────────────────────


_V2_FLOW_LIT = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase2"
    / "sprint_v2_flow_propagation.mlir"
)


def test_v2_flow_diagnostic_code_present() -> None:
    """The new SYMDIM_FLOW_INCONSISTENCY diagnostic code must be in
    the pass source so lit fixtures + future spec cross-links can
    match on it."""
    text = PASS_CPP.read_text()
    assert "SYMDIM_FLOW_INCONSISTENCY" in text, (
        "SymbolicDimEqualityPass.cpp must emit SYMDIM_FLOW_INCONSISTENCY "
        "when propagated dim-names disagree with explicit per-op "
        "annotations (Sprint V2-flow)"
    )


def test_v2_flow_arg_dim_names_attribute_documented() -> None:
    """The function-level attribute V2-flow seeds from must be named
    in the source."""
    text = PASS_CPP.read_text()
    assert "tessera.arg_dim_names" in text, (
        "Pass source must reference tessera.arg_dim_names — that's the "
        "function-level attribute V2-flow seeds the SSA-value dim map "
        "from"
    )


@pytest.mark.parametrize("symbol", [
    "ValueDimMap",
    "readArgDimNames",
    "crossCheck",
    "propagateThroughFunction",
])
def test_v2_flow_pipeline_helpers_present(symbol: str) -> None:
    """V2-flow added 4 new helpers — pin their names so a refactor
    can't silently drop the flow tracking machinery."""
    text = PASS_CPP.read_text()
    assert symbol in text, (
        f"V2-flow helper {symbol!r} missing from pass source"
    )


def test_v2_flow_propagation_rules_documented() -> None:
    """V2 documents which ops it propagates through.  Pin those
    decisions so a future contributor can't drop a rule without
    updating the doc."""
    text = PASS_CPP.read_text()
    # The pass-source comment block must mention each rule.
    for rule_marker in (
        "transpose:",
        "matmul:",
        "reshape:",
    ):
        assert rule_marker in text, (
            f"V2-flow rule documentation missing: {rule_marker!r}"
        )


def test_v2_flow_lit_fixture_present() -> None:
    """V2-flow ships with a positive + negative + backward-compat case."""
    assert _V2_FLOW_LIT.exists(), (
        f"V2-flow lit fixture missing: {_V2_FLOW_LIT.relative_to(REPO_ROOT)}"
    )
    text = _V2_FLOW_LIT.read_text()
    assert "tessera.arg_dim_names" in text
    assert "SYMDIM_FLOW_INCONSISTENCY" in text
    expected_errors = text.count("// expected-error @+1")
    assert expected_errors == 1, (
        f"V2-flow lit fixture must contain exactly 1 @+1-anchored "
        f"expected-error directive (the FLOW_INCONSISTENCY case); "
        f"got {expected_errors}"
    )
    # Positive cases (one with arg_dim_names, one without).
    assert "@flow_propagation_ok" in text
    assert "@flow_no_arg_names_falls_through" in text


def test_v2_flow_backward_compatible_with_v1() -> None:
    """V2-flow must NOT break functions that worked under V1 (no
    arg_dim_names declared).  The pass source's
    propagateThroughFunction returns success() early when the seed
    map is empty — pin that branch."""
    text = PASS_CPP.read_text()
    assert "fall through to V1 behaviour" in text, (
        "propagateThroughFunction must explicitly fall through to V1 "
        "when no flow seeds are present — backward compatibility lock"
    )


@pytest.mark.parametrize("pipeline", [
    "lowerToX86",
    "lowerToGPU",
    "buildCUDA13Pipeline",
])
def test_v6b_pass_wired_into_named_pipelines(pipeline: str) -> None:
    """The 3 named pipelines must each call
    createSymbolicDimEqualityPass() after createDistributionLoweringPass()."""
    cpp = PASSES_CPP.read_text()
    # Find the pipeline body and check ordering of two pass calls.
    idx = cpp.find(pipeline)
    assert idx >= 0, f"named pipeline {pipeline!r} not found in Passes.cpp"
    # Window over the pipeline body — generous to handle multi-line.
    window = cpp[idx: idx + 3000]
    dist_pos = window.find("createDistributionLoweringPass()")
    symdim_pos = window.find("createSymbolicDimEqualityPass()")
    assert dist_pos >= 0, (
        f"pipeline {pipeline!r} must call createDistributionLoweringPass()"
    )
    assert symdim_pos >= 0, (
        f"pipeline {pipeline!r} must call createSymbolicDimEqualityPass() "
        "(Sprint V6b integration)"
    )
    assert symdim_pos > dist_pos, (
        f"pipeline {pipeline!r}: SymbolicDimEqualityPass must run AFTER "
        "DistributionLoweringPass — the V6b contract is to catch breaks "
        "caused by distribution lowering"
    )
