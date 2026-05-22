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


def test_lit_fixture_covers_one_positive_two_negative() -> None:
    """V5 scope: 1 positive + 2 negative cases.  Pin that explicitly.
    V5 exercises the two diagnostic codes whose ops are registered in
    the Tessera dialect today: SYMDIM_BINDING_VIOLATION (function-level)
    and SYMDIM_MATMUL_CONTRACT_VIOLATION (tessera.matmul).  The reshape
    branch is in the pass body but not lit-exercised yet — tessera.reshape
    is a V2 ODS addition tracked in SHAPE_SYSTEM §11.2."""
    text = LIT_FIXTURE.read_text()
    # `expected-error` appears once per negative case AND once in the
    # explanatory header comment listing the codes — but each `// ----- `
    # split-input section can only have at most one `// expected-error`
    # directive (the one that's @-anchored).  Count those.
    expected_errors = text.count("// expected-error @+1")
    assert expected_errors == 2, (
        f"V5 lit fixture must contain exactly 2 @+1-anchored "
        f"`expected-error` directives (binding violation + matmul "
        f"contract violation); got {expected_errors}.  If V5.1 added "
        f"new cases, update this test to match."
    )
    # The positive case is the func.func without expected-error.
    assert "func.func @symdim_transpose_ok" in text, (
        "positive lit case 'symdim_transpose_ok' missing"
    )


def test_pass_argument_name_stable() -> None:
    """The pass's CLI argument must be `--tessera-symdim-equality`.
    Changing this silently would break lit fixtures + future
    pipeline composition."""
    text = PASS_CPP.read_text()
    assert '"tessera-symdim-equality"' in text, (
        "pass argument name must be tessera-symdim-equality"
    )
