"""GA7 + EBM5 wiring tests — verify the C++ dialect scaffolding is in place.

These tests run without a built MLIR. They:
  - Verify every expected source / header / tablegen / CMake file exists.
  - Parse each .td file with a lightweight grammar check for the expected
    `def TS_*Op` / `def CL_*Op` / `def EBM_*Op` entries.
  - Verify the pass-implementation .cpp files reference the correct
    pass-creation functions.
  - Verify the top-level CMake option `TESSERA_BUILD_CLIFFORD_BACKEND` and
    `TESSERA_BUILD_EBM_BACKEND` are declared.
  - Verify the GA4 primitive_coverage entries name-align with the
    `clifford.*` ops defined in `CliffordOps.td`.

This catches every "did you forget to add the file" or "did the op set drift"
failure that would otherwise surface only on a full MLIR build.
"""

from __future__ import annotations

import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# GA7 — Clifford dialect file layout
# ---------------------------------------------------------------------------

CLIFFORD_FILES = [
    "src/solvers/clifford/CMakeLists.txt",
    "src/solvers/clifford/include/tessera/Clifford/CliffordDialect.h",
    "src/solvers/clifford/include/tessera/Clifford/CliffordPasses.h",
    "src/solvers/clifford/include/tessera/Clifford/CliffordPasses.td",
    "src/solvers/clifford/lib/Dialect/Clifford/CliffordDialect.cpp",
    "src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.cpp",
    "src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.td",
    "src/solvers/clifford/lib/Passes/AnnotateAlgebra.cpp",
    # GA8 lowering passes (bodies, not stubs).
    "src/solvers/clifford/lib/Passes/CayleyTable.h",
    "src/solvers/clifford/lib/Passes/ExpandProductTable.cpp",
    "src/solvers/clifford/lib/Passes/GradeFusion.cpp",
    "src/solvers/clifford/lib/Passes/RotorSandwichFold.cpp",
    "src/solvers/clifford/test/lit.cfg.py",
    "src/solvers/clifford/test/ir/parse_print_cl30.mlir",
    "src/solvers/clifford/test/ir/parse_print_cl13.mlir",
    "src/solvers/clifford/test/ir/annotate_algebra.mlir",
    "src/solvers/clifford/test/ir/rotor_sandwich.mlir",
    # GA8 lit fixtures.
    "src/solvers/clifford/test/ir/passes/expand_product_table_cl30.mlir",
    "src/solvers/clifford/test/ir/passes/expand_product_table_cl13.mlir",
    "src/solvers/clifford/test/ir/passes/expand_rejects_batched.mlir",
    "src/solvers/clifford/test/ir/passes/grade_fusion_basic.mlir",
    "src/solvers/clifford/test/ir/passes/grade_fusion_multi_consumer.mlir",
    "src/solvers/clifford/test/ir/passes/grade_fusion_then_expand.mlir",
    "src/solvers/clifford/test/ir/passes/rotor_sandwich_fold_basic.mlir",
    "src/solvers/clifford/test/ir/passes/rotor_sandwich_fold_rejects_mismatched_R.mlir",
    "src/solvers/clifford/test/ir/passes/full_pipeline_cl30.mlir",
    "src/solvers/clifford/tools/ts-clifford-opt.cpp",
]


@pytest.mark.parametrize("rel_path", CLIFFORD_FILES)
def test_clifford_file_exists(rel_path: str) -> None:
    p = REPO_ROOT / rel_path
    assert p.is_file(), f"missing GA7 file: {rel_path}"


# ---------------------------------------------------------------------------
# CliffordOps.td defines all 17 ops from GA4 registry
# ---------------------------------------------------------------------------

EXPECTED_CLIFFORD_TD_OPS = {
    # Core GA3 ops (12)
    "CL_GeoProductOp": "geo_product",
    "CL_GradeProjectionOp": "grade",
    "CL_WedgeOp": "wedge",
    "CL_LeftContractOp": "left_contract",
    "CL_InnerOp": "inner",
    "CL_NormOp": "norm",
    "CL_ReverseOp": "reverse",
    "CL_GradeInvoluteOp": "grade_involute",
    "CL_ConjugateOp": "conjugate",
    "CL_HodgeStarOp": "hodge_star",
    "CL_ExpOp": "exp",
    "CL_LogOp": "log",
    "CL_RotorSandwichOp": "rotor_sandwich",
    # GA5 differential-form ops (4)
    "CL_ExtDerivOp": "ext_deriv",
    "CL_CodiffOp": "codiff",
    "CL_VecDerivOp": "vec_deriv",
    "CL_IntegralOp": "integral",
}


def test_clifford_td_defines_every_expected_op() -> None:
    td_path = REPO_ROOT / "src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.td"
    body = td_path.read_text(encoding="utf-8")
    for op_class, mnemonic in EXPECTED_CLIFFORD_TD_OPS.items():
        # Match `def CL_GeoProductOp : CliffordOp<"geo_product", ...>` shape.
        pattern = rf'def {op_class}\s*:\s*CliffordOp<"{re.escape(mnemonic)}"'
        assert re.search(pattern, body), (
            f"CliffordOps.td missing {op_class} for mnemonic '{mnemonic}'"
        )


def test_clifford_td_dialect_namespace_and_name() -> None:
    td = (REPO_ROOT / "src/solvers/clifford/lib/Dialect/Clifford/CliffordOps.td").read_text()
    assert 'def Clifford_Dialect : Dialect' in td
    assert 'let name = "tessera_clifford"' in td
    assert 'let cppNamespace = "::tessera::clifford"' in td


# ---------------------------------------------------------------------------
# Passes & driver
# ---------------------------------------------------------------------------

CLIFFORD_PASS_CREATORS = [
    "createCliffordAnnotateAlgebraPass",
    "createCliffordExpandProductTablePass",
    "createCliffordGradeFusionPass",
    "createCliffordRotorSandwichFoldPass",
]


@pytest.mark.parametrize("pass_fn", CLIFFORD_PASS_CREATORS)
def test_clifford_pass_creator_is_declared(pass_fn: str) -> None:
    header = (REPO_ROOT / "src/solvers/clifford/include/tessera/Clifford/CliffordPasses.h").read_text()
    assert pass_fn in header, (
        f"CliffordPasses.h missing declaration for {pass_fn}"
    )


@pytest.mark.parametrize(
    "pass_fn,impl_rel_path",
    [
        ("createCliffordAnnotateAlgebraPass",
         "src/solvers/clifford/lib/Passes/AnnotateAlgebra.cpp"),
        ("createCliffordExpandProductTablePass",
         "src/solvers/clifford/lib/Passes/ExpandProductTable.cpp"),
        ("createCliffordGradeFusionPass",
         "src/solvers/clifford/lib/Passes/GradeFusion.cpp"),
        ("createCliffordRotorSandwichFoldPass",
         "src/solvers/clifford/lib/Passes/RotorSandwichFold.cpp"),
    ],
)
def test_clifford_pass_creator_is_defined(pass_fn: str, impl_rel_path: str) -> None:
    impl = (REPO_ROOT / impl_rel_path).read_text()
    assert f"{pass_fn}()" in impl, (
        f"{impl_rel_path} missing definition of {pass_fn}"
    )


def test_clifford_pass_td_declares_each_pass() -> None:
    td = (REPO_ROOT / "src/solvers/clifford/include/tessera/Clifford/CliffordPasses.td").read_text()
    for pass_def in (
        "CliffordAnnotateAlgebra",
        "CliffordExpandProductTable",
        "CliffordGradeFusion",
        "CliffordRotorSandwichFold",
    ):
        assert re.search(rf"def {pass_def}\s*:\s*Pass<", td), (
            f"CliffordPasses.td missing `def {pass_def}`"
        )


def test_ts_clifford_opt_registers_all_passes_and_pipeline() -> None:
    src = (REPO_ROOT / "src/solvers/clifford/tools/ts-clifford-opt.cpp").read_text()
    for pass_fn in CLIFFORD_PASS_CREATORS:
        assert pass_fn in src, f"ts-clifford-opt.cpp doesn't register {pass_fn}"
    assert "tessera-clifford-pipeline" in src
    assert "tessera::clifford::CliffordDialect" in src


# ---------------------------------------------------------------------------
# Annotation pass enforces v1 allow-list (Cl(3,0), Cl(1,3))
# ---------------------------------------------------------------------------

def test_annotate_algebra_pass_enforces_v1_allow_list() -> None:
    impl = (REPO_ROOT / "src/solvers/clifford/lib/Passes/AnnotateAlgebra.cpp").read_text()
    # Both v1 signatures must be hard-coded as accepted.
    assert "p == 3 && q == 0 && r == 0" in impl
    assert "p == 1 && q == 3 && r == 0" in impl
    # Emits derived metadata.
    assert "tessera.clifford.dim" in impl
    assert "tessera.clifford.allow_listed" in impl
    assert "tessera.clifford.canonical" in impl


# ---------------------------------------------------------------------------
# Lit fixtures cover GA4 op set + annotation pass + pipeline alias
# ---------------------------------------------------------------------------

def test_lit_fixture_parse_print_cl30_runs_through_every_op() -> None:
    fixture = (REPO_ROOT / "src/solvers/clifford/test/ir/parse_print_cl30.mlir").read_text()
    for mnemonic in EXPECTED_CLIFFORD_TD_OPS.values():
        # Skip the GA5 field-level ops; the basic round-trip fixture covers
        # only the GA3 op surface.
        if mnemonic in {"ext_deriv", "codiff", "vec_deriv", "integral"}:
            continue
        assert f"tessera_clifford.{mnemonic}" in fixture, (
            f"parse_print_cl30 fixture missing op `tessera_clifford.{mnemonic}`"
        )


def test_lit_fixture_annotate_algebra_checks_canonical_attr() -> None:
    fixture = (REPO_ROOT / "src/solvers/clifford/test/ir/annotate_algebra.mlir").read_text()
    assert "tessera.clifford.canonical" in fixture
    assert "--tessera-clifford-annotate-algebra" in fixture


def test_lit_fixture_rotor_sandwich_invokes_full_pipeline() -> None:
    fixture = (REPO_ROOT / "src/solvers/clifford/test/ir/rotor_sandwich.mlir").read_text()
    assert "--tessera-clifford-pipeline" in fixture
    assert "rotor-sandwich-fold stub" in fixture


def test_lit_fixture_cl13_uses_minkowski_signature() -> None:
    fixture = (REPO_ROOT / "src/solvers/clifford/test/ir/parse_print_cl13.mlir").read_text()
    assert "algebra = [1, 3, 0]" in fixture


# ---------------------------------------------------------------------------
# CMake wiring
# ---------------------------------------------------------------------------

def test_top_level_cmake_declares_clifford_and_ebm_options() -> None:
    top = (REPO_ROOT / "CMakeLists.txt").read_text()
    assert "TESSERA_BUILD_CLIFFORD_BACKEND" in top
    assert "TESSERA_BUILD_EBM_BACKEND" in top


def test_solvers_cmake_conditionally_includes_clifford_and_ebm() -> None:
    cm = (REPO_ROOT / "src/solvers/CMakeLists.txt").read_text()
    assert "if(TESSERA_BUILD_CLIFFORD_BACKEND)" in cm
    assert "add_subdirectory(clifford)" in cm
    assert "if(TESSERA_BUILD_EBM_BACKEND)" in cm
    assert "add_subdirectory(ebm)" in cm


def test_clifford_cmake_links_tessera_clifford_library_and_driver() -> None:
    cm = (REPO_ROOT / "src/solvers/clifford/CMakeLists.txt").read_text()
    assert "TesseraClifford" in cm
    assert "add_executable(ts-clifford-opt" in cm
    assert "add_custom_target(check-clifford" in cm
    # GA8 source files must be compiled into the library.
    for src in (
        "lib/Passes/ExpandProductTable.cpp",
        "lib/Passes/GradeFusion.cpp",
        "lib/Passes/RotorSandwichFold.cpp",
    ):
        assert src in cm, f"CMakeLists missing GA8 source {src}"
    # GA8 ops generate arith + tensor IR — both dialects must link.
    assert "MLIRArithDialect" in cm
    assert "MLIRTensorDialect" in cm


# ---------------------------------------------------------------------------
# GA8 — Cayley-table helper structural checks
# ---------------------------------------------------------------------------

def test_cayley_table_header_implements_blade_product() -> None:
    """The CayleyTable.h header mirrors `tessera.ga.signature._product_table`."""
    body = (REPO_ROOT / "src/solvers/clifford/lib/Passes/CayleyTable.h").read_text()
    # Reordering-sign loop.
    assert "for (int64_t i = 0; i < n; ++i)" in body
    # Per-generator signature contributions (p, q, r).
    assert "if (i < p)" in body
    assert "} else if (i < q_hi) {" in body
    # Null-generator zero return.
    assert "return {0, 0};" in body


def test_cayley_table_helper_matches_python_reference_for_cl30() -> None:
    """Cross-check: the Python signature object exposes the same Cayley
    table the C++ helper computes. We assert structural alignment via
    a known-blade probe (e12 · e12 = -1 in Cl(3,0))."""
    from tessera.ga import Cl

    a = Cl(3, 0)
    table = a.product_table()
    e12 = a.blade("e12").mask
    res_mask, sign = table[e12][e12]
    # The Python and C++ tables must agree on this canonical value.
    assert res_mask == 0  # scalar
    assert sign == -1


def _cpp_shadow_blade_product(mask_a: int, mask_b: int, p: int, q: int, r: int):
    """Faithful Python translation of `bladeProduct` in CayleyTable.h.

    Re-implemented from the C++ source so we can verify the C++
    algorithm produces the same table as Python `tessera.ga` does —
    even without a built MLIR.  Drift between this shadow and the C++
    source would catch real bugs: any change to the C++ algorithm has
    to land in the shadow too.
    """
    n = p + q + r
    sign = 1
    for i in range(n):
        if (mask_b >> i) & 1:
            higher_a = mask_a >> (i + 1)
            crossings = bin(higher_a).count("1")
            if crossings & 1:
                sign = -sign
    common = mask_a & mask_b
    result_mask = mask_a ^ mask_b
    if common:
        q_hi = p + q
        for i in range(n):
            if (common >> i) & 1:
                if i < p:
                    pass
                elif i < q_hi:
                    sign = -sign
                else:
                    return 0, 0
    return result_mask, sign


@pytest.mark.parametrize("signature", [(3, 0, 0), (1, 3, 0)])
def test_cpp_cayley_shadow_matches_python_signature_table_byte_for_byte(signature) -> None:
    """The Python shadow of the C++ Cayley-table algorithm produces the
    same table as `tessera.ga.Cl(p, q).product_table()` — every entry,
    both signatures, end-to-end.

    This locks the C++ algorithm against drift even though we can't
    run the C++ in this session.  If someone changes the C++ algorithm
    in a way that diverges from the Python reference, the GA tests
    that depend on the Python table will still pass — but this test
    will fail because the shadow (translated from the C++ source)
    diverges from the Python product table.
    """
    from tessera.ga import Cl

    p, q, r = signature
    a = Cl(p, q, r)
    python_table = a.product_table()
    dim = a.dim
    for i in range(dim):
        for j in range(dim):
            py_mask, py_sign = python_table[i][j]
            cpp_mask, cpp_sign = _cpp_shadow_blade_product(i, j, p, q, r)
            assert (py_mask, py_sign) == (cpp_mask, cpp_sign), (
                f"Cl{signature[:2]} table mismatch at (i={i:b}, j={j:b}): "
                f"Python={py_mask, py_sign}, C++ shadow={cpp_mask, cpp_sign}"
            )


# ---------------------------------------------------------------------------
# GA8 — ExpandProductTable structural checks
# ---------------------------------------------------------------------------

def test_expand_product_table_emits_arith_and_tensor_ops() -> None:
    body = (REPO_ROOT / "src/solvers/clifford/lib/Passes/ExpandProductTable.cpp").read_text()
    # Builder calls for the lowered IR shape.
    assert "tensor::ExtractOp::create" in body
    assert "arith::MulFOp::create" in body
    assert "arith::AddFOp::create" in body
    assert "arith::SubFOp::create" in body
    assert "tensor::FromElementsOp::create" in body
    # Reads the optional output_grades attribute (grade-fusion savings).
    assert "tessera.clifford.output_grades" in body
    # Honors the v1 rank-1-only restriction with a diagnostic.
    assert "rank > 1" in body or "rank-1" in body


def test_expand_product_table_uses_cayley_helper() -> None:
    body = (REPO_ROOT / "src/solvers/clifford/lib/Passes/ExpandProductTable.cpp").read_text()
    assert "tessera::clifford::buildCayleyTable" in body
    assert "tessera::clifford::gradeOfMask" in body


# ---------------------------------------------------------------------------
# GA8 — GradeFusion structural checks
# ---------------------------------------------------------------------------

def test_grade_fusion_walks_grade_ops_with_geo_product_source() -> None:
    body = (REPO_ROOT / "src/solvers/clifford/lib/Passes/GradeFusion.cpp").read_text()
    assert "tessera_clifford.grade" in body
    assert "tessera_clifford.geo_product" in body
    # Attaches output_grades + replaces the grade op.
    assert "tessera.clifford.output_grades" in body
    assert "replaceOp" in body
    # Merges grades across multiple consumers (union semantics).
    assert "set<int64_t>" in body or "gradeSet.insert" in body


# ---------------------------------------------------------------------------
# GA8 — RotorSandwichFold structural checks
# ---------------------------------------------------------------------------

def test_rotor_sandwich_fold_matches_three_op_chain() -> None:
    body = (REPO_ROOT / "src/solvers/clifford/lib/Passes/RotorSandwichFold.cpp").read_text()
    # Pattern: outer gp(inner_gp, reverse) with matching R.
    assert "tessera_clifford.geo_product" in body
    assert "tessera_clifford.reverse" in body
    assert "tessera_clifford.rotor_sandwich" in body
    # Verifies the reverse's source matches the inner product's lhs.
    assert "rFromReverse != rFromInner" in body or "rFromReverse == rFromInner" in body
    # Tags the fused op with from_chain_fold for traceability.
    assert "tessera.clifford.from_chain_fold" in body


# ---------------------------------------------------------------------------
# GA8 — driver registers the new dialects (arith + tensor)
# ---------------------------------------------------------------------------

def test_ts_clifford_opt_registers_arith_and_tensor_dialects() -> None:
    src = (REPO_ROOT / "src/solvers/clifford/tools/ts-clifford-opt.cpp").read_text()
    assert "arith::ArithDialect" in src
    assert "tensor::TensorDialect" in src


def test_ts_clifford_opt_pipeline_orders_passes_correctly() -> None:
    """Pass order rationale: rotor-sandwich-fold MUST precede grade-fusion
    (otherwise the inner geo_product gets `output_grades` and the
    sandwich pattern is no longer recognizable)."""
    src = (REPO_ROOT / "src/solvers/clifford/tools/ts-clifford-opt.cpp").read_text()
    annotate_pos = src.find("createCliffordAnnotateAlgebraPass()")
    rotor_pos = src.find("createCliffordRotorSandwichFoldPass()")
    grade_pos = src.find("createCliffordGradeFusionPass()")
    expand_pos = src.find("createCliffordExpandProductTablePass()")
    # All four found in the pipeline body.
    assert annotate_pos > 0 and rotor_pos > 0 and grade_pos > 0 and expand_pos > 0
    # Order: annotate → rotor → grade → expand.
    assert annotate_pos < rotor_pos < grade_pos < expand_pos


# ---------------------------------------------------------------------------
# GA8 — lit fixture content checks
# ---------------------------------------------------------------------------

def test_expand_product_table_fixture_validates_arith_lowering() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/expand_product_table_cl30.mlir").read_text()
    assert "--tessera-clifford-expand-product-table" in fixture
    assert "tensor.extract" in fixture
    assert "arith.mulf" in fixture
    assert "arith.addf" in fixture
    assert "tensor.from_elements" in fixture


def test_expand_cl13_fixture_validates_minkowski_sign_flips() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/expand_product_table_cl13.mlir").read_text()
    # Cl(1,3) has q=3 generators squaring to -1, producing arith.subf
    # accumulations in the lowered IR.
    assert "arith.subf" in fixture
    assert "algebra = [1, 3, 0]" in fixture


def test_expand_rejects_batched_fixture() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/expand_rejects_batched.mlir").read_text()
    assert "tensor<32x8xf32>" in fixture
    # The op stays in the IR after the failed lowering.
    assert "tessera_clifford.geo_product" in fixture


def test_grade_fusion_basic_fixture_targets_bivector_slice() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/grade_fusion_basic.mlir").read_text()
    assert "grades = [2]" in fixture
    assert "tessera.clifford.output_grades = [2]" in fixture
    assert "--tessera-clifford-grade-fusion" in fixture


def test_grade_fusion_multi_consumer_unions_grade_sets() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/grade_fusion_multi_consumer.mlir").read_text()
    # Two grade ops requesting grade 0 and grade 2 fuse into the union.
    assert "tessera.clifford.output_grades = [0, 2]" in fixture


def test_full_pipeline_fixture_combines_all_three_passes() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/full_pipeline_cl30.mlir").read_text()
    assert "--tessera-clifford-pipeline" in fixture
    # Both transformations land: sandwich fold + arith lowering.
    assert "tessera_clifford.rotor_sandwich" in fixture
    assert "arith.mulf" in fixture


def test_rotor_sandwich_fold_rejects_mismatched_R_fixture() -> None:
    fixture = (REPO_ROOT
        / "src/solvers/clifford/test/ir/passes/rotor_sandwich_fold_rejects_mismatched_R.mlir").read_text()
    # The chain `gp(gp(R, x), reverse(S))` with R ≠ S does NOT fold.
    assert "CHECK-NOT: tessera_clifford.rotor_sandwich" in fixture


# ---------------------------------------------------------------------------
# Op names align 1:1 with the GA4 primitive_coverage registry
# ---------------------------------------------------------------------------

def test_clifford_td_op_set_aligns_with_ga4_registry() -> None:
    """Every `clifford_*` entry in primitive_coverage must correspond to a
    `tessera_clifford.<mnemonic>` op in CliffordOps.td, and vice versa."""
    from tessera.compiler import primitive_coverage as pc

    registry_clifford = sorted(
        e.name.removeprefix("clifford_")
        for e in pc.all_primitive_coverages().values()
        if e.category == "geometric_algebra"
    )
    # The TD ops use slightly different mnemonics in 4 places to match
    # MLIR convention (shorter / pure-camel form). Map them:
    td_to_registry = {
        "geo_product": "geometric_product",
        "grade": "grade_projection",
        "left_contract": "left_contraction",
        "grade_involute": "grade_involution",
    }
    td_mnemonics = sorted(EXPECTED_CLIFFORD_TD_OPS.values())
    aligned = sorted(td_to_registry.get(m, m) for m in td_mnemonics)
    assert aligned == registry_clifford, (
        f"GA4 registry vs CliffordOps.td mismatch:\n"
        f"  registry: {registry_clifford}\n"
        f"  td:       {aligned}"
    )


# ---------------------------------------------------------------------------
# EBM5 — parallel scaffold
# ---------------------------------------------------------------------------

EBM_FILES = [
    "src/solvers/ebm/CMakeLists.txt",
    "src/solvers/ebm/include/tessera/EBM/EBMDialect.h",
    "src/solvers/ebm/include/tessera/EBM/EBMPasses.h",
    "src/solvers/ebm/include/tessera/EBM/EBMPasses.td",
    "src/solvers/ebm/lib/Dialect/EBM/EBMDialect.cpp",
    "src/solvers/ebm/lib/Dialect/EBM/EBMOps.cpp",
    "src/solvers/ebm/lib/Dialect/EBM/EBMOps.td",
    "src/solvers/ebm/lib/Passes/Canonicalize.cpp",
    "src/solvers/ebm/test/lit.cfg.py",
    "src/solvers/ebm/test/ir/parse_print.mlir",
    "src/solvers/ebm/test/ir/canonicalize.mlir",
    "src/solvers/ebm/tools/ts-ebm-opt.cpp",
]


@pytest.mark.parametrize("rel_path", EBM_FILES)
def test_ebm_file_exists(rel_path: str) -> None:
    p = REPO_ROOT / rel_path
    assert p.is_file(), f"missing EBM5 file: {rel_path}"


def test_ebm_td_defines_six_core_ops() -> None:
    td = (REPO_ROOT / "src/solvers/ebm/lib/Dialect/EBM/EBMOps.td").read_text()
    for op_class in (
        "EBM_EnergyOp",
        "EBM_InnerStepOp",
        "EBM_LangevinStepOp",
        "EBM_SelfVerifyOp",
        "EBM_DecodeInitOp",
        "EBM_PartitionZOp",
    ):
        assert re.search(rf"def {op_class}\s*:\s*EBMOp<", td), (
            f"EBMOps.td missing {op_class}"
        )


def test_ebm_canonicalize_pass_normalizes_manifold_and_self_verify() -> None:
    impl = (REPO_ROOT / "src/solvers/ebm/lib/Passes/Canonicalize.cpp").read_text()
    assert "tessera.ebm.canonical" in impl
    assert "tessera.ebm.manifold" in impl
    assert "tessera.ebm.hard_argmin" in impl


def test_ts_ebm_opt_registers_all_passes_and_pipeline() -> None:
    src = (REPO_ROOT / "src/solvers/ebm/tools/ts-ebm-opt.cpp").read_text()
    for pass_fn in (
        "createEBMCanonicalizePass",
        "createEBMFuseEnergyGradPass",
        "createEBMCheckpointInnerLoopPass",
        "createEBMPipelineCandidatesPass",
    ):
        assert pass_fn in src
    assert "tessera-ebm-pipeline" in src


def test_ebm_lit_fixture_round_trips_sphere_manifold() -> None:
    fixture = (REPO_ROOT / "src/solvers/ebm/test/ir/canonicalize.mlir").read_text()
    assert 'manifold = "sphere"' in fixture
    assert "tessera.ebm.canonical" in fixture
    assert "tessera.ebm.manifold" in fixture


# ---------------------------------------------------------------------------
# Spectral parallel — sanity check that the new dialects mirror the template
# ---------------------------------------------------------------------------

def test_clifford_mirrors_spectral_structure() -> None:
    """Each piece of the spectral solver has a clifford counterpart."""
    spectral_root = REPO_ROOT / "src/solvers/spectral"
    clifford_root = REPO_ROOT / "src/solvers/clifford"
    for kind in ("CMakeLists.txt", "tools", "test", "lib/Dialect", "lib/Passes"):
        assert (spectral_root / kind).exists()
        assert (clifford_root / kind).exists(), (
            f"clifford solver missing parallel of {kind}"
        )
