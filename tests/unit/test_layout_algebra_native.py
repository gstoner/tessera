from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def native_layout_library(tmp_path_factory: pytest.TempPathFactory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("layout-algebra") / "libtessera_layout_algebra.so"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(root / "src/compiler/layout_algebra/include"),
            str(root / "src/compiler/layout_algebra/LayoutAlgebra.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    old = os.environ.get("TESSERA_LAYOUT_ALGEBRA_LIB")
    os.environ["TESSERA_LAYOUT_ALGEBRA_LIB"] = str(output)
    import tessera.compiler.layout_algebra as algebra

    algebra._LIB = None
    yield
    algebra._LIB = None
    if old is None:
        os.environ.pop("TESSERA_LAYOUT_ALGEBRA_LIB", None)
    else:
        os.environ["TESSERA_LAYOUT_ALGEBRA_LIB"] = old


def test_native_coordinate_algebra_exhausts_compact_layouts_to_64() -> None:
    from tessera.compiler.layout_algebra import crd2idx, idx2crd, size

    for a in range(1, 65):
        for b in range(1, 65 // a + 1):
            for c in range(1, 65 // (a * b) + 1):
                shape = (a, b, c)
                stride = (1, a, a * b)
                total = size(shape)
                assert total == a * b * c
                for linear in range(total):
                    assert crd2idx(shape, stride, idx2crd(shape, linear)) == linear


def test_native_rank2_plan_owns_emitter_coordinate_order() -> None:
    from tessera.compiler.layout_algebra import (
        LayoutAlgebraError,
        rank2_index_expression,
    )

    assert rank2_index_expression("row", "column", "columns") == (
        "row * columns + column"
    )
    assert rank2_index_expression(
        "row", "column", "rows", order="column_major"
    ) == "column * rows + row"
    with pytest.raises(LayoutAlgebraError, match="unsupported rank-2 order"):
        rank2_index_expression("row", "column", "stride", order="diagonal")


def test_native_coalesce_preserves_function_and_canonical_structure() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, coalesce

    assert coalesce(NestedLayout((2, (1, 6)), (1, (6, 2)))) == NestedLayout(12, 1)
    assert coalesce(NestedLayout((3, (4, 5)), (8, (1, 4)))) == NestedLayout(
        (3, 20), (8, 1)
    )


def test_native_composition_splits_crossing_modes_without_flattening() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, composition

    # CuTe's worked composition: B's extent-4 mode crosses A's radix-6
    # boundary, so it must remain a nested (2,2), not a flattened (2,2,3).
    lhs = NestedLayout((6, 2), (8, 2))
    rhs = NestedLayout((4, 3), (3, 1))
    assert composition(lhs, rhs) == NestedLayout(((2, 2), 3), ((24, 2), 8))


def test_native_coalesce_keeps_dynamic_residue_structured() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, coalesce

    assert coalesce(NestedLayout((2, -1, 4), (1, -1, 8))) == NestedLayout(
        (2, -1, 4), (1, -1, 8)
    )


def test_native_compact_inverses_and_complement_match_documented_contracts() -> None:
    from tessera.compiler.layout_algebra import (
        NestedLayout,
        complement,
        left_inverse,
        right_inverse,
    )

    source = NestedLayout((2, 4, 6), (4, 1, 8))
    expected_inverse = NestedLayout((4, 2, 6), (2, 1, 8))
    assert right_inverse(source) == expected_inverse
    assert left_inverse(source) == expected_inverse

    gaps = NestedLayout((2, 2), (4, 1))
    assert complement(gaps, 24) == NestedLayout((2, 3), (2, 8))
    assert complement(gaps) == NestedLayout((2, 1), (2, 8))


def test_native_product_variants_preserve_their_documented_groupings() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, product

    lhs = NestedLayout((3, 4), (4, 1))
    rhs = NestedLayout((2, 5), (1, 2))
    assert product(lhs, rhs) == NestedLayout(((3, 4), (2, 5)), ((4, 1), (12, 24)))
    assert product(lhs, rhs, variant="tiled") == NestedLayout(
        ((3, 4), 2, 5), ((4, 1), 12, 24)
    )
    assert product(lhs, rhs, variant="flat") == NestedLayout(
        (3, 4, 2, 5), (4, 1, 12, 24)
    )
    assert product(lhs, rhs, variant="blocked") == NestedLayout(
        ((3, 2), (4, 5)), ((4, 12), (1, 24))
    )
    assert product(lhs, rhs, variant="raked") == NestedLayout(
        ((2, 3), (5, 4)), ((12, 4), (24, 1))
    )


def test_native_divide_variants_preserve_their_documented_groupings() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, divide

    source = NestedLayout((6, 8), (8, 1))
    tiler = NestedLayout((3, 4), (1, 3))
    assert divide(source, tiler) == NestedLayout(((3, 2), (4, 2)), ((8, 24), (1, 4)))
    assert divide(source, tiler, variant="zipped") == NestedLayout(
        ((3, 4), (2, 2)), ((8, 1), (24, 4))
    )
    assert divide(source, tiler, variant="tiled") == NestedLayout(
        ((3, 4), 2, 2), ((8, 1), 24, 4)
    )
    assert divide(source, tiler, variant="flat") == NestedLayout(
        (3, 4, 2, 2), (8, 1, 24, 4)
    )


def test_native_general_composition_has_no_enumeration_ceiling_and_keeps_dynamic_tail() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, composition

    outer = NestedLayout((2, 2), (1, 2))
    assert composition(outer, NestedLayout(2**21, 1)) == NestedLayout(
        (2, 2**20), (1, 2)
    )

    boundary_outer = NestedLayout((6, 2), (8, 2))
    assert composition(boundary_outer, NestedLayout(-1, 3)) == NestedLayout(
        (2, -1), (24, 2)
    )


def test_native_noncompact_divide_routes_through_general_composition() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, divide

    source = NestedLayout((6, 8), (8, 1))
    noncompact_tiler = NestedLayout((3, 4), (1, 6))
    assert divide(source, noncompact_tiler) == NestedLayout(
        ((3, 4), (2, 2)), ((8, 1), (24, 4))
    )


def test_native_slice_carries_fixed_coordinate_offset() -> None:
    from tessera.compiler.layout_algebra import NestedLayout, SlicedLayout, slice_layout

    assert slice_layout(NestedLayout((3, 4, 5), (20, 5, 1)), (-1, 2, -1)) == SlicedLayout(
        NestedLayout((3, 5), (20, 1)), 10
    )
    assert slice_layout(NestedLayout((2, -1, 4), (1, -1, 8)), (-1, -1, 3)) == SlicedLayout(
        NestedLayout((2, -1), (1, -1)), 24
    )


def test_native_factorization_and_cosize_residency_proofs() -> None:
    from tessera.compiler.layout_algebra import (
        NestedLayout,
        factorizes,
        prove_residency,
    )

    block = NestedLayout((16, 16), (16, 1))
    coordinate = NestedLayout((1, 1), (16, 1))
    row = NestedLayout((1, 16), (16, 1))
    assert factorizes(coordinate, block).factorizes
    assert factorizes(row, block).factorizes
    assert not factorizes(NestedLayout((2, 2), (300, 1)), block).factorizes

    padded = prove_residency(
        NestedLayout((4, 4), (8, 1)), element_bytes=4, capacity_bytes=128
    )
    assert padded.elements == 28
    assert padded.bytes == 112
    assert padded.admitted
    assert not prove_residency(
        NestedLayout((4, 4), (8, 1)), element_bytes=4, capacity_bytes=64
    ).admitted


def test_gqa_fold_and_inverse_execute_through_the_native_plan() -> None:
    import tessera

    x = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    folded = tessera.ops.rearrange(x, "b h s d -> b s (h d)")
    assert folded.shape == (2, 5, 21)
    restored = tessera.ops.rearrange(
        folded,
        "b s (h d) -> b h s d",
        axes_lengths={"h": 3, "d": 7},
    )
    np.testing.assert_array_equal(restored, x)


def test_graph_shape_inference_consumes_the_same_native_plan() -> None:
    from tessera.compiler.graph_ir import _infer_result_type, tensor_ir_type

    result = _infer_result_type(
        "tessera.rearrange",
        [tensor_ir_type(("2", "3", "5", "7"), "fp32")],
        {"layout": "b h s d -> b s (h d)"},
    )
    assert result.shape == ("2", "5", "21")


def test_graph_shape_inference_preserves_dynamic_rearrange_rank() -> None:
    from tessera.compiler.graph_ir import _infer_result_type, tensor_ir_type

    result = _infer_result_type(
        "tessera.rearrange",
        [tensor_ir_type(("?", "3", "4"), "fp32")],
        {"layout": "a b c -> a (b c)"},
    )
    assert result.shape == ("?", "12")
    assert result.rank == 2


def test_explicit_missing_library_fails_closed_without_search_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tessera.compiler.layout_algebra as algebra

    selected = os.environ["TESSERA_LAYOUT_ALGEBRA_LIB"]
    monkeypatch.setenv("TESSERA_LAYOUT_ALGEBRA_LIB", selected + ".missing")
    algebra._LIB = None
    with pytest.raises(algebra.LayoutAlgebraUnavailableError, match="build target"):
        algebra.rearrange_plan((2, 3), "a b -> b a")
    monkeypatch.setenv("TESSERA_LAYOUT_ALGEBRA_LIB", selected)
    algebra._LIB = None
    assert algebra.native_available()


@pytest.mark.parametrize(
    "spec",
    (
        "b h -> h",
        "b h => h b",
        "b (h d) -> b h d",
        "b h -> b (h missing)",
    ),
)
def test_rearrange_remains_fail_closed_for_malformed_or_unresolved_specs(spec: str) -> None:
    import tessera
    with pytest.raises(ValueError):
        tessera.ops.rearrange(np.zeros((2, 6)), spec)
