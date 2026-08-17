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
