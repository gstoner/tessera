from __future__ import annotations

import pathlib

import pytest

import tessera as ts
from tessera import Tensor
from tessera.shape import (
    RuntimeShapeWitness,
    ShapeConstraintGraph,
    ShapeSystemError,
    broadcast_shape,
    check_schedule_tile,
    check_shard,
    matmul_shape,
    reshape_shape,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_symbolic_dims_are_tensor_annotation_ready():
    B, N, D = ts.sym("B N D")
    annotated = Tensor[B, N, D]

    assert annotated.__dims__ == (B, N, D)
    assert str(D) == "D"


def test_broadcast_shape_supports_symbolic_singleton_dims():
    B, N, D = ts.sym("B N D")

    result = broadcast_shape((B, 1, D), (B, N, D))

    assert result == (B, N, D)


def test_matmul_shape_broadcasts_batch_dims_and_checks_inner_dim():
    B, M, K, N = ts.sym("B M K N")

    assert matmul_shape((B, M, K), (1, K, N)) == (B, M, N)

    OtherK = ts.dim("OtherK")
    with pytest.raises(ShapeSystemError, match="inner dimensions differ"):
        matmul_shape((B, M, K), (OtherK, N))


def test_derived_dimension_constraint_models_head_splitting():
    H, Dh, D = ts.sym("H Dh D")
    graph = ShapeConstraintGraph().derived(D, H * Dh)

    graph.raise_if_errors({"H": 8, "Dh": 64, "D": 512})

    errors = graph.check_all({"H": 8, "Dh": 64, "D": 256})
    assert errors
    assert errors[0].code == "shape-equal"


def test_reshape_validates_element_count_when_bindings_are_known():
    B, T, H, Dh, D = ts.sym("B T H Dh D")

    assert reshape_shape((B, T, D), (B, T, H, Dh), {"B": 2, "T": 4, "D": 512, "H": 8, "Dh": 64})

    with pytest.raises(ShapeSystemError, match="element count"):
        reshape_shape((B, T, D), (B, T, H, Dh), {"B": 2, "T": 4, "D": 256, "H": 8, "Dh": 64})


def test_shard_checks_logical_dims_against_mesh_axes():
    B, N, D = ts.sym("B N D")

    assert check_shard((B, N, D), {"dp": 4, "tp": 8}, {"B": "dp", "D": "tp"}, {"B": 8, "D": 128}) == []

    errors = check_shard((B, N, D), {"dp": 4, "tp": 8}, {"B": "dp", "D": "tp"}, {"B": 8, "D": 130})
    assert errors
    assert errors[0].code == "shape-shard-divisible"
    assert "136" in errors[0].suggestion


def test_schedule_feasibility_reports_padding_suggestions():
    N = ts.dim("N")

    ok = check_schedule_tile(N, 256, {"N": 2304})
    assert ok.ok

    bad = check_schedule_tile(N, 256, {"N": 2305})
    assert not bad.ok
    assert bad.padded == 2560
    assert "pad N from 2305 to 2560" == bad.suggestion


def test_runtime_witness_refines_dynamic_dims_once_per_module():
    B, N, D = ts.sym("B N D")
    graph = ShapeConstraintGraph().divisible(N, 128)
    witness = RuntimeShapeWitness("attention.input", (B, N, D), graph)

    assert witness.refine({"B": 2, "N": 1024, "D": 128}) == (2, 1024, 128)

    with pytest.raises(ShapeSystemError, match="shape-divisible"):
        witness.refine({"B": 2, "N": 1000, "D": 128})


def test_check_shapes_decorator_marks_functions_for_frontend():
    B, N, M, D = ts.sym("B N M D")

    @ts.check_shapes
    def attention(q: Tensor[B, N, D], k: Tensor[B, M, D]) -> Tensor[B, N, M]:
        return q

    assert attention.__tessera_check_shapes__ is True


def test_shape_system_spec_documents_compiler_integration_points():
    spec = (ROOT / "docs/spec/SHAPE_SYSTEM.md").read_text()

    for needle in [
        "Derived dimensions",
        "Schedule Feasibility",
        "Runtime Shape Witnesses",
        "logical shape",
        "physical layout",
        "shard map",
    ]:
        assert needle in spec
