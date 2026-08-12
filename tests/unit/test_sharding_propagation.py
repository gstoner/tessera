from __future__ import annotations

from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IROp, tensor_ir_type
from tessera.compiler.sharding_propagation import (
    Placement,
    plan_explicit_reshards,
    propagate_sharding,
)


def _op(result: str, name: str, operands: list[str], **kwargs) -> IROp:
    ty = tensor_ir_type(("8", "16"), "fp32")
    return IROp(
        result=result,
        op_name=name,
        operands=operands,
        operand_types=[str(ty)] * len(operands),
        result_type=str(ty),
        inferred_type=ty,
        kwargs=kwargs,
    )


def test_fixed_point_propagates_tiling_and_creates_partial_reduction():
    ty = tensor_ir_type(("8", "16"), "fp32")
    add = _op("sum", "tessera.add", ["%x", "%bias"])
    reduce = _op("out", "tessera.reduce", ["%sum"], axis=0)
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("bias", ty)],
        result_types=[ty],
        body=[add, reduce],
        return_values=["%out"],
    )
    result = propagate_sharding(
        fn,
        {"x": Placement.tiled({0: "data"}), "bias": Placement.replicated()},
    )
    assert result.placements["sum"] == Placement.tiled({0: "data"})
    assert result.placements["out"] == Placement.partial_reduction(("data",))
    assert not result.conflicts
    assert len(result.digest) == 64


def test_incompatible_tiles_and_unknown_regions_fail_closed():
    ty = tensor_ir_type(("8", "16"), "fp32")
    add = _op("sum", "tessera.add", ["%x", "%y"])
    region = _op("out", "tessera.add", ["%sum", "%x"], region="nested")
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty],
        body=[add, region],
        return_values=["%out"],
    )
    result = propagate_sharding(
        fn,
        {"x": Placement.tiled({0: "data"}), "y": Placement.tiled({1: "model"})},
    )
    assert result.placements["sum"].kind == "unknown"
    assert result.placements["out"].kind == "unknown"
    assert result.conflicts[0].reason == "incompatible_or_underspecified_placement"


def test_catalog_pointwise_rule_and_explicit_all_gather_reshard():
    ty = tensor_ir_type(("8", "16"), "fp32")
    # sigmoid is not in the old handwritten ten-op table. Its canonical
    # elementwise catalog contract now participates in the fixed point.
    sigmoid = _op("local", "tessera.sigmoid", ["%x"])
    consume = _op("out", "tessera.add", ["%local", "%bias"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("bias", ty)],
        result_types=[ty],
        body=[sigmoid, consume],
        return_values=["%out"],
    )
    result = propagate_sharding(
        fn,
        {"x": Placement.tiled({0: "data"}), "bias": Placement.replicated()},
    )
    assert result.placements["local"] == Placement.tiled({0: "data"})
    plan = plan_explicit_reshards(
        fn,
        result,
        {1: (Placement.replicated(), Placement.replicated())},
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].collective == "all_gather"
    assert plan.actions[0].mesh_axes == ("data",)
    assert len(plan.digest) == 64


def test_partial_reduction_plans_all_reduce_or_reduce_scatter():
    ty = tensor_ir_type(("8", "16"), "fp32")
    reduce = _op("partial", "tessera.reduce", ["%x"], axis=0)
    consume = _op("out", "tessera.add", ["%partial", "%bias"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("bias", ty)],
        result_types=[ty],
        body=[reduce, consume],
        return_values=["%out"],
    )
    result = propagate_sharding(
        fn,
        {"x": Placement.tiled({0: "data"}), "bias": Placement.replicated()},
    )
    replicated = plan_explicit_reshards(
        fn, result, {1: (Placement.replicated(), Placement.replicated())}
    )
    assert replicated.actions[0].collective == "all_reduce"

    sharded = plan_explicit_reshards(
        fn,
        result,
        {1: (Placement.tiled({0: "data"}), Placement.replicated())},
    )
    assert sharded.actions[0].collective == "reduce_scatter"


def test_reshard_unknown_fails_closed():
    ty = tensor_ir_type(("8", "16"), "fp32")
    op = _op("out", "tessera.add", ["%x", "%y"])
    fn = GraphIRFunction(
        "f", args=[IRArg("x", ty), IRArg("y", ty)], result_types=[ty],
        body=[op], return_values=["%out"],
    )
    result = propagate_sharding(fn, {})
    try:
        plan_explicit_reshards(
            fn, result, {0: (Placement.replicated(), Placement.replicated())}
        )
    except ValueError as exc:
        assert "unknown placement" in str(exc)
    else:
        raise AssertionError("unknown placement must not choose a collective")


def test_registered_collectives_transform_placement_explicitly():
    ty = tensor_ir_type(("8", "16"), "fp32")
    ops = [
        _op("gathered", "tessera.all_gather", ["%x"], mesh_axis="data", axis=0),
        _op(
            "scattered", "tessera.reduce_scatter", ["%partial"],
            mesh_axis="data", axis=0,
        ),
        _op(
            "exchanged", "tessera.all_to_all", ["%two_d"],
            mesh_axis="data", axis=1, scatter_axis=0, gather_axis=1,
        ),
        _op("reduced", "tessera.all_reduce", ["%partial"], mesh_axis="data"),
    ]
    fn = GraphIRFunction(
        "collectives",
        args=[IRArg("x", ty), IRArg("partial", ty), IRArg("two_d", ty)],
        result_types=[ty], body=ops, return_values=["%reduced"],
    )
    result = propagate_sharding(
        fn,
        {
            "x": Placement.tiled({0: "data"}),
            "partial": Placement.partial_reduction(("data",)),
            "two_d": Placement.tiled({0: "data"}),
        },
    )
    assert result.placements["gathered"] == Placement.replicated()
    assert result.placements["scattered"] == Placement.tiled({0: "data"})
    assert result.placements["exchanged"] == Placement.tiled({1: "data"})
    assert result.placements["reduced"] == Placement.replicated()
