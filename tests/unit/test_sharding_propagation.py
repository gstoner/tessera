from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    tensor_ir_type,
)
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir
from tessera.compiler.sharding_propagation import (
    Placement,
    execute_resharded_graph_on_mock_mesh,
    materialize_reshard_plan,
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
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty],
        body=[op],
        return_values=["%out"],
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
            "scattered",
            "tessera.reduce_scatter",
            ["%partial"],
            mesh_axis="data",
            axis=0,
        ),
        _op(
            "exchanged",
            "tessera.all_to_all",
            ["%two_d"],
            mesh_axis="data",
            axis=1,
            scatter_axis=0,
            gather_axis=1,
        ),
        _op("reduced", "tessera.all_reduce", ["%partial"], mesh_axis="data"),
    ]
    fn = GraphIRFunction(
        "collectives",
        args=[IRArg("x", ty), IRArg("partial", ty), IRArg("two_d", ty)],
        result_types=[ty],
        body=ops,
        return_values=["%reduced"],
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


def test_reshard_plan_materializes_graph_schedule_and_tile_ssa():
    ty = tensor_ir_type(("8", "16"), "fp32")
    consume = _op("out", "tessera.sigmoid", ["%x"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[consume],
        return_values=["%out"],
    )
    propagated = propagate_sharding(fn, {"x": Placement.tiled({0: "data"})})
    plan = plan_explicit_reshards(
        fn, propagated, {0: (Placement.replicated(),)}, subgroup=(0, 1)
    )
    materialized = materialize_reshard_plan(fn, plan)
    assert [op.op_name for op in materialized.body] == [
        "tessera.all_gather",
        "tessera.sigmoid",
    ]
    collective, rewritten = materialized.body
    assert rewritten.operands == [f"%{collective.result}"]
    assert collective.kwargs["reshard_plan_digest"] == plan.digest
    assert collective.kwargs["subgroup"] == [0, 1]

    schedule = lower_graph_to_schedule_ir(
        GraphIRModule(functions=[materialized]), target_kind="cpu"
    )
    scheduled_collective = schedule.functions[0].body[0]
    assert scheduled_collective.op_name == "schedule.collective"
    assert scheduled_collective.result == collective.result
    assert scheduled_collective.operands == ["%x"]
    assert scheduled_collective.attrs["subgroup"] == [0, 1]
    tile = lower_schedule_to_tile_ir(schedule)
    assert tile.functions[0].body[0].op_name == "tile.all_gather"
    assert tile.functions[0].body[0].result == collective.result


def test_all_to_all_gets_verified_matching_rounds_and_region_identity():
    ty = tensor_ir_type(("9", "18"), "fp32")
    consume = _op("out", "tessera.sigmoid", ["%x"], _region_path=("loop0", "then"))
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[consume],
        return_values=["%out"],
    )
    propagated = propagate_sharding(fn, {"x": Placement.tiled({0: "data"})})
    plan = plan_explicit_reshards(
        fn,
        propagated,
        {0: (Placement.tiled({1: "data"}),)},
        subgroup=(2, 4, 7),
    )
    assert len(plan.matching_rounds) == 2
    expected = {(2, 4), (4, 7), (7, 2), (2, 7), (4, 2), (7, 4)}
    assert {edge for round_ in plan.matching_rounds for edge in round_} == expected
    assert all(
        len({source for source, _ in round_}) == 3
        and len({target for _, target in round_}) == 3
        for round_ in plan.matching_rounds
    )
    materialized = materialize_reshard_plan(fn, plan)
    collective = materialized.body[0]
    assert collective.op_name == "tessera.all_to_all"
    assert collective.kwargs["scatter_axis"] == 0
    assert collective.kwargs["gather_axis"] == 1
    assert collective.kwargs["region_path"] == ["loop0", "then"]
    schedule = lower_graph_to_schedule_ir(
        GraphIRModule(functions=[materialized]), target_kind="cpu"
    )
    tile = lower_schedule_to_tile_ir(schedule)
    assert tile.verify().ok
    tile_collective = tile.functions[0].body[0]
    tile_collective.attrs["matching_rounds"] = [[[2, 4]]]
    verification = tile.verify()
    assert not verification.ok
    assert "factor every directed peer edge once" in verification.format()


def test_reshard_rejects_sibling_escape_and_materializes_typed_local_shard():
    ty = tensor_ir_type(("8", "16"), "fp32")
    producer = _op("p", "tessera.sigmoid", ["%x"], _region_path=("then",))
    consume = _op("out", "tessera.sigmoid", ["%p"], _region_path=("else",))
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", ty), IRArg("y", ty)],
        result_types=[ty],
        body=[producer, consume],
        return_values=["%out"],
    )
    propagated = propagate_sharding(fn, {"x": Placement.tiled({0: "data"})})
    with pytest.raises(ValueError, match="sibling or escaping regions"):
        plan_explicit_reshards(
            fn, propagated, {1: (Placement.replicated(),)}, subgroup=(0, 1)
        )

    root_fn = GraphIRFunction(
        "root",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[_op("out", "tessera.sigmoid", ["%x"])],
        return_values=["%out"],
    )
    root_result = propagate_sharding(root_fn, {"x": Placement.replicated()})
    local_plan = plan_explicit_reshards(
        root_fn,
        root_result,
        {0: (Placement.tiled({0: "data"}),)},
        subgroup=(0, 1),
    )
    materialized = materialize_reshard_plan(root_fn, local_plan)
    local_slice, rewritten = materialized.body
    assert local_slice.op_name == "tessera.slice"
    assert local_slice.kwargs["reshard_kind"] == "local_shard"
    assert local_slice.inferred_type.shape == ("4", "16")
    assert local_slice.result_type == "tensor<4x16xf32>"
    assert rewritten.operand_types == ["tensor<4x16xf32>"]
    schedule = lower_graph_to_schedule_ir(
        GraphIRModule(functions=[materialized]), target_kind="cpu"
    )
    assert schedule.functions[0].body[0].attrs["reshard_kind"] == "local_shard"


def _execute_collective(op_name, rank_values, **kwargs):
    shape = tuple(str(dim) for dim in np.asarray(rank_values[0]).shape)
    ty = tensor_ir_type(shape, "fp32")
    op = IROp(
        result="out",
        op_name=op_name,
        operands=["%x"],
        operand_types=[str(ty)],
        result_type=str(ty),
        inferred_type=ty,
        kwargs=kwargs,
    )
    fn = GraphIRFunction(
        "mock_collective",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[op],
        return_values=["%out"],
    )
    return execute_resharded_graph_on_mock_mesh(
        fn, {"x": rank_values}, mesh_shape={"data": 2}
    )


def test_deterministic_mock_mesh_executes_every_movement_form():
    left = np.arange(8, dtype=np.float32).reshape(4, 2)
    right = left + 10

    reduced = _execute_collective("tessera.all_reduce", (left, right), op="sum")
    np.testing.assert_array_equal(reduced.returned["out"][0], left + right)
    np.testing.assert_array_equal(reduced.returned["out"][1], left + right)

    scattered = _execute_collective(
        "tessera.reduce_scatter", (left, right), axis=0, op="sum"
    )
    expected_sum = left + right
    np.testing.assert_array_equal(scattered.returned["out"][0], expected_sum[:2])
    np.testing.assert_array_equal(scattered.returned["out"][1], expected_sum[2:])

    gathered = _execute_collective("tessera.all_gather", (left[:2], right[:2]), axis=0)
    expected_gather = np.concatenate((left[:2], right[:2]), axis=0)
    np.testing.assert_array_equal(gathered.returned["out"][0], expected_gather)
    np.testing.assert_array_equal(gathered.returned["out"][1], expected_gather)

    exchanged = _execute_collective(
        "tessera.all_to_all", (left, right), scatter_axis=0, gather_axis=1
    )
    np.testing.assert_array_equal(
        exchanged.returned["out"][0],
        np.concatenate((left[:2], right[:2]), axis=1),
    )
    np.testing.assert_array_equal(
        exchanged.returned["out"][1],
        np.concatenate((left[2:], right[2:]), axis=1),
    )

    permuted = _execute_collective(
        "tessera.collective_permute",
        (left, right),
        source_peers=[0, 1],
        target_peers=[1, 0],
    )
    np.testing.assert_array_equal(permuted.returned["out"][0], right)
    np.testing.assert_array_equal(permuted.returned["out"][1], left)
    assert {
        *reduced.executed_reshards,
        *scattered.executed_reshards,
        *gathered.executed_reshards,
        *exchanged.executed_reshards,
        *permuted.executed_reshards,
    } == {
        "all_reduce",
        "reduce_scatter",
        "all_gather",
        "all_to_all",
        "collective_permute",
    }


def test_typed_local_shard_executes_without_hidden_reconstruction():
    ty = tensor_ir_type(("8", "4"), "fp32")
    fn = GraphIRFunction(
        "local",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[_op("out", "tessera.sigmoid", ["%x"])],
        return_values=["%out"],
    )
    propagation = propagate_sharding(fn, {"x": Placement.replicated()})
    plan = plan_explicit_reshards(
        fn,
        propagation,
        {0: (Placement.tiled({0: "data"}),)},
        subgroup=(0, 1),
    )
    materialized = materialize_reshard_plan(fn, plan, mesh_shape={"data": 2})
    full = np.arange(32, dtype=np.float32).reshape(8, 4)
    execution = execute_resharded_graph_on_mock_mesh(
        materialized,
        {"x": (full, full)},
        mesh_shape={"data": 2},
    )
    np.testing.assert_allclose(
        execution.returned["out"][0], 1.0 / (1.0 + np.exp(-full[:4]))
    )
    np.testing.assert_allclose(
        execution.returned["out"][1], 1.0 / (1.0 + np.exp(-full[4:]))
    )
    assert execution.executed_reshards == ("local_shard",)


def test_collective_permute_peer_map_survives_schedule_and_tile():
    values = (np.ones((2, 2), dtype=np.float32),) * 2
    shape = tuple(str(dim) for dim in values[0].shape)
    ty = tensor_ir_type(shape, "fp32")
    op = IROp(
        result="out",
        op_name="tessera.collective_permute",
        operands=["%x"],
        operand_types=[str(ty)],
        result_type=str(ty),
        inferred_type=ty,
        kwargs={
            "mesh_axis": "data",
            "axis": 0,
            "source_peers": [0, 1],
            "target_peers": [1, 0],
        },
    )
    fn = GraphIRFunction(
        "permute",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[op],
        return_values=["%out"],
    )
    schedule = lower_graph_to_schedule_ir(
        GraphIRModule(functions=[fn]), target_kind="cpu"
    )
    scheduled = schedule.functions[0].body[0]
    assert scheduled.attrs["source_peers"] == [0, 1]
    assert scheduled.attrs["target_peers"] == [1, 0]
    tile = lower_schedule_to_tile_ir(schedule)
    assert tile.functions[0].body[0].op_name == "tile.collective_permute"
    assert tile.verify().ok


def test_typed_local_shard_rejects_nondivisible_extent_and_mesh_mismatch():
    ty = tensor_ir_type(("7", "4"), "fp32")
    fn = GraphIRFunction(
        "bad_local",
        args=[IRArg("x", ty)],
        result_types=[ty],
        body=[_op("out", "tessera.sigmoid", ["%x"])],
        return_values=["%out"],
    )
    propagation = propagate_sharding(fn, {"x": Placement.replicated()})
    plan = plan_explicit_reshards(
        fn,
        propagation,
        {0: (Placement.tiled({0: "data"}),)},
        subgroup=(0, 1),
    )
    with pytest.raises(ValueError, match="not divisible"):
        materialize_reshard_plan(fn, plan, mesh_shape={"data": 2})
    with pytest.raises(ValueError, match="subgroup size"):
        materialize_reshard_plan(fn, plan, mesh_shape={"data": 4})
