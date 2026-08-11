from __future__ import annotations

import numpy as np
import pytest

from tessera import collectives
from tessera.compiler.distributed_contracts import (
    DISTRIBUTED_PRIMITIVE_CONTRACTS,
    collective_permute_record,
    collective_record_for_primitive,
)
from tessera.compiler.pipeline_runtime import (
    CollectiveDescriptor,
    CollectiveTransportRuntime,
)


def test_public_distributed_inventory_has_explicit_ownership() -> None:
    assert set(DISTRIBUTED_PRIMITIVE_CONTRACTS) == {
        "shard_map",
        "named_sharding",
        "partition_spec",
        "psum",
        "pmean",
        "pmax",
        "pmin",
        "collective_permute",
        "broadcast_to_axis",
    }
    assert {row.ownership for row in DISTRIBUTED_PRIMITIVE_CONTRACTS.values()} == {
        "compile_time_metadata",
        "compile_time_region",
        "collective_target",
        "point_to_point_target",
    }


@pytest.mark.parametrize(
    ("name", "kind", "reduction", "normalize"),
    (
        ("psum", "all_reduce", "sum", False),
        ("pmean", "all_reduce", "sum", True),
        ("pmax", "all_reduce", "max", False),
        ("pmin", "all_reduce", "min", False),
        ("broadcast_to_axis", "all_gather", "sum", False),
    ),
)
def test_collective_aliases_resolve_to_registered_transport(
    name: str, kind: str, reduction: str, normalize: bool
) -> None:
    descriptor = CollectiveDescriptor.from_sharding_primitive(
        name,
        tensor="x",
        result="y",
        mesh_axis="dp",
        tensor_axis=0,
        world_size=2,
    )
    assert descriptor.kind == kind
    assert descriptor.op == reduction
    assert descriptor.normalize is normalize
    assert descriptor.source_tensor == "x"
    assert descriptor.result_tensor == "y"


def test_sharding_metadata_do_not_masquerade_as_collectives() -> None:
    for name in (
        "shard_map",
        "named_sharding",
        "partition_spec",
    ):
        with pytest.raises(ValueError, match="not collective transport"):
            collective_record_for_primitive(name, tensor="x", mesh_axis="dp", world_size=2)


def test_collective_permute_requires_and_executes_explicit_peer_map() -> None:
    record = collective_permute_record(
        tensor="x",
        result="y",
        mesh_axis="dp",
        pairs=((0, 1), (1, 0)),
        world_size=2,
    )
    descriptor = CollectiveDescriptor.from_target_record(record)
    assert descriptor.kind == "collective_permute"
    assert descriptor.peer_pairs == ((0, 1), (1, 0))

    runtime = CollectiveTransportRuntime(
        collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2}),
        {
            "x": [
                np.array([1.0, 2.0], np.float32),
                np.array([3.0, 4.0], np.float32),
            ]
        },
    )
    runtime.execute(descriptor)
    np.testing.assert_array_equal(runtime.values("y")[0], [3.0, 4.0])
    np.testing.assert_array_equal(runtime.values("y")[1], [1.0, 2.0])


def test_collective_permute_rejects_duplicate_or_out_of_range_peers() -> None:
    with pytest.raises(ValueError, match="sources and targets must be unique"):
        collective_permute_record(
            tensor="x",
            result="y",
            mesh_axis="dp",
            pairs=((0, 1), (0, 2)),
            world_size=3,
        )
    with pytest.raises(ValueError, match="outside the communicator"):
        collective_permute_record(
            tensor="x",
            result="y",
            mesh_axis="dp",
            pairs=((0, 2),),
            world_size=2,
        )


def test_alias_descriptor_executes_through_portable_two_rank_runtime() -> None:
    descriptor = CollectiveDescriptor.from_sharding_primitive(
        "pmean",
        tensor="x",
        result="mean",
        mesh_axis="dp",
        world_size=2,
    )
    runtime = CollectiveTransportRuntime(
        collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2}),
        {
            "x": [
                np.array([1.0, 3.0], np.float32),
                np.array([5.0, 7.0], np.float32),
            ]
        },
    )
    runtime.execute(descriptor)
    for value in runtime.values("mean"):
        np.testing.assert_array_equal(value, np.array([3.0, 5.0], np.float32))
