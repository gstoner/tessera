"""DIST-NATIVE-1 exact two-rank CPU packet.

Provision a self-contained development MPI with
``.venv/bin/python -m pip install mpi4py-mpich``, then run
``.venv/bin/mpiexec -n 2 .venv/bin/python -m pytest -q`` on this file.  A
normal single-rank pytest invocation skips; it cannot promote mock or
singleton evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

# This lane needs AVX-512 (and MPI, below), NOT AMX. Marking it hardware_amx
# would skip it on Princess-Luna -- the only host that can run it -- because
# Zen 5 has AVX-512 and no AMX, and AMX is a retired target besides.
pytestmark = pytest.mark.hardware_avx512
mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI

from tessera.compiler.collective_target import lower_tile_collective_artifact
from tessera.compiler.schedule_ir import ScheduleFunction, ScheduleIRModule, ScheduleOp
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir
from tessera.mpi_collectives import mpi_rank_adapter


if MPI.COMM_WORLD.Get_size() != 2:
    pytest.skip("DIST-NATIVE-1 packet requires mpiexec -n 2", allow_module_level=True)


def _compiled_artifact(communicator_digest: str):
    subgroup = [0, 1]
    specs = (
        ("all_reduce", "x_ar", "y_ar", "sum"),
        ("all_gather", "x_ag", "y_ag", "none"),
        ("reduce_scatter", "x_rs", "y_rs", "sum"),
        ("all_to_all", "x_aa", "y_aa", "none"),
        ("collective_permute", "x_cp", "y_cp", "none"),
    )
    body = [ScheduleOp("schedule.artifact", {"hash": "dist-native-mpi-v1"})]
    for ordinal, (kind, source, result, reduction) in enumerate(specs):
        attrs: dict[str, object] = {
            "source": source,
            "result": result,
            "ordinal": ordinal,
            "kind": kind,
            "mesh_axis": "dp",
            "tensor_axis": 0,
            "reduction": reduction,
            "effect": "collective",
            "world_size": 2,
            "dtype": "f32",
            "subgroup": subgroup,
            "reshard_plan_digest": "a" * 64,
            "region_path": ["main"],
            "scatter_axis": 0,
            "gather_axis": 0,
        }
        if kind == "all_to_all":
            attrs["matching_rounds"] = [[[0, 1], [1, 0]]]
        if kind == "collective_permute":
            attrs["source_peers"] = [0, 1]
            attrs["target_peers"] = [1, 0]
        body.append(
            ScheduleOp(
                "schedule.collective",
                attrs,
                operands=[f"%{source}"],
                result=result,
            )
        )
    schedule = ScheduleIRModule(
        functions=[ScheduleFunction("transport", body, target="cpu")],
        attrs={
            "tessera.ir.level": "schedule",
            "target": "cpu",
            "collective.backend": "mpi",
            "collective.communicator_digest": communicator_digest,
        },
    )
    return lower_tile_collective_artifact(lower_schedule_to_tile_ir(schedule))


def test_two_rank_compiled_ssa_matches_numpy_oracles_and_binds_topology():
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    local = np.arange(4, dtype=np.float32) + 10.0 * rank
    artifact = _compiled_artifact(transport.communicator_digest)
    runtime = artifact.execute(
        adapter=transport,
        tensors={
            "x_ar": local,
            "x_ag": local,
            "x_rs": local,
            "x_aa": local,
            "x_cp": local,
        },
    )
    np.testing.assert_array_equal(
        runtime.value("y_ar"),
        np.arange(4, dtype=np.float32) * 2.0 + 10.0,
    )
    np.testing.assert_array_equal(
        runtime.value("y_ag"),
        np.concatenate((np.arange(4, dtype=np.float32),
                        np.arange(4, dtype=np.float32) + 10.0)),
    )
    reduced = runtime.value("y_rs")
    expected_reduced = (np.arange(4, dtype=np.float32) * 2.0 + 10.0).reshape(2, 2)[rank]
    np.testing.assert_array_equal(reduced, expected_reduced)

    exchanged = runtime.value("y_aa")
    expected_exchange = np.concatenate(
        (np.arange(4, dtype=np.float32).reshape(2, 2)[rank],
         (np.arange(4, dtype=np.float32) + 10.0).reshape(2, 2)[rank])
    )
    np.testing.assert_array_equal(exchanged, expected_exchange)
    np.testing.assert_array_equal(
        runtime.value("y_cp"),
        np.arange(4, dtype=np.float32) + 10.0 * (1 - rank),
    )

    snapshot = transport.capability_snapshot()
    assert snapshot["world_size"] == 2
    assert snapshot["rank_order"] == [0, 1]
    assert len(snapshot["digest"]) == 64
    assert transport.communicator_digest == MPI.COMM_WORLD.bcast(
        transport.communicator_digest if rank == 0 else None, root=0
    )
    assert artifact.communicator_digest == transport.communicator_digest
    assert artifact.records[0]["subgroup"] == [0, 1]
    assert artifact.records[0]["reshard_plan_digest"] == "a" * 64


def test_topology_digest_mismatch_fails_before_data_transport():
    with pytest.raises(RuntimeError, match="digest mismatch"):
        mpi_rank_adapter(
            expected_world_size=2,
            expected_communicator_digest="0" * 64,
        )


@pytest.mark.parametrize("mismatch", ["order", "shape", "dtype"])
def test_cross_rank_collective_mismatch_fails_before_transport(mismatch: str):
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    if mismatch == "order":
        if rank == 0:
            invoke = lambda: transport.all_reduce(np.ones(2, np.float32))
        else:
            invoke = lambda: transport.all_gather(np.ones(2, np.float32))
    elif mismatch == "shape":
        invoke = lambda: transport.all_reduce(
            np.ones(2 + rank, np.float32)
        )
    else:
        invoke = lambda: transport.all_reduce(
            np.ones(2, np.float32 if rank == 0 else np.float64)
        )
    with pytest.raises(RuntimeError, match="ranks disagree"):
        invoke()


def test_cross_rank_subgroup_and_artifact_identity_mismatch_fail_closed():
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    with pytest.raises(RuntimeError, match="subgroup membership"):
        transport.for_subgroup((0, 1) if rank == 0 else (1, 0))

    transport = mpi_rank_adapter(expected_world_size=2)
    with pytest.raises(RuntimeError, match="artifact or communicator identity"):
        transport.admit_artifact(
            ("a" if rank == 0 else "b") * 64,
            transport.communicator_digest,
        )


def test_reordered_two_rank_subgroup_uses_a_real_derived_communicator():
    transport = mpi_rank_adapter(expected_world_size=2)
    subgroup = transport.for_subgroup((1, 0))
    assert subgroup is not None
    assert subgroup.rank_order == (1, 0)
    result = subgroup.all_reduce(np.array([transport.rank + 1], np.float32))
    np.testing.assert_array_equal(result, np.array([3], np.float32))
mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI

from tessera.compiler.collective_target import lower_tile_collective_artifact
from tessera.compiler.schedule_ir import ScheduleFunction, ScheduleIRModule, ScheduleOp
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir
from tessera.mpi_collectives import mpi_rank_adapter


if MPI.COMM_WORLD.Get_size() != 2:
    pytest.skip("DIST-NATIVE-1 packet requires mpiexec -n 2", allow_module_level=True)


def _compiled_artifact(communicator_digest: str):
    subgroup = [0, 1]
    specs = (
        ("all_reduce", "x_ar", "y_ar", "sum"),
        ("all_gather", "x_ag", "y_ag", "none"),
        ("reduce_scatter", "x_rs", "y_rs", "sum"),
        ("all_to_all", "x_aa", "y_aa", "none"),
        ("collective_permute", "x_cp", "y_cp", "none"),
    )
    body = [ScheduleOp("schedule.artifact", {"hash": "dist-native-mpi-v1"})]
    for ordinal, (kind, source, result, reduction) in enumerate(specs):
        attrs: dict[str, object] = {
            "source": source,
            "result": result,
            "ordinal": ordinal,
            "kind": kind,
            "mesh_axis": "dp",
            "tensor_axis": 0,
            "reduction": reduction,
            "effect": "collective",
            "world_size": 2,
            "dtype": "f32",
            "subgroup": subgroup,
            "reshard_plan_digest": "a" * 64,
            "region_path": ["main"],
            "scatter_axis": 0,
            "gather_axis": 0,
        }
        if kind == "all_to_all":
            attrs["matching_rounds"] = [[[0, 1], [1, 0]]]
        if kind == "collective_permute":
            attrs["source_peers"] = [0, 1]
            attrs["target_peers"] = [1, 0]
        body.append(
            ScheduleOp(
                "schedule.collective",
                attrs,
                operands=[f"%{source}"],
                result=result,
            )
        )
    schedule = ScheduleIRModule(
        functions=[ScheduleFunction("transport", body, target="cpu")],
        attrs={
            "tessera.ir.level": "schedule",
            "target": "cpu",
            "collective.backend": "mpi",
            "collective.communicator_digest": communicator_digest,
        },
    )
    return lower_tile_collective_artifact(lower_schedule_to_tile_ir(schedule))


def test_two_rank_compiled_ssa_matches_numpy_oracles_and_binds_topology():
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    local = np.arange(4, dtype=np.float32) + 10.0 * rank
    artifact = _compiled_artifact(transport.communicator_digest)
    runtime = artifact.execute(
        adapter=transport,
        tensors={
            "x_ar": local,
            "x_ag": local,
            "x_rs": local,
            "x_aa": local,
            "x_cp": local,
        },
    )
    np.testing.assert_array_equal(
        runtime.value("y_ar"),
        np.arange(4, dtype=np.float32) * 2.0 + 10.0,
    )
    np.testing.assert_array_equal(
        runtime.value("y_ag"),
        np.concatenate((np.arange(4, dtype=np.float32),
                        np.arange(4, dtype=np.float32) + 10.0)),
    )
    reduced = runtime.value("y_rs")
    expected_reduced = (np.arange(4, dtype=np.float32) * 2.0 + 10.0).reshape(2, 2)[rank]
    np.testing.assert_array_equal(reduced, expected_reduced)

    exchanged = runtime.value("y_aa")
    expected_exchange = np.concatenate(
        (np.arange(4, dtype=np.float32).reshape(2, 2)[rank],
         (np.arange(4, dtype=np.float32) + 10.0).reshape(2, 2)[rank])
    )
    np.testing.assert_array_equal(exchanged, expected_exchange)
    np.testing.assert_array_equal(
        runtime.value("y_cp"),
        np.arange(4, dtype=np.float32) + 10.0 * (1 - rank),
    )

    snapshot = transport.capability_snapshot()
    assert snapshot["world_size"] == 2
    assert snapshot["rank_order"] == [0, 1]
    assert len(snapshot["digest"]) == 64
    assert transport.communicator_digest == MPI.COMM_WORLD.bcast(
        transport.communicator_digest if rank == 0 else None, root=0
    )
    assert artifact.communicator_digest == transport.communicator_digest
    assert artifact.records[0]["subgroup"] == [0, 1]
    assert artifact.records[0]["reshard_plan_digest"] == "a" * 64


def test_topology_digest_mismatch_fails_before_data_transport():
    with pytest.raises(RuntimeError, match="digest mismatch"):
        mpi_rank_adapter(
            expected_world_size=2,
            expected_communicator_digest="0" * 64,
        )


@pytest.mark.parametrize("mismatch", ["order", "shape", "dtype"])
def test_cross_rank_collective_mismatch_fails_before_transport(mismatch: str):
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    if mismatch == "order":
        if rank == 0:
            invoke = lambda: transport.all_reduce(np.ones(2, np.float32))
        else:
            invoke = lambda: transport.all_gather(np.ones(2, np.float32))
    elif mismatch == "shape":
        invoke = lambda: transport.all_reduce(
            np.ones(2 + rank, np.float32)
        )
    else:
        invoke = lambda: transport.all_reduce(
            np.ones(2, np.float32 if rank == 0 else np.float64)
        )
    with pytest.raises(RuntimeError, match="ranks disagree"):
        invoke()


def test_cross_rank_subgroup_and_artifact_identity_mismatch_fail_closed():
    transport = mpi_rank_adapter(expected_world_size=2)
    rank = transport.rank
    with pytest.raises(RuntimeError, match="subgroup membership"):
        transport.for_subgroup((0, 1) if rank == 0 else (1, 0))

    transport = mpi_rank_adapter(expected_world_size=2)
    with pytest.raises(RuntimeError, match="artifact or communicator identity"):
        transport.admit_artifact(
            ("a" if rank == 0 else "b") * 64,
            transport.communicator_digest,
        )


def test_reordered_two_rank_subgroup_uses_a_real_derived_communicator():
    transport = mpi_rank_adapter(expected_world_size=2)
    subgroup = transport.for_subgroup((1, 0))
    assert subgroup is not None
    assert subgroup.rank_order == (1, 0)
    result = subgroup.all_reduce(np.array([transport.rank + 1], np.float32))
    np.testing.assert_array_equal(result, np.array([3], np.float32))
