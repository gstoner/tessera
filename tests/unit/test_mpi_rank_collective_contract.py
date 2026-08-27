from __future__ import annotations

import pytest
import numpy as np

from tessera.mpi_collectives import MPIRankAdapter


class _FakeMPI:
    SUM = object()
    PROC_NULL = -1
    UNDEFINED = -32766

    @staticmethod
    def Get_processor_name():
        return "host-a"

    @staticmethod
    def Get_library_version():
        return "FakeMPI 1.0"


class _FakeComm:
    def __init__(
        self,
        *,
        size=2,
        rank=0,
        mismatch=False,
        topology_mismatch=False,
        admission_mismatch: str | None = None,
        artifact_mismatch=False,
        subgroup_mismatch=False,
    ):
        self.size = size
        self.rank = rank
        self.mismatch = mismatch
        self.topology_mismatch = topology_mismatch
        self.admission_mismatch = admission_mismatch
        self.artifact_mismatch = artifact_mismatch
        self.subgroup_mismatch = subgroup_mismatch
        self.records = []

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def allgather(self, value):
        self.records.append(value)
        if self.topology_mismatch and isinstance(value, tuple) and len(value) == 7:
            changed = list(value)
            changed[1] = (("tp", self.size),)
            return [value, tuple(changed)]
        if self.artifact_mismatch and isinstance(value, tuple) and len(value) == 4:
            changed = list(value)
            changed[1] = "b" * 64
            return [value, tuple(changed)]
        if (
            self.subgroup_mismatch
            and isinstance(value, tuple)
            and len(value) == 5
            and value[1] == "subgroup"
        ):
            changed = list(value)
            changed[2] = tuple(reversed(value[2]))
            return [value, tuple(changed)]
        if self.admission_mismatch and isinstance(value, tuple) and len(value) == 8:
            changed = list(value)
            index = {"order": 1, "shape": 4, "dtype": 5}[self.admission_mismatch]
            changed[index] = {
                "order": "all_gather",
                "shape": (999,),
                "dtype": "float64",
            }[self.admission_mismatch]
            return [value, tuple(changed)]
        if (
            self.mismatch and isinstance(value, tuple) and len(value) >= 2
            and isinstance(value[1], str)
        ):
            return [value, (999, "wrong", (), "float32")]
        return [value] * self.size


def test_mpi_topology_binds_process_rank_ownership_and_digest():
    adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(),
        _mpi=_FakeMPI,
    )
    snapshot = adapter.capability_snapshot()
    assert snapshot["rank_order"] == [0, 1]
    assert snapshot["local_ranks"] == [0, 1]
    assert snapshot["rank_hosts"] == ["host-a", "host-a"]
    assert len(adapter.communicator_digest) == 64

    with pytest.raises(RuntimeError, match="digest mismatch"):
        MPIRankAdapter(
            expected_world_size=2,
            communicator=_FakeComm(),
            expected_communicator_digest="0" * 64,
            _mpi=_FakeMPI,
        )


def test_mpi_world_and_cross_rank_issue_order_fail_closed():
    with pytest.raises(ValueError, match="world size is invalid"):
        MPIRankAdapter(
            expected_world_size="two",
            communicator=_FakeComm(),
            _mpi=_FakeMPI,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        MPIRankAdapter(
            expected_world_size=2,
            communicator=_FakeComm(size=1),
            _mpi=_FakeMPI,
        )
    adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(mismatch=True),
        _mpi=_FakeMPI,
    )
    with pytest.raises(RuntimeError, match="disagree"):
        adapter.all_reduce(np.array([1.0, 2.0], dtype=np.float32))


def test_mpi_dtype_and_mesh_envelope_fail_closed():
    with pytest.raises(ValueError, match="mesh extent"):
        MPIRankAdapter(
            expected_world_size=2,
            mesh_axes={"dp": 3},
            communicator=_FakeComm(),
            _mpi=_FakeMPI,
        )
    adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(),
        _mpi=_FakeMPI,
    )
    with pytest.raises(ValueError, match="float32"):
        adapter.all_reduce([1, 2])


def test_mpi_topology_and_collective_options_are_cross_rank_admitted():
    with pytest.raises(RuntimeError, match="ranks disagree"):
        MPIRankAdapter(
            expected_world_size=2,
            communicator=_FakeComm(topology_mismatch=True),
            _mpi=_FakeMPI,
        )

    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    pairs = ((0, 1), (1, 0))
    with pytest.raises(AttributeError):
        adapter.collective_permute(np.ones(2, dtype=np.float32), pairs=pairs)
    assert pairs in comm.records[-1][2]


@pytest.mark.parametrize(
    "invoke,match",
    [
        (lambda adapter: adapter.all_reduce(np.ones(2, np.float32), op="max"), "SUM"),
        (lambda adapter: adapter.all_gather(np.ones(2, np.float32), axis=1), "axis=0"),
        (lambda adapter: adapter.all_reduce(np.ones(2, np.float64)), "float32"),
    ],
)
def test_mpi_local_rejections_happen_after_shared_admission(invoke, match):
    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    before = len(comm.records)
    with pytest.raises(ValueError, match=match):
        invoke(adapter)
    assert len(comm.records) == before + 1


class _InvalidOption:
    def __str__(self):
        raise ValueError("not printable")


def test_mpi_invalid_reduction_option_is_admitted_before_rejection():
    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    before = len(comm.records)
    with pytest.raises(ValueError, match="operation is invalid"):
        adapter.all_reduce(np.ones(2, np.float32), op=_InvalidOption())
    assert len(comm.records) == before + 1


@pytest.mark.parametrize("field", ["order", "shape", "dtype"])
def test_mpi_cross_rank_collective_mismatch_fails_before_transport(field):
    adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(admission_mismatch=field),
        _mpi=_FakeMPI,
    )
    with pytest.raises(RuntimeError, match="ranks disagree"):
        adapter.all_reduce(np.ones(2, np.float32))


def test_mpi_artifact_and_subgroup_mismatch_fail_before_transport():
    artifact_adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(artifact_mismatch=True),
        _mpi=_FakeMPI,
    )
    with pytest.raises(RuntimeError, match="artifact or communicator identity"):
        artifact_adapter.admit_artifact(
            "a" * 64, artifact_adapter.communicator_digest
        )

    subgroup_adapter = MPIRankAdapter(
        expected_world_size=2,
        communicator=_FakeComm(subgroup_mismatch=True),
        _mpi=_FakeMPI,
    )
    with pytest.raises(RuntimeError, match="subgroup membership"):
        subgroup_adapter.for_subgroup((0, 1))


def test_mpi_artifact_dtype_is_admitted_then_rejected():
    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    before = len(comm.records)
    with pytest.raises(ValueError, match="unsupported dtype"):
        adapter.all_reduce(
            np.ones(2, np.float32), _expected_dtype="f16"
        )
    assert len(comm.records) == before + 1


def test_mpi_malformed_artifact_and_subgroup_are_admitted_then_rejected():
    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    before = len(comm.records)
    with pytest.raises(ValueError, match="64 hex"):
        adapter.admit_artifact("z" * 64, adapter.communicator_digest)
    assert len(comm.records) == before + 1

    before = len(comm.records)
    with pytest.raises(ValueError, match="membership is invalid"):
        adapter.for_subgroup((0, object()))
    assert len(comm.records) == before + 1


def test_mpi_subgroup_permute_peer_error_is_admitted_before_rejection():
    comm = _FakeComm()
    adapter = MPIRankAdapter(
        expected_world_size=2, communicator=comm, _mpi=_FakeMPI,
    )
    subgroup = adapter.for_subgroup((0, 1))
    assert subgroup is not None
    before = len(comm.records)
    with pytest.raises(ValueError, match="non-empty one-to-one peer map"):
        subgroup.collective_permute(
            np.ones(2, np.float32), pairs=((0, 2),)
        )
    assert len(comm.records) == before + 1
    assert "outside its subgroup" in comm.records[-1][3][-2]
