from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = (
    _ROOT
    / "src/collectives/tools/tessera-rccl-gin-smoke/gin_smoke.cpp"
).read_text()


def test_gin_harness_binds_common_native_launchers_without_mpi_dependency() -> None:
    for variable in (
        "TESSERA_GIN_RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "TESSERA_GIN_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "TESSERA_GIN_LOCAL_RANK",
        "TESSERA_GIN_RENDEZVOUS",
        "TESSERA_GIN_RUN_ID",
    ):
        assert variable in _SOURCE
    assert "ncclCommInitRank" in _SOURCE
    assert "mpi.h" not in _SOURCE


def test_gin_packet_executes_all_registered_one_sided_operations() -> None:
    for call in (
        "registerSymmetricWindow",
        "putSignalAsync",
        "signalAsync",
        "waitSignalAsync",
    ):
        assert call in _SOURCE
    assert '\\"put_signal\\",\\"signal\\",\\"wait_signal\\"' in _SOURCE
    assert "hipMemcpyDeviceToHost" in _SOURCE


def test_gin_packet_keeps_timing_domains_and_promotion_identity_explicit() -> None:
    for field in (
        "host_wall_ns_per_iteration",
        "hip_event_ns_per_iteration",
        "artifact_digest",
        "communicator_digest",
        "eligible_for_promotion",
        "rccl_version",
        "hip_runtime_version",
        "hip_driver_version",
    ):
        assert field in _SOURCE
    assert "tessera.rccl_gin_packet.v1" in _SOURCE
    assert "maxEventNs > 0.0" in _SOURCE
