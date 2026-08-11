from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera import collectives
from tessera.compiler.collective_capabilities import (
    CollectiveExecutionContract,
    NativeCollectiveCapabilities,
    RcclTransportRequest,
    RcclTransportEvidence,
    assess_rccl_transport,
    communicator_capability_snapshot,
    probe_native_collective_capabilities,
)
from tessera.compiler.collective_target import (
    CollectiveTargetArtifact,
    build_native_collective_product_artifact,
    lower_tile_collective_artifact,
    package_one_sided_target_artifact,
)
from tessera.compiler.pipeline_runtime import (
    OptimizerShardTransport,
    execute_pipeline_steps,
    execute_target_collectives,
)
from tessera.compiler.tile_ir import TileFunction, TileIRModule, TileOp


_ROOT = Path(__file__).resolve().parents[2]


def _record(kind: str, tensor: str, *, reduction: str = "none") -> dict[str, object]:
    return {
        "op": f"tessera_collective.{kind}",
        "tensor": tensor,
        "mesh_axis": "dp",
        "tensor_axis": 0,
        "reduction": reduction,
        "world_size": 2,
    }


def test_all_four_target_collectives_execute_on_functional_adapter() -> None:
    adapter = collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2})
    runtime = execute_target_collectives(
        [
            _record("all_reduce", "ar", reduction="sum"),
            _record("reduce_scatter", "rs", reduction="sum"),
            _record("all_gather", "ag"),
            _record("all_to_all", "aa"),
        ],
        adapter=adapter,
        tensors={
            "ar": [np.array([1, 2], np.float32), np.array([3, 4], np.float32)],
            "rs": [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 4],
            "ag": [np.array([0, 1], np.float32), np.array([2, 3], np.float32)],
            "aa": [np.array([0, 1, 2, 3], np.float32), np.array([10, 11, 12, 13], np.float32)],
        },
    )

    for value in runtime.values("ar"):
        np.testing.assert_array_equal(value, np.array([4, 6], np.float32))
    np.testing.assert_array_equal(runtime.values("rs")[0], np.array([4, 6], np.float32))
    np.testing.assert_array_equal(runtime.values("rs")[1], np.array([8, 10], np.float32))
    for value in runtime.values("ag"):
        np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(runtime.values("aa")[0], np.array([0, 1, 10, 11], np.float32))
    np.testing.assert_array_equal(runtime.values("aa")[1], np.array([2, 3, 12, 13], np.float32))


def test_native_collective_product_requires_linear_hardware_transport() -> None:
    contract = CollectiveExecutionContract(backend="rccl")
    artifact = CollectiveTargetArtifact(
        "rocm_gfx1151", (_record("all_reduce", "x", reduction="sum"),),
        contract, "a" * 64,
    )
    product = build_native_collective_product_artifact(artifact)

    class Status:
        status = "hardware_runtime"

    class FakeRCCL:
        backend = "rccl"
        world_size = 2
        mesh_axes = {"dp": 2}

        def status(self):
            return Status()

        def all_reduce(self, values, *, op="sum"):
            total = values[0] + values[1]
            return [total.copy(), total.copy()]

    primal_runtime, tangent_runtime = product.execute(
        adapter=FakeRCCL(),
        primals={"x": [np.array([1.0]), np.array([2.0])]},
        tangents={"x": [np.array([3.0]), np.array([5.0])]},
    )
    np.testing.assert_array_equal(primal_runtime.values("x")[0], [3.0])
    np.testing.assert_array_equal(tangent_runtime.values("x")[0], [8.0])

    nonlinear = CollectiveTargetArtifact(
        "rocm_gfx1151", (_record("all_reduce", "x", reduction="max"),),
        contract, "b" * 64,
    )
    with pytest.raises(ValueError, match="nonlinear"):
        build_native_collective_product_artifact(nonlinear)


def test_target_collective_rejects_runtime_topology_mismatch() -> None:
    adapter = collectives.adapter(backend="single_process", world_size=1)
    with pytest.raises(RuntimeError, match="targets world_size=2"):
        execute_target_collectives(
            [_record("all_reduce", "x", reduction="sum")],
            adapter=adapter,
            tensors={"x": [np.ones(2, np.float32)]},
        )


def test_target_collective_rejects_unknown_or_subgroup_mesh_axis() -> None:
    adapter = collectives.adapter(
        backend="mock", world_size=4, mesh_axes={"dp": 2, "tp": 2}
    )
    with pytest.raises(RuntimeError, match="subgroup transport is not implemented"):
        execute_target_collectives(
            [
                {
                    **_record("all_reduce", "x", reduction="sum"),
                    "world_size": 2,
                }
            ],
            adapter=adapter,
            tensors={"x": [np.ones(1, np.float32) for _ in range(4)]},
        )
    with pytest.raises(RuntimeError, match="unknown mesh axis"):
        execute_target_collectives(
            [
                {
                    **_record("all_reduce", "x", reduction="sum"),
                    "mesh_axis": "missing",
                    "world_size": 4,
                }
            ],
            adapter=adapter,
            tensors={"x": [np.ones(1, np.float32) for _ in range(4)]},
        )


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("mean", np.array([2, 3], dtype=np.float32)),
        ("prod", np.array([3, 8], dtype=np.float32)),
    ],
)
def test_functional_reduction_modes(
    reduction: str, expected: np.ndarray
) -> None:
    runtime = execute_target_collectives(
        [_record("all_reduce", "x", reduction=reduction)],
        adapter=collectives.adapter(
            backend="mock", world_size=2, mesh_axes={"dp": 2}
        ),
        tensors={
            "x": [
                np.array([1, 2], dtype=np.float32),
                np.array([3, 4], dtype=np.float32),
            ]
        },
    )
    for value in runtime.values("x"):
        np.testing.assert_array_equal(value, expected)


def test_target_collective_rejects_non_target_operation() -> None:
    adapter = collectives.adapter(backend="single_process", world_size=1)
    with pytest.raises(ValueError, match=r"tessera_collective\.\*"):
        execute_target_collectives(
            [{"op": "tile.all_reduce", "tensor": "x"}],
            adapter=adapter,
            tensors={"x": [np.ones(2, np.float32)]},
        )


def test_tile_artifact_preserves_lineage_and_executes_distinct_outputs() -> None:
    operations = []
    for ordinal, (kind, source, result, reduction) in enumerate(
        (
            ("all_reduce", "x_ar", "y_ar", "sum"),
            ("reduce_scatter", "x_rs", "y_rs", "sum"),
            ("all_gather", "x_ag", "y_ag", "none"),
            ("all_to_all", "x_aa", "y_aa", "none"),
        )
    ):
        operations.append(
            TileOp(
                f"tile.{kind}",
                {
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
                    "chunk_bytes": 4096,
                },
                operands=[f"%{source}"],
                result=result,
            )
        )
    module = TileIRModule(
        functions=[TileFunction("transport", operations, target="rocm_gfx1151")]
    )
    artifact = lower_tile_collective_artifact(module)
    assert artifact.to_dict()["schema"] == "tessera.collective.target.v2"
    assert artifact.execution.initiation == "host_collective"
    assert len(artifact.digest) == 64
    assert [row["output"] for row in artifact.records] == [
        "y_ar", "y_rs", "y_ag", "y_aa"
    ]
    assert artifact.records[0]["dtype"] == "f32"
    assert artifact.records[0]["chunk_bytes"] == 4096

    runtime = artifact.execute(
        adapter=collectives.adapter(backend="mock", world_size=2, mesh_axes={"dp": 2}),
        tensors={
            "x_ar": [np.array([1], np.float32), np.array([2], np.float32)],
            "x_rs": [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 4],
            "x_ag": [np.array([0, 1], np.float32), np.array([2, 3], np.float32)],
            "x_aa": [np.array([0, 1, 2, 3], np.float32), np.array([10, 11, 12, 13], np.float32)],
        },
    )
    for value in runtime.values("y_ar"):
        np.testing.assert_array_equal(value, np.array([3], np.float32))
    np.testing.assert_array_equal(runtime.values("y_rs")[0], np.array([4, 6], np.float32))
    for value in runtime.values("y_ag"):
        np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32))


def test_pipeline_step_submits_each_collective_once() -> None:
    class CountingAdapter:
        world_size = 1

        def __init__(self) -> None:
            self.calls = 0

        def all_reduce(
            self, values: list[np.ndarray], *, op: str
        ) -> list[np.ndarray]:
            del op
            self.calls += 1
            return values

    adapter = CountingAdapter()
    transport = OptimizerShardTransport(
        adapter, {"x": [np.array([1], dtype=np.float32)]}
    )
    result = execute_pipeline_steps(
        [
            {
                "clock": 0,
                "micro_batch": 0,
                "phase": "backward",
                "region": "steady",
                "stage": 0,
                "collectives": [
                    {"kind": "all_reduce", "tensor": "x", "op": "sum"}
                ],
            }
        ],
        run_stage=lambda _step: None,
        collective_runtime=transport,
    )

    assert result.collective_count == 1
    assert adapter.calls == 1


def test_device_initiated_artifact_binds_capabilities_and_ordering() -> None:
    capability_digest = "a" * 64
    module = TileIRModule(
        functions=[
            TileFunction(
                "transport",
                [
                    TileOp(
                        "tile.all_reduce",
                        {
                            "source": "x",
                            "result": "y",
                            "ordinal": 0,
                            "kind": "all_reduce",
                            "mesh_axis": "dp",
                            "tensor_axis": 0,
                            "reduction": "sum",
                            "effect": "collective",
                            "world_size": 2,
                        },
                        operands=["%x"],
                        result="y",
                    )
                ],
                target="rocm_gfx1151",
            )
        ],
        attrs={
            "tessera.ir.level": "tile",
            "collective.backend": "rccl",
            "collective.initiation": "device_kernel",
            "collective.window_registration": "symmetric",
            "collective.ordering": "strict",
            "collective.graph_capture_compatible": True,
            "collective.capability_digest": capability_digest,
            "collective.backend_version": "2.30.4",
            "collective.source_revision": "5bc651a82683",
        },
    )
    artifact = lower_tile_collective_artifact(module)
    payload = artifact.to_dict()
    assert payload["execution"]["initiation"] == "device_kernel"
    assert payload["execution"]["ordering"] == "strict"
    assert payload["execution"]["capability_digest"] == capability_digest


def test_device_initiated_contract_fails_without_window_or_capability() -> None:
    with pytest.raises(ValueError, match="symmetric window"):
        CollectiveExecutionContract(backend="rccl", initiation="device_kernel")
    with pytest.raises(ValueError, match="capability digest"):
        CollectiveExecutionContract(
            backend="rccl",
            initiation="device_kernel",
            window_registration="symmetric",
        )
    with pytest.raises(ValueError, match="not graph-capture compatible"):
        CollectiveExecutionContract(
            backend="rccl",
            initiation="copy_engine_zero_cu",
            window_registration="symmetric",
            capability_digest="b" * 64,
        )


def test_missing_native_library_reports_declarations_not_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ctypes.util.find_library", lambda _name: None)
    capabilities = probe_native_collective_capabilities(
        "rccl", target_arch="gfx1151"
    )
    assert not capabilities.host_collectives_declared
    assert not capabilities.generic_lsa_candidate
    assert len(capabilities.digest) == 64


def test_communicator_snapshot_is_rank_ordered_and_content_addressed() -> None:
    topology = {
        "schema": "tessera.collective_topology.v1",
        "backend": "rccl",
        "world_size": 2,
        "device_ordinals": [0, 1],
        "rank_order": [0, 1],
    }
    ranks = [
        {"rank": 1, "world_size": 2, "device": 1,
         "device_api_support": True, "lsa_team_count": 1},
        {"rank": 0, "world_size": 2, "device": 0,
         "device_api_support": True, "lsa_team_count": 1},
    ]
    snapshot = communicator_capability_snapshot(
        backend="rccl", topology=topology, version_code=23004, ranks=ranks
    )
    assert [row["rank"] for row in snapshot["ranks"]] == [0, 1]
    assert len(snapshot["digest"]) == 64


def test_device_artifact_rejects_communicator_digest_mismatch() -> None:
    contract = CollectiveExecutionContract(
        backend="rccl",
        initiation="device_kernel",
        window_registration="symmetric",
        capability_digest="a" * 64,
    )
    artifact = CollectiveTargetArtifact("rocm_gfx1151", (), contract, "c" * 64)

    class WrongCommunicator:
        def communicator_capability_digest(self) -> str:
            return "b" * 64

        def communicator_capability_snapshot(self) -> dict[str, object]:
            return {"digest": "b" * 64, "ranks": [], "topology": {}}

    with pytest.raises(RuntimeError, match="capability digest mismatch"):
        artifact.execute(adapter=WrongCommunicator(), tensors={})


def _rccl_capabilities(*, arch: str = "gfx1151") -> NativeCollectiveCapabilities:
    return NativeCollectiveCapabilities(
        backend="rccl",
        target_arch=arch,
        library="librccl.so",
        version_code=23004,
        driver_version=71200000,
        host_collectives_declared=True,
        symmetric_windows_declared=True,
        strict_ordering_declared=True,
        one_sided_rma_declared=True,
        device_api_query_declared=True,
        generic_lsa_candidate=arch == "gfx1151",
        dda_candidate=arch == "gfx1250",
    )


def _communicator_rows(*, gin: int = 0) -> dict[str, object]:
    return {
        "digest": "d" * 64,
        "ranks": [
            {
                "rank": rank,
                "world_size": 2,
                "device_api_support": True,
                "host_rma_support": True,
                "gin_type": gin,
            }
            for rank in range(2)
        ]
    }


def test_copy_engine_and_gin_rma_have_independent_legality() -> None:
    copy_engine = assess_rccl_transport(
        RcclTransportRequest(
            lane="copy_engine",
            target_arch="gfx1151",
            op="all_gather",
            message_bytes=4096,
            symmetric_buffers=True,
            cta_policy_zero=True,
        ),
        library=_rccl_capabilities(),
        communicator=_communicator_rows(),
    )
    assert copy_engine.legal
    assert not copy_engine.promotion_eligible

    gin_without_network = assess_rccl_transport(
        RcclTransportRequest(
            lane="gin_rma",
            target_arch="gfx1151",
            op="put_signal",
            message_bytes=4096,
            node_count=2,
            symmetric_buffers=True,
        ),
        library=_rccl_capabilities(),
        communicator=_communicator_rows(gin=0),
    )
    assert not gin_without_network.legal
    assert any("host RMA plus GIN" in reason for reason in gin_without_network.reasons)

    gin = assess_rccl_transport(
        RcclTransportRequest(
            lane="gin_rma",
            target_arch="gfx1151",
            op="put_signal",
            message_bytes=4096,
            node_count=2,
            symmetric_buffers=True,
        ),
        library=_rccl_capabilities(),
        communicator=_communicator_rows(gin=2),
    )
    assert gin.legal

    ce_evidence = RcclTransportEvidence(
        lane="copy_engine",
        target_arch="gfx1151",
        artifact_digest="e" * 64,
        communicator_digest="d" * 64,
        exact_device=True,
        correctness_passed=True,
        route_proven=True,
        provenance="bare_metal_rccl_route_packet",
    )
    gin_with_ce_evidence = assess_rccl_transport(
        RcclTransportRequest(
            lane="gin_rma",
            target_arch="gfx1151",
            op="put_signal",
            message_bytes=4096,
            node_count=2,
            symmetric_buffers=True,
            artifact_digest="e" * 64,
        ),
        library=_rccl_capabilities(),
        communicator=_communicator_rows(gin=2),
        evidence=ce_evidence,
    )
    assert gin_with_ce_evidence.legal
    assert not gin_with_ce_evidence.promotion_eligible


def test_gfx1250_dda_is_selector_evidence_gated() -> None:
    with pytest.raises(ValueError, match="selector evidence digest"):
        CollectiveExecutionContract(
            backend="rccl",
            initiation="selector",
            transport_lane="gfx1250_dda",
            target_arch="gfx1250",
            capability_digest="a" * 64,
        )
    contract = CollectiveExecutionContract(
        backend="rccl",
        initiation="selector",
        transport_lane="gfx1250_dda",
        target_arch="gfx1250",
        capability_digest="a" * 64,
        selector_evidence_digest="b" * 64,
    )
    assert contract.target_arch == "gfx1250"

    wrong_arch = assess_rccl_transport(
        RcclTransportRequest(
            lane="gfx1250_dda",
            target_arch="gfx1151",
            op="all_reduce",
            message_bytes=4096,
            reduction="sum",
        ),
        library=_rccl_capabilities(),
        communicator=_communicator_rows(),
    )
    assert not wrong_arch.legal

    candidate = assess_rccl_transport(
        RcclTransportRequest(
            lane="gfx1250_dda",
            target_arch="gfx1250",
            op="all_reduce",
            message_bytes=4096,
            reduction="sum",
        ),
        library=_rccl_capabilities(arch="gfx1250"),
        communicator=_communicator_rows(),
    )
    assert candidate.legal
    assert not candidate.promotion_eligible


def test_copy_engine_artifact_rejects_nonzero_cta_runtime() -> None:
    contract = CollectiveExecutionContract(
        backend="rccl",
        initiation="copy_engine_zero_cu",
        transport_lane="copy_engine",
        target_arch="gfx1151",
        window_registration="symmetric",
        graph_capture_compatible=False,
        capability_digest="a" * 64,
    )
    artifact = CollectiveTargetArtifact("rocm_gfx1151", (), contract, "c" * 64)

    class DefaultCtaCommunicator:
        def communicator_capability_digest(self) -> str:
            return "a" * 64

        def communicator_capability_snapshot(self) -> dict[str, object]:
            return {
                "digest": "a" * 64,
                "topology": {"cta_policy": "default"},
                "ranks": [
                    {"device_api_support": True},
                    {"device_api_support": True},
                ],
            }

    with pytest.raises(RuntimeError, match="zero-CTA communicator"):
        artifact.execute(adapter=DefaultCtaCommunicator(), tensors={})


def test_gin_artifact_rejects_host_rma_without_gin() -> None:
    contract = CollectiveExecutionContract(
        backend="rccl",
        initiation="gpu_network_rma",
        transport_lane="gin_rma",
        target_arch="gfx1151",
        window_registration="symmetric",
        ordering="strict",
        capability_digest="a" * 64,
    )
    artifact = CollectiveTargetArtifact(
        "rocm_gfx1151",
        ({"op": "tessera_collective.put_signal"},),
        contract,
        "c" * 64,
    )

    class LocalOnlyRmaCommunicator:
        def communicator_capability_digest(self) -> str:
            return "a" * 64

        def communicator_capability_snapshot(self) -> dict[str, object]:
            return {
                "digest": "a" * 64,
                "topology": {"world_size": 2},
                "ranks": [
                    {"host_rma_support": True, "gin_type": 0},
                    {"host_rma_support": True, "gin_type": 0},
                ],
            }

    with pytest.raises(RuntimeError, match="nonzero GIN type"):
        artifact.execute(adapter=LocalOnlyRmaCommunicator(), tensors={})


def _gin_execution_contract() -> CollectiveExecutionContract:
    return CollectiveExecutionContract(
        backend="rccl",
        initiation="gpu_network_rma",
        transport_lane="gin_rma",
        target_arch="gfx1151",
        window_registration="symmetric",
        ordering="strict",
        capability_digest="d" * 64,
    )


def _gin_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "op": "tessera_collective.window.register",
            "window": "remote",
            "buffer": "destination",
            "bytes": 32,
            "strict_ordering": True,
        },
        {
            "op": "tessera_collective.put_signal",
            "window": "remote",
            "source": "source",
            "count": 8,
            "dtype": "f32",
            "peer": 1,
            "peer_window_offset": 0,
            "signal_index": 3,
            "rma_context": 0,
        },
        {
            "op": "tessera_collective.signal",
            "peer": 1,
            "signal_index": 4,
            "rma_context": 0,
        },
        {
            "op": "tessera_collective.wait_signal",
            "peer": 1,
            "operation_count": 2,
            "signal_index": 3,
            "rma_context": 0,
        },
        {
            "op": "tessera_collective.window.deregister",
            "window": "remote",
        },
    )


def test_registered_one_sided_artifact_executes_ordered_runtime_calls() -> None:
    calls: list[tuple[object, ...]] = []

    class Window:
        def close(self) -> None:
            calls.append(("window.deregister",))

    class GinAdapter:
        def communicator_capability_snapshot(self) -> dict[str, object]:
            return {
                "digest": "d" * 64,
                "topology": {"world_size": 2},
                "ranks": [
                    {"host_rma_support": True, "gin_type": 2},
                    {"host_rma_support": True, "gin_type": 2},
                ],
            }

        def register_symmetric_window(
            self, buffer, bytes_: int, *, strict_ordering: bool
        ) -> Window:
            calls.append(("window.register", buffer, bytes_, strict_ordering))
            return Window()

        def put_signal(
            self, source, count, dtype, peer, window, offset,
            *, signal_index, context,
        ) -> None:
            calls.append(
                ("put_signal", source, count, dtype, peer, offset,
                 signal_index, context)
            )

        def signal(self, peer, *, signal_index, context) -> None:
            calls.append(("signal", peer, signal_index, context))

        def wait_signal(
            self, peer, *, operation_count, signal_index, context
        ) -> None:
            calls.append(
                ("wait_signal", peer, operation_count, signal_index, context)
            )

    artifact = package_one_sided_target_artifact(
        target="rocm_gfx1151",
        execution=_gin_execution_contract(),
        records=_gin_records(),
    )
    source = object()
    destination = object()
    artifact.execute(
        adapter=GinAdapter(),
        tensors={"source": source, "destination": destination},
    )
    assert [call[0] for call in calls] == [
        "window.register", "put_signal", "signal", "wait_signal",
        "window.deregister",
    ]
    assert calls[0][1:] == (destination, 32, True)
    assert calls[1][1] is source
    assert len(artifact.digest) == 64


def test_one_sided_packaging_rejects_window_lifetime_errors() -> None:
    with pytest.raises(ValueError, match="unregistered window"):
        package_one_sided_target_artifact(
            target="rocm_gfx1151",
            execution=_gin_execution_contract(),
            records=(_gin_records()[1],),
        )
    with pytest.raises(ValueError, match="leaks registered windows"):
        package_one_sided_target_artifact(
            target="rocm_gfx1151",
            execution=_gin_execution_contract(),
            records=(_gin_records()[0],),
        )


def test_active_compiler_sources_do_not_reintroduce_collective_pseudo_dialect() -> None:
    roots = (
        _ROOT / "src" / "transforms",
        _ROOT / "src" / "solvers" / "scaling_resilience",
        _ROOT / "tests" / "tessera-ir",
    )
    forbidden = (
        "tessera.collective.all_reduce",
        "tessera.collective.reduce_scatter",
        "tessera.collective.all_gather",
        "tessera.collective.all_to_all",
        "tessera.collective.await",
        "!tessera.collective.future",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".td", ".mlir"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []
