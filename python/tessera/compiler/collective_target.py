"""Content-addressed portable packages for Tile collective transport."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, cast

from .collective_capabilities import CollectiveExecutionContract
from .pipeline_runtime import (
    CollectiveTransportRuntime,
    OneSidedDescriptor,
    OneSidedTransportRuntime,
    RankLocalCollectiveTransport,
    execute_one_sided_target_records,
    execute_target_collectives,
)
from .tile_ir import TILE_COLLECTIVE_OPS, TILE_METADATA_OPS, TileIRModule, TileOp


_SCHEMA = "tessera.collective.target.v3"
_PRODUCT_SCHEMA = "tessera.collective.product.v1"


def _walk(ops: Iterable[TileOp]) -> Iterable[TileOp]:
    for op in ops:
        yield op
        yield from _walk(op.body)


@dataclass(frozen=True)
class CollectiveTargetArtifact:
    """Ordered transport queue separated from architecture-owned compute."""

    target: str
    records: tuple[Mapping[str, Any], ...]
    execution: CollectiveExecutionContract
    digest: str
    communicator_digest: str = "unbound"
    schedule_artifacts: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "target": self.target,
            "digest": self.digest,
            "communicator_digest": self.communicator_digest,
            "schedule_artifacts": [dict(row) for row in self.schedule_artifacts],
            "execution": self.execution.to_dict(),
            "collectives": [dict(row) for row in self.records],
        }

    def execute(
        self, *, adapter: Any, tensors: Mapping[str, Iterable[Any]]
    ) -> CollectiveTransportRuntime | RankLocalCollectiveTransport | OneSidedTransportRuntime:
        if self.execution.backend == "mpi":
            if str(getattr(adapter, "backend", "")) != "mpi" or not bool(
                getattr(adapter, "rank_local", False)
            ):
                raise RuntimeError(
                    "MPI collective artifact requires a rank-local MPI adapter"
                )
            admit_artifact = getattr(adapter, "admit_artifact", None)
            if not callable(admit_artifact):
                raise RuntimeError(
                    "MPI rank adapter does not implement artifact admission"
                )
            admit_artifact(self.digest, self.communicator_digest)
        if self.execution.initiation != "host_collective":
            snapshot_query = getattr(adapter, "communicator_capability_snapshot", None)
            if not callable(snapshot_query):
                raise RuntimeError("advanced collective artifact requires communicator properties")
            snapshot = snapshot_query()
            runtime_digest = str(snapshot.get("digest", ""))
            if runtime_digest != self.execution.capability_digest:
                raise RuntimeError("collective communicator capability digest mismatch")
            rows = list(snapshot.get("ranks", ()))
            topology = dict(snapshot.get("topology", {}))
            lane = self.execution.transport_lane
            if lane == "copy_engine":
                if topology.get("cta_policy") != "zero":
                    raise RuntimeError("Copy Engine artifact requires a zero-CTA communicator")
                if not rows or not all(bool(row.get("device_api_support", False)) for row in rows):
                    raise RuntimeError("Copy Engine artifact requires symmetric device API support")
            elif lane == "gin_rma":
                one_sided_ops = {
                    "tessera_collective.window.register",
                    "tessera_collective.window.deregister",
                    "tessera_collective.put_signal",
                    "tessera_collective.signal",
                    "tessera_collective.wait_signal",
                }
                if not self.records or any(str(record.get("op")) not in one_sided_ops for record in self.records):
                    raise RuntimeError("GIN/RMA requires explicit one-sided target records")
                if not rows or not all(
                    bool(row.get("host_rma_support", False)) and int(row.get("gin_type", 0)) != 0 for row in rows
                ):
                    raise RuntimeError("GIN/RMA artifact requires host RMA and a nonzero GIN type")
                for record in self.records:
                    descriptor = OneSidedDescriptor.from_target_record(record)
                    if descriptor.kind in {"put_signal", "signal", "wait_signal"} and descriptor.peer >= len(rows):
                        raise RuntimeError(
                            f"one-sided peer {descriptor.peer} is outside the {len(rows)}-rank communicator"
                        )
                return execute_one_sided_target_records(self.records, adapter=adapter, buffers=tensors)
            elif lane == "gfx1250_dda":
                evidence_query = getattr(adapter, "selector_evidence_digest", None)
                if not callable(evidence_query):
                    raise RuntimeError("gfx1250 DDA artifact requires selector evidence")
                if str(evidence_query()) != self.execution.selector_evidence_digest:
                    raise RuntimeError("gfx1250 DDA selector evidence digest mismatch")
        return execute_target_collectives(
            self.records,
            adapter=adapter,
            tensors=tensors,
            artifact_digest=self.digest,
        )


@dataclass(frozen=True)
class NativeCollectiveProductArtifact:
    """A linear primal/tangent pair over one exact native transport queue."""

    collective: CollectiveTargetArtifact
    digest: str

    def execute(
        self,
        *,
        adapter: Any,
        primals: Mapping[str, Iterable[Any]],
        tangents: Mapping[str, Iterable[Any]],
    ) -> tuple[CollectiveTransportRuntime, CollectiveTransportRuntime]:
        backend = str(getattr(adapter, "backend", ""))
        world_size = int(getattr(adapter, "world_size", 0))
        if backend != self.collective.execution.backend or backend not in {"nccl", "rccl"}:
            raise RuntimeError("native collective product requires its bound NCCL/RCCL adapter")
        if world_size < 2:
            raise RuntimeError("native collective product requires multiple ranks")
        status_query = getattr(adapter, "status", None)
        if not callable(status_query) or getattr(status_query(), "status", "") != "hardware_runtime":
            raise RuntimeError("native collective product requires an available hardware runtime")
        primal_runtime = self.collective.execute(adapter=adapter, tensors=primals)
        tangent_runtime = self.collective.execute(adapter=adapter, tensors=tangents)
        return (
            cast(CollectiveTransportRuntime, primal_runtime),
            cast(CollectiveTransportRuntime, tangent_runtime),
        )


def build_native_collective_product_artifact(
    collective: CollectiveTargetArtifact,
) -> NativeCollectiveProductArtifact:
    """Seal the JVP of a native multi-rank linear collective artifact."""
    if collective.execution.backend not in {"nccl", "rccl"}:
        raise ValueError("native collective products require an NCCL/RCCL artifact")
    if not collective.records:
        raise ValueError("native collective product requires transport records")
    for record in collective.records:
        world_size = int(record.get("world_size", 0))
        if world_size < 2:
            raise ValueError("native collective product records require world_size >= 2")
        reduction = str(record.get("reduction", "none"))
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError(f"collective reduction {reduction!r} is nonlinear and has no JVP")
    payload = {
        "schema": _PRODUCT_SCHEMA,
        "collective_digest": collective.digest,
        "backend": collective.execution.backend,
        "target": collective.target,
        "linearity": "same_transport_on_primal_and_tangent",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return NativeCollectiveProductArtifact(collective, digest)


def package_one_sided_target_artifact(
    *,
    target: str,
    execution: CollectiveExecutionContract,
    records: Iterable[Mapping[str, Any]],
) -> CollectiveTargetArtifact:
    """Seal an ordered, rank-local GIN/RMA Target record stream."""
    if execution.transport_lane != "gin_rma":
        raise ValueError("one-sided artifact requires transport_lane=gin_rma")
    normalized = tuple(dict(record) for record in records)
    if not normalized:
        raise ValueError("one-sided artifact requires at least one record")
    live_windows: set[str] = set()
    for record in normalized:
        descriptor = OneSidedDescriptor.from_target_record(record)
        if descriptor.kind == "window.register":
            if not descriptor.strict_ordering:
                raise ValueError("GIN/RMA window registration requires strict ordering")
            if descriptor.window in live_windows:
                raise ValueError(f"one-sided window {descriptor.window!r} is registered twice")
            live_windows.add(descriptor.window)
        elif descriptor.kind == "window.deregister":
            if descriptor.window not in live_windows:
                raise ValueError(f"one-sided window {descriptor.window!r} is not live")
            live_windows.remove(descriptor.window)
        elif descriptor.kind == "put_signal" and descriptor.window not in live_windows:
            raise ValueError(f"put_signal references unregistered window {descriptor.window!r}")
    if live_windows:
        names = ", ".join(sorted(live_windows))
        raise ValueError(f"one-sided artifact leaks registered windows: {names}")
    payload = {
        "schema": _SCHEMA,
        "target": target,
        "communicator_digest": "unbound",
        "schedule_artifacts": [],
        "execution": execution.to_dict(),
        "collectives": normalized,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CollectiveTargetArtifact(target, normalized, execution, digest)


def lower_tile_collective_artifact(module: TileIRModule) -> CollectiveTargetArtifact:
    """Package every Tile collective without re-entering Graph/Schedule IR."""
    verification = module.verify()
    if not verification.ok:
        raise ValueError(verification.format())
    targets = {function.target for function in module.functions}
    if len(targets) != 1:
        raise ValueError("collective artifact requires one exact target")
    target = next(iter(targets), str(module.attrs.get("target", "cpu")))
    execution = CollectiveExecutionContract(
        backend=str(module.attrs.get("collective.backend", "portable")),
        initiation=str(module.attrs.get("collective.initiation", "host_collective")),
        transport_lane=str(module.attrs.get("collective.transport_lane", "standard")),
        target_arch=str(module.attrs.get("collective.target_arch", "unbound")),
        window_registration=str(module.attrs.get("collective.window_registration", "none")),
        ordering=str(module.attrs.get("collective.ordering", "default")),
        graph_capture_compatible=bool(module.attrs.get("collective.graph_capture_compatible", True)),
        capability_digest=str(module.attrs.get("collective.capability_digest", "unbound")),
        selector_evidence_digest=str(module.attrs.get("collective.selector_evidence_digest", "unbound")),
        backend_version=str(module.attrs.get("collective.backend_version", "unbound")),
        source_revision=str(module.attrs.get("collective.source_revision", "unbound")),
    )
    communicator_digest = str(
        module.attrs.get("collective.communicator_digest", "unbound")
    )
    if execution.backend == "mpi" and (
        len(communicator_digest) != 64
        or any(char not in "0123456789abcdef" for char in communicator_digest)
    ):
        raise ValueError(
            "MPI collective artifact requires a 64-hex communicator digest"
        )
    schedule_artifacts = tuple(
        {
            "function": function.name,
            "hash": str(op.attrs.get("hash", "")),
            **(
                {"schedule_hash": str(op.attrs["schedule_hash"])}
                if op.attrs.get("schedule_hash") is not None
                else {}
            ),
            **(
                {
                    "collective_binding_digest": str(
                        op.attrs["collective_binding_digest"]
                    )
                }
                if op.attrs.get("collective_binding_digest") is not None
                else {}
            ),
        }
        for function in module.functions
        for op in _walk(function.body)
        if op.op_name == "tile.artifact"
    )
    if execution.backend == "mpi" and (
        not schedule_artifacts
        or any(
            len(str(row["hash"])) != 64
            or any(
                char not in "0123456789abcdef" for char in str(row["hash"])
            )
            for row in schedule_artifacts
        )
    ):
        raise ValueError(
            "MPI collective artifact requires a 64-hex Schedule→Tile artifact identity"
        )
    records: list[dict[str, Any]] = []
    for function in module.functions:
        for op in _walk(function.body):
            if op.op_name in TILE_METADATA_OPS:
                continue
            if op.op_name not in TILE_COLLECTIVE_OPS:
                continue
            kind = op.op_name.removeprefix("tile.")
            source = str(op.attrs.get("source", op.operands[0] if op.operands else ""))
            result = str(op.attrs.get("result", op.result or source))
            if not source or not result:
                raise ValueError(f"{op.op_name} requires stable input/output identities")
            reduction = str(op.attrs.get("reduction", "none"))
            records.append(
                {
                    "op": f"tessera_collective.{kind}",
                    "function": function.name,
                    "ordinal": int(op.attrs.get("ordinal", len(records))),
                    "tensor": result,
                    "input": source,
                    "output": result,
                    "mesh_axis": str(op.attrs["mesh_axis"]),
                    "tensor_axis": int(op.attrs["tensor_axis"]),
                    "reduction": reduction,
                    "reshard_plan_digest": str(
                        op.attrs.get("reshard_plan_digest", "")
                    ),
                    "subgroup": list(op.attrs.get("subgroup", ())),
                    "region_path": list(op.attrs.get("region_path", ())),
                    "matching_rounds": list(op.attrs.get("matching_rounds", ())),
                    **({"world_size": int(op.attrs["world_size"])} if op.attrs.get("world_size") is not None else {}),
                    **({"dtype": str(op.attrs["dtype"])} if op.attrs.get("dtype") is not None else {}),
                    **(
                        {"chunk_bytes": int(op.attrs["chunk_bytes"])} if op.attrs.get("chunk_bytes") is not None else {}
                    ),
                    **(
                        {"scatter_axis": int(op.attrs["scatter_axis"])}
                        if op.attrs.get("scatter_axis") is not None
                        else {}
                    ),
                    **(
                        {"gather_axis": int(op.attrs["gather_axis"])} if op.attrs.get("gather_axis") is not None else {}
                    ),
                    **(
                        {
                            "source_peers": list(op.attrs["source_peers"]),
                            "target_peers": list(op.attrs["target_peers"]),
                        }
                        if op.op_name == "tile.collective_permute"
                        else {}
                    ),
                }
            )
    if not records:
        raise ValueError("Tile module contains no collective transport")
    if execution.backend == "mpi":
        ordinals = [int(record["ordinal"]) for record in records]
        if ordinals != list(range(len(records))):
            raise ValueError(
                "MPI collective artifact requires contiguous source-order ordinals"
            )
    payload = {
        "schema": _SCHEMA,
        "target": target,
        "communicator_digest": communicator_digest,
        "schedule_artifacts": schedule_artifacts,
        "execution": execution.to_dict(),
        "collectives": records,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CollectiveTargetArtifact(
        target,
        tuple(records),
        execution,
        digest,
        communicator_digest,
        schedule_artifacts,
    )


__all__ = [
    "CollectiveTargetArtifact",
    "NativeCollectiveProductArtifact",
    "build_native_collective_product_artifact",
    "lower_tile_collective_artifact",
    "package_one_sided_target_artifact",
]
