"""Truthful native-collective capability and artifact contracts.

The runtime library, communicator topology, and exact target jointly determine
whether device-initiated or one-sided transport is legal.  A library symbol or
an architecture name is therefore recorded as *declared* capability, never as
execution evidence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


_CAPABILITY_SCHEMA = "tessera.collective.capabilities.v1"
_EXECUTION_SCHEMA = "tessera.collective.execution.v2"
_COMMUNICATOR_SCHEMA = "tessera.collective.communicator.v1"
_INITIATION_MODES = {
    "host_collective",
    "device_kernel",
    "copy_engine_zero_cu",
    "gpu_network_rma",
    "selector",
}


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class NativeCollectiveCapabilities:
    backend: str
    target_arch: str
    library: str
    version_code: int
    driver_version: int
    host_collectives_declared: bool
    symmetric_windows_declared: bool
    strict_ordering_declared: bool
    one_sided_rma_declared: bool
    device_api_query_declared: bool
    generic_lsa_candidate: bool
    dda_candidate: bool
    provenance: str = "runtime_symbol_probe"

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema": _CAPABILITY_SCHEMA, **asdict(self)}
        payload["digest"] = _digest(payload)
        return payload

    @property
    def digest(self) -> str:
        return str(self.to_dict()["digest"])


@dataclass(frozen=True)
class CollectiveExecutionContract:
    """Content-addressed choice at the Schedule/Tile-to-runtime boundary."""

    backend: str = "portable"
    initiation: str = "host_collective"
    transport_lane: str = "standard"
    target_arch: str = "unbound"
    window_registration: str = "none"
    ordering: str = "default"
    graph_capture_compatible: bool = True
    capability_digest: str = "unbound"
    selector_evidence_digest: str = "unbound"
    backend_version: str = "unbound"
    source_revision: str = "unbound"

    def __post_init__(self) -> None:
        if self.backend not in {"portable", "mpi", "nccl", "rccl", "metal"}:
            raise ValueError(f"unknown collective backend {self.backend!r}")
        if self.initiation not in _INITIATION_MODES:
            raise ValueError(f"unknown collective initiation {self.initiation!r}")
        if self.transport_lane not in {
            "standard", "copy_engine", "gin_rma", "gfx1250_dda"
        }:
            raise ValueError(f"unknown collective transport lane {self.transport_lane!r}")
        if self.transport_lane != "standard" and self.backend != "rccl":
            raise ValueError(
                f"{self.transport_lane} is an RCCL-owned transport lane"
            )
        if self.window_registration not in {"none", "symmetric"}:
            raise ValueError("collective window registration must be none or symmetric")
        if self.ordering not in {"default", "strict"}:
            raise ValueError("collective ordering must be default or strict")
        requires_window = self.initiation in {
            "device_kernel", "copy_engine_zero_cu", "gpu_network_rma"
        }
        if requires_window and self.window_registration != "symmetric":
            raise ValueError(f"{self.initiation} requires symmetric window registration")
        if self.initiation == "copy_engine_zero_cu" and self.graph_capture_compatible:
            raise ValueError("copy-engine zero-CU transport is not graph-capture compatible")
        lane_initiation = {
            "copy_engine": "copy_engine_zero_cu",
            "gin_rma": "gpu_network_rma",
            "gfx1250_dda": "selector",
        }
        expected = lane_initiation.get(self.transport_lane)
        if expected is not None and self.initiation != expected:
            raise ValueError(
                f"{self.transport_lane} requires initiation={expected}"
            )
        if self.transport_lane == "gin_rma" and self.ordering != "strict":
            raise ValueError("GIN/RMA requires strict ordering")
        if self.transport_lane == "gfx1250_dda" and self.target_arch != "gfx1250":
            raise ValueError("gfx1250 DDA cannot be selected on another architecture")
        if (
            self.transport_lane == "gfx1250_dda"
            and len(self.selector_evidence_digest) != 64
        ):
            raise ValueError("gfx1250 DDA requires a selector evidence digest")
        if self.initiation != "host_collective" and len(self.capability_digest) != 64:
            raise ValueError(
                "non-host collective initiation requires a capability digest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _EXECUTION_SCHEMA, **asdict(self)}


def communicator_capability_snapshot(
    *,
    backend: str,
    topology: dict[str, Any],
    version_code: int,
    ranks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Seal properties returned by initialized communicators.

    Unlike the library probe, this is topology-specific runtime evidence. It
    still is not an exact-device correctness or performance packet.
    """
    if backend not in {"nccl", "rccl"}:
        raise ValueError("communicator snapshot requires nccl or rccl")
    world_size = int(topology["world_size"])
    if len(ranks) != world_size:
        raise ValueError("communicator snapshot requires one row per rank")
    ordered = sorted((dict(row) for row in ranks), key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in ordered] != list(range(world_size)):
        raise ValueError("communicator ranks must be contiguous and unique")
    if any(int(row["world_size"]) != world_size for row in ordered):
        raise ValueError("communicator properties disagree with topology world size")
    payload = {
        "schema": _COMMUNICATOR_SCHEMA,
        "backend": backend,
        "version_code": int(version_code),
        "topology": dict(topology),
        "ranks": ordered,
    }
    payload["digest"] = _digest(payload)
    return payload


@dataclass(frozen=True)
class RcclTransportRequest:
    lane: str
    target_arch: str
    op: str
    message_bytes: int
    dtype: str = "f32"
    reduction: str = "none"
    world_size: int = 2
    node_count: int = 1
    graph_capture: bool = False
    symmetric_buffers: bool = False
    cta_policy_zero: bool = False
    ce_allreduce_enabled: bool = False
    artifact_digest: str = "unbound"


@dataclass(frozen=True)
class RcclTransportAssessment:
    lane: str
    legal: bool
    promotion_eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RcclTransportEvidence:
    lane: str
    target_arch: str
    artifact_digest: str
    communicator_digest: str
    exact_device: bool
    correctness_passed: bool
    route_proven: bool
    provenance: str

    def __post_init__(self) -> None:
        if self.lane not in {"copy_engine", "gin_rma", "gfx1250_dda"}:
            raise ValueError(f"unknown RCCL evidence lane {self.lane!r}")
        if len(self.artifact_digest) != 64 or len(self.communicator_digest) != 64:
            raise ValueError("RCCL transport evidence requires content digests")
        if not self.provenance:
            raise ValueError("RCCL transport evidence requires provenance")


def assess_rccl_transport(
    request: RcclTransportRequest,
    *,
    library: NativeCollectiveCapabilities,
    communicator: dict[str, Any] | None,
    evidence: RcclTransportEvidence | None = None,
) -> RcclTransportAssessment:
    """Evaluate public legality without reimplementing RCCL's selector."""
    reasons: list[str] = []
    if request.lane not in {"copy_engine", "gin_rma", "gfx1250_dda"}:
        raise ValueError(f"unknown RCCL transport lane {request.lane!r}")
    if library.backend != "rccl":
        reasons.append("requires RCCL")
    if request.message_bytes <= 0 or request.world_size < 2:
        reasons.append("requires a non-empty multi-rank operation")

    rows = list((communicator or {}).get("ranks", ()))
    if len(rows) != request.world_size:
        reasons.append("requires initialized communicator properties for every rank")
    symmetric_support = bool(rows) and all(
        bool(row.get("device_api_support", False)) for row in rows
    )
    host_rma = bool(rows) and all(bool(row.get("host_rma_support", False)) for row in rows)
    gin = bool(rows) and all(int(row.get("gin_type", 0)) != 0 for row in rows)

    if request.lane == "copy_engine":
        if library.driver_version < 71200000:
            reasons.append("requires ROCm driver 7.12 or newer")
        if request.node_count != 1:
            reasons.append("Copy Engine collectives are single-node only")
        if request.graph_capture:
            reasons.append("Copy Engine collectives are not graph-capture safe")
        if not request.symmetric_buffers or not symmetric_support:
            reasons.append("requires registered symmetric buffers and communicator support")
        if not request.cta_policy_zero:
            reasons.append("requires NCCL_CTA_POLICY_ZERO at communicator initialization")
        supported = {"all_gather", "all_to_all", "gather", "scatter", "all_reduce"}
        if request.op not in supported:
            reasons.append(f"Copy Engine does not implement {request.op}")
        if request.op == "all_reduce":
            if not request.ce_allreduce_enabled:
                reasons.append("CE AllReduce requires RCCL_CE_ALLREDUCE=1")
            if not (4 << 20) <= request.message_bytes <= (256 << 20):
                reasons.append("CE AllReduce requires a 4 MiB to 256 MiB message")
            element_bytes = {
                "i8": 1, "u8": 1, "f16": 2, "bf16": 2,
                "i32": 4, "u32": 4, "f32": 4,
                "i64": 8, "u64": 8, "f64": 8,
            }.get(request.dtype)
            if element_bytes is None:
                reasons.append("CE AllReduce dtype has no element-width contract")
            elif (
                request.message_bytes % element_bytes
                or (request.message_bytes // element_bytes) % request.world_size
            ):
                reasons.append("CE AllReduce requires equal per-rank shards")
            if request.reduction not in {"sum", "prod", "min", "max"}:
                reasons.append("CE AllReduce requires a standard reduction")
            if request.dtype in {"fp8e4m3", "fp8e5m2"}:
                reasons.append("CE AllReduce does not support FP8")
    elif request.lane == "gin_rma":
        if request.node_count < 2:
            reasons.append("GIN/RMA requires a multi-node communicator")
        if not library.one_sided_rma_declared:
            reasons.append("RCCL one-sided RMA symbols are unavailable")
        if not host_rma or not gin:
            reasons.append("communicator does not expose host RMA plus GIN")
        if not request.symmetric_buffers:
            reasons.append("GIN/RMA requires registered peer windows")
        if request.op not in {"put_signal", "signal", "wait_signal"}:
            reasons.append("GIN/RMA request must be an explicit one-sided operation")
    else:
        if request.target_arch != "gfx1250":
            reasons.append("DDA lane is architecture-owned by gfx1250")
        if not library.dda_candidate:
            reasons.append("installed RCCL is not a declared gfx1250 DDA candidate")
        if request.op not in {
            "all_reduce", "all_gather", "reduce_scatter", "all_to_all"
        }:
            reasons.append(f"gfx1250 DDA does not implement {request.op}")
        if request.dtype not in {"f32", "f16", "bf16"}:
            reasons.append("gfx1250 DDA supports f32/f16/bf16 in this contract")
        if request.op in {"all_reduce", "reduce_scatter"} and request.reduction != "sum":
            reasons.append("gfx1250 DDA reductions require sum")
        if request.message_bytes % 16:
            reasons.append("gfx1250 DDA requires a 16-byte-aligned payload")

    legal = not reasons
    evidence_matches = bool(
        evidence is not None
        and evidence.lane == request.lane
        and evidence.target_arch == request.target_arch
        and evidence.exact_device
        and evidence.correctness_passed
        and evidence.route_proven
        and evidence.artifact_digest == request.artifact_digest
        and communicator is not None
        and evidence.communicator_digest == communicator.get("digest")
    )
    return RcclTransportAssessment(
        lane=request.lane,
        legal=legal,
        promotion_eligible=legal and evidence_matches,
        reasons=tuple(reasons),
    )


def probe_native_collective_capabilities(
    backend: str, *, target_arch: str
) -> NativeCollectiveCapabilities:
    """Probe declarations without upgrading them to execution evidence."""
    if backend not in {"nccl", "rccl"}:
        raise ValueError("native collective capability probe requires nccl or rccl")
    library = ctypes.util.find_library(backend)
    if not library:
        return NativeCollectiveCapabilities(
            backend, target_arch, "unavailable", 0, 0,
            False, False, False, False, False,
            generic_lsa_candidate=False, dda_candidate=False,
            provenance="library_not_found",
        )
    handle = ctypes.CDLL(library)

    def declared(name: str) -> bool:
        return getattr(handle, name, None) is not None

    version = ctypes.c_int()
    get_version = getattr(handle, "ncclGetVersion", None)
    version_code = 0
    if get_version is not None:
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        if get_version(ctypes.byref(version)) == 0:
            version_code = int(version.value)
    # Driver discovery belongs to the native evidence collector. Calling the
    # HIP version entry points from a declaration-only Python probe aborts on
    # some WSL runtimes, and a library SONAME is not a driver contract. Zero
    # therefore means unknown and fails every driver-version eligibility gate.
    driver_version = 0

    host = all(
        declared(name)
        for name in (
            "ncclAllReduce", "ncclReduceScatter", "ncclAllGather",
            "ncclSend", "ncclRecv",
        )
    )
    windows = all(
        declared(name)
        for name in ("ncclCommWindowRegister", "ncclCommWindowDeregister")
    )
    one_sided = all(
        declared(name)
        for name in ("ncclPutSignal", "ncclSignal", "ncclWaitSignal")
    )
    query = declared("ncclCommQueryProperties")

    # These are source-routing candidates, not eligibility decisions. Runtime
    # communicator properties and exact-device evidence remain mandatory.
    generic_lsa = backend == "rccl" and target_arch == "gfx1151" and windows
    dda = backend == "rccl" and target_arch in {"gfx942", "gfx950", "gfx1250"}
    return NativeCollectiveCapabilities(
        backend=backend,
        target_arch=target_arch,
        library=library,
        version_code=version_code,
        driver_version=driver_version,
        host_collectives_declared=host,
        symmetric_windows_declared=windows,
        strict_ordering_declared=windows and version_code >= 23000,
        one_sided_rma_declared=one_sided,
        device_api_query_declared=query,
        generic_lsa_candidate=generic_lsa,
        dda_candidate=dda,
    )


__all__ = [
    "CollectiveExecutionContract",
    "NativeCollectiveCapabilities",
    "RcclTransportAssessment",
    "RcclTransportEvidence",
    "RcclTransportRequest",
    "assess_rccl_transport",
    "communicator_capability_snapshot",
    "probe_native_collective_capabilities",
]
