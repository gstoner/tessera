"""Runtime-facing collective adapter with explicit topology binding.

The NCCL path is a one-process/multiple-device executor: every rank is bound to
one CUDA ordinal, calls are issued in one NCCL group, and host arrays cross the
ABI only at the collective boundary.  Exact multi-device promotion still
requires evidence from a host exposing every device in the topology.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .testing.mock_collective import MockRankGroup


COLLECTIVE_STATUSES = {"mock", "single_process", "backend_unavailable", "hardware_runtime"}


def _communicator_capability_snapshot(
    *, backend: str, topology: dict[str, Any], version_code: int,
    ranks: list[dict[str, Any]],
) -> dict[str, Any]:
    world_size = int(topology["world_size"])
    ordered = sorted((dict(row) for row in ranks), key=lambda row: int(row["rank"]))
    if len(ordered) != world_size or [int(row["rank"]) for row in ordered] != list(range(world_size)):
        raise RuntimeError("native communicator returned an invalid rank set")
    if any(int(row["world_size"]) != world_size for row in ordered):
        raise RuntimeError("native communicator world size disagrees with topology")
    payload = {
        "schema": "tessera.collective.communicator.v1",
        "backend": backend,
        "version_code": int(version_code),
        "topology": dict(topology),
        "ranks": ordered,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


@dataclass(frozen=True)
class CollectiveTopology:
    backend: str
    world_size: int
    device_ordinals: tuple[int, ...]
    rank_order: tuple[int, ...]
    cta_policy: str = "default"

    def __post_init__(self) -> None:
        if self.backend not in {"nccl", "rccl"}:
            raise ValueError("native collective topology requires nccl or rccl")
        if self.world_size < 2:
            raise ValueError("native multi-rank topology requires world_size >= 2")
        if len(self.device_ordinals) != self.world_size or len(set(self.device_ordinals)) != self.world_size:
            raise ValueError("collective topology requires one unique device ordinal per rank")
        if self.rank_order != tuple(range(self.world_size)):
            raise ValueError("collective rank order must be the canonical contiguous order")
        if any(device < 0 for device in self.device_ordinals):
            raise ValueError("collective device ordinals must be nonnegative")
        if self.cta_policy not in {"default", "zero"}:
            raise ValueError("collective CTA policy must be default or zero")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tessera.collective_topology.v2",
            "backend": self.backend,
            "world_size": self.world_size,
            "device_ordinals": list(self.device_ordinals),
            "rank_order": list(self.rank_order),
            "cta_policy": self.cta_policy,
        }


def collective_topology(*, backend: str, world_size: int,
                        device_ordinals: tuple[int, ...] | None = None,
                        cta_policy: str = "default") -> CollectiveTopology:
    devices = tuple(range(world_size)) if device_ordinals is None else tuple(device_ordinals)
    return CollectiveTopology(
        backend, int(world_size), devices, tuple(range(int(world_size))), cta_policy
    )


def _native_probe(topology: CollectiveTopology) -> tuple[bool, str]:
    collective_name = ctypes.util.find_library(topology.backend)
    runtime_library = "cudart" if topology.backend == "nccl" else "amdhip64"
    runtime_name = ctypes.util.find_library(runtime_library)
    runtime_label = "CUDA" if topology.backend == "nccl" else "HIP"
    if not collective_name or not runtime_name:
        return False, (
            f"{topology.backend} and {runtime_label} runtime libraries must both be loadable"
        )
    runtime = ctypes.CDLL(runtime_name)
    count = ctypes.c_int()
    get_count = (
        runtime.cudaGetDeviceCount
        if topology.backend == "nccl"
        else runtime.hipGetDeviceCount
    )
    if get_count(ctypes.byref(count)) != 0:
        return False, f"{runtime_label} device enumeration failed"
    highest = max(topology.device_ordinals)
    if count.value <= highest:
        return False, (
            f"topology requires {runtime_label} device ordinal {highest}, "
            f"but only {count.value} device(s) are visible"
        )
    return True, ""


class _NCCLExecutor:
    """Synchronous NCCL-compatible executor for an explicit local topology.

    RCCL intentionally exposes the NCCL collective ABI. The device runtime is
    selected independently: CUDA for NCCL and HIP for RCCL.
    """

    _F32 = 7
    _SUM = 0
    _H2D = 1
    _D2H = 2

    class _CommProperties(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_size_t),
            ("magic", ctypes.c_uint),
            ("version", ctypes.c_uint),
            ("rank", ctypes.c_int),
            ("world_size", ctypes.c_int),
            ("device", ctypes.c_int),
            ("management_device", ctypes.c_int),
            ("device_api_support", ctypes.c_bool),
            ("multimem_support", ctypes.c_bool),
            ("gin_type", ctypes.c_int),
            ("lsa_team_count", ctypes.c_int),
            ("host_rma_support", ctypes.c_bool),
            ("railed_gin_type", ctypes.c_int),
        ]

    class _UniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_char * 128)]

    class _Config(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_size_t),
            ("magic", ctypes.c_uint),
            ("version", ctypes.c_uint),
            ("blocking", ctypes.c_int),
            ("cga_cluster_size", ctypes.c_int),
            ("min_ctas", ctypes.c_int),
            ("max_ctas", ctypes.c_int),
            ("net_name", ctypes.c_char_p),
            ("split_share", ctypes.c_int),
            ("traffic_class", ctypes.c_int),
            ("comm_name", ctypes.c_char_p),
            ("collnet_enable", ctypes.c_int),
            ("cta_policy", ctypes.c_int),
            ("shrink_share", ctypes.c_int),
            ("nvls_ctas", ctypes.c_int),
            ("channels_per_net_peer", ctypes.c_int),
            ("nvlink_centric_sched", ctypes.c_int),
            ("graph_usage_mode", ctypes.c_int),
            ("num_rma_contexts", ctypes.c_int),
            ("max_p2p_peers", ctypes.c_int),
        ]

    def __init__(self, topology: CollectiveTopology) -> None:
        ready, reason = _native_probe(topology)
        if not ready:
            raise RuntimeError(reason)
        self.topology = topology
        is_hip = topology.backend == "rccl"
        runtime_library = "amdhip64" if is_hip else "cudart"
        self.cuda = ctypes.CDLL(ctypes.util.find_library(runtime_library))
        self.nccl = ctypes.CDLL(ctypes.util.find_library(topology.backend))
        self._runtime_label = "HIP" if is_hip else "CUDA"
        self._collective_label = topology.backend.upper()
        void_p = ctypes.c_void_p
        prefix = "hip" if is_hip else "cuda"
        self._set_device = getattr(self.cuda, f"{prefix}SetDevice")
        self._malloc = getattr(self.cuda, f"{prefix}Malloc")
        self._free = getattr(self.cuda, f"{prefix}Free")
        self._memcpy = getattr(self.cuda, f"{prefix}Memcpy")
        self._stream_create = getattr(self.cuda, f"{prefix}StreamCreate")
        self._stream_synchronize = getattr(self.cuda, f"{prefix}StreamSynchronize")
        self._stream_destroy = getattr(self.cuda, f"{prefix}StreamDestroy")
        self._set_device.argtypes = [ctypes.c_int]
        self._malloc.argtypes = [ctypes.POINTER(void_p), ctypes.c_size_t]
        self._free.argtypes = [void_p]
        self._memcpy.argtypes = [void_p, void_p, ctypes.c_size_t, ctypes.c_int]
        self._stream_create.argtypes = [ctypes.POINTER(void_p)]
        self._stream_synchronize.argtypes = [void_p]
        self._stream_destroy.argtypes = [void_p]
        self.nccl.ncclCommInitAll.argtypes = [ctypes.POINTER(void_p), ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int)]
        self.nccl.ncclCommDestroy.argtypes = [void_p]
        self._version_code = ctypes.c_int()
        self.nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._check_nccl(
            self.nccl.ncclGetVersion(ctypes.byref(self._version_code)),
            "version query",
        )
        self._symmetric_alloc = None
        self._symmetric_free = None
        self._window_register = None
        self._window_deregister = None
        if topology.cta_policy == "zero":
            self._symmetric_alloc = getattr(self.nccl, "ncclMemAlloc", None)
            self._symmetric_free = getattr(self.nccl, "ncclMemFree", None)
            self._window_register = getattr(
                self.nccl, "ncclCommWindowRegister", None
            )
            self._window_deregister = getattr(
                self.nccl, "ncclCommWindowDeregister", None
            )
            if not all(
                (
                    self._symmetric_alloc,
                    self._symmetric_free,
                    self._window_register,
                    self._window_deregister,
                )
            ):
                raise RuntimeError(
                    "zero-CTA communicator requires RCCL symmetric-memory APIs"
                )
            self._symmetric_alloc.argtypes = [
                ctypes.POINTER(void_p), ctypes.c_size_t
            ]
            self._symmetric_free.argtypes = [void_p]
            self._window_register.argtypes = [
                void_p, void_p, ctypes.c_size_t, ctypes.POINTER(void_p),
                ctypes.c_int,
            ]
            self._window_deregister.argtypes = [void_p, void_p]
        for name in ("ncclAllReduce", "ncclReduceScatter"):
            getattr(self.nccl, name).argtypes = [void_p, void_p, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_int, void_p, void_p]
        self.nccl.ncclAllGather.argtypes = [void_p, void_p, ctypes.c_size_t,
            ctypes.c_int, void_p, void_p]
        for name in ("ncclSend", "ncclRecv"):
            getattr(self.nccl, name).argtypes = [void_p, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_int, void_p, void_p]
        self.comms = (ctypes.c_void_p * topology.world_size)()
        devices = (ctypes.c_int * topology.world_size)(*topology.device_ordinals)
        if topology.cta_policy == "zero":
            self._init_rank_configured_communicators()
        else:
            self._check_nccl(
                self.nccl.ncclCommInitAll(
                    self.comms, topology.world_size, devices
                ),
                "communicator init",
            )
        self._communicator_property_error = ""
        self._communicator_snapshot = self._query_communicator_properties()

    def _init_rank_configured_communicators(self) -> None:
        self.nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(self._UniqueId)]
        self.nccl.ncclCommInitRankConfig.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, self._UniqueId,
            ctypes.c_int, ctypes.POINTER(self._Config),
        ]
        unique_id = self._UniqueId()
        self._check_nccl(
            self.nccl.ncclGetUniqueId(ctypes.byref(unique_id)), "unique ID query"
        )
        undefined = -(2**31)

        def initialize(rank: int) -> tuple[int, int, int | None]:
            self._check_device(
                self._set_device(self.topology.device_ordinals[rank]),
                f"set device for configured communicator rank {rank}",
            )
            config = self._Config()
            config.size = ctypes.sizeof(self._Config)
            config.magic = 0xCAFEBEEF
            config.version = int(self._version_code.value)
            for field, _ctype in self._Config._fields_[3:]:
                if field not in {"net_name", "comm_name"}:
                    setattr(config, field, undefined)
            config.net_name = None
            config.comm_name = None
            config.cta_policy = 0x02
            communicator = ctypes.c_void_p()
            rc = int(
                self.nccl.ncclCommInitRankConfig(
                    ctypes.byref(communicator), self.topology.world_size,
                    unique_id, rank, ctypes.byref(config),
                )
            )
            return rank, rc, communicator.value

        with ThreadPoolExecutor(max_workers=self.topology.world_size) as pool:
            rows = list(pool.map(initialize, range(self.topology.world_size)))
        for rank, rc, communicator in rows:
            self._check_nccl(rc, f"configured communicator init for rank {rank}")
            self.comms[rank] = communicator

    def _query_communicator_properties(self) -> dict[str, Any] | None:
        query = getattr(self.nccl, "ncclCommQueryProperties", None)
        if query is None:
            self._communicator_property_error = (
                f"{self._collective_label} communicator property query is unavailable"
            )
            return None
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(self._CommProperties)]
        rows: list[dict[str, Any]] = []
        for expected_rank, comm in enumerate(self.comms):
            properties = self._CommProperties()
            properties.size = ctypes.sizeof(self._CommProperties)
            properties.magic = 0xCAFEBEEF
            properties.version = int(self._version_code.value)
            rc = int(query(comm, ctypes.byref(properties)))
            if rc:
                self.nccl.ncclGetErrorString.restype = ctypes.c_char_p
                detail = self.nccl.ncclGetErrorString(rc)
                self._communicator_property_error = (
                    f"{self._collective_label} communicator property query for "
                    f"rank {expected_rank} failed: "
                    f"{(detail or b'unknown').decode()}"
                )
                return None
            rows.append(
                {
                    "rank": int(properties.rank),
                    "world_size": int(properties.world_size),
                    "device": int(properties.device),
                    "management_device": int(properties.management_device),
                    "device_api_support": bool(properties.device_api_support),
                    "multimem_support": bool(properties.multimem_support),
                    "gin_type": int(properties.gin_type),
                    "lsa_team_count": int(properties.lsa_team_count),
                    "host_rma_support": bool(properties.host_rma_support),
                    "railed_gin_type": int(properties.railed_gin_type),
                }
            )
        return _communicator_capability_snapshot(
            backend=self.topology.backend,
            topology=self.topology.to_dict(),
            version_code=int(self._version_code.value),
            ranks=rows,
        )

    @property
    def communicator_snapshot(self) -> dict[str, Any]:
        if self._communicator_snapshot is None:
            raise RuntimeError(self._communicator_property_error)
        return dict(self._communicator_snapshot)

    def _check_device(self, rc: int, where: str) -> None:
        if rc:
            raise RuntimeError(f"{self._runtime_label} {where} failed with rc={rc}")

    def _check_nccl(self, rc: int, where: str) -> None:
        if rc:
            self.nccl.ncclGetErrorString.restype = ctypes.c_char_p
            detail = self.nccl.ncclGetErrorString(rc)
            raise RuntimeError(
                f"{self._collective_label} {where} failed: "
                f"{(detail or b'unknown').decode()}"
            )

    def _register_symmetric_windows(
        self,
        sends: list[ctypes.c_void_p],
        recvs: list[ctypes.c_void_p],
        send_bytes: int,
        receive_bytes: int,
    ) -> tuple[list[ctypes.c_void_p], list[ctypes.c_void_p]]:
        assert self._window_register is not None

        def register(rank: int) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
            self._check_device(
                self._set_device(self.topology.device_ordinals[rank]),
                "set device for symmetric window registration",
            )
            send_window = ctypes.c_void_p()
            receive_window = ctypes.c_void_p()
            self._check_nccl(
                self._window_register(
                    self.comms[rank], sends[rank], send_bytes,
                    ctypes.byref(send_window), 0x01,
                ),
                "send-window registration",
            )
            self._check_nccl(
                self._window_register(
                    self.comms[rank], recvs[rank], receive_bytes,
                    ctypes.byref(receive_window), 0x01,
                ),
                "receive-window registration",
            )
            return send_window, receive_window

        with ThreadPoolExecutor(max_workers=self.topology.world_size) as pool:
            rows = list(pool.map(register, range(self.topology.world_size)))
        return [row[0] for row in rows], [row[1] for row in rows]

    def _allocate(self, arrays: list[np.ndarray], output_shapes: list[tuple[int, ...]]):
        sends: list[ctypes.c_void_p] = []
        recvs: list[ctypes.c_void_p] = []
        streams: list[ctypes.c_void_p] = []
        for rank, (array, output_shape) in enumerate(zip(arrays, output_shapes)):
            self._check_device(self._set_device(self.topology.device_ordinals[rank]), "set device")
            send = ctypes.c_void_p(); recv = ctypes.c_void_p(); stream = ctypes.c_void_p()
            if self._symmetric_alloc is not None:
                self._check_nccl(
                    self._symmetric_alloc(ctypes.byref(send), array.nbytes),
                    "symmetric send allocation",
                )
            else:
                self._check_device(
                    self._malloc(ctypes.byref(send), array.nbytes),
                    "send allocation",
                )
            output_bytes = int(np.prod(output_shape, dtype=np.int64)) * 4
            if self._symmetric_alloc is not None:
                self._check_nccl(
                    self._symmetric_alloc(ctypes.byref(recv), output_bytes),
                    "symmetric receive allocation",
                )
            else:
                self._check_device(
                    self._malloc(ctypes.byref(recv), output_bytes),
                    "receive allocation",
                )
            self._check_device(self._stream_create(ctypes.byref(stream)), "stream creation")
            self._check_device(self._memcpy(send, ctypes.c_void_p(array.ctypes.data), array.nbytes, self._H2D), "host-to-device copy")
            sends.append(send); recvs.append(recv); streams.append(stream)
        send_windows: list[ctypes.c_void_p] = []
        receive_windows: list[ctypes.c_void_p] = []
        if self._window_register is not None:
            send_windows, receive_windows = self._register_symmetric_windows(
                sends, recvs, arrays[0].nbytes,
                int(np.prod(output_shapes[0], dtype=np.int64)) * 4,
            )
        return sends, recvs, streams, send_windows, receive_windows

    def _finish(self, recvs, streams, output_shapes):
        outputs: list[np.ndarray] = []
        for rank, shape in enumerate(output_shapes):
            self._check_device(self._set_device(self.topology.device_ordinals[rank]), "set device")
            self._check_device(self._stream_synchronize(streams[rank]), "stream synchronization")
            output = np.empty(shape, np.float32)
            self._check_device(self._memcpy(ctypes.c_void_p(output.ctypes.data), recvs[rank], output.nbytes, self._D2H), "device-to-host copy")
            outputs.append(output)
        return outputs

    def _cleanup(
        self, sends, recvs, streams, send_windows, receive_windows
    ) -> None:
        if self._window_deregister is not None:
            def deregister(rank: int) -> None:
                self._check_device(
                    self._set_device(self.topology.device_ordinals[rank]),
                    "set device for symmetric window deregistration",
                )
                if send_windows[rank]:
                    self._check_nccl(
                        self._window_deregister(
                            self.comms[rank], send_windows[rank]
                        ),
                        "send-window deregistration",
                    )
                if receive_windows[rank]:
                    self._check_nccl(
                        self._window_deregister(
                            self.comms[rank], receive_windows[rank]
                        ),
                        "receive-window deregistration",
                    )

            with ThreadPoolExecutor(max_workers=self.topology.world_size) as pool:
                list(pool.map(deregister, range(self.topology.world_size)))
        for rank in range(self.topology.world_size):
            self._set_device(self.topology.device_ordinals[rank])
            if self._symmetric_free is not None:
                if sends[rank]:
                    self._check_nccl(
                        self._symmetric_free(sends[rank]), "symmetric send free"
                    )
                if recvs[rank]:
                    self._check_nccl(
                        self._symmetric_free(recvs[rank]),
                        "symmetric receive free",
                    )
            else:
                if sends[rank]: self._free(sends[rank])
                if recvs[rank]: self._free(recvs[rank])
            if streams[rank]: self._stream_destroy(streams[rank])

    def run(self, kind: str, values: list[Any], *, op: str = "sum") -> list[np.ndarray]:
        if op != "sum":
            raise ValueError("native collective v1 executor supports sum reductions only")
        source_arrays = [np.asarray(value) for value in values]
        if any(array.dtype != np.float32 for array in source_arrays):
            raise ValueError(
                "native collective v1 supports fp32 storage only; implicit dtype "
                "conversion is forbidden"
            )
        arrays = [np.ascontiguousarray(value) for value in source_arrays]
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("collective rank inputs must have identical shapes")
        world = self.topology.world_size
        if kind == "all_gather":
            output_shapes = [(arrays[0].shape[0] * world,) + arrays[0].shape[1:]] * world
        elif kind == "reduce_scatter":
            if arrays[0].ndim < 1 or arrays[0].shape[0] % world:
                raise ValueError("reduce_scatter axis 0 must divide evenly across ranks")
            output_shapes = [(arrays[0].shape[0] // world,) + arrays[0].shape[1:]] * world
        else:
            output_shapes = [arrays[0].shape] * world
        sends, recvs, streams, send_windows, receive_windows = self._allocate(
            arrays, output_shapes
        )
        try:
            self._check_nccl(self.nccl.ncclGroupStart(), "group start")
            if kind == "all_reduce":
                count = arrays[0].size
                for rank in range(world):
                    self._check_nccl(self.nccl.ncclAllReduce(sends[rank], recvs[rank], count, self._F32, self._SUM, self.comms[rank], streams[rank]), "all_reduce")
            elif kind == "all_gather":
                count = arrays[0].size
                for rank in range(world):
                    self._check_nccl(self.nccl.ncclAllGather(sends[rank], recvs[rank], count, self._F32, self.comms[rank], streams[rank]), "all_gather")
            elif kind == "reduce_scatter":
                count = arrays[0].size // world
                for rank in range(world):
                    self._check_nccl(self.nccl.ncclReduceScatter(sends[rank], recvs[rank], count, self._F32, self._SUM, self.comms[rank], streams[rank]), "reduce_scatter")
            elif kind == "all_to_all":
                if arrays[0].ndim < 1 or arrays[0].shape[0] % world:
                    raise ValueError("all_to_all scatter axis 0 must divide evenly across ranks")
                count = arrays[0].size // world
                stride = count * 4
                for source in range(world):
                    for peer in range(world):
                        send = ctypes.c_void_p(sends[source].value + peer * stride)
                        recv = ctypes.c_void_p(recvs[source].value + peer * stride)
                        self._check_nccl(self.nccl.ncclSend(send, count, self._F32, peer, self.comms[source], streams[source]), "send")
                        self._check_nccl(self.nccl.ncclRecv(recv, count, self._F32, peer, self.comms[source], streams[source]), "receive")
            else:
                raise ValueError(f"unsupported native collective {kind!r}")
            self._check_nccl(self.nccl.ncclGroupEnd(), "group end")
            return self._finish(recvs, streams, output_shapes)
        finally:
            self._cleanup(
                sends, recvs, streams, send_windows, receive_windows
            )

    def close(self) -> None:
        for comm in self.comms:
            if comm:
                self.nccl.ncclCommDestroy(comm)


@dataclass(frozen=True)
class CollectiveBackendStatus:
    backend: str
    status: str
    reason: str = ""
    world_size: int = 1

    @property
    def available(self) -> bool:
        return self.status in {"mock", "single_process", "hardware_runtime"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "reason": self.reason,
            "world_size": self.world_size,
            "available": self.available,
        }


class CollectiveAdapter:
    """Small collective adapter with explicit backend status."""

    def __init__(self, *, backend: str = "mock", world_size: int = 1, mesh_axes: dict[str, int] | None = None,
                 device_ordinals: tuple[int, ...] | None = None,
                 cta_policy: str = "default") -> None:
        self.backend = backend
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError("collective adapter world_size must be positive")
        self.mesh_axes = dict(mesh_axes or {"dp": self.world_size})
        if not self.mesh_axes or any(
            not str(axis) or int(size) < 1 for axis, size in self.mesh_axes.items()
        ):
            raise ValueError("collective mesh axes require non-empty names and positive sizes")
        if self.backend == "mock" and self.world_size > 1:
            self._group: MockRankGroup | None = MockRankGroup(self.world_size, self.mesh_axes)
        else:
            self._group = None
        self._topology = (collective_topology(backend=backend, world_size=self.world_size,
                          device_ordinals=device_ordinals, cta_policy=cta_policy)
                          if backend in {"nccl", "rccl"} and self.world_size > 1 else None)
        self._native: _NCCLExecutor | None = None

    def status(self) -> CollectiveBackendStatus:
        if self.backend == "mock" and self._group is not None:
            return CollectiveBackendStatus(self.backend, "mock", world_size=self.world_size)
        if self.backend in {"mock", "single_process"} and self.world_size == 1:
            return CollectiveBackendStatus(self.backend, "single_process", world_size=1)
        if self.backend in {"nccl", "rccl"} and self._topology is not None:
            ready, reason = _native_probe(self._topology)
            return CollectiveBackendStatus(
                self.backend,
                "hardware_runtime" if ready else "backend_unavailable",
                reason=(
                    ""
                    if ready
                    else f"{self.backend} native collective runtime is not wired on this host: {reason}"
                ),
                world_size=self.world_size,
            )
        if self.backend == "mpi":
            return CollectiveBackendStatus(self.backend, "backend_unavailable",
                reason=f"{self.backend} native collective runtime is not wired on this host",
                world_size=self.world_size)
        return CollectiveBackendStatus(
            self.backend,
            "backend_unavailable",
            reason=f"unknown collective backend {self.backend!r}",
            world_size=self.world_size,
        )

    def all_reduce(self, values, *, op: str = "sum"):
        if self.backend in {"nccl", "rccl"} and self.world_size > 1:
            return self._run_native("all_reduce", values, op=op)
        return self._run(lambda rank, value: rank.all_reduce(value, op=op), values)

    def reduce_scatter(self, values, *, axis: int = 0, op: str = "sum"):
        if self.backend in {"nccl", "rccl"} and self.world_size > 1:
            if axis != 0: raise ValueError("NCCL reduce_scatter v1 requires axis=0")
            return self._run_native("reduce_scatter", values, op=op)
        return self._run(lambda rank, value: rank.reduce_scatter(value, axis=axis, op=op), values)

    def all_gather(self, values, *, axis: int = 0):
        if self.backend in {"nccl", "rccl"} and self.world_size > 1:
            if axis != 0: raise ValueError("NCCL all_gather v1 requires axis=0")
            return self._run_native("all_gather", values)
        return self._run(lambda rank, value: rank.all_gather(value, axis=axis), values)

    def all_to_all(self, values, *, scatter_axis: int = 0, gather_axis: int = 0):
        if self.backend in {"nccl", "rccl"} and self.world_size > 1:
            if scatter_axis != 0 or gather_axis != 0:
                raise ValueError("NCCL all_to_all v1 requires scatter_axis=gather_axis=0")
            return self._run_native("all_to_all", values)
        return self._run(lambda rank, value: rank.all_to_all(value, scatter_axis=scatter_axis, gather_axis=gather_axis), values)

    def _run_native(self, kind: str, values, *, op: str = "sum"):
        status = self.status()
        if not status.available:
            raise RuntimeError(status.reason)
        assert self._topology is not None
        if self._native is None:
            self._native = _NCCLExecutor(self._topology)
        return self._native.run(kind, _per_rank_values(values, self.world_size), op=op)

    def close(self) -> None:
        if self._native is not None:
            self._native.close()
            self._native = None

    def communicator_capability_snapshot(self) -> dict[str, Any]:
        if self.backend not in {"nccl", "rccl"} or self._topology is None:
            raise RuntimeError("communicator properties require a native topology")
        status = self.status()
        if not status.available:
            raise RuntimeError(status.reason)
        if self._native is None:
            self._native = _NCCLExecutor(self._topology)
        return self._native.communicator_snapshot

    def communicator_capability_digest(self) -> str:
        return str(self.communicator_capability_snapshot()["digest"])

    def _run(self, fn, values):
        status = self.status()
        if status.status == "backend_unavailable":
            raise RuntimeError(status.reason)
        if status.status == "single_process":
            value = _single_value(values)
            return [np.asarray(value)]
        assert self._group is not None
        per_rank = _per_rank_values(values, self.world_size)
        return self._group.run(lambda rank: fn(rank, np.asarray(per_rank[rank.rank])))


def adapter(*, backend: str = "mock", world_size: int = 1, mesh_axes: dict[str, int] | None = None,
            device_ordinals: tuple[int, ...] | None = None,
            cta_policy: str = "default") -> CollectiveAdapter:
    return CollectiveAdapter(backend=backend, world_size=world_size, mesh_axes=mesh_axes,
                             device_ordinals=device_ordinals, cta_policy=cta_policy)


def query_backend(backend: str = "mock", *, world_size: int = 1) -> dict[str, Any]:
    return adapter(backend=backend, world_size=world_size).status().to_dict()


def _per_rank_values(values, world_size: int) -> list[Any]:
    if isinstance(values, (list, tuple)) and len(values) == world_size:
        return list(values)
    return [values for _ in range(world_size)]


def _single_value(values):
    if isinstance(values, (list, tuple)):
        return values[0]
    return values


__all__ = [
    "COLLECTIVE_STATUSES",
    "CollectiveAdapter",
    "CollectiveBackendStatus",
    "CollectiveTopology",
    "adapter",
    "collective_topology",
    "query_backend",
]
