"""Runtime-facing collective adapter with explicit topology binding.

The NCCL path is a one-process/multiple-device executor: every rank is bound to
one CUDA ordinal, calls are issued in one NCCL group, and host arrays cross the
ABI only at the collective boundary.  Exact multi-device promotion still
requires evidence from a host exposing every device in the topology.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .testing.mock_collective import MockRankGroup


COLLECTIVE_STATUSES = {"mock", "single_process", "backend_unavailable", "hardware_runtime"}


@dataclass(frozen=True)
class CollectiveTopology:
    backend: str
    world_size: int
    device_ordinals: tuple[int, ...]
    rank_order: tuple[int, ...]

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

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tessera.collective_topology.v1",
            "backend": self.backend,
            "world_size": self.world_size,
            "device_ordinals": list(self.device_ordinals),
            "rank_order": list(self.rank_order),
        }


def collective_topology(*, backend: str, world_size: int,
                        device_ordinals: tuple[int, ...] | None = None) -> CollectiveTopology:
    devices = tuple(range(world_size)) if device_ordinals is None else tuple(device_ordinals)
    return CollectiveTopology(backend, int(world_size), devices, tuple(range(int(world_size))))


def _native_probe(topology: CollectiveTopology) -> tuple[bool, str]:
    nccl_name = ctypes.util.find_library("nccl" if topology.backend == "nccl" else "rccl")
    cuda_name = ctypes.util.find_library("cudart")
    if not nccl_name or not cuda_name:
        return False, f"{topology.backend} and CUDA runtime libraries must both be loadable"
    cuda = ctypes.CDLL(cuda_name)
    count = ctypes.c_int()
    if cuda.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return False, "CUDA device enumeration failed"
    highest = max(topology.device_ordinals)
    if count.value <= highest:
        return False, (
            f"topology requires CUDA device ordinal {highest}, but only {count.value} device(s) are visible"
        )
    return True, ""


class _NCCLExecutor:
    """Synchronous host-array NCCL executor for an explicit local topology."""

    _F32 = 7
    _SUM = 0
    _H2D = 1
    _D2H = 2

    def __init__(self, topology: CollectiveTopology) -> None:
        ready, reason = _native_probe(topology)
        if not ready:
            raise RuntimeError(reason)
        self.topology = topology
        self.cuda = ctypes.CDLL(ctypes.util.find_library("cudart"))
        self.nccl = ctypes.CDLL(ctypes.util.find_library("nccl"))
        void_p = ctypes.c_void_p
        self.cuda.cudaSetDevice.argtypes = [ctypes.c_int]
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(void_p), ctypes.c_size_t]
        self.cuda.cudaFree.argtypes = [void_p]
        self.cuda.cudaMemcpy.argtypes = [void_p, void_p, ctypes.c_size_t, ctypes.c_int]
        self.cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(void_p)]
        self.cuda.cudaStreamSynchronize.argtypes = [void_p]
        self.cuda.cudaStreamDestroy.argtypes = [void_p]
        self.nccl.ncclCommInitAll.argtypes = [ctypes.POINTER(void_p), ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int)]
        self.nccl.ncclCommDestroy.argtypes = [void_p]
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
        self._check_nccl(self.nccl.ncclCommInitAll(self.comms, topology.world_size, devices), "communicator init")

    def _check_cuda(self, rc: int, where: str) -> None:
        if rc:
            raise RuntimeError(f"CUDA {where} failed with rc={rc}")

    def _check_nccl(self, rc: int, where: str) -> None:
        if rc:
            self.nccl.ncclGetErrorString.restype = ctypes.c_char_p
            detail = self.nccl.ncclGetErrorString(rc)
            raise RuntimeError(f"NCCL {where} failed: {(detail or b'unknown').decode()}")

    def _allocate(self, arrays: list[np.ndarray], output_shapes: list[tuple[int, ...]]):
        sends: list[ctypes.c_void_p] = []
        recvs: list[ctypes.c_void_p] = []
        streams: list[ctypes.c_void_p] = []
        for rank, (array, output_shape) in enumerate(zip(arrays, output_shapes)):
            self._check_cuda(self.cuda.cudaSetDevice(self.topology.device_ordinals[rank]), "set device")
            send = ctypes.c_void_p(); recv = ctypes.c_void_p(); stream = ctypes.c_void_p()
            self._check_cuda(self.cuda.cudaMalloc(ctypes.byref(send), array.nbytes), "send allocation")
            output_bytes = int(np.prod(output_shape, dtype=np.int64)) * 4
            self._check_cuda(self.cuda.cudaMalloc(ctypes.byref(recv), output_bytes), "receive allocation")
            self._check_cuda(self.cuda.cudaStreamCreate(ctypes.byref(stream)), "stream creation")
            self._check_cuda(self.cuda.cudaMemcpy(send, ctypes.c_void_p(array.ctypes.data), array.nbytes, self._H2D), "host-to-device copy")
            sends.append(send); recvs.append(recv); streams.append(stream)
        return sends, recvs, streams

    def _finish(self, recvs, streams, output_shapes):
        outputs: list[np.ndarray] = []
        for rank, shape in enumerate(output_shapes):
            self._check_cuda(self.cuda.cudaSetDevice(self.topology.device_ordinals[rank]), "set device")
            self._check_cuda(self.cuda.cudaStreamSynchronize(streams[rank]), "stream synchronization")
            output = np.empty(shape, np.float32)
            self._check_cuda(self.cuda.cudaMemcpy(ctypes.c_void_p(output.ctypes.data), recvs[rank], output.nbytes, self._D2H), "device-to-host copy")
            outputs.append(output)
        return outputs

    def _cleanup(self, sends, recvs, streams) -> None:
        for rank in range(self.topology.world_size):
            self.cuda.cudaSetDevice(self.topology.device_ordinals[rank])
            if sends[rank]: self.cuda.cudaFree(sends[rank])
            if recvs[rank]: self.cuda.cudaFree(recvs[rank])
            if streams[rank]: self.cuda.cudaStreamDestroy(streams[rank])

    def run(self, kind: str, values: list[Any], *, op: str = "sum") -> list[np.ndarray]:
        if op != "sum":
            raise ValueError("NCCL v1 executor currently supports sum reductions only")
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in values]
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
        sends, recvs, streams = self._allocate(arrays, output_shapes)
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
            self._cleanup(sends, recvs, streams)

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
                 device_ordinals: tuple[int, ...] | None = None) -> None:
        self.backend = backend
        self.world_size = int(world_size)
        self.mesh_axes = dict(mesh_axes or {"dp": self.world_size})
        if self.backend == "mock" and self.world_size > 1:
            self._group: MockRankGroup | None = MockRankGroup(self.world_size, self.mesh_axes)
        else:
            self._group = None
        self._topology = (collective_topology(backend=backend, world_size=self.world_size,
                          device_ordinals=device_ordinals)
                          if backend in {"nccl", "rccl"} and self.world_size > 1 else None)
        self._native: _NCCLExecutor | None = None

    def status(self) -> CollectiveBackendStatus:
        if self.backend == "mock" and self._group is not None:
            return CollectiveBackendStatus(self.backend, "mock", world_size=self.world_size)
        if self.backend in {"mock", "single_process"} and self.world_size == 1:
            return CollectiveBackendStatus(self.backend, "single_process", world_size=1)
        if self.backend == "nccl" and self._topology is not None:
            ready, reason = _native_probe(self._topology)
            return CollectiveBackendStatus(
                self.backend,
                "hardware_runtime" if ready else "backend_unavailable",
                reason=reason,
                world_size=self.world_size,
            )
        if self.backend in {"rccl", "mpi"}:
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
        if self.backend == "nccl" and self.world_size > 1:
            return self._run_native("all_reduce", values, op=op)
        return self._run(lambda rank, value: rank.all_reduce(value, op=op), values)

    def reduce_scatter(self, values, *, axis: int = 0, op: str = "sum"):
        if self.backend == "nccl" and self.world_size > 1:
            if axis != 0: raise ValueError("NCCL reduce_scatter v1 requires axis=0")
            return self._run_native("reduce_scatter", values, op=op)
        return self._run(lambda rank, value: rank.reduce_scatter(value, axis=axis, op=op), values)

    def all_gather(self, values, *, axis: int = 0):
        if self.backend == "nccl" and self.world_size > 1:
            if axis != 0: raise ValueError("NCCL all_gather v1 requires axis=0")
            return self._run_native("all_gather", values)
        return self._run(lambda rank, value: rank.all_gather(value, axis=axis), values)

    def all_to_all(self, values, *, scatter_axis: int = 0, gather_axis: int = 0):
        if self.backend == "nccl" and self.world_size > 1:
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
            device_ordinals: tuple[int, ...] | None = None) -> CollectiveAdapter:
    return CollectiveAdapter(backend=backend, world_size=world_size, mesh_axes=mesh_axes,
                             device_ordinals=device_ordinals)


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
