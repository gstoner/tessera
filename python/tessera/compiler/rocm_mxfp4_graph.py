"""Opt-in kernel-only HIP graph for the exact gfx1201 packed MXFP4 ABI.

The session owns all five device buffers and one stream. A producer may fill
the leased A/As pointers on that stream; capture/replay never allocates,
transfers, or synchronizes. This remains a manual route, not selector entry.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rocm_mxfp4_packed_folded import PackedFoldedPayload
from .rocm_mxfp4_resident import PackedFoldedResidentSession
from .rocm_native import ROCMNativePackage


@dataclass(frozen=True)
class DeviceBufferLease:
    """Borrowed pointer; invalid as soon as its owning graph session closes."""

    owner: PackedFoldedGraphSession
    ordinal: int
    name: str
    nbytes: int

    @property
    def pointer(self) -> int:
        self.owner._require_open()
        value = self.owner._resident._device[self.ordinal].value
        if value is None:
            raise RuntimeError(f"device buffer {self.name} is unavailable")
        return value


class PackedFoldedGraphSession:
    """Capture one stable-pointer kernel node and replay it on the owned stream.

    External producers must enqueue writes to A/As on ``stream_pointer`` and
    call ``mark_device_inputs_ready`` with that same pointer. They retain
    responsibility for data correctness; no cross-stream dependency is made.
    ``read_output`` is deliberately outside the graph and synchronizes.
    """

    def __init__(
        self, package: ROCMNativePackage, payload: PackedFoldedPayload, m: int,
        *, hip: Any | None = None,
    ) -> None:
        self._resident = PackedFoldedResidentSession(package, payload, m, hip=hip)
        self._hip: Any = self._resident._hip
        self._graph_exec = ctypes.c_void_p()
        self._closed = False
        self._inputs_ready = False
        self._output_valid = False
        self._replays = 0
        self._capture_nodes: tuple[int, ...] = ()
        try:
            self._bind_graph_api()
        except BaseException:
            self._resident.close()
            raise

    def _bind_graph_api(self) -> None:
        hip = self._hip
        hip.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        hip.hipStreamEndCapture.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        hip.hipGraphGetNodes.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        hip.hipGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        hip.hipGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
        ]
        hip.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        hip.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
        hip.hipGraphDestroy.argtypes = [ctypes.c_void_p]

    def _call(self, name: str, *args: Any) -> None:
        self._resident._call(name, *args)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("packed MXFP4 graph session is closed")

    @property
    def stream_pointer(self) -> int:
        self._require_open()
        value = self._resident._stream.value
        if value is None:
            raise RuntimeError("packed MXFP4 graph stream is unavailable")
        return value

    def buffers(self) -> dict[str, DeviceBufferLease]:
        """Expose only the mutable A, As, and output device buffers."""
        self._require_open()
        resident = self._resident
        return {
            name: DeviceBufferLease(self, ordinal, name, resident._sizes[ordinal])
            for name, ordinal in (("a", 0), ("a_scale", 2), ("output", 4))
        }

    def upload_inputs(self, a: np.ndarray, a_scale: np.ndarray) -> None:
        """One-time host bootstrap or explicit later refresh outside capture."""
        self._require_open()
        self._output_valid = False
        self._resident.upload_activations(a, a_scale)
        self._resident.synchronize()
        self._inputs_ready = True

    def mark_device_inputs_ready(self, *, stream_pointer: int) -> None:
        """Assert A/As writes were enqueued on this same stream."""
        if stream_pointer != self.stream_pointer:
            raise ValueError("device producer must use the graph session stream")
        self._output_valid = False
        self._inputs_ready = True

    def _enqueue_kernel(self) -> None:
        resident = self._resident
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in resident._device),
            ctypes.c_int64(resident.m), ctypes.c_int64(resident.n),
            ctypes.c_int64(resident.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        self._call(
            "hipModuleLaunchKernel", resident._function,
            (resident.n + 63) // 64, (resident.m + 255) // 256, 1,
            256, 1, 1, 0, resident._stream, arguments, None,
        )

    def capture(self) -> None:
        """Require a single HIP kernel node; refuse copy/sync graph nodes."""
        self._require_open()
        if not self._inputs_ready:
            raise RuntimeError("graph capture requires initialized device inputs")
        if self._graph_exec.value:
            raise RuntimeError("packed MXFP4 graph was already captured")
        stream = self._resident._stream
        graph = ctypes.c_void_p()
        began = False
        try:
            self._call("hipStreamBeginCapture", stream, 1)  # thread-local mode
            began = True
            self._enqueue_kernel()
            self._call("hipStreamEndCapture", stream, ctypes.byref(graph))
            began = False
            count = ctypes.c_size_t()
            self._call("hipGraphGetNodes", graph, None, ctypes.byref(count))
            if count.value != 1:
                raise RuntimeError(f"packed MXFP4 graph needs one node; got {count.value}")
            nodes = (ctypes.c_void_p * count.value)()
            self._call("hipGraphGetNodes", graph, nodes, ctypes.byref(count))
            node_type = ctypes.c_int()
            self._call("hipGraphNodeGetType", nodes[0], ctypes.byref(node_type))
            if node_type.value != 0:  # hipGraphNodeTypeKernel
                raise RuntimeError("packed MXFP4 graph contains a non-kernel node")
            self._call(
                "hipGraphInstantiate", ctypes.byref(self._graph_exec), graph,
                None, None, 0,
            )
            self._capture_nodes = (node_type.value,)
        except BaseException:
            if began:
                self._hip.hipStreamEndCapture(stream, ctypes.byref(graph))
            raise
        finally:
            if graph.value:
                self._hip.hipGraphDestroy(graph)

    def replay(self) -> None:
        """Enqueue only the instantiated kernel graph; do not synchronize."""
        self._require_open()
        if not self._graph_exec.value or not self._inputs_ready:
            raise RuntimeError("packed MXFP4 graph is not ready for replay")
        self._call("hipGraphLaunch", self._graph_exec, self._resident._stream)
        self._replays += 1
        self._output_valid = True

    def synchronize(self) -> None:
        self._require_open()
        self._resident.synchronize()

    def read_output(self, output: np.ndarray | None = None) -> np.ndarray:
        """Copy the stable device output outside capture after at least one replay."""
        from tessera import runtime as rt

        self._require_open()
        if not self._output_valid:
            raise RuntimeError("packed MXFP4 graph has not replayed these inputs")
        bf16 = rt._bfloat16_dtype()
        if bf16 is None:
            raise RuntimeError("packed MXFP4 graph requires ml_dtypes.bfloat16")
        resident = self._resident
        if output is None:
            output = np.empty((resident.m, resident.n), dtype=bf16)
        if (output.dtype != np.dtype(bf16) or output.shape != (resident.m, resident.n)
                or not output.flags.c_contiguous):
            raise ValueError("graph output must be contiguous BF16 [M,N]")
        resident._pending_host.append(output)
        self._call(
            "hipMemcpyAsync", output.ctypes.data_as(ctypes.c_void_p),
            resident._device[4], int(output.nbytes), 2, resident._stream,
        )
        resident.synchronize()
        return output

    def receipt(self) -> dict[str, object]:
        return {
            **self._resident.receipt(),
            "route": "packed_folded_device_graph_manual",
            "capture_nodes": self._capture_nodes,
            "graph_captures": int(bool(self._graph_exec.value)),
            "graph_replays": self._replays,
            "input_ownership": "session_device_buffers",
            "output_ownership": "session_device_buffer",
            "automatic_selection": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._resident.synchronize()
        finally:
            try:
                if self._graph_exec.value:
                    self._hip.hipGraphExecDestroy(self._graph_exec)
                    self._graph_exec.value = None
            finally:
                self._resident.close()

    def __enter__(self) -> PackedFoldedGraphSession:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DeviceBufferLease", "PackedFoldedGraphSession"]
