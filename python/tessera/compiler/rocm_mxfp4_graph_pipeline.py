"""Manual gfx1201 FP32→E4M3→packed MXFP4→BF16 graph pipeline.

The producer, GEMM, and consumer operate on retained device buffers. The
FP32 input and BF16 result may be borrowed from model-owned ROCm tensors.
The graph is fixed-shape; a different M requires a distinct package/session.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

import numpy as np

from .rocm_mxfp4_graph import PackedFoldedGraphSession
from .rocm_mxfp4_graph_tensor import ROCMGraphTensor
from .rocm_mxfp4_packed_folded import PackedFoldedPayload
from .rocm_mxfp4_native import _extract_gfx1201_hsaco, _rocm_hipcc
from .rocm_native import ROCMNativePackage, _rocm_path


_PIPELINE_HIP = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bfloat16.h>
#include <math.h>

extern "C" __global__ void tessera_graph_e4m3_producer(
    const float* X, unsigned char* A, float* As, long M, long K) {
  const long m = blockIdx.x;
  if (m >= M) return;
  const int t = threadIdx.x;
  __shared__ float maxima[256];
  __shared__ float row_scale;
  float local_max = 0.0f;
  for (long k = t; k < K; k += 256)
    local_max = fmaxf(local_max, fabsf(X[m * K + k]));
#ifdef TESSERA_PRODUCER_WAVE_REDUCE
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  if ((t % warpSize) == 0) maxima[t / warpSize] = local_max;
  __syncthreads();
  if (t == 0) {
    float block_max = 0.0f;
    for (int wave = 0; wave < 256 / warpSize; ++wave)
      block_max = fmaxf(block_max, maxima[wave]);
    row_scale = fmaxf(1.0f, block_max / 448.0f);
    As[m] = row_scale;
  }
#else
  maxima[t] = local_max;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) maxima[t] = fmaxf(maxima[t], maxima[t + stride]);
    __syncthreads();
  }
  if (t == 0) {
    row_scale = fmaxf(1.0f, maxima[0] / 448.0f);
    As[m] = row_scale;
  }
#endif
  __syncthreads();
  for (long k = t; k < K; k += 256)
    A[m * K + k] = __hip_cvt_float_to_fp8(
        X[m * K + k] / row_scale, __HIP_SATFINITE, __HIP_E4M3);
}

extern "C" __global__ void tessera_graph_bf16_relu(
    const __bf16* input, __bf16* output, long count) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = (float)input[i];
    output[i] = (__bf16)fmaxf(value, 0.0f);
  }
}
"""


def _compile_auxiliary_hsaco(producer_variant: str = "block") -> bytes:
    if producer_variant not in ("block", "wave"):
        raise ValueError("graph producer variant must be block or wave")
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("gfx1201 graph pipeline requires hipcc")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-graph-pipeline-") as directory:
        source = Path(directory) / "pipeline.hip"
        bundle = Path(directory) / "pipeline.hipfb"
        image = Path(directory) / "pipeline.hsaco"
        source.write_text(_PIPELINE_HIP)
        flags = ["-DTESSERA_PRODUCER_WAVE_REDUCE=1"] if producer_variant == "wave" else []
        result = subprocess.run(
            [str(compiler), "-x", "hip", "-O3", *flags, "--genco",
             "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
             str(source), "-o", str(bundle)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode or not bundle.is_file():
            raise RuntimeError(
                "gfx1201 graph pipeline compilation failed: "
                + (result.stderr.strip() or f"hipcc exited {result.returncode}")
            )
        return _extract_gfx1201_hsaco(bundle, image, rocm_path)


class PackedFoldedGraphPipeline(PackedFoldedGraphSession):
    """Fixed-M three-kernel graph with retained FP32 input and BF16 result."""

    def __init__(
        self, package: ROCMNativePackage, payload: PackedFoldedPayload, m: int,
        *, hip: Any | None = None,
        input_tensor: ROCMGraphTensor | None = None,
        output_tensor: ROCMGraphTensor | None = None,
        producer_variant: str = "block",
    ) -> None:
        if (input_tensor is None) != (output_tensor is None):
            raise ValueError("model-owned graph input and output must be supplied together")
        if producer_variant not in ("block", "wave"):
            raise ValueError("graph producer variant must be block or wave")
        super().__init__(package, payload, m, hip=hip)
        self._aux_module = ctypes.c_void_p()
        self._producer = ctypes.c_void_p()
        self._consumer = ctypes.c_void_p()
        self._input = ctypes.c_void_p()
        self._final = ctypes.c_void_p()
        self._aux_sha256 = ""
        self._pending_input: np.ndarray | None = None
        self._owns_io = input_tensor is None
        self._borrowed: list[ROCMGraphTensor] = []
        self._producer_variant = producer_variant
        try:
            image = _compile_auxiliary_hsaco(producer_variant)
            self._aux_sha256 = hashlib.sha256(image).hexdigest()
            self._call("hipModuleLoadData", ctypes.byref(self._aux_module), image)
            self._call(
                "hipModuleGetFunction", ctypes.byref(self._producer),
                self._aux_module, b"tessera_graph_e4m3_producer",
            )
            self._call(
                "hipModuleGetFunction", ctypes.byref(self._consumer),
                self._aux_module, b"tessera_graph_bf16_relu",
            )
            resident = self._resident
            if input_tensor is None or output_tensor is None:
                self._call("hipMalloc", ctypes.byref(self._input), resident.m * resident.k * 4)
                self._call("hipMalloc", ctypes.byref(self._final), resident.m * resident.n * 2)
            else:
                ordinal = ctypes.c_int()
                self._call("hipGetDevice", ctypes.byref(ordinal))
                self._input.value = input_tensor.borrow(
                    (resident.m, resident.k), np.float32, ordinal.value,
                )
                self._borrowed.append(input_tensor)
                from tessera import runtime as rt

                self._final.value = output_tensor.borrow(
                    (resident.m, resident.n), rt._bfloat16_dtype(), ordinal.value,
                )
                self._borrowed.append(output_tensor)
        except BaseException:
            self.close()
            raise

    @property
    def input_pointer(self) -> int:
        self._require_open()
        if self._input.value is None:
            raise RuntimeError("graph pipeline input buffer is unavailable")
        return self._input.value

    @property
    def final_output_pointer(self) -> int:
        self._require_open()
        if self._final.value is None:
            raise RuntimeError("graph pipeline output buffer is unavailable")
        return self._final.value

    def upload_fp32(self, x: np.ndarray) -> None:
        """Bootstrap or refresh the owned device input outside capture."""
        self._require_open()
        resident = self._resident
        if (x.dtype != np.float32 or x.shape != (resident.m, resident.k)
                or not x.flags.c_contiguous or not np.isfinite(x).all()):
            raise ValueError("pipeline input must be finite contiguous FP32 [M,K]")
        self._inputs_ready = False
        self._output_valid = False
        self._pending_input = x
        self._call(
            "hipMemcpyAsync", self._input, x.ctypes.data_as(ctypes.c_void_p),
            int(x.nbytes), 1, resident._stream,
        )
        self._resident.synchronize()
        self._pending_input = None
        self._inputs_ready = True

    def mark_device_input_ready(self, *, stream_pointer: int) -> None:
        """Assert an external FP32 producer wrote the leased input on this stream."""
        if stream_pointer != self.stream_pointer:
            raise ValueError("device FP32 producer must use the pipeline stream")
        self._output_valid = False
        self._inputs_ready = True

    def _launch(self, function: ctypes.c_void_p, grid_x: int, values: list[Any]) -> None:
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        self._call(
            "hipModuleLaunchKernel", function, grid_x, 1, 1, 256, 1, 1,
            0, self._resident._stream, arguments, None,
        )

    def _enqueue_producer(self) -> None:
        resident = self._resident
        self._launch(
            self._producer, resident.m,
            [ctypes.c_void_p(self._input.value),
             ctypes.c_void_p(resident._device[0].value),
             ctypes.c_void_p(resident._device[2].value),
             ctypes.c_int64(resident.m), ctypes.c_int64(resident.k)],
        )

    def _enqueue_consumer(self) -> None:
        resident = self._resident
        count = resident.m * resident.n
        self._launch(
            self._consumer, (count + 255) // 256,
            [ctypes.c_void_p(resident._device[4].value),
             ctypes.c_void_p(self._final.value), ctypes.c_int64(count)],
        )

    def capture(self) -> None:
        """Capture producer→GEMM→consumer and verify three kernel nodes."""
        self._require_open()
        if not self._inputs_ready:
            raise RuntimeError("pipeline capture requires initialized FP32 device input")
        if self._graph_exec.value:
            raise RuntimeError("packed MXFP4 pipeline graph was already captured")
        stream = self._resident._stream
        graph = ctypes.c_void_p()
        began = False
        try:
            self._call("hipStreamBeginCapture", stream, 1)
            began = True
            self._enqueue_producer()
            self._enqueue_kernel()
            self._enqueue_consumer()
            self._call("hipStreamEndCapture", stream, ctypes.byref(graph))
            began = False
            count = ctypes.c_size_t()
            self._call("hipGraphGetNodes", graph, None, ctypes.byref(count))
            if count.value != 3:
                raise RuntimeError(f"packed MXFP4 pipeline needs three nodes; got {count.value}")
            nodes = (ctypes.c_void_p * count.value)()
            self._call("hipGraphGetNodes", graph, nodes, ctypes.byref(count))
            kinds = []
            for node in nodes:
                kind = ctypes.c_int()
                self._call("hipGraphNodeGetType", node, ctypes.byref(kind))
                kinds.append(kind.value)
            if kinds != [0, 0, 0]:
                raise RuntimeError("packed MXFP4 pipeline contains a non-kernel node")
            self._call(
                "hipGraphInstantiate", ctypes.byref(self._graph_exec), graph,
                None, None, 0,
            )
            self._capture_nodes = tuple(kinds)
        except BaseException:
            if began:
                self._hip.hipStreamEndCapture(stream, ctypes.byref(graph))
            raise
        finally:
            if graph.value:
                self._hip.hipGraphDestroy(graph)

    def read_final(self, output: np.ndarray | None = None) -> np.ndarray:
        """Read the post-consumer BF16 tensor only after replay."""
        from tessera import runtime as rt

        self._require_open()
        if not self._output_valid:
            raise RuntimeError("packed MXFP4 pipeline has not replayed these inputs")
        resident = self._resident
        bf16 = rt._bfloat16_dtype()
        if bf16 is None:
            raise RuntimeError("packed MXFP4 pipeline requires BF16 support")
        if output is None:
            output = np.empty((resident.m, resident.n), dtype=bf16)
        if (output.dtype != np.dtype(bf16) or output.shape != (resident.m, resident.n)
                or not output.flags.c_contiguous):
            raise ValueError("pipeline output must be contiguous BF16 [M,N]")
        resident._pending_host.append(output)
        self._call(
            "hipMemcpyAsync", output.ctypes.data_as(ctypes.c_void_p),
            self._final, int(output.nbytes), 2, resident._stream,
        )
        resident.synchronize()
        return output

    def receipt(self) -> dict[str, object]:
        return {
            **super().receipt(),
            "route": "packed_folded_fp32_e4m3_gemm_relu_graph_manual",
            "producer": "fp32_to_ocp_e4m3_per_token_scale_v1",
            "producer_variant": self._producer_variant,
            "consumer": "bf16_relu_v1",
            "aux_hsaco_sha256": self._aux_sha256,
            "shape": [self._resident.m, self._resident.n, self._resident.k],
            "io_ownership": "session" if self._owns_io else "model_borrowed",
            "automatic_selection": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        synchronized = False
        try:
            super().close()
            synchronized = True
        finally:
            try:
                if self._owns_io:
                    if self._input.value:
                        self._hip.hipFree(self._input)
                    if self._final.value:
                        self._hip.hipFree(self._final)
                else:
                    for tensor in reversed(self._borrowed):
                        tensor.release(synchronized=synchronized)
                    self._borrowed.clear()
                self._input.value = None
                self._final.value = None
            finally:
                if self._aux_module.value:
                    self._hip.hipModuleUnload(self._aux_module)
                    self._aux_module.value = None


class PackedFoldedGraphPipelinePool:
    """Bounded exact-M graph cache; never retarget a captured pointer or grid.

    Each shape owns an independent package, stream, buffers, and graph. A
    cache miss beyond ``max_shapes`` fails instead of evicting a live lease.
    """

    def __init__(
        self, payload: PackedFoldedPayload,
        package_for_m: Callable[[int, PackedFoldedPayload], ROCMNativePackage],
        *, max_shapes: int = 2,
    ) -> None:
        if max_shapes < 1:
            raise ValueError("graph pipeline pool requires max_shapes >= 1")
        self._payload = payload
        self._package_for_m = package_for_m
        self._max_shapes = max_shapes
        self._sessions: dict[int, PackedFoldedGraphPipeline] = {}
        self._closed = False

    def get(self, m: int) -> PackedFoldedGraphPipeline:
        if self._closed:
            raise RuntimeError("graph pipeline pool is closed")
        if m in self._sessions:
            return self._sessions[m]
        if len(self._sessions) >= self._max_shapes:
            raise RuntimeError("graph pipeline shape budget exhausted; release a shape first")
        package = self._package_for_m(m, self._payload)
        session = PackedFoldedGraphPipeline(package, self._payload, m)
        self._sessions[m] = session
        return session

    def release(self, m: int) -> None:
        if self._closed:
            raise RuntimeError("graph pipeline pool is closed")
        session = self._sessions.pop(m)
        session.close()

    def receipt(self) -> dict[str, object]:
        return {
            "route": "packed_folded_graph_exact_m_pool_manual",
            "dynamic_m_strategy": "separate_exact_shape_graphs",
            "active_m": sorted(self._sessions),
            "max_shapes": self._max_shapes,
            "automatic_selection": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for m, session in tuple(self._sessions.items()):
            try:
                session.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                del self._sessions[m]
        if first_error is not None:
            raise first_error

    def __enter__(self) -> PackedFoldedGraphPipelinePool:
        if self._closed:
            raise RuntimeError("graph pipeline pool is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["PackedFoldedGraphPipeline", "PackedFoldedGraphPipelinePool"]
