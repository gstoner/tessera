"""Resident gfx1201 RMSNorm -> matmul handoff for one checked tensor edge."""

from __future__ import annotations

import copy
import ctypes
from typing import Any, cast

import numpy as np


class ResidentROCmNormMatmul:
    """Keep an f16/bf16 RMSNorm result resident between scheduled gfx1201 images.

    The API admits one static, contiguous low-precision last-axis RMSNorm
    producer and one same-storage, f32-accumulating matmul consumer. Both launches use
    one private HIP stream and one owned intermediate allocation. Host arrays
    are snapshotted at construction; close synchronizes before releasing any
    image, event, stream, or allocation.
    """

    def __init__(self, norm_package: Any, matmul_package: Any, x: Any, rhs: Any):
        from tessera import runtime as rt
        from tessera.compiler.rocm_native import (
            GFX_MATMUL_BF16_F32_ABI, GFX_MATMUL_F16_F32_ABI,
            GFX_NORM_BF16_ABI, GFX_NORM_F16_ABI,
        )

        self._closed = False
        self._hip: ctypes.CDLL | None = None
        self._stream = ctypes.c_void_p()
        self._modules: list[ctypes.c_void_p] = []
        self._events: list[ctypes.c_void_p] = []
        self._buffers: dict[str, ctypes.c_void_p] = {}
        self._image_storage: list[Any] = []
        self._norm_function = ctypes.c_void_p()
        self._gemm_function = ctypes.c_void_p()
        self._norm_package, self._matmul_package = norm_package, matmul_package
        self._norm, self._gemm = norm_package.descriptor, matmul_package.descriptor
        if any(
            package.image.target != "rocm_gfx1201"
            or package.image.architecture != "gfx1201"
            or package.image.binary_format != "hsaco"
            for package in (norm_package, matmul_package)
        ):
            raise ValueError("resident norm-matmul requires gfx1201 HSACO packages")
        abi_pairs = {
            GFX_NORM_F16_ABI: ("fp16", GFX_MATMUL_F16_F32_ABI),
            GFX_NORM_BF16_ABI: ("bf16", GFX_MATMUL_BF16_F32_ABI),
        }
        pair = abi_pairs.get(self._norm.abi_id)
        if pair is None or self._gemm.abi_id != pair[1]:
            raise ValueError("resident norm-matmul requires matching f16 or bf16 producer/consumer ABIs")
        self.storage_dtype_name = pair[0]
        if self.storage_dtype_name == "bf16":
            import ml_dtypes
            self.storage_dtype = np.dtype(ml_dtypes.bfloat16)
        else:
            self.storage_dtype = np.dtype(np.float16)
        if self._norm.provenance.get("family") != "norm" or self._norm.provenance.get("kind") != "rmsnorm":
            raise ValueError("resident norm-matmul producer must be scheduled RMSNorm")
        if self._gemm.provenance.get("activation") != "none" or self._gemm.provenance.get("bias"):
            raise ValueError("resident norm-matmul consumer must be a plain scheduled matmul")
        self.rows, self.k = tuple(int(v) for v in self._norm.provenance["shape"])
        self.m, self.n, gemm_k = (int(v) for v in self._gemm.provenance["shape"])
        if self.m != self.rows or self.k != gemm_k:
            raise ValueError("resident RMSNorm output shape must match matmul A shape")
        self.epsilon = float(self._norm.provenance["epsilon"])
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("resident RMSNorm epsilon must be finite and positive")
        self.x = np.array(x, dtype=self.storage_dtype, order="C", copy=True)
        self.rhs = np.array(rhs, dtype=self.storage_dtype, order="C", copy=True)
        if self.x.shape != (self.rows, self.k) or self.rhs.shape != (self.k, self.n):
            raise ValueError("resident inputs disagree with package shapes")
        if not np.isfinite(self.x).all() or not np.isfinite(self.rhs).all():
            raise ValueError("resident inputs must contain finite values")
        self.output = np.empty((self.m, self.n), dtype=np.float32)
        self._tmp_host = np.empty((self.rows, self.k), dtype=self.storage_dtype)
        self._norm_input = self._binding_name(self._norm, "input")
        self._norm_output = self._binding_name(self._norm, "output")
        self._gemm_a = self._binding_name(self._gemm, "input", ordinal=0)
        self._gemm_b = self._binding_name(self._gemm, "input", ordinal=1)
        self._gemm_output = self._binding_name(self._gemm, "output")
        self._dynamic_n = any(
            guard.binding == self._gemm_b
            and guard.dimension == 1
            and guard.predicate == "max"
            and guard.value == self.n
            for guard in self._gemm.shape_guards
        )
        norm_scalars = {item.name for item in self._norm.scalars}
        gemm_scalars = {item.name for item in self._gemm.scalars}
        if norm_scalars != {"Rows", "K", "Epsilon"} or gemm_scalars != {"M", "N", "K"}:
            raise ValueError("resident norm-matmul package scalar ABI is not canonical")
        self._validate_descriptors()
        self._grid_norm = (self.rows, 1, 1)
        self._block_norm = tuple(self._norm.provenance.get("workgroup", ()))
        macro = self._gemm.provenance.get("macro_tile")
        workgroup = self._gemm.provenance.get("workgroup")
        if (
            not isinstance(self._block_norm, tuple)
            or len(self._block_norm) != 3
            or self._block_norm[0] <= 0
            or self._block_norm[1:] != (1, 1)
            or not isinstance(macro, list)
            or len(macro) != 2
            or not isinstance(workgroup, list)
            or len(workgroup) != 3
            or workgroup[1:] != [1, 1]
        ):
            raise ValueError("resident packages lack checked fixed launch geometry")
        self._macro_tile = tuple(int(v) for v in macro)
        self._grid_gemm = (
            (self.n + self._macro_tile[1] - 1) // self._macro_tile[1],
            (self.m + self._macro_tile[0] - 1) // self._macro_tile[0],
            1,
        )
        self._block_gemm = cast(
            tuple[int, int, int], tuple(int(v) for v in workgroup)
        )
        if rt._rocm_live_arch() != "gfx1201" or rt._rocm_chip() != "gfx1201":
            raise RuntimeError("resident norm-matmul requires the exact gfx1201 owning device")
        hip = rt._load_hip_for_launch()
        if hip is None or hip.hipInit(0) != 0:
            raise RuntimeError("resident norm-matmul requires initialized HIP")
        self._hip = hip
        current_device = ctypes.c_int()
        if hip.hipGetDevice(ctypes.byref(current_device)) != 0:
            raise RuntimeError("resident norm-matmul cannot identify the owning device")
        self._device_id = current_device.value
        self._bind_hip_api()
        try:
            if hip.hipStreamCreateWithFlags(ctypes.byref(self._stream), 1) != 0:
                raise RuntimeError("resident norm-matmul stream creation failed")
            self._norm_function = self._load_module(
                norm_package.image.payload, self._norm.entry_symbol
            )
            self._gemm_function = self._load_module(
                matmul_package.image.payload, self._gemm.entry_symbol
            )
            for name, array in (
                ("x", self.x),
                ("rhs", self.rhs),
                ("intermediate", self._tmp_host),
                ("output", self.output),
            ):
                self._allocate(name, array.nbytes)
            for pointer, array in ((self._buffers["x"], self.x), (self._buffers["rhs"], self.rhs)):
                if (
                    hip.hipMemcpyAsync(pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1, self._stream)
                    != 0
                ):
                    raise RuntimeError("resident norm-matmul input upload failed")
            for _ in range(4):
                event = ctypes.c_void_p()
                if hip.hipEventCreate(ctypes.byref(event)) != 0:
                    raise RuntimeError("resident norm-matmul event creation failed")
                self._events.append(event)
            if hip.hipStreamSynchronize(self._stream) != 0:
                raise RuntimeError("resident norm-matmul input upload synchronization failed")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _binding_name(descriptor: Any, direction: str, ordinal: int | None = None) -> str:
        items = [item for item in descriptor.buffers if item.direction == direction]
        if ordinal is not None:
            items = [item for item in items if item.ordinal == ordinal]
        if len(items) != 1:
            raise ValueError(f"resident norm-matmul expected one {direction} binding at {ordinal}")
        return items[0].name

    def _validate_descriptors(self) -> None:
        from tessera.compiler.native_artifact import BufferArgument

        norm_args = {
            self._norm_input: BufferArgument(self.storage_dtype_name, (self.rows, self.k), "row_major", 2),
            self._norm_output: BufferArgument(self.storage_dtype_name, (self.rows, self.k), "row_major", 2),
        }
        gemm_args = {
            self._gemm_a: BufferArgument(self.storage_dtype_name, (self.m, self.k), "row_major", 2),
            self._gemm_b: BufferArgument(self.storage_dtype_name, (self.k, self.n), "row_major", 2),
            self._gemm_output: BufferArgument("fp32", (self.m, self.n), "row_major", 4),
        }
        self._norm.validate_invocation(
            self._norm_package.image,
            norm_args,
            {"Rows": self.rows, "K": self.k, "Epsilon": self.epsilon},
        )
        self._gemm.validate_invocation(
            self._matmul_package.image,
            gemm_args,
            {"M": self.m, "N": self.n, "K": self.k},
        )

    def _require_hip(self) -> ctypes.CDLL:
        hip = self._hip
        if hip is None:
            raise RuntimeError("resident norm-matmul HIP library is not initialized")
        return hip

    def _bind_hip_api(self) -> None:
        hip = self._require_hip()
        hip.hipStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        hip.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        hip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        hip.hipEventDestroy.argtypes = [ctypes.c_void_p]
        hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        hip.hipModuleUnload.argtypes = [ctypes.c_void_p]

    def _load_module(self, payload: bytes, entry: str) -> ctypes.c_void_p:
        hip = self._require_hip()
        storage = ctypes.create_string_buffer(payload)
        module = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(module), ctypes.cast(storage, ctypes.c_void_p)) != 0:
            raise RuntimeError("resident norm-matmul HSACO load failed")
        self._image_storage.append(storage)
        self._modules.append(module)
        function = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(function), module, entry.encode()) != 0:
            raise RuntimeError(f"resident norm-matmul entry {entry!r} not found")
        return function

    def _allocate(self, name: str, size: int) -> None:
        hip = self._require_hip()
        pointer = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(pointer), max(int(size), 1)) != 0 or pointer.value is None:
            raise RuntimeError(f"resident norm-matmul {name} allocation failed")
        self._buffers[name] = pointer

    @staticmethod
    def _memref(pointer: ctypes.c_void_p, elements: int) -> list[Any]:
        return [
            ctypes.c_void_p(pointer.value),
            ctypes.c_void_p(pointer.value),
            ctypes.c_int64(0),
            ctypes.c_int64(elements),
            ctypes.c_int64(1),
        ]

    def _launch(
        self, function: ctypes.c_void_p, grid: tuple[int, int, int], block: tuple[int, int, int], values: list[Any]
    ) -> None:
        holders = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            holders[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        gx, gy, gz = grid
        bx, by, bz = block
        hip = self._require_hip()
        rc = hip.hipModuleLaunchKernel(function, gx, gy, gz, bx, by, bz, 0, self._stream, holders, None)
        if rc != 0:
            raise RuntimeError(f"resident norm-matmul kernel launch failed rc={rc}")

    def _run_once(self, rhs: Any | None = None) -> tuple[np.ndarray, float, float]:
        if self._closed:
            raise RuntimeError("resident norm-matmul session is closed")
        hip = self._require_hip()
        active_rhs = self.rhs if rhs is None else np.array(rhs, dtype=self.storage_dtype, order="C", copy=True)
        if (
            active_rhs.ndim != 2
            or active_rhs.shape[0] != self.k
            or active_rhs.shape[1] <= 0
            or active_rhs.shape[1] > self.n
            or (active_rhs.shape[1] != self.n and not self._dynamic_n)
            or not np.isfinite(active_rhs).all()
        ):
            raise ValueError("runtime RHS must satisfy the scheduled matmul N bound")
        active_n = int(active_rhs.shape[1])
        active_output = np.empty((self.m, active_n), dtype=np.float32)
        from tessera.compiler.native_artifact import BufferArgument

        runtime_buffers = {
            self._gemm_a: BufferArgument(self.storage_dtype_name, (self.m, self.k), "row_major", 2),
            self._gemm_b: BufferArgument(self.storage_dtype_name, (self.k, active_n), "row_major", 2),
            self._gemm_output: BufferArgument("fp32", (self.m, active_n), "row_major", 4),
        }
        runtime_scalars = {"M": self.m, "N": active_n, "K": self.k}
        self._gemm.validate_invocation(self._matmul_package.image, runtime_buffers, runtime_scalars)
        hip = self._require_hip()
        start_norm, stop_norm, start_gemm, stop_gemm = self._events
        if hip.hipMemcpyAsync(
            self._buffers["rhs"], active_rhs.ctypes.data_as(ctypes.c_void_p), active_rhs.nbytes, 1, self._stream
        ) != 0:
            raise RuntimeError("resident matmul RHS upload failed")
        if hip.hipEventRecord(start_norm, self._stream) != 0:
            raise RuntimeError("resident RMSNorm start event failed")
        norm_args = (
            self._memref(self._buffers["x"], self.rows * self.k)
            + self._memref(self._buffers["intermediate"], self.rows * self.k)
            + [ctypes.c_int64(self.rows), ctypes.c_int64(self.k), ctypes.c_float(self.epsilon)]
        )
        self._launch(self._norm_function, self._grid_norm, self._block_norm, norm_args)
        if hip.hipEventRecord(stop_norm, self._stream) != 0:
            raise RuntimeError("resident RMSNorm stop event failed")
        if hip.hipEventRecord(start_gemm, self._stream) != 0:
            raise RuntimeError("resident matmul start event failed")
        gemm_args = (
            self._memref(self._buffers["intermediate"], self.m * self.k)
            + self._memref(self._buffers["rhs"], self.k * active_n)
            + self._memref(self._buffers["output"], self.m * active_n)
            + [ctypes.c_int64(self.m), ctypes.c_int64(active_n), ctypes.c_int64(self.k)]
        )
        grid_gemm = (
            (active_n + self._macro_tile[1] - 1) // self._macro_tile[1],
            self._grid_gemm[1],
            1,
        )
        self._launch(self._gemm_function, grid_gemm, self._block_gemm, gemm_args)
        if hip.hipEventRecord(stop_gemm, self._stream) != 0 or hip.hipEventSynchronize(stop_gemm) != 0:
            raise RuntimeError("resident matmul completion event failed")
        norm_ms, gemm_ms = ctypes.c_float(), ctypes.c_float()
        if (
            hip.hipEventElapsedTime(ctypes.byref(norm_ms), start_norm, stop_norm) != 0
            or hip.hipEventElapsedTime(ctypes.byref(gemm_ms), start_gemm, stop_gemm) != 0
        ):
            raise RuntimeError("resident norm-matmul event timing failed")
        if (
            hip.hipMemcpyAsync(
                active_output.ctypes.data_as(ctypes.c_void_p),
                self._buffers["output"],
                active_output.nbytes,
                2,
                self._stream,
            )
            != 0
        ):
            raise RuntimeError("resident norm-matmul output copyback failed")
        if hip.hipStreamSynchronize(self._stream) != 0:
            raise RuntimeError("resident norm-matmul output synchronization failed")
        return active_output, float(norm_ms.value), float(gemm_ms.value)

    def run(
        self, *, warmup: int = 2, iterations: int = 5, rhs: Any | None = None
    ) -> dict[str, Any]:
        if warmup < 0 or iterations <= 0:
            raise ValueError("warmup must be nonnegative and iterations positive")
        for _ in range(warmup):
            self._run_once(rhs)
        producer, consumer, outputs = [], [], []
        for _ in range(iterations):
            output, norm_ms, gemm_ms = self._run_once(rhs)
            outputs.append(output)
            producer.append(norm_ms)
            consumer.append(gemm_ms)
        return {
            "outputs": outputs,
            "producer_device_event_ms": producer,
            "consumer_device_event_ms": consumer,
            "producer_median_ms": float(np.median(producer)),
            "consumer_median_ms": float(np.median(consumer)),
            "buffer_addresses": {
                name: int(ptr.value) if ptr.value is not None else 0
                for name, ptr in self._buffers.items()
            },
            "device_id": self._device_id,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        hip = self._hip
        if hip is None:
            return
        if self._stream.value:
            hip.hipStreamSynchronize(self._stream)
        for event in reversed(self._events):
            if event.value:
                hip.hipEventDestroy(event)
        self._events.clear()
        for module in reversed(self._modules):
            if module.value:
                hip.hipModuleUnload(module)
        self._modules.clear()
        for pointer in reversed(tuple(self._buffers.values())):
            if pointer.value:
                hip.hipFree(pointer)
        self._buffers.clear()
        if self._stream.value:
            hip.hipStreamDestroy(self._stream)
            self._stream = ctypes.c_void_p()

    def __enter__(self) -> "ResidentROCmNormMatmul":
        if self._closed:
            raise RuntimeError("resident norm-matmul session is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _with_bounded_dynamic_n(matmul_module: Any, rhs: Any, bound: int) -> Any:
    """Give one traced Graph matmul a bounded dynamic N contract."""
    from .graph_ir import tensor_ir_type

    if bound <= 0:
        raise ValueError("dynamic N bound must be positive")
    module = copy.deepcopy(matmul_module)
    if len(module.functions) != 1:
        raise ValueError("bounded dynamic N requires one Graph function")
    function = module.functions[0]
    if len(function.args) != 2 or len(function.result_types) != 1:
        raise ValueError("bounded dynamic N requires a two-input, one-result matmul")
    matmuls = [op for op in function.body if op.op_name == "tessera.matmul"]
    if len(matmuls) != 1:
        raise ValueError("bounded dynamic N requires one Graph matmul operation")
    lhs_type, rhs_type = function.args[0].ir_type, function.args[1].ir_type
    output_type = function.result_types[0]
    try:
        m, k = (int(str(dim)) for dim in lhs_type.shape)
        rhs_k, rhs_n = (int(str(dim)) for dim in rhs_type.shape)
        out_m, out_n = (int(str(dim)) for dim in output_type.shape)
        actual_rhs_shape = tuple(int(dim) for dim in rhs.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("bounded dynamic N requires static traced tensor shapes") from exc
    if (rhs_k, rhs_n) != (k, bound) or actual_rhs_shape != (k, bound):
        raise ValueError("dynamic N bound must match the traced RHS capacity")
    if (out_m, out_n) != (m, bound):
        raise ValueError("dynamic N bound must match the traced matmul result")
    dynamic_rhs = tensor_ir_type(
        (str(k), "?"), rhs_type.dtype, layout=rhs_type.layout,
    )
    dynamic_output = tensor_ir_type(
        (str(m), "?"), output_type.dtype, layout=output_type.layout,
    )
    function.args[1].ir_type = dynamic_rhs
    function.result_types[0] = dynamic_output
    op = matmuls[0]
    op.operand_types[1] = str(dynamic_rhs)
    op.result_type = str(dynamic_output)
    op.inferred_type = dynamic_output
    op.kwargs["shape_bounds"] = [m, bound, k]
    return module


def lower_graph_rmsnorm_matmul(norm_module: Any, matmul_module: Any) -> tuple[Any, Any]:
    """Lower traced Graph IR modules through gfx1201 Schedule and Tile IR."""
    from .scheduled_kernel import lower_scheduled_kernel
    from .scheduled_matmul import lower_scheduled_matmul

    norm = lower_scheduled_kernel(norm_module, target="rocm_gfx1201")
    matmul = lower_scheduled_matmul(matmul_module, target="rocm_gfx1201")
    if norm.architecture != "gfx1201" or matmul.architecture != "gfx1201":
        raise ValueError("Graph RMSNorm-matmul lowering must remain on gfx1201")
    return norm, matmul


def package_graph_rmsnorm_matmul(
    norm_module: Any,
    matmul_module: Any,
    x: Any,
    rhs: Any,
    *,
    dynamic_n_bound: int | None = None,
    pipeline_name: str = "tessera-lower-to-rocm",
) -> ResidentROCmNormMatmul:
    """Create the resident gfx1201 package directly from traced Graph IR.

    When supplied, dynamic_n_bound turns the traced consumer's static N extent
    into a bounded runtime N guard while retaining its maximum storage.
    """
    from . import rocm_native

    if dynamic_n_bound is not None:
        matmul_module = _with_bounded_dynamic_n(matmul_module, rhs, dynamic_n_bound)
    norm, matmul = lower_graph_rmsnorm_matmul(norm_module, matmul_module)
    norm_package = rocm_native.package_scheduled_kernel(
        norm, pipeline_name=pipeline_name,
    )
    matmul_package = rocm_native.package_scheduled_matmul(
        matmul, pipeline_name=pipeline_name,
    )
    return ResidentROCmNormMatmul(norm_package, matmul_package, x, rhs)
