"""Opt-in resident HIP lifecycle for the gfx1201 packed-folded prefill ABI.

This keeps the proved code object and fragment-order weights on the selected
device across launches. It is a manual executor, not capability admission.
"""
from __future__ import annotations

import ctypes
import hashlib
from typing import Any

import numpy as np

from .rocm_mxfp4_packed_folded import (
    PACKED_FOLDED_PHYSICAL_V1,
    PACKED_FOLDED_SCALE_PLANE_V1,
    PACKED_FOLDED_TARGET_ABI_V1,
    PACKED_FOLDED_WEIGHT_LAYOUT_V1,
    PackedFoldedPayload,
)
from .rocm_native import ROCMNativePackage


def _check_package(
    package: ROCMNativePackage, payload: PackedFoldedPayload, m: int,
) -> tuple[int, int]:
    image, descriptor = package.image, package.descriptor
    n, k = payload.shape
    if m <= 64 or n % 16 or k % 64:
        raise ValueError("resident packed-folded prefill requires M>64, N%16=0, K%64=0")
    if image.target != "rocm_gfx1201" or image.architecture != "gfx1201":
        raise ValueError("resident MXFP4 requires an exact gfx1201 image")
    if descriptor.abi_id != PACKED_FOLDED_TARGET_ABI_V1:
        raise ValueError("resident MXFP4 requires the packed-folded target ABI")
    if descriptor.image_digest != image.image_digest:
        raise ValueError("resident MXFP4 descriptor/image digest mismatch")
    if not any(
        entry.symbol == descriptor.entry_symbol and entry.abi_id == descriptor.abi_id
        for entry in image.entry_points
    ):
        raise ValueError("resident MXFP4 entry symbol is absent from the image")
    if tuple(item.name for item in sorted(descriptor.buffers, key=lambda item: item.ordinal)) != (
        "a", "b_packed", "a_scale", "scale_plane", "output",
    ):
        raise ValueError("resident MXFP4 buffer order disagrees with the ABI")
    if tuple(item.name for item in sorted(descriptor.scalars, key=lambda item: item.ordinal)) != (
        "M", "N", "K",
    ):
        raise ValueError("resident MXFP4 scalar order disagrees with the ABI")
    provenance = descriptor.provenance
    receipt = payload.receipt()
    expected = {
        "weight_layout": PACKED_FOLDED_WEIGHT_LAYOUT_V1,
        "scale_layout": PACKED_FOLDED_SCALE_PLANE_V1,
        "numeric_policy": "folded_row_reference_explicit_approximate",
        "physical_contract": PACKED_FOLDED_PHYSICAL_V1,
        "weight_sha256": receipt["weight_sha256"],
        "scale_plane_sha256": receipt["scale_plane_sha256"],
        "execution_state": "manual_executable_candidate",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"resident MXFP4 {key} disagrees with the package")
    guards = {(guard.binding, guard.dimension, guard.predicate, guard.value)
              for guard in descriptor.shape_guards}
    for name, dimension, value in (
        ("a", 0, m), ("a", 1, k), ("b_packed", 0, n),
        ("b_packed", 1, k // 2), ("a_scale", 0, m),
        ("scale_plane", 0, k // 32 + 1), ("scale_plane", 1, n),
        ("output", 0, m), ("output", 1, n),
    ):
        if (name, dimension, "eq", value) not in guards:
            raise ValueError(f"resident MXFP4 {name} shape guard mismatch")
    geometry = descriptor.geometry
    if geometry.grid != ((n + 63) // 64, (m + 255) // 256, 1):
        raise ValueError("resident MXFP4 grid disagrees with the package shape")
    if geometry.workgroup != (256, 1, 1):
        raise ValueError("resident MXFP4 requires a 256-thread workgroup")
    return n, k


class PackedFoldedResidentSession:
    """One fixed-shape HIP stream/module and five persistent device buffers.

    Host uploads may use pageable NumPy memory; HIP can make those calls
    synchronous. ``launch_resident`` itself performs no allocation, copy, or
    synchronization. The session owns every pointer until ``close``.
    """

    def __init__(
        self, package: ROCMNativePackage, payload: PackedFoldedPayload, m: int,
        *, hip: Any | None = None,
    ) -> None:
        from tessera import runtime as rt

        n, k = _check_package(package, payload, m)
        if rt._rocm_live_arch() != "gfx1201":
            raise RuntimeError("resident MXFP4 requires the selected gfx1201 device")
        loaded_hip = hip if hip is not None else rt._load_hip_for_launch()
        if loaded_hip is None or loaded_hip.hipInit(0) != 0:
            raise RuntimeError("resident MXFP4 requires a usable HIP context")
        self._hip: Any = loaded_hip
        self.m, self.n, self.k = m, n, k
        self.image_sha256 = hashlib.sha256(package.image.payload).hexdigest()
        self.weight_sha256 = str(payload.receipt()["weight_sha256"])
        self._stream = ctypes.c_void_p()
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._device = [ctypes.c_void_p() for _ in range(5)]
        self._pending_host: list[np.ndarray] = []
        self._activations_loaded = False
        self._launched = False
        self._closed = False
        self._upload_count = 0
        self._launch_count = 0
        self._bind_hip()
        try:
            self._call(
                "hipStreamCreateWithFlags",
                ctypes.byref(self._stream), 1,
            )
            self._call(
                "hipModuleLoadData", ctypes.byref(self._module), package.image.payload,
            )
            self._call(
                "hipModuleGetFunction", ctypes.byref(self._function),
                self._module, package.descriptor.entry_symbol.encode(),
            )
            self._sizes = (
                m * k, payload.weight_bytes.nbytes, m * 4,
                payload.scale_plane.nbytes, m * n * 2,
            )
            for pointer, size in zip(self._device, self._sizes, strict=True):
                self._call("hipMalloc", ctypes.byref(pointer), size)
            self._copy_to_device(1, payload.weight_bytes)
            self._copy_to_device(3, payload.scale_plane)
            self.synchronize()
        except BaseException:
            self.close()
            raise

    def _bind_hip(self) -> None:
        hip = self._hip
        hip.hipStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        hip.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_void_p,
        ]

    def _call(self, name: str, *args: Any) -> None:
        rc = getattr(self._hip, name)(*args)
        if rc != 0:
            raise RuntimeError(f"resident MXFP4 {name} failed rc={rc}")

    def _copy_to_device(self, ordinal: int, array: np.ndarray) -> None:
        self._pending_host.append(array)
        self._call(
            "hipMemcpyAsync", self._device[ordinal],
            array.ctypes.data_as(ctypes.c_void_p), int(array.nbytes), 1,
            self._stream,
        )

    def upload_activations(self, a: np.ndarray, a_scale: np.ndarray) -> None:
        """Update the fixed-shape A/As buffers; wait before another host upload."""
        self._require_open()
        if self._pending_host:
            raise RuntimeError("synchronize before reusing resident host staging")
        if (a.dtype != np.uint8 or a.shape != (self.m, self.k)
                or not a.flags.c_contiguous):
            raise ValueError("resident MXFP4 A must be contiguous uint8 [M,K]")
        if (a_scale.dtype != np.float32 or a_scale.shape != (self.m,)
                or not a_scale.flags.c_contiguous):
            raise ValueError("resident MXFP4 As must be contiguous float32 [M]")
        self._activations_loaded = False
        self._launched = False
        self._copy_to_device(0, a)
        self._copy_to_device(2, a_scale)
        self._activations_loaded = True
        self._upload_count += 1

    def launch_resident(self) -> None:
        """Enqueue only the kernel; A, As, B, scales, and output stay resident."""
        self._require_open()
        if not self._activations_loaded:
            raise RuntimeError("resident MXFP4 activations have not been uploaded")
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in self._device),
            ctypes.c_int64(self.m), ctypes.c_int64(self.n), ctypes.c_int64(self.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        self._call(
            "hipModuleLaunchKernel", self._function,
            (self.n + 63) // 64, (self.m + 255) // 256, 1,
            256, 1, 1, 0, self._stream, arguments, None,
        )
        self._launched = True
        self._launch_count += 1

    def synchronize(self) -> None:
        self._require_open()
        self._call("hipStreamSynchronize", self._stream)
        self._pending_host.clear()

    def read_output(self, output: np.ndarray | None = None) -> np.ndarray:
        """Download BF16 output and wait only for this session's stream."""
        from tessera import runtime as rt

        self._require_open()
        if not self._launched:
            raise RuntimeError("resident MXFP4 has not launched")
        bf16 = rt._bfloat16_dtype()
        if bf16 is None:
            raise RuntimeError("resident MXFP4 requires ml_dtypes.bfloat16")
        if output is None:
            output = np.empty((self.m, self.n), dtype=bf16)
        if (output.dtype != np.dtype(bf16) or output.shape != (self.m, self.n)
                or not output.flags.c_contiguous):
            raise ValueError("resident MXFP4 output must be contiguous BF16 [M,N]")
        self._pending_host.append(output)
        self._call(
            "hipMemcpyAsync", output.ctypes.data_as(ctypes.c_void_p),
            self._device[4], int(output.nbytes), 2, self._stream,
        )
        self.synchronize()
        return output

    def run_host(
        self, a: np.ndarray, a_scale: np.ndarray, output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compatibility control: reusable module/buffers with host I/O."""
        self.upload_activations(a, a_scale)
        self.launch_resident()
        return self.read_output(output)

    def receipt(self) -> dict[str, object]:
        return {
            "route": "packed_folded_resident_hip_manual",
            "target": "rocm_gfx1201",
            "abi_id": PACKED_FOLDED_TARGET_ABI_V1,
            "image_sha256": self.image_sha256,
            "weight_sha256": self.weight_sha256,
            "module_loads": 1,
            "weight_uploads": 1,
            "activation_uploads": self._upload_count,
            "kernel_launches": self._launch_count,
            "automatic_selection": False,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("resident MXFP4 session is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        hip = self._hip
        if self._stream.value:
            hip.hipStreamSynchronize(self._stream)
        self._pending_host.clear()
        for pointer in reversed(self._device):
            if pointer.value:
                hip.hipFree(pointer)
                pointer.value = None
        if self._module.value:
            hip.hipModuleUnload(self._module)
            self._module.value = None
        if self._stream.value:
            hip.hipStreamDestroy(self._stream)
            self._stream.value = None

    def __enter__(self) -> PackedFoldedResidentSession:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["PackedFoldedResidentSession"]
