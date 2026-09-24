"""Opt-in RDNA W4A4 numerical probe for bounded Quark projection slices.

This is a distinct activation/weight ABI, not a checkpoint converter or an
automatic production selection. The kernel is scalar HIP (no WMMA), so it is
packaged for whichever supported chip is live -- gfx1201 or gfx1151 -- and each
chip's result is its own exact-device proof; neither transfers. The caller supplies already packed E2M1
activation and weight planes in row-major, low-even K order. E8M0 edge codes
0 and 255 remain outside this probe's proved domain.
"""

from __future__ import annotations

import hashlib
import ctypes
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from .rocm_mxfp4_native import _rocm_hipcc, _rocm_offload_bundler
from .rocm_native import (
    ROCMNativePackage,
    _driver_selected_device_libraries,
    _rocm_path,
    _version_fingerprint,
)


GFX1201_QUARK_W4A4_PROBE_ABI = "tessera.rocm.quark_w4a4.a_b_sa_sb_o_m_n_k.e2m1_low_even_e8m0_k32_bf16.exact_probe.v1"
QUARK_W4A4_SYNC_KEY = "GFX1201-QUARK-INDEPENDENT-W4A4-2026-09-23"
QUARK_W4A4_PROBE_ARCHS = ("gfx1151", "gfx1201")


def _probe_arch(arch: str) -> str:
    if arch not in QUARK_W4A4_PROBE_ARCHS:
        raise ValueError(
            f"Quark W4A4 probe supports exact {' or '.join(QUARK_W4A4_PROBE_ARCHS)}; got {arch!r}"
        )
    return arch


def _extract_hsaco(compiled: Path, output: Path, rocm_path: Path, arch: str) -> bytes:
    """Return the raw HSACO for ``arch`` from raw or bundled HIP output."""
    payload = compiled.read_bytes()
    if payload.startswith(b"\x7fELF"):
        return payload
    if not payload.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        raise RuntimeError("Quark W4A4 compiler output is neither ELF nor a HIP bundle")
    bundler = _rocm_offload_bundler(rocm_path)
    if bundler is None:
        raise RuntimeError("Quark W4A4 HIP bundle requires clang-offload-bundler")
    result = subprocess.run(
        [
            str(bundler), "-unbundle", "-type=o",
            f"-targets=hipv4-amdgcn-amd-amdhsa--{arch}",
            f"-input={compiled}", f"-output={output}",
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or not output.is_file():
        detail = result.stderr.strip() or f"clang-offload-bundler exited {result.returncode}"
        raise RuntimeError(f"Quark W4A4 HSACO extraction failed: {detail}")
    image = output.read_bytes()
    if not image.startswith(b"\x7fELF"):
        raise RuntimeError("Quark W4A4 extracted device image is not an ELF HSACO")
    return image


def validate_quark_w4a4_buffers(buffers: dict[str, np.ndarray]) -> tuple[int, int, int]:
    """Check the bounded byte-level input contract before a probe launch."""
    names = ("a_packed", "b_packed", "a_scale", "b_scale", "output")
    if set(buffers) != set(names):
        raise ValueError("Quark W4A4 probe requires exactly five named buffers")
    a, b, sa, sb, out = (np.asarray(buffers[name]) for name in names)
    if any(array.ndim != 2 or not array.flags.c_contiguous for array in (a, b, sa, sb, out)):
        raise ValueError("Quark W4A4 probe requires contiguous rank-two buffers")
    if any(array.dtype != np.uint8 for array in (a, b, sa, sb)):
        raise TypeError("Quark W4A4 packed and scale planes must be uint8")
    if out.dtype.name != "bfloat16":
        raise TypeError("Quark W4A4 output must be bfloat16")
    m, packed_k = a.shape
    n, weight_k = b.shape
    k = packed_k * 2
    if min(m, n, packed_k) <= 0 or packed_k != weight_k or k % 32:
        raise ValueError("Quark W4A4 requires positive M/N and K divisible by 32")
    if sa.shape != (m, k // 32) or sb.shape != (n, k // 32) or out.shape != (m, n):
        raise ValueError("Quark W4A4 scale or output shape disagrees with packed planes")
    if np.any((sa == 0) | (sa == 255)) or np.any((sb == 0) | (sb == 255)):
        raise ValueError("Quark W4A4 scale-code 0/255 semantics are not proved")
    return m, n, k


def emit_quark_w4a4_probe_hip(*, entry: str = "tessera_quark_w4a4_probe") -> str:
    if not entry.isidentifier() or not entry.isascii():
        raise ValueError("Quark W4A4 entry must be an ASCII C identifier")
    return f"""#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float tessera_e2m1(uint8_t code) {{
  const float table[8] = {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}};
  const float value = table[code & 7u];
  return (code & 8u) ? -value : value;
}}

__device__ __forceinline__ uint16_t tessera_bf16_rne(float value) {{
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}}

extern "C" __global__ void {entry}(
    const uint8_t *__restrict__ a_packed,
    const uint8_t *__restrict__ b_packed,
    const uint8_t *__restrict__ a_scale,
    const uint8_t *__restrict__ b_scale,
    uint16_t *__restrict__ output,
    int64_t M, int64_t N, int64_t K) {{
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float accum = 0.0f;
  for (int64_t group = 0; group < K / 32; ++group) {{
    float partial = 0.0f;
    for (int64_t offset = 0; offset < 32; ++offset) {{
      const int64_t k = group * 32 + offset;
      const uint8_t a_byte = a_packed[m * (K / 2) + (k >> 1)];
      const uint8_t b_byte = b_packed[n * (K / 2) + (k >> 1)];
      const uint8_t ac = (k & 1) ? (a_byte >> 4) : (a_byte & 15u);
      const uint8_t bc = (k & 1) ? (b_byte >> 4) : (b_byte & 15u);
      partial = fmaf(tessera_e2m1(ac), tessera_e2m1(bc), partial);
    }}
    const int exponent = (int)a_scale[m * (K / 32) + group]
                       + (int)b_scale[n * (K / 32) + group] - 254;
    accum += ldexpf(partial, exponent);
  }}
  output[m * N + n] = tessera_bf16_rne(accum);
}}
"""


def package_quark_w4a4_probe(
    m: int,
    n: int,
    k: int,
    *,
    arch: str = "gfx1201",
    entry: str = "tessera_quark_w4a4_probe",
) -> ROCMNativePackage:
    """Compile only the exact-device scalar probe; never select it implicitly."""
    arch = _probe_arch(arch)
    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("Quark W4A4 requires positive M/N and K divisible by 32")
    source = emit_quark_w4a4_probe_hip(entry=entry)
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("Quark W4A4 probe requires the HIP compiler driver")
    device_libraries = _driver_selected_device_libraries(arch=arch)
    with tempfile.TemporaryDirectory(prefix="tessera-quark-w4a4-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler),
            "-x",
            "hip",
            "-O3",
            "--genco",
            f"--offload-arch={arch}",
            f"--rocm-path={rocm_path}",
            str(source_path),
            "-o",
            str(bundle_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle_path.is_file():
            detail = result.stderr.strip() or f"AMD clang exited {result.returncode}"
            raise RuntimeError(f"Quark W4A4 HSACO compilation failed: {detail}")
        payload = _extract_hsaco(bundle_path, image_path, rocm_path, arch)
    image = NativeImageArtifact(
        target=f"rocm_{arch}",
        architecture=arch,
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=hashlib.sha256(
            (str(rocm_path) + f"|{arch}|O3|quark_w4a4_exact_probe").encode()
        ).hexdigest(),
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX1201_QUARK_W4A4_PROBE_ABI),),
        compile_state="cold",
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=GFX1201_QUARK_W4A4_PROBE_ABI,
        buffers=(
            BufferBinding(0, "a_packed", "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, "b_packed", "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, "a_scale", "input", "uint8", 2, "row_major", 1),
            BufferBinding(3, "b_scale", "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, "output", "output", "bf16", 2, "row_major", 2),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard("a_packed", 0, "eq", m),
            ShapeGuard("a_packed", 1, "eq", k // 2),
            ShapeGuard("b_packed", 0, "eq", n),
            ShapeGuard("b_packed", 1, "eq", k // 2),
            ShapeGuard("a_scale", 0, "eq", m),
            ShapeGuard("a_scale", 1, "eq", k // 32),
            ShapeGuard("b_scale", 0, "eq", n),
            ShapeGuard("b_scale", 1, "eq", k // 32),
            ShapeGuard("output", 0, "eq", m),
            ShapeGuard("output", 1, "eq", n),
        ),
        geometry=LaunchGeometry(grid=((n + 15) // 16, (m + 15) // 16, 1), workgroup=(16, 16, 1)),
        ordering=OrderingSemantics(True, "none", ("completion",)),
        provenance={
            "work_item": "ROCM-MXFP4-W4A8-1",
            "sync_key": QUARK_W4A4_SYNC_KEY,
            "route": "manual_quark_w4a4_scalar_probe",
            # ``pipeline_name`` must name a registered pipeline and every
            # hipcc-built MXFP4 package reuses tessera-lower-to-rocm; this
            # kernel is hand-emitted HIP, not a Tile IR lowering.
            "lowering": "hand_emitted_hip_hipcc",
            "activation_layout": "e2m1_row_major_low_even_k_v1",
            "weight_layout": "quark_sampled_row_major_low_even_k_v1",
            "activation_scale": "e8m0_m_k32",
            "weight_scale": "e8m0_n_k32",
            "scale_edge_policy": "refuse_0_255",
            "accum": "fp32_per_k32_then_scale",
            "output": "bf16",
        },
    )
    semantic_ir = f"rocm.quark_w4a4_probe arch={arch} M={m} N={n} K={k} A=e2m1_low_even B=e2m1_low_even scales=e8m0_k32 output=bf16"
    return ROCMNativePackage(semantic_ir, source, " ".join(command[:-2]), image, descriptor)


def launch_quark_w4a4_probe(buffers: dict[str, np.ndarray]) -> dict[str, object]:
    """Guard raw planes before the opt-in host-array launch."""
    from tessera import runtime as rt

    m, n, k = validate_quark_w4a4_buffers(buffers)
    live = rt._rocm_live_arch()
    if live not in QUARK_W4A4_PROBE_ARCHS:
        raise RuntimeError(
            f"Quark W4A4 probe requires an exact gfx1151 or gfx1201 device; live is {live!r}"
        )
    package = package_quark_w4a4_probe(m, n, k, arch=live)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(artifact, {"buffers": buffers, "scalars": {"M": m, "N": n, "K": k}})
    result["hsaco_sha256"] = hashlib.sha256(package.image.payload).hexdigest()
    result["probe_arch"] = live
    return result


def submit_quark_w4a4_probe(
    image: NativeImageArtifact,
    descriptor: LaunchDescriptor,
    buffers: Mapping[str, Any],
    scalars: Mapping[str, object],
) -> np.ndarray:
    """Runtime submission for this ABI only, with an independent byte guard."""
    from tessera import runtime as rt

    arch = image.architecture
    if arch not in QUARK_W4A4_PROBE_ARCHS or image.target != f"rocm_{arch}":
        raise ValueError("Quark W4A4 probe requires an exact gfx1151 or gfx1201 image")
    live = rt._rocm_live_arch()
    if live != arch:
        raise RuntimeError(f"Quark W4A4 {arch} image cannot launch on live device {live!r}")
    if descriptor.abi_id != GFX1201_QUARK_W4A4_PROBE_ABI:
        raise ValueError("Quark W4A4 probe ABI mismatch")
    ordered = sorted(descriptor.buffers, key=lambda item: item.ordinal)
    expected_names = ("a_packed", "b_packed", "a_scale", "b_scale", "output")
    if tuple(item.name for item in ordered) != expected_names:
        raise ValueError("Quark W4A4 descriptor buffer order mismatch")
    arrays = {name: np.asarray(buffers[name]) for name in expected_names}
    m, n, k = validate_quark_w4a4_buffers(arrays)
    if (m, n, k) != tuple(cast(int, scalars[name]) for name in ("M", "N", "K")):
        raise ValueError("Quark W4A4 descriptor scalar shape mismatch")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError(f"Quark W4A4 HIP runtime or {arch} device is unavailable")
    module = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), image.payload) != 0:
        raise RuntimeError("Quark W4A4 HSACO module load failed")
    device = [ctypes.c_void_p() for _ in ordered]
    try:
        function = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(function), module, descriptor.entry_symbol.encode()) != 0:
            raise RuntimeError("Quark W4A4 kernel symbol not found")
        for pointer, binding in zip(device, ordered, strict=True):
            array = arrays[binding.name]
            if hip.hipMalloc(ctypes.byref(pointer), int(array.nbytes)) != 0:
                raise RuntimeError("Quark W4A4 hipMalloc failed")
        for pointer, binding in zip(device[:4], ordered[:4], strict=True):
            array = arrays[binding.name]
            if hip.hipMemcpy(pointer, array.ctypes.data_as(ctypes.c_void_p), int(array.nbytes), 1) != 0:
                raise RuntimeError("Quark W4A4 host-to-device copy failed")
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in device),
            ctypes.c_int64(m),
            ctypes.c_int64(n),
            ctypes.c_int64(k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        if descriptor.geometry.grid is None or descriptor.geometry.workgroup is None:
            raise RuntimeError("Quark W4A4 requires fixed launch geometry")
        gx, gy, gz = descriptor.geometry.grid
        wx, wy, wz = descriptor.geometry.workgroup
        rc = hip.hipModuleLaunchKernel(function, gx, gy, gz, wx, wy, wz, 0, None, arguments, None)
        if rc != 0:
            raise RuntimeError(f"Quark W4A4 kernel launch failed rc={rc}")
        sync = hip.hipDeviceSynchronize()
        if sync != 0:
            raise RuntimeError(f"Quark W4A4 kernel execution failed rc={sync}")
        output = arrays["output"]
        if hip.hipMemcpy(output.ctypes.data_as(ctypes.c_void_p), device[4], int(output.nbytes), 2) != 0:
            raise RuntimeError("Quark W4A4 device-to-host copy failed")
        return output
    finally:
        for pointer in reversed(device):
            if pointer.value:
                hip.hipFree(pointer)
        unload = getattr(hip, "hipModuleUnload", None)
        if unload is not None and module.value:
            unload(module)
