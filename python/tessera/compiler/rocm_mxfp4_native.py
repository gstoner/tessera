"""Exact gfx1201 MXFP4 W4A8 native package.

This correctness-first kernel is the executable specification for the later
WMMA schedule.  It keeps the OCP K=32 scale boundary explicit on device and
therefore never uses the lossy row-reference fold.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

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
from .rocm_native import (
    ROCMNativePackage,
    _driver_selected_device_libraries,
    _rocm_path,
    _version_fingerprint,
)


GFX_MXFP4_W4A8_EXACT_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.exact.v1"
)


def _rocm_hipcc(rocm_path: Path) -> Path | None:
    """Return the HIP driver that owns ``--genco`` code-object emission.

    ``amdclang++`` is still used by :func:`_driver_selected_device_libraries`
    to fingerprint the exact OCML/OCKL/OCLC selection, but it is not a drop-in
    replacement for the HIP wrapper on split ROCm installations: the raw
    driver on Tajasarus deliberately rejects the wrapper-only ``--genco``
    option.
    """

    configured = os.environ.get("TESSERA_ROCM_HIPCC")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    candidates = [rocm_path / "bin" / "hipcc"]
    if rocm_path.name == "core":
        candidates.append(rocm_path.parent / "bin" / "hipcc")
    candidates.append(Path("/opt/rocm/bin/hipcc"))
    found = shutil.which("hipcc")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def _rocm_offload_bundler(rocm_path: Path) -> Path | None:
    candidates = [
        rocm_path / "lib" / "llvm" / "bin" / "clang-offload-bundler",
        rocm_path / "llvm" / "bin" / "clang-offload-bundler",
        rocm_path / "bin" / "clang-offload-bundler",
    ]
    found = shutil.which("clang-offload-bundler")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def _extract_gfx1201_hsaco(compiled: Path, output: Path, rocm_path: Path) -> bytes:
    """Return a raw HSACO from raw or LLVM 23 bundled HIP output."""

    payload = compiled.read_bytes()
    if payload.startswith(b"\x7fELF"):
        return payload
    if not payload.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        raise RuntimeError("MXFP4 W4A8 compiler output is neither ELF nor a HIP bundle")
    bundler = _rocm_offload_bundler(rocm_path)
    if bundler is None:
        raise RuntimeError("MXFP4 W4A8 HIP bundle requires clang-offload-bundler")
    result = subprocess.run(
        [
            str(bundler), "-unbundle", "-type=o",
            "-targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
            f"-input={compiled}", f"-output={output}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not output.is_file():
        detail = result.stderr.strip() or f"clang-offload-bundler exited {result.returncode}"
        raise RuntimeError(f"MXFP4 W4A8 HSACO extraction failed: {detail}")
    payload = output.read_bytes()
    if not payload.startswith(b"\x7fELF"):
        raise RuntimeError("MXFP4 W4A8 extracted device image is not an ELF HSACO")
    return payload


def emit_mxfp4_w4a8_exact_hip(*, entry: str = "tessera_mxfp4_w4a8_exact") -> str:
    """Emit the independent scalar baseline for the exact per-group route."""

    if not entry or not entry.replace("_", "a").isalnum():
        raise ValueError("MXFP4 entry must be a C identifier")
    return f'''#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float tessera_e2m1(uint8_t code) {{
  const float table[8] = {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}};
  float value = table[code & 7u];
  return (code & 8u) ? -value : value;
}}

__device__ __forceinline__ float tessera_e4m3fn(uint8_t code) {{
  const unsigned sign = code >> 7;
  const unsigned exponent = (code >> 3) & 15u;
  const unsigned mantissa = code & 7u;
  float value;
  if (exponent == 0u)
    value = ldexpf((float)mantissa, -9);
  else if (exponent == 15u && mantissa == 7u)
    value = __uint_as_float(0x7fc00000u);
  else
    value = ldexpf(1.0f + (float)mantissa * 0.125f,
                   (int)exponent - 7);
  return sign ? -value : value;
}}

__device__ __forceinline__ uint16_t tessera_bf16_rne(float value) {{
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}}

extern "C" __global__ void {entry}(
    const uint8_t *__restrict__ a_e4m3,
    const uint8_t *__restrict__ b_e2m1_packed,
    const float *__restrict__ a_scale,
    const uint8_t *__restrict__ b_scale_e8m0,
    uint16_t *__restrict__ output_bf16,
    int64_t M, int64_t N, int64_t K) {{
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  float accum = 0.0f;
  const float row_scale = a_scale[m];
  for (int64_t group = 0; group < K / 32; ++group) {{
    float partial = 0.0f;
    const int64_t k0 = group * 32;
    for (int64_t offset = 0; offset < 32; ++offset) {{
      const int64_t k = k0 + offset;
      const uint8_t packed = b_e2m1_packed[(k >> 1) * N + n];
      const uint8_t weight = (k & 1) ? (packed >> 4) : (packed & 15u);
      partial = fmaf(tessera_e4m3fn(a_e4m3[m * K + k]),
                     tessera_e2m1(weight), partial);
    }}
    const uint8_t exponent = b_scale_e8m0[group * N + n];
    const float scale = exponent == 0u
        ? 0.0f : ldexpf(row_scale, (int)exponent - 127);
    accum = fmaf(partial, scale, accum);
  }}
  output_bf16[m * N + n] = tessera_bf16_rne(accum);
}}
'''


def mxfp4_w4a8_descriptor(
    image: NativeImageArtifact,
    *,
    m: int,
    n: int,
    k: int,
    entry: str = "tessera_mxfp4_w4a8_exact",
) -> LaunchDescriptor:
    """Build the exact five-buffer W4A8 launch contract."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    if image.target != "rocm_gfx1201" or image.architecture != "gfx1201":
        raise ValueError("MXFP4 W4A8 native packages are exact gfx1201 artifacts")
    return LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=GFX_MXFP4_W4A8_EXACT_ABI,
        buffers=(
            BufferBinding(0, "a", "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, "b_packed", "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, "a_scale", "input", "fp32", 1, "row_major", 4),
            BufferBinding(3, "b_scale", "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, "output", "output", "bf16", 2, "row_major", 2),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard("a", 0, "eq", m), ShapeGuard("a", 1, "eq", k),
            ShapeGuard("b_packed", 0, "eq", k // 2), ShapeGuard("b_packed", 1, "eq", n),
            ShapeGuard("a_scale", 0, "eq", m),
            ShapeGuard("b_scale", 0, "eq", k // 32), ShapeGuard("b_scale", 1, "eq", n),
            ShapeGuard("output", 0, "eq", m), ShapeGuard("output", 1, "eq", n),
        ),
        geometry=LaunchGeometry(
            grid=((n + 15) // 16, (m + 15) // 16, 1),
            workgroup=(16, 16, 1),
        ),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-MXFP4-W4A8-1",
            "sync_key": "ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21",
            "route": "exact_per_block_scalar_baseline",
            "architecture": "gfx1201",
            "shape": [m, n, k],
            "activation_storage": "e4m3_raw_uint8",
            "weight_storage": "e2m1_packed_low_nibble_even_k",
            "activation_scale": "fp32_per_token",
            "weight_scale": "e8m0_k32_kn_major",
            "scale_group_k": 32,
            "accum": "fp32",
            "output": "bf16",
            "numeric_policy": "exact_per_block",
        },
    )


def package_mxfp4_w4a8_exact(
    m: int,
    n: int,
    k: int,
    *,
    pipeline_name: str = "tessera-lower-to-rocm",
    entry: str = "tessera_mxfp4_w4a8_exact",
) -> ROCMNativePackage:
    """Compile the exact scalar gfx1201 baseline to one HSACO package."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    source = emit_mxfp4_w4a8_exact_hip(entry=entry)
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("MXFP4 W4A8 packaging requires the HIP compiler driver")
    device_libraries = _driver_selected_device_libraries(arch="gfx1201")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler), "-x", "hip", "-O3", "--genco",
            "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
            str(source_path), "-o", str(bundle_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle_path.is_file():
            detail = result.stderr.strip() or f"AMD clang exited {result.returncode}"
            raise RuntimeError(f"MXFP4 W4A8 HSACO compilation failed: {detail}")
        payload = _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)
    toolchain_fingerprint = hashlib.sha256(
        (str(rocm_path) + "|gfx1201|O3|exact_per_block").encode()
    ).hexdigest()
    image = NativeImageArtifact(
        target="rocm_gfx1201",
        architecture="gfx1201",
        pipeline_name=pipeline_name,
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=toolchain_fingerprint,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX_MXFP4_W4A8_EXACT_ABI),),
        compile_state="cold",
        device_libraries=device_libraries,
    )
    descriptor = mxfp4_w4a8_descriptor(image, m=m, n=n, k=k, entry=entry)
    semantic_ir = (
        f"rocm.mxfp4_w4a8 exact_per_block M={m} N={n} K={k} "
        "activation=e4m3 scale_a=fp32_per_token weight=e2m1_packed "
        "scale_b=e8m0_k32 accum=fp32 output=bf16"
    )
    return ROCMNativePackage(semantic_ir, source, " ".join(command[:-2]), image, descriptor)


__all__ = [
    "GFX_MXFP4_W4A8_EXACT_ABI",
    "emit_mxfp4_w4a8_exact_hip",
    "mxfp4_w4a8_descriptor",
    "package_mxfp4_w4a8_exact",
]
