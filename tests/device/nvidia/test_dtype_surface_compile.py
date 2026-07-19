"""CUDA 13.3 compile proof for the SM120 scalar/vector dtype surface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tessera.compiler.nvidia_native import (
    _cuda_libdevice,
    _link_cuda_device_library_if_needed,
)
from tests._support.nvidia import nvidia_cuda_host_ready


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/device/nvidia/fixtures/sm120_dtype_surface.cu"
INTRINSIC_FIXTURE = ROOT / "tests/device/nvidia/fixtures/sm120_intrinsic_surface.cu"
F64_MMA_FIXTURE = ROOT / (
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/"
    "sm120_f64_mma_kernel.mlir"
)
MX_MMA_FIXTURE = ROOT / (
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/"
    "sm120_mx_block_scale_mma_kernel.mlir"
)
NVIDIA_OPT = ROOT / (
    "build-nvidia-cuda/src/compiler/codegen/tessera_gpu_backend_NVIDIA/"
    "tools/tessera-nvidia-opt"
)


@pytest.mark.hardware_nvidia
def test_sm120_scalar_vector_dtype_surface_compiles(tmp_path: Path) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    result = subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc", "-std=c++17", "-arch=sm_120a",
            "-c", str(FIXTURE), "-o", str(tmp_path / "dtype-surface.o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.hardware_nvidia
def test_sm120_integer_cast_and_simd_intrinsic_surface_compiles(
    tmp_path: Path,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    result = subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc", "-std=c++17", "-arch=sm_120a",
            "-c", str(INTRINSIC_FIXTURE),
            "-o", str(tmp_path / "intrinsic-surface.o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _pipe(command: list[str], source: bytes) -> bytes:
    result = subprocess.run(command, input=source, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def _assemble_ptx(source: str, tmp_path: Path, stem: str) -> subprocess.CompletedProcess[str]:
    ptx_path = tmp_path / f"{stem}.ptx"
    ptx_path.write_text(source)
    return subprocess.run(
        [
            "/usr/local/cuda/bin/ptxas", "-arch=sm_120a", str(ptx_path),
            "-o", str(tmp_path / f"{stem}.cubin"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.hardware_nvidia
def test_sm120_ptx_fundamental_register_surface_assembles(tmp_path: Path) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    source = r'''
.version 9.0
.target sm_120a
.address_size 64
.visible .entry tessera_ptx_fundamental_types() {
  .reg .pred %p;
  .reg .b8 %b8;
  .reg .s8 %s8;
  .reg .b16 %b16;
  .reg .f16 %f16;
  .reg .f16x2 %f16x2;
  .reg .b32 %b32;
  .reg .f32 %f32;
  .reg .b64 %b64;
  .reg .f64 %f64;
  ret;
}
'''
    result = _assemble_ptx(source, tmp_path, "fundamental-types")
    assert result.returncode == 0, result.stderr


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("non_fundamental", ["bf16", "tf32", "e4m3", "u8x4"])
def test_sm120_ptx_alternate_and_packed_formats_reject_register_declaration(
    tmp_path: Path, non_fundamental: str,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    source = f'''
.version 9.0
.target sm_120a
.address_size 64
.visible .entry tessera_bad_ptx_type() {{
  .reg .{non_fundamental} %value;
  ret;
}}
'''
    result = _assemble_ptx(source, tmp_path, f"bad-{non_fundamental}")
    assert result.returncode != 0


@pytest.mark.hardware_nvidia
def test_sm120_fp64_tensor_core_target_ir_assembles(tmp_path: Path) -> None:
    if not nvidia_cuda_host_ready() or not NVIDIA_OPT.is_file():
        pytest.skip("host WSL CUDA device/compiler toolchain unavailable")
    nvvm = _pipe(
        [str(NVIDIA_OPT), "--lower-tessera-nvidia-to-nvvm"],
        F64_MMA_FIXTURE.read_bytes(),
    )
    llvm_ir = _pipe(["/usr/lib/llvm-23/bin/mlir-translate", "--mlir-to-llvmir"], nvvm)
    ptx = _pipe([
        "/usr/lib/llvm-23/bin/llc", "-mtriple=nvptx64-nvidia-cuda",
        "-mcpu=sm_120a", "-O3",
    ], llvm_ir)
    assert b"mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64" in ptx
    ptx_path = tmp_path / "sm120-f64.ptx"
    ptx_path.write_bytes(ptx)
    assembled = subprocess.run(
        [
            "/usr/local/cuda/bin/ptxas", "-arch=sm_120a", str(ptx_path),
            "-o", str(tmp_path / "sm120-f64.cubin"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assembled.returncode == 0, assembled.stderr


@pytest.mark.hardware_nvidia
def test_sm120_fp6_and_mxfp4_tensor_core_target_ir_assemble(
    tmp_path: Path,
) -> None:
    if not nvidia_cuda_host_ready() or not NVIDIA_OPT.is_file():
        pytest.skip("host WSL CUDA device/compiler toolchain unavailable")
    nvvm = _pipe(
        [str(NVIDIA_OPT), "--lower-tessera-nvidia-to-nvvm"],
        MX_MMA_FIXTURE.read_bytes(),
    )
    llvm_ir = _pipe(["/usr/lib/llvm-23/bin/mlir-translate", "--mlir-to-llvmir"], nvvm)
    ptx = _pipe([
        "/usr/lib/llvm-23/bin/llc", "-mtriple=nvptx64-nvidia-cuda",
        "-mcpu=sm_120a", "-O3",
    ], llvm_ir)
    for mnemonic in (
        b"kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m3.e2m3.f32.ue8m0",
        b"kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e3m2.f32.ue8m0",
        b"kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0",
    ):
        assert mnemonic in ptx
    ptx_path = tmp_path / "sm120-mx-low-precision.ptx"
    ptx_path.write_bytes(ptx)
    assembled = subprocess.run(
        [
            "/usr/local/cuda/bin/ptxas", "-arch=sm_120a", str(ptx_path),
            "-o", str(tmp_path / "sm120-mx-low-precision.cubin"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assembled.returncode == 0, assembled.stderr


@pytest.mark.hardware_nvidia
def test_sm120_llvm_stage_links_cuda_libdevice(tmp_path: Path) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    libdevice = _cuda_libdevice()
    if libdevice is None:
        pytest.skip("CUDA libdevice.10.bc unavailable")
    llvm_ir = b'''target triple = "nvptx64-nvidia-cuda"
declare float @__nv_sinf(float)
define void @tessera_libdevice_probe(ptr %out, float %x) {
entry:
  %value = call float @__nv_sinf(float %x)
  store float %value, ptr %out
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @tessera_libdevice_probe, !"kernel", i32 1}
'''
    linked, libraries = _link_cuda_device_library_if_needed(
        llvm_ir,
        llvm_link=Path("/usr/lib/llvm-23/bin/llvm-link"),
        libdevice=libdevice,
    )
    assert [library.logical_name for library in libraries] == ["cuda.libdevice"]
    assert libraries[0].link_mode == "llvm_link_only_needed"
    ptx = _pipe([
        "/usr/lib/llvm-23/bin/llc", "-mtriple=nvptx64-nvidia-cuda",
        "-mcpu=sm_120a", "-O3",
    ], linked)
    assert b".visible .entry tessera_libdevice_probe" in ptx
    ptx_path = tmp_path / "sm120-libdevice.ptx"
    ptx_path.write_bytes(ptx)
    assembled = subprocess.run(
        [
            "/usr/local/cuda/bin/ptxas", "-arch=sm_120a", str(ptx_path),
            "-o", str(tmp_path / "sm120-libdevice.cubin"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assembled.returncode == 0, assembled.stderr
