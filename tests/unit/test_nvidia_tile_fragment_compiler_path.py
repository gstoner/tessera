"""Execute the canonical Tile fragment path produced by the C++ NVIDIA passes.

This is intentionally different from the shipped handwritten/NVRTC GEMM lane:
the test lowers the layout-bearing Tile fixture to NVVM, translates it through
LLVM to PTX/cubin, and launches that exact compiler artifact with the CUDA
Driver API.  It is the numerical oracle for the sm_120 m16n8k16 lane mapping
and f32 accumulator unpack/store contract.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
           / "sm120_pointer_fragment_store.mlir")
KERNEL_FIXTURE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
                  / "sm120_matmul_kernel.mlir")
NVIDIA_OPT = (ROOT / "build/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools"
              / "tessera-nvidia-opt")


def _tool(path: str) -> str | None:
    return path if Path(path).is_file() else shutil.which(Path(path).name)


def _run(command: list[str], *, stdin: Path | None = None,
         stdout: Path | None = None) -> None:
    with (stdin.open("rb") if stdin else open(os.devnull, "rb")) as src, \
            (stdout.open("wb") if stdout else open(os.devnull, "wb")) as dst:
        subprocess.run(command, stdin=src, stdout=dst, check=True)


def _compile_cubin(work: Path, fixture: Path = FIXTURE,
                   entries: tuple[str, ...] = ("pointer_fragment_store",)) -> Path:
    tools = {
        "opt": str(NVIDIA_OPT) if NVIDIA_OPT.is_file() else None,
        "mlir-opt": _tool("/usr/lib/llvm-22/bin/mlir-opt"),
        "mlir-translate": _tool("/usr/lib/llvm-22/bin/mlir-translate"),
        "llc": _tool("/usr/lib/llvm-22/bin/llc"),
        "ptxas": _tool("/usr/local/cuda/bin/ptxas"),
    }
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        pytest.skip(f"missing compiler-path tools: {', '.join(missing)}")

    lowered = work / "tile-kernel.mlir"
    llvm_mlir = work / "tile-kernel-llvm.mlir"
    llvm_ir = work / "tile-kernel.ll"
    ptx = work / "tile-kernel.ptx"
    cubin = work / "tile-kernel.cubin"
    _run([tools["opt"], "--lower-tile-to-nvidia=sm=120",
          "--lower-tessera-nvidia-to-nvvm", str(fixture)], stdout=lowered)
    _run([tools["mlir-opt"], "--convert-scf-to-cf", "--convert-arith-to-llvm",
          "--convert-cf-to-llvm",
          "--reconcile-unrealized-casts"], stdin=lowered, stdout=llvm_mlir)
    _run([tools["mlir-translate"], "--mlir-to-llvmir"],
         stdin=llvm_mlir, stdout=llvm_ir)
    _run([tools["llc"], "-mtriple=nvptx64-nvidia-cuda", "-mcpu=sm_120",
          "-O3"], stdin=llvm_ir, stdout=ptx)
    ptx_text = ptx.read_text()
    for entry in entries:
        assert f".visible .entry {entry}" in ptx_text
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx_text
    _run([tools["ptxas"], "-arch=sm_120", str(ptx), "-o", str(cubin)])
    return cubin


class _CudaDriver:
    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError:
            pytest.skip("CUDA driver library is unavailable")
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(major), 75, device),
                    "cuDeviceGetAttribute(major)")
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, device),
                    "cuDeviceGetAttribute(minor)")
        if (major.value, minor.value) != (12, 0):
            pytest.skip(f"requires sm_120, found sm_{major.value}{minor.value}")
        self.context = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, device),
                    "cuCtxCreate")

    def _bind(self) -> None:
        void_p = ctypes.c_void_p
        u64 = ctypes.c_uint64
        size = ctypes.c_size_t
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuGetErrorString.argtypes = [ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                  ctypes.c_int, ctypes.c_int]
        self.lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint,
                                            ctypes.c_int]
        self.lib.cuModuleLoad.argtypes = [ctypes.POINTER(void_p), ctypes.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(void_p), void_p,
                                                 ctypes.c_char_p]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), size]
        self.lib.cuMemcpyHtoD_v2.argtypes = [u64, void_p, size]
        self.lib.cuMemcpyDtoH_v2.argtypes = [void_p, u64, size]
        self.lib.cuLaunchKernel.argtypes = ([void_p] + [ctypes.c_uint] * 6
                                            + [ctypes.c_uint, void_p,
                                               ctypes.POINTER(void_p),
                                               ctypes.POINTER(void_p)])
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuMemFree_v2.argtypes = [u64]
        self.lib.cuModuleUnload.argtypes = [void_p]
        self.lib.cuCtxDestroy_v2.argtypes = [void_p]

    def _check(self, result: int, operation: str) -> None:
        if result:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorString(result, ctypes.byref(name))
            detail = name.value.decode() if name.value else f"CUDA error {result}"
            raise RuntimeError(f"{operation}: {detail}")

    def execute(self, cubin: Path, a: np.ndarray, b_col: np.ndarray) -> np.ndarray:
        out = np.zeros((16, 8), dtype=np.float32)
        return self.launch(cubin, "pointer_fragment_store", [a, b_col, out], 2,
                           [0], (1, 1))

    def launch(self, cubin: Path, entry: str, buffers: list[np.ndarray],
               output_index: int, scalar_args: list[int],
               grid: tuple[int, int], block_threads: int = 32) -> np.ndarray:
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        allocations: list[ctypes.c_uint64] = []
        try:
            self._check(self.lib.cuModuleLoad(ctypes.byref(module),
                                              os.fsencode(cubin)), "cuModuleLoad")
            self._check(self.lib.cuModuleGetFunction(
                ctypes.byref(function), module, entry.encode()),
                "cuModuleGetFunction")
            for host in buffers:
                ptr = ctypes.c_uint64()
                self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), host.nbytes),
                            "cuMemAlloc")
                allocations.append(ptr)
                self._check(self.lib.cuMemcpyHtoD_v2(
                    ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes),
                    "cuMemcpyHtoD")
            scalars = [ctypes.c_int64(value) for value in scalar_args]
            storage = [*allocations, *scalars]
            params = (ctypes.c_void_p * len(storage))(*[
                ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
                for value in storage])
            self._check(self.lib.cuLaunchKernel(
                function, grid[0], grid[1], 1, block_threads, 1, 1,
                0, None, params, None),
                "cuLaunchKernel")
            self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
            out = buffers[output_index]
            self._check(self.lib.cuMemcpyDtoH_v2(
                out.ctypes.data_as(ctypes.c_void_p), allocations[output_index],
                out.nbytes),
                "cuMemcpyDtoH")
            return out
        finally:
            for ptr in reversed(allocations):
                self.lib.cuMemFree_v2(ptr)
            if module:
                self.lib.cuModuleUnload(module)

    def benchmark(self, cubin: Path, entry: str, buffers: list[np.ndarray],
                  scalar_args: list[int], grid: tuple[int, int],
                  block_threads: int, iterations: int = 200) -> float:
        """Return steady-state kernel launch time in milliseconds.

        Allocations, module loading, and host transfers are outside the timed
        region so direct-global and shared/multi-warp materializations can be
        compared without Python/CUDA setup noise.
        """
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        allocations: list[ctypes.c_uint64] = []
        try:
            self._check(self.lib.cuModuleLoad(ctypes.byref(module), os.fsencode(cubin)),
                        "cuModuleLoad")
            self._check(self.lib.cuModuleGetFunction(
                ctypes.byref(function), module, entry.encode()),
                "cuModuleGetFunction")
            for host in buffers:
                ptr = ctypes.c_uint64()
                self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), host.nbytes),
                            "cuMemAlloc")
                allocations.append(ptr)
                self._check(self.lib.cuMemcpyHtoD_v2(
                    ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes),
                    "cuMemcpyHtoD")
            scalars = [ctypes.c_int64(value) for value in scalar_args]
            storage = [*allocations, *scalars]
            params = (ctypes.c_void_p * len(storage))(*[
                ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
                for value in storage])

            def launch_once() -> None:
                self._check(self.lib.cuLaunchKernel(
                    function, grid[0], grid[1], 1, block_threads, 1, 1,
                    0, None, params, None), "cuLaunchKernel")

            for _ in range(20):
                launch_once()
            self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
            start = time.perf_counter()
            for _ in range(iterations):
                launch_once()
            self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
            return (time.perf_counter() - start) * 1e3 / iterations
        finally:
            for ptr in reversed(allocations):
                self.lib.cuMemFree_v2(ptr)
            if module:
                self.lib.cuModuleUnload(module)

    def close(self) -> None:
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None


def test_sm120_tile_fragment_compiler_path_matches_numpy() -> None:
    rng = np.random.default_rng(20260714)
    a = np.ascontiguousarray(rng.uniform(-1, 1, (16, 16)), dtype=np.float16)
    b = np.asfortranarray(rng.uniform(-1, 1, (16, 8)), dtype=np.float16)
    reference = a.astype(np.float32) @ b.astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="tessera-sm120-") as tmp:
        cubin = _compile_cubin(Path(tmp))
        driver = _CudaDriver()
        try:
            actual = driver.execute(cubin, a, b)
        finally:
            driver.close()
    max_error = float(np.max(np.abs(actual - reference)))
    assert max_error < 1e-2, f"compiler-generated Tile GEMM max_error={max_error}"


def test_sm120_launch_level_matmul_handles_grid_and_ragged_k() -> None:
    entries = ("tile_matmul_f32", "tile_matmul_bias_relu_f16",
               "tile_matmul_bf16", "tile_matmul_f32_direct")
    shapes = ((16, 8, 16), (31, 13, 19), (33, 17, 37))
    rng = np.random.default_rng(20260715)
    with tempfile.TemporaryDirectory(prefix="tessera-sm120-grid-") as tmp:
        cubin = _compile_cubin(Path(tmp), KERNEL_FIXTURE, entries)
        driver = _CudaDriver()
        try:
            for m, n, k in shapes:
                a = np.ascontiguousarray(
                    rng.uniform(-1, 1, (m, k)), dtype=np.float16)
                b = np.asfortranarray(
                    rng.uniform(-1, 1, (k, n)), dtype=np.float16)
                out = np.zeros((m, n), dtype=np.float32)
                actual = driver.launch(
                    cubin, entries[0], [a, b, out], 2, [m, n, k],
                    ((n + 31) // 32, (m + 31) // 32), block_threads=128)
                reference = a.astype(np.float32) @ b.astype(np.float32)
                np.testing.assert_allclose(actual, reference, atol=1e-2, rtol=0)

                direct_out = np.zeros((m, n), dtype=np.float32)
                direct = driver.launch(
                    cubin, entries[3], [a, b, direct_out], 2, [m, n, k],
                    ((n + 7) // 8, (m + 15) // 16), block_threads=32)
                np.testing.assert_allclose(direct, reference, atol=1e-2, rtol=0)

                bias = np.ascontiguousarray(
                    rng.uniform(-.5, .5, (n,)), dtype=np.float32)
                out_f16 = np.zeros((m, n), dtype=np.float16)
                fused = driver.launch(
                    cubin, entries[1], [a, b, bias, out_f16], 3, [m, n, k],
                    ((n + 31) // 32, (m + 31) // 32), block_threads=128)
                fused_reference = np.maximum(reference + bias, 0).astype(np.float16)
                np.testing.assert_allclose(fused, fused_reference, atol=1e-2,
                                           rtol=0)

            ml_dtypes = pytest.importorskip("ml_dtypes")
            m, n, k = 35, 21, 29
            a = np.ascontiguousarray(
                rng.uniform(-1, 1, (m, k)).astype(ml_dtypes.bfloat16))
            b = np.asfortranarray(
                rng.uniform(-1, 1, (k, n)).astype(ml_dtypes.bfloat16))
            out = np.zeros((m, n), dtype=np.float32)
            actual = driver.launch(
                cubin, entries[2], [a, b, out], 2, [m, n, k],
                ((n + 31) // 32, (m + 31) // 32), block_threads=128)
            reference = a.astype(np.float32) @ b.astype(np.float32)
            np.testing.assert_allclose(actual, reference, atol=1e-1, rtol=0)
        finally:
            driver.close()
