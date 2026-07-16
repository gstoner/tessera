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
from tests._support.nvidia_numerics import assert_matches


pytestmark = [pytest.mark.compiler_tool, pytest.mark.hardware_nvidia]


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
           / "sm120_pointer_fragment_store.mlir")
KERNEL_FIXTURE = (ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
                  / "sm120_matmul_kernel.mlir")
_NVIDIA_OPT_CANDIDATES = tuple(
    Path(value) for value in (os.environ.get("TESSERA_NVIDIA_OPT"),) if value
) + (
    ROOT / "build-nvidia-cuda/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools"
    / "tessera-nvidia-opt",
    ROOT / "build-nvidia/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools"
    / "tessera-nvidia-opt",
    ROOT / "build/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools"
    / "tessera-nvidia-opt",
)
NVIDIA_OPT = next((path for path in _NVIDIA_OPT_CANDIDATES if path.is_file()),
                  _NVIDIA_OPT_CANDIDATES[0])


def _tool(path: str) -> str | None:
    return path if Path(path).is_file() else shutil.which(Path(path).name)


def _run(command: list[str], *, stdin: Path | None = None,
         stdout: Path | None = None) -> None:
    with (stdin.open("rb") if stdin else open(os.devnull, "rb")) as src, \
            (stdout.open("wb") if stdout else open(os.devnull, "wb")) as dst:
        subprocess.run(command, stdin=src, stdout=dst, check=True)


def _compile_cubin(work: Path, fixture: Path = FIXTURE,
                   entries: tuple[str, ...] = ("pointer_fragment_store",),
                   expected_mma: str =
                   "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32") -> Path:
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
          "--convert-cf-to-llvm", "--convert-math-to-llvm",
          "--reconcile-unrealized-casts"], stdin=lowered, stdout=llvm_mlir)
    _run([tools["mlir-translate"], "--mlir-to-llvmir"],
         stdin=llvm_mlir, stdout=llvm_ir)
    _run([tools["llc"], "-mtriple=nvptx64-nvidia-cuda", "-mcpu=sm_120",
          "-O3"], stdin=llvm_ir, stdout=ptx)
    ptx_text = ptx.read_text()
    for entry in entries:
        assert f".visible .entry {entry}" in ptx_text
    assert expected_mma in ptx_text
    _run([tools["ptxas"], "-arch=sm_120", str(ptx), "-o", str(cubin)])
    return cubin


def _write_dtype_fixture(work: Path, *, entry: str, dtype: str, k: int,
                         accumulator: str) -> Path:
    """Create the same layout-bearing Tile kernel for a physical MMA dtype."""
    fixture = work / f"{entry}.mlir"
    text = f'''module {{
  llvm.func @{entry}(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
                     %d_ptr: !llvm.ptr, %zero: i64) attributes {{nvvm.kernel}} {{
    %a_tile = tile.view %a_ptr, %zero, %zero {{
      tile.layout = #tile.layout<shard = [16, {k}] : [{k}, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = {k}>
    }} : (!llvm.ptr, i64, i64) -> !tile.tile
    %b_tile = tile.view %b_ptr, %zero, %zero {{
      tile.layout = #tile.layout<shard = [{k}, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "col_major", leading_dim = {k}>
    }} : (!llvm.ptr, i64, i64) -> !tile.tile
    %a = tile.fragment_pack %a_tile {{role = "a", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {k}, a = "{dtype}", b = "{dtype}", acc = "{accumulator}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>}} : (!tile.tile) -> !tile.fragment
    %b = tile.fragment_pack %b_tile {{role = "b", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {k}, a = "{dtype}", b = "{dtype}", acc = "{accumulator}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>}} : (!tile.tile) -> !tile.fragment
    %c = tile.fragment_zero {{role = "acc", mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {k}, a = "{dtype}", b = "{dtype}", acc = "{accumulator}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>}} : !tile.fragment
    %d = tile.mma %a, %b, %c {{mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {k}, a = "{dtype}", b = "{dtype}", acc = "{accumulator}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>}} : (!tile.fragment, !tile.fragment, !tile.fragment) -> !tile.fragment
    %out = tile.fragment_unpack %d {{
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {k}, a = "{dtype}", b = "{dtype}", acc = "{accumulator}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>
    }} : (!tile.fragment) -> !tile.tile
    "tile.store"(%out, %d_ptr, %zero, %zero) {{
      tile.layout = #tile.layout<shard = [16, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 8>
    }} : (!tile.tile, !llvm.ptr, i64, i64) -> ()
    llvm.return
  }}
}}
'''
    fixture.write_text(text)
    return fixture


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


@pytest.mark.parametrize(
    "dtype,k,accumulator,storage,ptx_dtype,atol",
    [
        ("tf32", 8, "f32", "tf32", "tf32", 1e-2),
        ("e4m3", 32, "f32", "e4m3", "e4m3", 1.0),
        ("e5m2", 32, "f32", "e5m2", "e5m2", 2.0),
    ],
)
def test_sm120_tile_dtype_fragment_paths_match_numpy(
        dtype: str, k: int, accumulator: str, storage: str,
        ptx_dtype: str, atol: float) -> None:
    rng = np.random.default_rng(20260718 + k)
    a_source = rng.uniform(-1, 1, (16, k)).astype(np.float32)
    b_source = np.asfortranarray(rng.uniform(-1, 1, (k, 8)), dtype=np.float32)
    if storage == "tf32":
        a, b = a_source, b_source
        reference = a @ b
    else:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        fp8 = (ml_dtypes.float8_e4m3fn if storage == "e4m3"
               else ml_dtypes.float8_e5m2)
        a_quant = np.asarray(a_source, dtype=fp8)
        b_quant = np.asfortranarray(b_source, dtype=fp8)
        reference = a_quant.astype(np.float32) @ b_quant.astype(np.float32)
        a = a_quant.view(np.uint8)
        b = b_quant.view(np.uint8)
    out = np.zeros((16, 8), dtype=np.float32)
    entry = f"pointer_fragment_store_{dtype}"
    with tempfile.TemporaryDirectory(prefix=f"tessera-sm120-{dtype}-") as tmp:
        work = Path(tmp)
        fixture = _write_dtype_fixture(
            work, entry=entry, dtype=dtype, k=k, accumulator=accumulator)
        cubin = _compile_cubin(
            work, fixture, (entry,),
            f"mma.sync.aligned.m16n8k{k}.row.col.f32.{ptx_dtype}.{ptx_dtype}.f32")
        driver = _CudaDriver()
        try:
            actual = driver.launch(cubin, entry, [a, b, out], 2, [0], (1, 1))
        finally:
            driver.close()
    np.testing.assert_allclose(actual, reference, rtol=0, atol=atol)


def test_sm120_tile_int8_fragment_path_matches_numpy() -> None:
    rng = np.random.default_rng(20260719)
    a = rng.integers(-8, 9, size=(16, 32), dtype=np.int8)
    b = np.asfortranarray(rng.integers(-8, 9, size=(32, 8), dtype=np.int8))
    reference = a.astype(np.int32) @ b.astype(np.int32)
    out = np.zeros((16, 8), dtype=np.int32)
    entry = "pointer_fragment_store_s8"
    with tempfile.TemporaryDirectory(prefix="tessera-sm120-s8-") as tmp:
        work = Path(tmp)
        fixture = _write_dtype_fixture(
            work, entry=entry, dtype="s8", k=32, accumulator="s32")
        cubin = _compile_cubin(
            work, fixture, (entry,),
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
        driver = _CudaDriver()
        try:
            actual = driver.launch(cubin, entry, [a, b, out], 2, [0], (1, 1))
        finally:
            driver.close()
    np.testing.assert_array_equal(actual, reference)


def test_sm120_launch_level_matmul_handles_grid_and_ragged_k() -> None:
    entries = ("tile_matmul_f32", "tile_matmul_bias_relu_f16",
               "tile_matmul_bf16", "tile_matmul_f32_direct",
               "tile_matmul_gelu", "tile_matmul_silu")
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
                assert_matches(actual, reference, "f16", reduction_length=k)

                direct_out = np.zeros((m, n), dtype=np.float32)
                direct = driver.launch(
                    cubin, entries[3], [a, b, direct_out], 2, [m, n, k],
                    ((n + 7) // 8, (m + 15) // 16), block_threads=32)
                assert_matches(direct, reference, "f16", reduction_length=k)

                bias = np.ascontiguousarray(
                    rng.uniform(-.5, .5, (n,)), dtype=np.float32)
                out_f16 = np.zeros((m, n), dtype=np.float16)
                fused = driver.launch(
                    cubin, entries[1], [a, b, bias, out_f16], 3, [m, n, k],
                    ((n + 31) // 32, (m + 31) // 32), block_threads=128)
                fused_reference = np.maximum(reference + bias, 0).astype(np.float16)
                assert_matches(fused, fused_reference, "f16", reduction_length=k)

                gelu_out = np.zeros((m, n), dtype=np.float32)
                gelu = driver.launch(
                    cubin, entries[4], [a, b, gelu_out], 2, [m, n, k],
                    ((n + 7) // 8, (m + 15) // 16), block_threads=32)
                gelu_reference = (0.5 * reference * (1 + np.tanh(
                    np.sqrt(2 / np.pi) *
                    (reference + 0.044715 * reference ** 3))))
                assert_matches(gelu, gelu_reference, "f16", reduction_length=k)

                silu_out = np.zeros((m, n), dtype=np.float32)
                silu = driver.launch(
                    cubin, entries[5], [a, b, silu_out], 2, [m, n, k],
                    ((n + 7) // 8, (m + 15) // 16), block_threads=32)
                silu_reference = reference / (1 + np.exp(-reference))
                assert_matches(silu, silu_reference, "f16", reduction_length=k)

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
            assert_matches(actual, reference, "bf16", reduction_length=k)
        finally:
            driver.close()
