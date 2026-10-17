"""Native BF16 arithmetic and compiler-free generic tensor ABI replay on SM120."""
import inspect
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage, build_native_gpu_storage
from tessera.compiler.native_gpu_tensor import IndexSpec, TensorSpec
from tessera.compiler.native_storage_contract import attach_tensor_contract, generate_tensor_binding
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]

SOURCE = """module {
 gpu.module @bf16_tensor {
  gpu.func @bf16_add_one(%x: !llvm.ptr<1>, %y: !llvm.ptr<1>, %scratch: index) kernel {
   %buffer = memref.alloca(%scratch) : memref<?xf32>
   "tile.alloc_shared"(%buffer) : (memref<?xf32>) -> ()
   %i = gpu.thread_id x
   %offset = arith.index_cast %i : index to i64
   %px = llvm.getelementptr %x[%offset] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, bf16
   %a = llvm.load %px {alignment = 2 : i64} : !llvm.ptr<1> -> bf16
   %f = arith.extf %a : bf16 to f32
   memref.store %f, %buffer[%i] : memref<?xf32>
   gpu.barrier
   %saved = memref.load %buffer[%i] : memref<?xf32>
   %one = arith.constant 1.0 : f32
   %sum = arith.addf %saved, %one : f32
   %result = arith.truncf %sum : f32 to bf16
   %py = llvm.getelementptr %y[%offset] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, bf16
   llvm.store %result, %py {alignment = 2 : i64} : bf16, !llvm.ptr<1>
   gpu.return
  }
 }
}
"""


def test_native_bf16_arithmetic_replays_through_semantic_tensor_binding(compiler_toolchain, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    compiler = compiler_toolchain.require_tessera_opt("tessera-tile-buffer-arena")
    llvm_bin = Path(os.environ.get("TESSERA_LLVM_BIN", ""))
    if not (llvm_bin / "mlir-opt").is_file():
        pytest.skip("matched native LLVM/MLIR tools required")
    specs = (TensorSpec("x", "bf16", (32,), False), TensorSpec("y", "bf16", (32,), True),
             IndexSpec("scratch", 32, 32))
    source = attach_tensor_contract(SOURCE, specs, grid=(1, 1, 1), block=(32, 1, 1))
    toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    package = build_native_gpu_storage(source, compiler=compiler, llvm_bin=llvm_bin,
        backend="nvidia", chip="sm_120", toolkit=toolkit)
    from tessera.compiler import native_gpu_storage
    def forbidden(*args, **kwargs):
        pytest.fail("portable BF16 tensor replay invoked a compiler")
    monkeypatch.setattr(native_gpu_storage, "_run", forbidden)
    restored = NativeGPUStoragePackage.from_json(package.to_json(), expected_digest=package.binding_digest)
    call = generate_tensor_binding(restored, inspect.signature(lambda x, y, scratch: None))
    with NvidiaDeviceSession() as session:
        for seed in (20261017, 20261018):
            host = np.random.default_rng(seed).normal(0, .7, 32).astype(ml_dtypes.bfloat16)
            host[:8] = np.array([0., -0., 1., -1., 128., -128., 1e-5, -1e-5], dtype=ml_dtypes.bfloat16)
            x = session.upload(host)
            y = session.empty((32,), host.dtype)
            call(x, y, 32)
            actual = download(session, y)
            expected = (host.astype(np.float32) + np.float32(1)).astype(ml_dtypes.bfloat16)
            assert actual.dtype == host.dtype
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
