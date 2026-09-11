"""Replay-validated native f32 cotangent addition for resident DAG tracing."""
import inspect
import math
from pathlib import Path
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_gpu_storage import build_native_gpu_storage, _run
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def gradient_sum_source(shape):
    if (type(shape) is not tuple or not shape or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape) * 4 > 64 * 1024 * 1024):
        raise ValueError('gradient sum requires a positive shape within the native buffer bound')
    count = math.prod(shape)
    specs = tuple(TensorSpec(n, 'fp32', shape, n == 'out') for n in ('left', 'right', 'out')) + (IndexSpec('scratch', 1, 1),)
    source = f'''module {{
 gpu.module @native_tape {{
  gpu.func @product(%left: !llvm.ptr<1>, %right: !llvm.ptr<1>, %out: !llvm.ptr<1>, %scratch: index) kernel attributes {{known_block_size = array<i32: 128, 1, 1>}} {{
   %marker = memref.alloca(%scratch) : memref<?xf32>
   "tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()
   %block = gpu.block_id x
   %thread = gpu.thread_id x
   %width = arith.constant 128 : index
   %limit = arith.constant {count} : index
   %base = arith.muli %block, %width : index
   %index = arith.addi %base, %thread : index
   %inside = arith.cmpi ult, %index, %limit : index
   scf.if %inside {{
    %i = arith.index_cast %index : index to i64
    %lp = llvm.getelementptr %left[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %rp = llvm.getelementptr %right[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %op = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %l = llvm.load %lp : !llvm.ptr<1> -> f32
    %r = llvm.load %rp : !llvm.ptr<1> -> f32
    %sum = arith.addf %l, %r : f32
    llvm.store %sum, %op : f32, !llvm.ptr<1>
   }}
   gpu.return
  }}
 }}
}}'''
    return attach_tensor_contract(source, specs, grid=((count + 127) // 128, 1, 1), block=(128, 1, 1)), specs


def bind_gradient_sum(shape, *, compiler, llvm_bin, backend, chip):
    source, specs = gradient_sum_source(shape)
    compiler = Path(compiler)
    package = build_native_gpu_storage(source, compiler=compiler, llvm_bin=Path(llvm_bin), backend=backend, chip=chip)
    replay = _run(compiler, '--allow-unregistered-dialect', '--tessera-tile-buffer-reuse',
                  '--tessera-tile-buffer-arena', '--canonicalize', source=source)
    if package.arena_ir != replay:
        raise ValueError('gradient sum native replay disagrees')
    return generate_tensor_binding(package, inspect.Signature([
        inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs]))
