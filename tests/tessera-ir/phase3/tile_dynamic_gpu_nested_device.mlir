// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
// CHECK: tile.dynamic_shared_size
// CHECK: gpu.dynamic_shared_memory
// CHECK: scf.for
// CHECK: memref.view
module {
  gpu.module @dynamic {
    gpu.func @scratch(%output: !llvm.ptr<1>, %n: index, %rounds: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %initial = arith.constant 0.0 : f32
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %next = arith.addi %tid, %one : index
      %neighbor = arith.remui %next, %n : index
      %base = arith.muli %bid, %n : index
      %position = arith.addi %base, %tid : index
      %index = arith.index_cast %position : index to i64
      %dst = llvm.getelementptr %output[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %sum = scf.for %i = %zero to %rounds step %one iter_args(%acc = %initial) -> f32 {
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()

        %value_index = arith.addi %tid, %i : index
        %integer = arith.index_cast %value_index : index to i32
        %value = arith.sitofp %integer : i32 to f32
        memref.store %value, %a[%tid] : memref<?xf32>
        gpu.barrier
        %read = memref.load %a[%neighbor] : memref<?xf32>
        gpu.barrier
        %total = arith.addf %acc, %read : f32
        scf.yield %total : f32
      }
      llvm.store %sum, %dst : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
