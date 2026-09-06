// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
// CHECK: tile.dynamic_shared_size
// CHECK: gpu.dynamic_shared_memory
// CHECK: llvm.load
// CHECK: scf.for
// CHECK: scf.if
// CHECK: gpu.barrier
module {
  gpu.module @dynamic {
    gpu.func @scratch(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>, %n: index, %rounds: index) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %initial = arith.constant 0.0 : f32
      %two = arith.constant 2.0 : f32
      %unit = arith.constant 1.0 : f32
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %blocks = gpu.grid_dim x
      %plane = arith.muli %blocks, %n : index
      %base = arith.muli %bid, %n : index
      %position = arith.addi %base, %tid : index
      %index = arith.index_cast %position : index to i64
      %src = llvm.getelementptr %input[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %dst = llvm.getelementptr %output[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %prime = llvm.load %src : !llvm.ptr<1> -> f32
      %next = arith.addi %tid, %one : index
      %neighbor = arith.remui %next, %n : index
      %result:2 = scf.for %i = %zero to %rounds step %one iter_args(%current = %prime, %acc = %initial) -> (f32, f32) {
        %a = memref.alloca(%n) : memref<?xf32, 3>
        "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
        memref.store %current, %a[%tid] : memref<?xf32, 3>
        gpu.barrier
        %read = memref.load %a[%neighbor] : memref<?xf32, 3>
        %generation = arith.addi %i, %one : index
        %has_next = arith.cmpi ult, %generation, %rounds : index
        %prefetched = scf.if %has_next -> f32 {
          %offset = arith.muli %generation, %plane : index
          %pos = arith.addi %offset, %position : index
          %idx = arith.index_cast %pos : index to i64
          %ptr = llvm.getelementptr %input[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
          %loaded = llvm.load %ptr : !llvm.ptr<1> -> f32
          scf.yield %loaded : f32
        } else {
          scf.yield %current : f32
        }
        %scaled = arith.mulf %read, %two : f32
        %shifted = arith.addf %scaled, %unit : f32
        %total = arith.addf %acc, %shifted : f32
        gpu.barrier
        scf.yield %prefetched, %total : f32, f32
      }
      llvm.store %result#1, %dst : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
