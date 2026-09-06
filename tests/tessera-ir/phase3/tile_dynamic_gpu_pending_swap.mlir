// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
// CHECK: tile.dynamic_shared_size
// CHECK: gpu.dynamic_shared_memory
// CHECK: nvgpu.device_async_copy
// CHECK: scf.for
// CHECK: nvgpu.device_async_wait
// CHECK: gpu.barrier
module {
  gpu.module @dynamic {
    gpu.func @scratch(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>, %n: index, %rounds: index) kernel {
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
      %total_blocks = gpu.grid_dim x
      %length = arith.muli %total_blocks, %n : index
      %generations = arith.addi %rounds, %one : index
      %all_length = arith.muli %length, %generations : index
      %len = arith.index_cast %all_length : index to i64
      %z64 = arith.constant 0 : i64
      %o64 = arith.constant 1 : i64
      %d0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d1 = llvm.insertvalue %input, %d0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d2 = llvm.insertvalue %input, %d1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d3 = llvm.insertvalue %z64, %d2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d4 = llvm.insertvalue %len, %d3[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d5 = llvm.insertvalue %o64, %d4[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %source = builtin.unrealized_conversion_cast %d5 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf32, 1>
      %a = memref.alloca(%n) : memref<?xf32, 3>
      "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()
      %other = memref.alloca(%n) : memref<?xf32, 3>
      "tile.alloc_shared"(%other) : (memref<?xf32, 3>) -> ()
      %seed = nvgpu.device_async_copy %source[%position], %a[%tid], 1 : memref<?xf32, 1> to memref<?xf32, 3>
      %initial_group = nvgpu.device_async_create_group %seed
      %last:4 = scf.for %i = %zero to %rounds step %one iter_args(%token = %initial_group, %acc = %initial, %read_slot = %a, %write_slot = %other) -> (!nvgpu.device.async.token, f32, memref<?xf32, 3>, memref<?xf32, 3>) {
        nvgpu.device_async_wait %token
        gpu.barrier
        %next_generation = arith.addi %i, %one : index
        %generation_base = arith.muli %next_generation, %length : index
        %refill_position = arith.addi %generation_base, %position : index
        %refill = nvgpu.device_async_copy %source[%refill_position], %write_slot[%tid], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %new_group = nvgpu.device_async_create_group %refill
        %read = memref.load %read_slot[%neighbor] : memref<?xf32, 3>
        gpu.barrier
        %total = arith.addf %acc, %read : f32
        scf.yield %new_group, %total, %write_slot, %read_slot : !nvgpu.device.async.token, f32, memref<?xf32, 3>, memref<?xf32, 3>
      }
      nvgpu.device_async_wait %last#0
      gpu.barrier
      %b = memref.alloca(%n) : memref<?xf32, 3>
      "tile.alloc_shared"(%b) : (memref<?xf32, 3>) -> ()
      memref.store %last#1, %b[%tid] : memref<?xf32, 3>
      gpu.barrier
      %result = memref.load %b[%tid] : memref<?xf32, 3>
      gpu.barrier
      llvm.store %result, %dst : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
