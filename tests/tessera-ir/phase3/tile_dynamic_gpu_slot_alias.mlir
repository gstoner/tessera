// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
// CHECK: tile.dynamic_shared_size
// CHECK: gpu.dynamic_shared_memory
// CHECK: nvgpu.device_async_copy
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
      %slots:3 = scf.for %i = %zero to %rounds step %one iter_args(%read_slot = %a, %write_slot = %other, %acc = %initial) -> (memref<?xf32, 3>, memref<?xf32, 3>, f32) {
        %generation_base = arith.muli %i, %length : index
        %copy_position = arith.addi %generation_base, %position : index
        %copy = nvgpu.device_async_copy %source[%copy_position], %write_slot[%tid], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %group = nvgpu.device_async_create_group %copy
        nvgpu.device_async_wait %group
        gpu.barrier
        %read = memref.load %write_slot[%neighbor] : memref<?xf32, 3>
        %total = arith.addf %acc, %read : f32
        gpu.barrier
        scf.yield %write_slot, %read_slot, %total : memref<?xf32, 3>, memref<?xf32, 3>, f32
      }
      %b = memref.alloca(%n) : memref<?xf32, 3>
      "tile.alloc_shared"(%b) : (memref<?xf32, 3>) -> ()
      memref.store %slots#2, %b[%tid] : memref<?xf32, 3>
      gpu.barrier
      %result = memref.load %b[%tid] : memref<?xf32, 3>
      gpu.barrier
      llvm.store %result, %dst : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
