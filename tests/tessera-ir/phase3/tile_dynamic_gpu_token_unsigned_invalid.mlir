// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --verify-diagnostics
// CHECK: tile.dynamic_shared_size
// CHECK: gpu.dynamic_shared_memory
// CHECK: scf.for
// CHECK: nvgpu.device_async_copy
// CHECK: nvgpu.device_async_wait
// CHECK: gpu.barrier
module {
  gpu.module @dynamic {
    // expected-error @+1 {{dynamic GPU arena requires uniform structured kernel regions}}
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
      %len = arith.index_cast %length : index to i64
      %z64 = arith.constant 0 : i64
      %o64 = arith.constant 1 : i64
      %d0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d1 = llvm.insertvalue %input, %d0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d2 = llvm.insertvalue %input, %d1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d3 = llvm.insertvalue %z64, %d2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d4 = llvm.insertvalue %len, %d3[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %d5 = llvm.insertvalue %o64, %d4[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %source = builtin.unrealized_conversion_cast %d5 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf32, 1>
      %sum = scf.for %i = %zero to %rounds step %one iter_args(%acc = %initial) -> f32 {
      %a = memref.alloca(%n) : memref<?xf32, 3>
      "tile.alloc_shared"(%a) : (memref<?xf32, 3>) -> ()

        %value_index = arith.addi %tid, %i : index
        %integer = arith.index_cast %value_index : index to i32
        %value = arith.sitofp %integer : i32 to f32
        %copy = nvgpu.device_async_copy %source[%position], %a[%tid], 1 : memref<?xf32, 1> to memref<?xf32, 3>
        %condition = arith.cmpi ult, %n, %rounds : index
        %selected = scf.if %condition -> !nvgpu.device.async.token {
          scf.yield %copy : !nvgpu.device.async.token
        } else {
          scf.yield %copy : !nvgpu.device.async.token
        }
        %group = nvgpu.device_async_create_group %selected
        %old = "nvgpu.device_async_create_group"() : () -> !nvgpu.device.async.token
        %negative = arith.constant -1 : index
        %forwarded = scf.for unsigned %j = %negative to %one step %one iter_args(%token = %old) -> !nvgpu.device.async.token {
          scf.yield %group : !nvgpu.device.async.token
        }
        nvgpu.device_async_wait %forwarded
        gpu.barrier
        %read = memref.load %a[%neighbor] : memref<?xf32, 3>
        gpu.barrier
        %total = arith.addf %acc, %read : f32
        scf.yield %total : f32
      }
      llvm.store %sum, %dst : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
