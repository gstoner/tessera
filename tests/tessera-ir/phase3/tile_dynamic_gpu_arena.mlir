// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
module attributes {gpu.container_module} {
  // CHECK: func.func @__tessera_shared_bytes_dynamic_scratch
  // CHECK: cf.cond_br
  // CHECK: return
  gpu.module @dynamic {
    // CHECK-LABEL: gpu.func @scratch
    // CHECK-SAME: tile.dynamic_shared_size = @__tessera_shared_bytes_dynamic_scratch
    // CHECK-SAME: tile.smem_arena_materialized
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
      %a = memref.alloca(%n) : memref<?xf32>
      "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
      // CHECK: gpu.dynamic_shared_memory
      // CHECK: memref.view
      %sum = scf.for %i = %zero to %rounds step %one iter_args(%acc = %initial) -> f32 {
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
  // CHECK-LABEL: func.func @launch
  func.func @launch(%output: !llvm.ptr<1>, %n: index, %rounds: index) {
    %one = arith.constant 1 : index
    // CHECK: %[[BYTES:.*]] = call @__tessera_shared_bytes_dynamic_scratch
    // CHECK: cf.assert
    // CHECK-SAME: dynamic launch size exceeds nonnegative i32 range
    // CHECK: %[[COUNT:.*]] = arith.index_cast %[[BYTES]] : index to i32
    // CHECK: gpu.launch_func @dynamic::@scratch
    // CHECK-SAME: dynamic_shared_memory_size %[[COUNT]]
    gpu.launch_func @dynamic::@scratch blocks in (%one, %one, %one) threads in (%n, %one, %one) args(%output : !llvm.ptr<1>, %n : index, %rounds : index)
    return
  }
  // CHECK-LABEL: func.func @explicit_launch
  func.func @explicit_launch(%output: !llvm.ptr<1>, %n: index, %rounds: index, %bytes: i32) {
    %one = arith.constant 1 : index
    // CHECK: call @__tessera_shared_bytes_dynamic_scratch
    // CHECK: arith.cmpi eq
    // CHECK: cf.assert
    // CHECK-SAME: explicit dynamic launch bytes disagree with arena
    gpu.launch_func @dynamic::@scratch blocks in (%one, %one, %one) threads in (%n, %one, %one) dynamic_shared_memory_size %bytes args(%output : !llvm.ptr<1>, %n : index, %rounds : index)
    return
  }

}
