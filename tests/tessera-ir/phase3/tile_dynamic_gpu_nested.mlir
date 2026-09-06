// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --canonicalize | FileCheck %s
module {
  // CHECK: func.func @__tessera_shared_bytes_nested_scratch
  // CHECK: arith.maxui
  gpu.module @nested {
    // CHECK-LABEL: gpu.func @scratch
    // CHECK-SAME: tile.buffer_reuse.groups = 1
    // CHECK-SAME: tile.dynamic_shared_size
    gpu.func @scratch(%n: index, %m: index, %rounds: index, %choose: i1) kernel {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %value = arith.constant 1.0 : f32
      // CHECK: gpu.dynamic_shared_memory
      // CHECK: scf.for
      scf.for %i = %zero to %rounds step %one {
        scf.if %choose {
          %a = memref.alloca(%n) : memref<?xf32>
          "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
          memref.store %value, %a[%zero] : memref<?xf32>
          gpu.barrier
        } else {
          %size = arith.addi %m, %one : index
          %b = memref.alloca(%size) : memref<?xf32>
          "tile.alloc_shared"(%b) : (memref<?xf32>) -> ()
          memref.store %value, %b[%zero] : memref<?xf32>
          gpu.barrier
        }
      }
      gpu.return
    }
  }
}
