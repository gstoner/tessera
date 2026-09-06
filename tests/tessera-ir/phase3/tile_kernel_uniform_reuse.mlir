// RUN: tessera-opt --tessera-tile-buffer-reuse --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s --check-prefix=ARENA

module {
gpu.module @kernels {
// CHECK-LABEL: gpu.func @symbolic
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: gpu.func @symbolic
// ARENA-SAME: tile.smem_arena_bytes = 64
// ARENA-SAME: tile.smem_arena_materialized
  gpu.func @symbolic(%a: memref<16xf32>, %b: memref<16xf32>, %n: index) kernel {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tid = gpu.thread_id x
    scf.for %i = %zero to %n step %one {
      "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
      "tile.async_copy"(%a) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
      "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
      "tile.async_copy"(%b) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @divergent
// CHECK-SAME: tile.buffer_reuse.groups = 2
// ARENA-LABEL: gpu.func @divergent
// ARENA-SAME: tile.smem_arena_bytes = 128
// ARENA-SAME: tile.smem_arena_materialized
  gpu.func @divergent(%a: memref<16xf32>, %b: memref<16xf32>, %n: index) kernel {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tid = gpu.thread_id x
    scf.for %i = %zero to %tid step %one {
      "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
      "tile.async_copy"(%a) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
      "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
      "tile.async_copy"(%b) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @helper
// CHECK-SAME: tile.buffer_reuse.groups = 2
// ARENA-LABEL: gpu.func @helper
// ARENA-SAME: tile.smem_arena_bytes = 128
// ARENA-SAME: tile.smem_arena_materialized
  gpu.func @helper(%a: memref<16xf32>, %b: memref<16xf32>, %n: index)  {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tid = gpu.thread_id x
    scf.for %i = %zero to %n step %one {
      "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
      "tile.async_copy"(%a) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
      "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
      "tile.async_copy"(%b) : (memref<16xf32>) -> ()
      "tile.wait_async"() : () -> ()
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @missing_release
// CHECK-SAME: tile.buffer_reuse.groups = 2
// ARENA-LABEL: gpu.func @missing_release
// ARENA-SAME: tile.smem_arena_bytes = 128
// ARENA-SAME: tile.smem_arena_materialized
  gpu.func @missing_release(%a: memref<16xf32>, %b: memref<16xf32>, %n: index) kernel {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tid = gpu.thread_id x
    scf.for %i = %zero to %n step %one {
      "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
      "tile.async_copy"(%a) : (memref<16xf32>) -> ()
      "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
      "tile.async_copy"(%b) : (memref<16xf32>) -> ()
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @induction_branch
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: gpu.func @induction_branch
// ARENA-SAME: tile.smem_arena_bytes = 64
  gpu.func @induction_branch(%a: memref<16xf32>, %b: memref<16xf32>, %n: index) kernel {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %first = arith.cmpi eq, %i, %zero : index
      scf.if %first {
        "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
        "tile.async_copy"(%a) : (memref<16xf32>) -> ()
        "tile.wait_async"() : () -> ()
      } else {
        "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
        "tile.async_copy"(%b) : (memref<16xf32>) -> ()
        "tile.wait_async"() : () -> ()
      }
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @native_barrier
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: gpu.func @native_barrier
// ARENA-SAME: tile.smem_arena_bytes = 64
// ARENA-SAME: tile.smem_arena_materialized
// ARENA: memref.get_global @__tessera_smem_arena_native_barrier : memref<64xi8, 3>
// ARENA: gpu.barrier
  gpu.func @native_barrier(%a: memref<16xf32>, %b: memref<16xf32>, %out: memref<16xf32>, %n: index) kernel {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tid = gpu.thread_id x
    %value = arith.constant 1.0 : f32
    scf.for %i = %zero to %n step %one {
      "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
      memref.store %value, %a[%tid] : memref<16xf32>
      gpu.barrier
      %first = memref.load %a[%tid] : memref<16xf32>
      gpu.barrier
      "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
      memref.store %value, %b[%tid] : memref<16xf32>
      gpu.barrier
      %second = memref.load %b[%tid] : memref<16xf32>
      gpu.barrier
      %sum = arith.addf %first, %second : f32
      memref.store %sum, %out[%tid] : memref<16xf32>
    }
    gpu.return
  }

// CHECK-LABEL: gpu.func @dynamic_storage
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: gpu.func @dynamic_storage
// ARENA-SAME: tile.dynamic_shared_size = @__tessera_shared_bytes_kernels_dynamic_storage
// ARENA-SAME: tile.smem_arena_dynamic
// ARENA-SAME: tile.smem_arena_materialized
// ARENA: gpu.dynamic_shared_memory
// ARENA-NOT: memref.alloca
// ARENA: gpu.return
  gpu.func @dynamic_storage(%a: memref<?xf32>) kernel {
    "tile.alloc_shared"(%a) : (memref<?xf32>) -> ()
    gpu.return
  }
}
}

// A kernel-like marker on an ordinary function is not a launch-ABI proof.
// CHECK-LABEL: func.func @marked_function
// CHECK-SAME: tile.buffer_reuse.groups = 2
// ARENA-LABEL: func.func @marked_function
// ARENA-SAME: tile.smem_arena_bytes = 128
func.func @marked_function(%a: memref<16xf32>, %b: memref<16xf32>, %n: index) attributes {gpu.kernel} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
    "tile.async_copy"(%a) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
    "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
    "tile.async_copy"(%b) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
  }
  return
}
