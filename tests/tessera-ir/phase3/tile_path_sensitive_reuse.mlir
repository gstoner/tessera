// RUN: tessera-opt --tessera-tile-buffer-reuse --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s --check-prefix=ARENA

// CHECK-LABEL: func.func @both_paths_complete
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: func.func @both_paths_complete
// ARENA-SAME: tile.smem_arena_bytes = 64
func.func @both_paths_complete(%a: memref<16xf32>, %b: memref<16xf32>, %c: i1) {
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  "tile.async_copy"(%a) {stage = 1 : i32} : (memref<16xf32>) -> ()
  scf.if %c {
    "tile.wait_async"() {stage = 1 : i32} : () -> ()
  } else {
    "tile.wait_async"() {stage = 1 : i32} : () -> ()
  }
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @uniform_exclusive_arms
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: func.func @uniform_exclusive_arms
// ARENA-SAME: tile.smem_arena_bytes = 64
func.func @uniform_exclusive_arms(%a: memref<16xf32>, %b: memref<16xf32>) {
  %block = gpu.block_id x
  %zero = arith.constant 0 : index
  %c = arith.cmpi eq, %block, %zero : index
  scf.if %c {
    "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
    "tile.async_copy"(%a) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
  } else {
    "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
    "tile.async_copy"(%b) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
  }
  return
}

// The threads in a workgroup may take different arms.
// CHECK-LABEL: func.func @divergent_arms
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @divergent_arms(%a: memref<16xf32>, %b: memref<16xf32>) {
  %thread = gpu.thread_id x
  %zero = arith.constant 0 : index
  %c = arith.cmpi eq, %thread, %zero : index
  scf.if %c {
    "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  } else {
    "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  }
  return
}

// CHECK-LABEL: func.func @loop_local_reuse
// CHECK-SAME: tile.buffer_reuse.groups = 1
// ARENA-LABEL: func.func @loop_local_reuse
// ARENA-SAME: tile.smem_arena_bytes = 64
func.func @loop_local_reuse(%a: memref<16xf32>, %b: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
    "tile.async_copy"(%a) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
    "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
    "tile.async_copy"(%b) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
  }
  return
}

// A copy still in flight at the backedge prohibits sharing with the next iteration.
// CHECK-LABEL: func.func @loop_missing_release
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @loop_missing_release(%a: memref<16xf32>, %b: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
    "tile.async_copy"(%a) : (memref<16xf32>) -> ()
    "tile.wait_async"() : () -> ()
    "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
    "tile.async_copy"(%b) : (memref<16xf32>) -> ()
  }
  return
}
