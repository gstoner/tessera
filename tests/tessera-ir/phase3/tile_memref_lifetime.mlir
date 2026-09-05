// RUN: tessera-opt --tessera-tile-buffer-reuse --allow-unregistered-dialect %s | FileCheck %s

// A missing wait must retain the buffer through exit.
// CHECK-LABEL: func.func @missing_wait
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @missing_wait(%a: memref<16xf32>, %b: memref<16xf32>) {
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  "tile.async_copy"(%a) : (memref<16xf32>) -> ()
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}

// Matching stage alone cannot complete a different barrier's copy.
// CHECK-LABEL: func.func @wrong_barrier
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @wrong_barrier(%a: memref<16xf32>, %b: memref<16xf32>) {
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  "tile.async_copy"(%a) {stage = 0 : i32, tile.barrier_id = 0 : i32} : (memref<16xf32>) -> ()
  "tile.wait_async"() {stage = 0 : i32, tile.barrier_id = 1 : i32} : () -> ()
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}

// Cast and subview users keep the whole backing allocation live.
// CHECK-LABEL: func.func @alias_late_use
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @alias_late_use(%a: memref<16xf32>, %b: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  %cast = memref.cast %a : memref<16xf32> to memref<?xf32>
  %view = memref.subview %cast[0] [8] [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  %x = memref.load %view[%c0] : memref<8xf32, strided<[1]>>
  return
}

// Views whose uses finish before the next allocation allow real reuse.
// CHECK-LABEL: func.func @alias_completed
// CHECK-SAME: tile.buffer_reuse.bytes_after = 64
// CHECK-SAME: tile.buffer_reuse.groups = 1
func.func @alias_completed(%a: memref<16xf32>, %b: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  %cast = memref.cast %a : memref<16xf32> to memref<?xf32>
  %x = memref.load %cast[%c0] : memref<?xf32>
  "tile.cta_sync"() : () -> ()
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}

// A wait in one branch cannot establish unconditional completion.
// CHECK-LABEL: func.func @conditional_wait
// CHECK-SAME: tile.buffer_reuse.groups = 2
func.func @conditional_wait(%a: memref<16xf32>, %b: memref<16xf32>, %cond: i1) {
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  "tile.async_copy"(%a) : (memref<16xf32>) -> ()
  scf.if %cond { "tile.wait_async"() : () -> () }
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}
