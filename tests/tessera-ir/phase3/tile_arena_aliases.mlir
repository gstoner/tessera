// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s
// The arena rewrite must retarget the whole view chain to workgroup space.
// CHECK-LABEL: func.func @arena_alias_chain
// CHECK-SAME: tile.buffer_reuse.groups = 1
// CHECK-SAME: tile.smem_arena_bytes = 64
// CHECK: memref.view {{.*}} to memref<16xf32, 3>
// CHECK: memref.cast {{.*}} : memref<16xf32, 3> to memref<?xf32, 3>
// CHECK: memref.subview {{.*}} : memref<?xf32, 3> to memref<8xf32, strided<[1]>, 3>
// CHECK: memref.load {{.*}} : memref<8xf32, strided<[1]>, 3>
// CHECK: memref.view {{.*}} to memref<16xf32, 3>
func.func @arena_alias_chain(%a: memref<16xf32>, %b: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  "tile.alloc_shared"(%a) : (memref<16xf32>) -> ()
  %cast = memref.cast %a : memref<16xf32> to memref<?xf32>
  %sub = memref.subview %cast[0] [8] [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
  %v = memref.load %sub[%c0] : memref<8xf32, strided<[1]>>
  "tile.cta_sync"() : () -> ()
  "tile.alloc_shared"(%b) : (memref<16xf32>) -> ()
  return
}
