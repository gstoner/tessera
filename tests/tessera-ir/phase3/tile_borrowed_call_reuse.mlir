// RUN: tessera-opt --tessera-tile-buffer-reuse --tessera-tile-buffer-arena --allow-unregistered-dialect %s | FileCheck %s
func.func private @read_leaf(%a: memref<16xf32, 3>) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0] : memref<16xf32, 3>
  return %v : f32
}
func.func private @read_wrapper(%a: memref<16xf32, 3>) -> f32 {
  %v = call @read_leaf(%a) : (memref<16xf32, 3>) -> f32
  %again = call @read_leaf(%a) : (memref<16xf32, 3>) -> f32
  %sum = arith.addf %v, %again : f32
  return %sum : f32
}
// CHECK-LABEL: func.func @borrowed_call_reuse
// CHECK-SAME: tile.buffer_reuse.groups = 1
// CHECK-SAME: tile.smem_arena_bytes = 64
// CHECK: %[[VIEW:.*]] = memref.view {{.*}} to memref<16xf32, 3>
// CHECK: call @read_wrapper(%[[VIEW]])
func.func @borrowed_call_reuse(%a: memref<16xf32, 3>, %b: memref<16xf32, 3>) {
  "tile.alloc_shared"(%a) : (memref<16xf32, 3>) -> ()
  %v = call @read_wrapper(%a) : (memref<16xf32, 3>) -> f32
  "tile.cta_sync"() : () -> ()
  "tile.alloc_shared"(%b) : (memref<16xf32, 3>) -> ()
  return
}
