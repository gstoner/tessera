// RUN: %tnv --tessera-lower-to-nvidia-sm120 --allow-unregistered-dialect %s | FileCheck %s
//
// E2E-SPINE-3 NVIDIA memory proof: the shared reuse plan must become one
// address-space-3 object.  The two disjoint-live buffers alias offset zero;
// the overlapping buffer occupies the second 512-byte slot.

module {
  func.func @sm120_shared_arena(
      %a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf16>) {
    "tile.alloc_shared"(%a) : (memref<16x16xf16>) -> ()
    "tile.async_copy"(%a) : (memref<16x16xf16>) -> ()
    "tile.alloc_shared"(%b) : (memref<16x16xf16>) -> ()
    "tile.async_copy"(%b) : (memref<16x16xf16>) -> ()
    "tile.async_copy"(%a) : (memref<16x16xf16>) -> ()
    "tile.alloc_shared"(%c) : (memref<16x16xf16>) -> ()
    "tile.async_copy"(%c) : (memref<16x16xf16>) -> ()
    return
  }
}

// CHECK: memref.global "private" @__tessera_smem_arena_sm120_shared_arena
// CHECK-SAME: memref<1024xi8, 3>
// CHECK-SAME: alignment = 16
// CHECK-LABEL: func.func @sm120_shared_arena
// CHECK-SAME: tile.buffer_reuse.bytes_after = 1024
// CHECK-SAME: tile.buffer_reuse.bytes_before = 1536
// CHECK-SAME: tile.buffer_reuse.groups = 2
// CHECK-SAME: tile.smem_arena_bytes = 1024
// CHECK-SAME: tile.smem_arena_materialized
// CHECK: memref.get_global @__tessera_smem_arena_sm120_shared_arena
// CHECK: arith.constant 0 : index
// CHECK: memref.view
// CHECK: arith.constant 512 : index
// CHECK: memref.view
// CHECK: arith.constant 0 : index
// CHECK: memref.view
// CHECK-NOT: tile.alloc_shared
