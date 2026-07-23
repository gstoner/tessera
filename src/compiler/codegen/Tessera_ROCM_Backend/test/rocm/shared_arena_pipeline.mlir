// RUN: %trop --tessera-rocm-backend --allow-unregistered-dialect %s | FileCheck %s
//
// The production ROCm backend executes shared-buffer reuse and arena
// materialization before Tile-to-ROCm. Two disjoint f16 tiles alias one 512-byte
// LDS allocation through address-space-3 typed views.

module {
  // CHECK-LABEL: func.func @shared_arena
  // CHECK-SAME: tile.smem_arena_bytes = 512
  // CHECK-SAME: tile.smem_arena_materialized
  // CHECK: memref.alloca() {alignment = 16 : i64} : memref<512xi8, 3>
  // CHECK-COUNT-2: memref.view{{.*}}to memref<16x16xf16, 3>
  // CHECK-NOT: tile.alloc_shared
  func.func @shared_arena(%a: memref<16x16xf16>,
                          %b: memref<16x16xf16>) {
    "tile.alloc_shared"(%a) : (memref<16x16xf16>) -> ()
    "tile.alloc_shared"(%b) : (memref<16x16xf16>) -> ()
    return
  }
}
