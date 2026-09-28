// RUN: not %tnv --allow-unregistered-dialect --split-input-file --lower-tile-to-nvidia='sm=100' %s 2>&1 | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. Only the three declared Tile TMEM ops lower.
// Anything else under the `tile.tmem.` prefix used to fall into a default
// branch and silently become a `tessera_nvidia.tmem_store` contract; it now
// fails closed with a diagnostic naming the op and the target (Decision #21).

// The unregistered legacy spelling of the allocation is not an alias.
module {
  func.func @legacy_alloc_spelling(%buf: memref<16xf32>) {
    "tile.tmem.alloc"(%buf) : (memref<16xf32>) -> ()
    return
  }
}

// CHECK: NVIDIA_TMEM_UNKNOWN_OP: 'tile.tmem.alloc' is not a declared Tile TMEM op
// CHECK-SAME: NVIDIA sm_100 lowering
// CHECK-NOT: tessera_nvidia.tmem_store

// -----

// A misspelled / invented op must not become a store.
module {
  func.func @invented_op(%x: f32) {
    "tile.tmem.relinquish"(%x) : (f32) -> ()
    return
  }
}

// CHECK: NVIDIA_TMEM_UNKNOWN_OP: 'tile.tmem.relinquish' is not a declared Tile TMEM op
// CHECK-NOT: tessera_nvidia.tmem_store
