// RUN: tessera-opt %s --allow-unregistered-dialect --tessera-tile-buffer-arena | FileCheck %s
module {
  // CHECK: func.func @__tessera_shared_bytes_spaces_mixed
  // CHECK: arith.maxui
  gpu.module @spaces {
    // CHECK-LABEL: gpu.func @mixed
    // CHECK-SAME: tile.smem_arena_materialized
    // CHECK-SAME: tile.tmem_arena_bytes = 64
    gpu.func @mixed(%n: index, %m: index, %tmem: memref<16xf32>) kernel {
      %a = memref.alloca(%n) : memref<?xf32>
      %b = memref.alloca(%m) : memref<?xf32>
      // Disjoint empty lifetimes permit a supplied shared group; its size is
      // the maximum extent, while tensor memory retains an independent arena.
      "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<?xf32>) -> ()
      "tile.alloc_shared"(%b) {tile.buffer_group = 0 : i64} : (memref<?xf32>) -> ()
      // CHECK: gpu.dynamic_shared_memory
      // CHECK-COUNT-2: memref.view
      // CHECK: "tile.tmem.alloc"{{.*}}tile.tmem_offset = 0
      "tile.tmem.alloc"(%tmem) {tile.buffer_group = 1 : i64} : (memref<16xf32>) -> ()
      gpu.return
    }
  }
}
