// RUN: tessera-rocm-opt %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' | FileCheck %s --check-prefix=TARGET
// RUN: tessera-rocm-opt %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-paged-kv-read-kernel)' | FileCheck %s --check-prefix=KERNEL

module {
  llvm.func @tessera_tile_paged_kv_read_f32_direct(
      %pages: !llvm.ptr, %table: !llvm.ptr, %o: !llvm.ptr,
      %p: i64, %lp: i64, %ps: i64, %h: i64, %d: i64,
      %start: i64, %tokens: i64) {
    tile.paged_kv_read_kernel %pages, %table, %o, %p, %lp, %ps, %h, %d, %start, %tokens {
      storage = "f32", table_storage = "i32", route = "direct"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.paged_kv_read
// TARGET-SAME: source = "tile.paged_kv_read_kernel"
// TARGET-NOT: tile.paged_kv_read_kernel

// KERNEL: gpu.module @tessera_tile_paged_kv_read_f32_direct_mod
// KERNEL: gpu.func @tessera_tile_paged_kv_read_f32_direct
// KERNEL-SAME: memref<?xf32>, %{{.*}}: memref<?xi32>, %{{.*}}: memref<?xf32>
// KERNEL-NOT: tessera_rocm.paged_kv_read
