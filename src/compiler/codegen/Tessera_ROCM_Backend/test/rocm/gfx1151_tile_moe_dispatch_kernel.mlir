// RUN: %trop %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' | FileCheck %s --check-prefix=TARGET
// RUN: %trop %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-moe-kernel)' | FileCheck %s --check-prefix=KERNEL

module {
  llvm.func @tessera_tile_moe_dispatch_f32_direct(
      %x: !llvm.ptr, %token: !llvm.ptr, %o: !llvm.ptr,
      %t: i64, %s: i64, %h: i64) {
    tile.moe_dispatch_kernel %x, %token, %o, %t, %s, %h {
      storage = "f32", index_storage = "i32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.moe_dispatch
// TARGET-SAME: source = "tile.moe_dispatch_kernel"
// TARGET-NOT: tile.moe_dispatch_kernel

// KERNEL: gpu.module @tessera_tile_moe_dispatch_f32_direct_mod
// KERNEL: gpu.func @tessera_tile_moe_dispatch_f32_direct
// KERNEL-SAME: memref<?xf32>, %{{.*}}: memref<?xi32>, %{{.*}}: memref<?xf32>
// KERNEL-NOT: tessera_rocm.moe_dispatch
