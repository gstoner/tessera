// RUN: tessera-rocm-opt %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' | FileCheck %s --check-prefix=TARGET
// RUN: tessera-rocm-opt %s --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-reduce-kernel)' | FileCheck %s --check-prefix=KERNEL

module {
  llvm.func @tessera_tile_reduce_sum_f32(%x: !llvm.ptr, %o: !llvm.ptr,
                                         %outer: i64, %axis_extent: i64,
                                         %inner: i64) {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "f32", accum = "f32", kind = "sum", axis = 1 : i64,
      keepdims = false, schedule = "serial", nan_mode = "propagate"
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.reduce
// TARGET-SAME: layout = "outer_axis_inner"
// TARGET-SAME: source = "tile.reduce_kernel"
// TARGET-NOT: tile.reduce_kernel

// KERNEL: gpu.module @tessera_tile_reduce_sum_f32_mod
// KERNEL: gpu.func @tessera_tile_reduce_sum_f32
// KERNEL-SAME: %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index
// KERNEL-NOT: tessera_rocm.reduce
