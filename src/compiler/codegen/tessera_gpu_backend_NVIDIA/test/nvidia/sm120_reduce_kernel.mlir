// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @tessera_tile_reduce_sum_f32(
      %x: !llvm.ptr, %o: !llvm.ptr, %outer: i64, %axis_extent: i64, %inner: i64)
      attributes {nvvm.kernel} {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "f32", accum = "f32", kind = "sum",
      axis = 1 : i64, keepdims = true, schedule = "serial",
      nan_mode = "propagate"
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tessera_tile_reduce_max_f16(
      %x: !llvm.ptr, %o: !llvm.ptr, %outer: i64, %axis_extent: i64, %inner: i64)
      attributes {nvvm.kernel} {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "f16", accum = "f32", kind = "max",
      axis = 0 : i64, keepdims = false, schedule = "cooperative_128",
      nan_mode = "propagate"
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tessera_tile_reduce_min_bf16(
      %x: !llvm.ptr, %o: !llvm.ptr, %outer: i64, %axis_extent: i64, %inner: i64)
      attributes {nvvm.kernel} {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "bf16", accum = "f32", kind = "min",
      axis = 2 : i64, keepdims = true, schedule = "serial",
      nan_mode = "propagate"
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tessera_tile_reduce_mean_bf16_cooperative(
      %x: !llvm.ptr, %o: !llvm.ptr, %outer: i64, %axis_extent: i64, %inner: i64)
      attributes {nvvm.kernel} {
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {
      storage = "bf16", accum = "f32", kind = "mean",
      axis = 1 : i64, keepdims = false, schedule = "cooperative_128",
      nan_mode = "propagate"
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK: llvm.func @tessera_tile_reduce_sum_f32
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: arith.addf
// CHECK: llvm.func @tessera_tile_reduce_max_f16
// CHECK: llvm.fpext
// CHECK: arith.maximumf
// CHECK: nvvm.barrier
// CHECK: llvm.func @tessera_tile_reduce_min_bf16
// CHECK: llvm.fpext
// CHECK: arith.minimumf
// CHECK: llvm.func @tessera_tile_reduce_mean_bf16_cooperative
// CHECK: llvm.fpext
// CHECK: nvvm.barrier
// CHECK: arith.divf
// CHECK-NOT: tile.reduce_kernel
