// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module attributes {tessera.arch = "sm_120"} {
  llvm.func @tessera_tile_rmsnorm_bf16(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    %epsilon = arith.constant 1.000000e-05 : f32
    tile.norm_kernel %x, %o, %rows, %columns, %epsilon {
      storage = "bf16", accum = "f32", kind = "rmsnorm",
      axis = -1 : i64, affine = false
    } : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }

  llvm.func @tessera_tile_layernorm_f16(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    %epsilon = arith.constant 1.000000e-05 : f32
    tile.norm_kernel %x, %o, %rows, %columns, %epsilon {
      storage = "f16", accum = "f32", kind = "layernorm",
      axis = -1 : i64, affine = false
    } : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }

  llvm.func @tessera_tile_rmsnorm_bf16_cooperative(
      %x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    %epsilon = arith.constant 1.000000e-05 : f32
    tile.norm_kernel %x, %o, %rows, %columns, %epsilon {
      storage = "bf16", accum = "f32", kind = "rmsnorm",
      axis = -1 : i64, affine = false, schedule = "cooperative_128"
    } : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }

}

// CHECK-LABEL: llvm.func @tessera_tile_rmsnorm_bf16
// CHECK: llvm.fpext
// CHECK: arith.mulf
// CHECK: math.absf
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: llvm.fptrunc
// CHECK-LABEL: llvm.func @tessera_tile_layernorm_f16
// CHECK: llvm.fpext
// CHECK: arith.subf
// CHECK: math.absf
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: llvm.fptrunc
// CHECK-NOT: tile.norm_kernel

// CHECK-LABEL: llvm.func @tessera_tile_rmsnorm_bf16_cooperative
// CHECK-COUNT-9: nvvm.barrier
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: llvm.fptrunc
// CHECK-NOT: nvvm.rsqrt
// CHECK-NOT: tile.norm_kernel
