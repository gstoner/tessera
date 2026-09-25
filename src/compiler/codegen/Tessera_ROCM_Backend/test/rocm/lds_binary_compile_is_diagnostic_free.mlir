// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1151 staging=lds})' %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201 staging=lds})' %s 2>&1 | FileCheck %s --implicit-check-not=error:
//
// A successful LDS-staged compile must not print `error:`. The pipeline used to
// run upstream `convert-vector-to-llvm` on the GPU module only to materialize
// `vector.create_mask`; that pass's partial LLVM conversion has no GPU
// address-space mapping, so every workgroup memref printed "error: conversion
// of memref memory space #gpu.address_space<workgroup> to integer address
// space failed" while the compile -- correctly, as `convert-gpu-to-rocdl` then
// did the conversion -- exited 0. The pipeline now runs only that pass's
// vector-to-vector stage (LowerVectorToVectorForROCDLPass); every emitted
// binary is byte-identical to before. stderr is folded into the check so any
// diagnostic on a successful compile fails this fixture.

module {
  func.func @gemm(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr, %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 32 : i64, tessera.macro_tile_n = 64 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK: gpu.binary
