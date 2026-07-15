// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Launch-level canonical tensor-core GEMM. One warp owns each 16x8 output tile;
// block IDs select the tile, masked vector loads/stores cover M/N/K tails, and
// the scf.for carries four f32 accumulator registers across K panels.

module {
  // Baseline retained for kernel-only comparisons: one warp and direct global
  // fragment loads, with the same masks/K-loop/epilogue contract.
  llvm.func @tile_matmul_f32_direct(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                                    %m: i64, %n: i64, %k: i64)
      attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_matmul_f32(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                             %m: i64, %n: i64, %k: i64)
      attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 4 : i64, staging = "shared"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_matmul_bias_relu_f16(
      %a: !llvm.ptr, %b: !llvm.ptr, %bias: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %bias, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = true, activation = "relu", output = "f16">,
      warps = 4 : i64, staging = "shared"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_matmul_bf16(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                              %m: i64, %n: i64, %k: i64)
      attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "bf16", b = "bf16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 4 : i64, staging = "shared"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_matmul_gelu(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                              %m: i64, %n: i64, %k: i64)
      attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "gelu", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @tile_matmul_silu(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                              %m: i64, %n: i64, %k: i64)
      attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "silu", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tile_matmul_f32_direct
// CHECK: scf.for
// CHECK: nvvm.mma.sync
// CHECK-LABEL: llvm.func @tile_matmul_f32
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.ctaid.y
// CHECK: scf.for
// CHECK: llvm.intr.masked.load
// CHECK: nvvm.mma.sync
// CHECK: llvm.intr.masked.store
// CHECK-LABEL: llvm.func @tile_matmul_bias_relu_f16
// CHECK: scf.for
// CHECK: nvvm.mma.sync
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: arith.truncf
// CHECK: llvm.intr.masked.store
// CHECK-LABEL: llvm.func @tile_matmul_bf16
// CHECK: scf.for
// CHECK: nvvm.mma.sync
// CHECK: llvm.intr.masked.store
// CHECK-LABEL: llvm.func @tile_matmul_gelu
// CHECK: nvvm.mma.sync
// CHECK: math.tanh
// CHECK: llvm.intr.masked.store
// CHECK-LABEL: llvm.func @tile_matmul_silu
// CHECK: nvvm.mma.sync
// CHECK: math.exp
// CHECK: llvm.intr.masked.store
