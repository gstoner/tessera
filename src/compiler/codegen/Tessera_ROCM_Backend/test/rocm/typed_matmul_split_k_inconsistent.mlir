// RUN: not %trop --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=0 --generate-wmma-gemm-kernel='via-tile=true' %s 2>&1 | FileCheck %s
//
// ROCM-SPLIT-K-1: the generator's own consistency check. An ordered reduction
// stated with split_k = 1 names no split at all; the generator refuses it
// rather than guessing which half is right. Unreachable from verified IR (the
// tile.matmul_kernel verifier refuses split_k < 2), so the verifier is off.

module attributes {tessera.arch = "gfx1201"} {
  func.func @order_without_split(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
    %m = arith.constant 16 : i64
    %n = arith.constant 256 : i64
    %k = arith.constant 2048 : i64
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.canonical_k_loop = true,
      tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64,
      tessera.split_k = 1 : i64, tessera.split_k_reduction = "ordered"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK: ROCM_SPLIT_K_{{UNSUPPORTED}}: inconsistent split-K contract (split_k=1, reduction='ordered')
