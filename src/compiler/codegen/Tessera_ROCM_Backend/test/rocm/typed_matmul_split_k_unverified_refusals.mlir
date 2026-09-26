// RUN: not %trop --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=0 --lower-tile-to-rocm='arch=gfx1201' %s 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: not %trop --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=0 --generate-wmma-gemm-kernel='via-tile=true' %s 2>&1 | FileCheck %s --check-prefix=GEN
//
// ROCM-SPLIT-K-1 defence in depth. The tile.matmul_kernel verifier already
// refuses a split without its reduction order (TILE_SPLIT_K_BAD_CONTRACT, see
// tests/tessera-ir/phase2/rocm_split_k_contract_invalid.mlir), so a VERIFIED
// module never reaches these two consumers in this state. They still refuse on
// their own, because a consumer that trusted its producer here would lower a
// split with no stated order -- a semantic key defaulted (Decision #21a). The
// verifier is switched off to reach them; that flag is the point of the test,
// not a workaround.

module attributes {tessera.arch = "gfx1201"} {
  func.func @half_pair(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
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
      tessera.split_k = 2 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// TARGET: ROCM_SPLIT_K_{{UNSUPPORTED}}: tessera.split_k without tessera.split_k_reduction reached the ROCm Target consumer
// GEN: ROCM_SPLIT_K_{{UNSUPPORTED}}: the only admitted reduction is 'ordered'
