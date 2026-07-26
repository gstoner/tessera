// RUN: not %tnv --lower-tile-to-nvidia=sm=120 %s 2>&1 | FileCheck %s

module {
  llvm.func @canonical_k_tile_must_match_mma(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
// CHECK: error: 'tile.matmul_kernel' op canonical K-loop tile sizes must match the physical MMA descriptor
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.canonical_k_loop = true,
      tessera.tile_m = 16 : i64,
      tessera.tile_n = 8 : i64,
      tessera.tile_k = 8 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}
