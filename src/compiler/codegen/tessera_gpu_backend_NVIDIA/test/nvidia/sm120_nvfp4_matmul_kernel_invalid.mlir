// RUN: not %tnv --lower-tile-to-nvidia=sm=120 %s 2>&1 | FileCheck %s

module {
  llvm.func @nvfp4_bias_is_not_in_the_launch_abi(
      %a: !llvm.ptr, %b: !llvm.ptr, %scale_a: !llvm.ptr,
      %scale_b: !llvm.ptr, %bias: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
// CHECK: error: 'tile.matmul_kernel' op block-scaled launch-level matmul does not support fused bias/residual yet
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %bias, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = true, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}
