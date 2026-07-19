// RUN: %tnv --lower-tile-to-nvidia=sm=120 %s | FileCheck %s

module {
  llvm.func @mxfp4_is_not_nvfp4(
      %a: !llvm.ptr, %b: !llvm.ptr, %scale_a: !llvm.ptr,
      %scale_b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) attributes {nvvm.kernel} {
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 64, a = "fp4_e2m1", b = "fp4_e2m1", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK: tessera_nvidia.mx_block_scale_mma
// CHECK-SAME: dtype_ab = "e2m1"
// CHECK-SAME: scale_dtype = "ue8m0"
// CHECK-SAME: scale_vector = "2X"
// CHECK-NOT: mxf4nvf4
