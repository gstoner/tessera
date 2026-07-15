// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel %s | FileCheck %s

module {
  func.func @tile_matmul_f32(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                             %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// CHECK-NOT: tile.matmul_kernel
// CHECK: gpu.module @tile_matmul_f32_mod
// CHECK: gpu.func @tile_matmul_f32
// CHECK-SAME: memref<?xf16>
// CHECK: gpu.block_id x
// CHECK: gpu.block_id y
// CHECK: scf.for
// CHECK: vector.load
// CHECK: tessera_rocm.wmma
// CHECK: scf.if
// CHECK: memref.store
