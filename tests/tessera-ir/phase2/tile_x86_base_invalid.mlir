// RUN: not tessera-opt --tessera-tile-to-x86='architecture=base prefer-amx=false' %s 2>&1 | FileCheck %s

module {
  llvm.func @matmul(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                    %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "f32", b = "f32", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK: error: x86 base architecture currently supports only softmax and reduction launch envelopes
