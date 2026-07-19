// RUN: not %tnv --tessera-lower-to-nvidia-sm120 %s 2>&1 | FileCheck %s

module {
  llvm.func @bad(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "bf16", accum = "f32", axis = -1 : i64,
      exp_mode = "approx_exp2", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// CHECK: 'tile.softmax_kernel' op requires storage="f16" or storage="f32"
