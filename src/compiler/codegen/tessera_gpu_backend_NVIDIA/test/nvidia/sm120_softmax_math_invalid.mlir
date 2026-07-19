// RUN: not %tnv --tessera-lower-to-nvidia-sm120 %s 2>&1 | FileCheck %s

module {
  llvm.func @bad_math(%x: !llvm.ptr, %o: !llvm.ptr,
                      %rows: i64, %columns: i64) attributes {nvvm.kernel} {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "ieee_exp", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// CHECK: sm_120 softmax_kernel requires f16/f32 storage, f32 accum, axis=-1, exp_mode=approx_exp2, and ftz=false
