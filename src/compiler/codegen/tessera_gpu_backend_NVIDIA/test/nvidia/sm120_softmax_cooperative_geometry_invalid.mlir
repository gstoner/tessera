// RUN: not %tnv --tessera-lower-to-nvidia-sm120 %s 2>&1 | FileCheck %s
module {
  llvm.func @bad(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %columns: i64)
      attributes {nvvm.kernel} {
    tile.softmax_kernel %x, %o, %rows, %columns {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "approx_exp2", ftz = false,
      schedule = "cooperative_128", tessera.workgroup_size = 32 : i64
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}
// CHECK: cooperative softmax requires an NVVM kernel with workgroup size 128
