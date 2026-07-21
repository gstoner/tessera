// RUN: tessera-opt --tessera-tile-to-x86='architecture=base prefer-amx=false' %s | FileCheck %s

module {
  llvm.func @softmax(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %k: i64) {
    tile.softmax_kernel %x, %o, %rows, %k {
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
  llvm.func @reduce(%x: !llvm.ptr, %o: !llvm.ptr, %outer: i64,
                    %extent: i64, %inner: i64) {
    tile.reduce_kernel %x, %o, %outer, %extent, %inner {
      storage = "f32", accum = "f32", kind = "sum", axis = 1 : i64,
      keepdims = false, schedule = "serial", nan_mode = "propagate",
      inner_is_one = true
    } : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-DAG: func.func private @tessera_x86_base_softmax_f32
// CHECK-DAG: func.func private @tessera_x86_base_reduce_f32
// CHECK-LABEL: llvm.func @softmax
// CHECK: call @tessera_x86_base_softmax_f32
// CHECK-NOT: avx512
// CHECK-LABEL: llvm.func @reduce
// CHECK: call @tessera_x86_base_reduce_f32
// CHECK-NOT: avx512
