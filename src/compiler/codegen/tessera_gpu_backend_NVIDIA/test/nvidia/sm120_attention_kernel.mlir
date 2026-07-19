// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @attention(%q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr,
                       %o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64,
                       %sq: i64, %sk: i64, %d: i64, %dv: i64)
      attributes {nvvm.kernel} {
    tile.attention_kernel %q, %k, %v, %o, %b, %hq, %hkv, %sq, %sk, %d, %dv {
      storage = "f16", accum = "f32", scale = 0.35355338454246521 : f32,
      causal = true, bias = false, window_left = -1 : i64,
      window_right = -1 : i64, softcap = 0.000000e+00 : f32,
      dropout_p = 0.000000e+00 : f32, dropout_seed = 0 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @attention
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: llvm.fpext
// CHECK: nvvm.ex2
// CHECK: llvm.store
// CHECK-NOT: tile.attention_kernel
