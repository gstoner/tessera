// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s
//
// AR-DECODE-PROGRAM-1 S1c: the embedding gather lowers to one CTA per token
// row. The token is loaded once, sign-extended and checked unsigned against
// the vocabulary (so a negative token fails), the row is copied with a
// CTA-width stride, and an out-of-range token makes thread 0 store 1 into the
// status word instead of writing the row.

module {
  llvm.func @embed(%table: !llvm.ptr, %tokens: !llvm.ptr, %out: !llvm.ptr, %status: !llvm.ptr,
                   %v: i64, %h: i64, %b: i64)
      attributes {nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} {
    tile.embedding_gather_kernel %table, %tokens, %out, %status, %v, %h, %b {
      storage = "f32", index_storage = "i32", index_bounds = "kernel_checked_status_word",
      schedule = "cta_128_per_token_row"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @embed
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: arith.cmpi ult, %{{.*}}, %arg6 : i64
// CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: arith.cmpi ult, %{{.*}}, %arg4 : i64
// CHECK: scf.for
// CHECK: llvm.load {{.*}} : !llvm.ptr -> f32
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr
// CHECK: } else {
// CHECK: arith.cmpi eq
// CHECK: llvm.store %{{.*}}, %arg3 {{.*}} : i32, !llvm.ptr
// CHECK-NOT: tile.embedding_gather_kernel
