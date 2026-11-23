// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s
//
// AR-DECODE-PROGRAM-1 S1b: the row argmax lowers to one CTA per row: a strided
// thread scan, five butterfly shuffles of (value, index), then a shared-memory
// combine of the four warp partials by thread 0, which stores one i32 index.

module {
  llvm.func @argmax(%x: !llvm.ptr, %out: !llvm.ptr, %rows: i64, %cols: i64)
      attributes {nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} {
    tile.argreduce_kernel %x, %out, %rows, %cols {
      kind = "argmax", storage = "f32", output_storage = "i32", tie_break = "first",
      nan_mode = "first_nan", schedule = "cta_128_warp_shuffle"
    } : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

// CHECK-DAG: llvm.mlir.global internal @__tessera_sm120_argreduce_values_f32() {addr_space = 3 : i32
// CHECK-DAG: llvm.mlir.global internal @__tessera_sm120_argreduce_indices_i32() {addr_space = 3 : i32
// CHECK-LABEL: llvm.func @argmax
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: arith.cmpi ult, %{{.*}}, %arg2 : i64
// CHECK: scf.if
// The scan starts at the thread index and strides by the CTA width.
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, i32)
// CHECK: llvm.load {{.*}} : !llvm.ptr -> f32
// NaN detection, strict comparison and the lower-index tie-break.
// CHECK: arith.cmpf uno
// CHECK: arith.cmpi slt
// CHECK: arith.cmpf ogt
// CHECK: arith.cmpf oeq
// CHECK-COUNT-10: nvvm.shfl.sync bfly
// CHECK: nvvm.barrier
// CHECK: llvm.store %{{.*}}, %{{.*}} {{.*}} : i32, !llvm.ptr
// CHECK-NOT: tile.argreduce_kernel
