// RUN: %tnv --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

module {
  llvm.func @tessera_sm120_dmma_f64(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr)
      attributes {nvvm.kernel} {
    %av = llvm.load %a {alignment = 8 : i64} : !llvm.ptr -> f64
    %bv = llvm.load %b {alignment = 8 : i64} : !llvm.ptr -> f64
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    %result = tessera_nvidia.mma_sync %av, %bv, %zero, %zero
        {arch = "sm_120", shape = "m8n8k4", dtype_ab = "f64", dtype_c = "f64"}
        : (f64, f64, f64, f64) -> !llvm.struct<(f64, f64)>
    %d0 = llvm.extractvalue %result[0] : !llvm.struct<(f64, f64)>
    %d1 = llvm.extractvalue %result[1] : !llvm.struct<(f64, f64)>
    llvm.store %d0, %d {alignment = 8 : i64} : f64, !llvm.ptr
    %one = llvm.mlir.constant(1 : index) : i64
    %d_next = llvm.getelementptr %d[%one] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %d1, %d_next {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tessera_sm120_dmma_f64
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: shape = #nvvm.shape<m = 8, n = 8, k = 4>
// CHECK-SAME: -> !llvm.struct<(f64, f64)>
