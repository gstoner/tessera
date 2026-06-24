// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality)' %s 2>&1 | FileCheck %s

module {
  func.func @missing_wait(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64, %a: f16, %b: f16) -> f16 {
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    %m = "tile.mma"(%a, %b) : (f16, f16) -> f16
    return %m : f16
  }
}

// CHECK: ROCM_WAVE_LDS_MISSING_WAITCNT
