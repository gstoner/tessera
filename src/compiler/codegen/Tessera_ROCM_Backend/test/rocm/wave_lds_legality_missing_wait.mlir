// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality)' %s 2>&1 | FileCheck %s

module {
  func.func @missing_wait(%dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
    return %m : tensor<16x16xf16>
  }
}

// CHECK: ROCM_WAVE_LDS_MISSING_WAITCNT
