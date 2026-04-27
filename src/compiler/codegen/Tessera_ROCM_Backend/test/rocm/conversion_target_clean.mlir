// RUN: %trop -pass-pipeline='builtin.module(tessera-rocm-backend)' %s | FileCheck %s

module {
  func.func @my_kernel(%src: !llvm.ptr<1>, %dst: !llvm.ptr<3>) attributes {tessera_rocm.kernel = "true"} {
    %bytes = llvm.mlir.constant(64 : i64) : i64
    %tok = "tessera_rocm.async_copy"(%dst, %src, %bytes) : (!llvm.ptr<3>, !llvm.ptr<1>, i64) -> !tessera_rocm.token
    "tessera_rocm.wait"(%tok) : (!tessera_rocm.token) -> ()
    return
  }
}

// CHECK-LABEL: func.func @my_kernel
// CHECK-NOT: tessera_rocm.async_copy
// CHECK-NOT: tessera_rocm.wait
// CHECK: llvm.call @llvm.amdgcn.raw.buffer.load
// CHECK: llvm.call @llvm.amdgcn.ds.write
