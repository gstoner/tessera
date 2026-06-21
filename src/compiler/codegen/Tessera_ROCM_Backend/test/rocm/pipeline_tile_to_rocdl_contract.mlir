// RUN: %trop --allow-unregistered-dialect --tessera-lower-to-rocm %s | FileCheck %s

module {
  func.func @kernel(%a: f32, %b: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = "tile.mma"(%a, %b) : (f32, f32) -> f32
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    "tile.wait_async"() : () -> ()
    return
  }
}

// The async global→LDS copy lowers to a targeted vmcnt wait (NOT a full
// barrier), so the matrix core can keep issuing while the copy is in flight.
// CHECK: llvm.call @llvm.amdgcn.mfma.contract
// CHECK: llvm.call @llvm.amdgcn.raw.buffer.copy.contract
// CHECK: llvm.call @llvm.amdgcn.s.waitcnt.vmcnt.contract
// CHECK-NOT: llvm.amdgcn.s.barrier.contract
// CHECK-NOT: tessera_rocm.
