// RUN: %trop --allow-unregistered-dialect --tessera-lower-to-rocm %s | FileCheck %s

module {
  func.func @kernel(%a: f32, %b: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = "tile.mma"(%a, %b) : (f32, f32) -> f32
    %tok = "tile.async_copy"(%dst, %src, %bytes) : (!llvm.ptr, !llvm.ptr, i64) -> !tessera_rocm.token
    "tile.wait_async"() : () -> ()
    return
  }
}

// The compatibility alias stops at typed Target IR. The selected counter and
// SSA token remain inspectable without fabricating LLVM symbols.
// CHECK: tessera_rocm.mfma
// CHECK: tessera_rocm.async_copy
// CHECK: tessera_rocm.wait
// CHECK-SAME: counter = "vmcnt"
// CHECK-NOT: .contract
