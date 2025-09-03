// RUN: %trop --lower-tessera-target-to-rocdl %s | FileCheck %s

module {
  func.func @k(%A: memref<*xf32>, %B: memref<*xf32>) {
    %c64 = arith.constant 64 : i64
    %t = "tessera_rocm.async_copy"(%A, %B, %c64) : (memref<*xf32>, memref<*xf32>, i64) -> !tessera_rocm.token
    "tessera_rocm.wait"(%t) : (!tessera_rocm.token) -> ()
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %c = arith.constant 0.0 : f32
    %r = "tessera_rocm.mfma"(%a,%b,%c) {gelu} : (f32,f32,f32) -> f32
    // CHECK: call @llvm.amdgcn.mfma
    return
  }
}
