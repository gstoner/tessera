// RUN: %trop %s | FileCheck %s --check-prefix=TARGET
// RUN: not %trop --lower-tessera-target-to-rocdl %s 2>&1 | FileCheck %s --check-prefix=STRICT

module {
  func.func @k(%A: memref<*xf32>, %B: memref<*xf32>) {
    %c64 = arith.constant 64 : i64
    %t = "tessera_rocm.async_copy"(%A, %B, %c64) : (memref<*xf32>, memref<*xf32>, i64) -> !tessera_rocm.token
    "tessera_rocm.wait"(%t) : (!tessera_rocm.token) -> ()
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %c = arith.constant 0.0 : f32
    %r = "tessera_rocm.mfma"(%a,%b,%c) {gelu} : (f32,f32,f32) -> f32
    // TARGET: tessera_rocm.async_copy
    // TARGET: tessera_rocm.wait
    // TARGET: tessera_rocm.mfma
    // TARGET-NOT: .contract
    // STRICT: executable ROCm async operations must pass through lower-rocm-async-copy
    return
  }
}
