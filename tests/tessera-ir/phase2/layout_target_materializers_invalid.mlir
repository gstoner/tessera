// REQUIRES: tessera-apple-backend
//
// RUN: not %tessera_strict_opt --tessera-apple-materialize-layout-casts %s 2>&1 | FileCheck %s

// Apple runtime bindings do not silently reinterpret a column-major tensor as
// row-major. NVIDIA owns a separate column-major staging contract.
// CHECK: error: Apple Graph layout materializer does not support 'col_major'
func.func @unsupported_apple_layout(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = "tessera.cast"(%arg0) {tessera.layout = "col_major"} :
      (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
