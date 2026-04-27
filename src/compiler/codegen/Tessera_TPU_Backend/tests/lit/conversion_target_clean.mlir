// RUN: tessera-tpu-opt -pass-pipeline='builtin.module(tessera-tpu-backend)' %s | FileCheck %s

module {
  func.func @batched(%a: tensor<2x4x8xbf16>, %b: tensor<2x8x16xbf16>) -> tensor<2x4x16xbf16> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<2x4x8xbf16>, tensor<2x8x16xbf16>) -> tensor<2x4x16xbf16>
    return %0 : tensor<2x4x16xbf16>
  }
}

// CHECK-LABEL: func.func @batched
// CHECK-NOT: tessera.matmul
// CHECK: stablehlo.dot_general
// CHECK-SAME: tensor<2x4x16xbf16>
