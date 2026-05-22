\
// RUN: tessera-tpu-opt %s -tessera-lower-to-stablehlo | FileCheck %s

// A tiny synthetic op: replace tessera.matmul with stablehlo.dot_general
module {
  func.func @f(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }
}

// CHECK: stablehlo.dot_general
