\
// RUN: tessera-tpu-opt %s -tessera-lower-to-stablehlo -tessera-export-shardy | FileCheck %s
module {
  func.func @f(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }
}
// CHECK: "sdy.mesh"
// CHECK: "sdy.tensor_sharding"
