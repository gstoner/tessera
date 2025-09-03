
// RUN: tessera-opt -tessera-cleanup %s | FileCheck %s
module {
  func.func @noop(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c = "tessera.cast"(%x) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %c : tensor<2x2xf32>
  }
}
// CHECK-NOT: tessera.cast
