
// RUN: tessera-opt -tessera-canonicalize %s | FileCheck %s
module {
  func.func @g(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = "tessera.cast"(%x) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %c : tensor<4x4xf32>
  }
}
// CHECK-NOT: tessera.cast
