// RUN: not tessera-opt %s --tessera-autodiff-forward 2>&1 | FileCheck %s

module {
  func.func @unsupported(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "forward"} {
    %y = tessera.sin %x : (tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }
}

// CHECK: error: tessera-autodiff-forward: active operation has no TangentInterface
