// RUN: not tessera-opt %s --tessera-autodiff-forward 2>&1 | FileCheck %s

module {
  func.func @bad_wrt(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [1]} {
    return %x : tensor<4xf32>
  }
}

// CHECK: error: tessera-autodiff-forward: invalid wrt_indices entry
