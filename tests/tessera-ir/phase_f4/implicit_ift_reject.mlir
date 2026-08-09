// RUN: not tessera-opt --allow-unregistered-dialect --tessera-newton-autodiff %s 2>&1 | FileCheck %s

module {
  func.func @missing_residual(%theta: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: error: 'tessera_solver.implicit' op requires attribute 'residual'
    %x = "tessera_solver.implicit"(%theta)
        : (tensor<4xf32>) -> tensor<4xf32>
    return %x : tensor<4xf32>
  }
}
