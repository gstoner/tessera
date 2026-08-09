// RUN: not tessera-opt --allow-unregistered-dialect --tessera-newton-autodiff %s 2>&1 | FileCheck %s

module {
  func.func private @wrong_residual(%theta: tensor<4xf32>) -> tensor<4xf32> {
    return %theta : tensor<4xf32>
  }

  func.func @bad_signature(%theta: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: error: residual @wrong_residual must have type (parameters..., solution...) -> residuals with one result matching each solution
    %x = "tessera_solver.implicit"(%theta) <{residual = @wrong_residual}>
        : (tensor<4xf32>) -> tensor<4xf32>
    return %x : tensor<4xf32>
  }
}
