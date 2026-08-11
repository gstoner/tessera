// RUN: not tessera-opt --tessera-newton-autodiff %s 2>&1 | FileCheck %s

module {
  func.func private @general_residual(%theta: tensor<4xf32>,
                                      %x: tensor<4xf32>) -> tensor<4xf32> {
    %r = arith.subf %x, %theta : tensor<4xf32>
    return %r : tensor<4xf32>
  }

  func.func @bad_cg(%theta: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: linear_solver='cg' requires residual function attribute
    %x = "tessera_solver.implicit"(%theta) {
      residual = @general_residual,
      linear_solver = "cg"
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %x : tensor<4xf32>
  }
}
