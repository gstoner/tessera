module {
  func.func @axis(%arg0: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> attributes {tessera.autodiff = "reverse", tessera.autodiff.paired = @axis__bwd, tessera.autodiff.residual_policy = "recompute_all", tessera.autodiff.wrt_indices = [0]} {
    %0 = tessera.transpose %arg0 {permutation = array<i64: 2, 0, 1>, tessera.autodiff.activity = "active"} : (tensor<2x3x5xf32>) -> tensor<5x2x3xf32>
    return %0 : tensor<5x2x3xf32>
  }
  func.func @axis__bwd(%arg0: tensor<2x3x5xf32>, %arg1: tensor<5x2x3xf32>) -> tensor<2x3x5xf32> attributes {tessera.autodiff.forward = @axis, tessera.autodiff.residual_policy = "recompute_all", tessera.autodiff.role = "backward"} {
    %0 = tessera.transpose %arg0 {permutation = array<i64: 2, 0, 1>, tessera.autodiff.activity = "active"} : (tensor<2x3x5xf32>) -> tensor<5x2x3xf32>
    %1 = tessera.transpose %arg1 : (tensor<5x2x3xf32>) -> tensor<2x3x5xf32>
    return %1 : tensor<2x3x5xf32>
  }
}

