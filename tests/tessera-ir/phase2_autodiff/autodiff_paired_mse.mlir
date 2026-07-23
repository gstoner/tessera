// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// The first compiler-native loss adjoint keeps dynamic mean extent and the
// target gradient visible in Graph IR.

module {
  func.func @mse(%prediction: tensor<?x5xf32>,
                 %target: tensor<?x5xf32>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.mse"(%prediction, %target)
        {reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @mse__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DP:.+]], %[[DT:.+]] = tessera.loss.mse_backward
  // CHECK: return %[[DP]], %[[DT]]
}
