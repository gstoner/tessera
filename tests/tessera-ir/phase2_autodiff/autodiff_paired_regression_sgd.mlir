// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @mae(%prediction: tensor<?x5xf32>,
                 %target: tensor<?x5xf32>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.mae"(%prediction, %target) {reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @mae__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DP:.+]], %[[DT:.+]] = tessera.loss.regression_backward
  // CHECK-SAME: kind = "mae"
  // CHECK: return %[[DP]], %[[DT]]

  func.func @huber(%prediction: tensor<?x5xf32>,
                   %target: tensor<?x5xf32>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.huber"(%prediction, %target)
        {delta = 0.75 : f64, reduction = "sum"} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @huber__bwd
  // CHECK: tessera.loss.regression_backward
  // CHECK-SAME: kind = "huber"
  // CHECK-SAME: parameter = 7.500000e-01

  func.func @sgd(%param: tensor<?x5xf32>,
                 %grad: tensor<?x5xf32>) -> tensor<?x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %updated = "tessera.sgd"(%param, %grad) {lr = 0.25 : f64} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<?x5xf32>
    return %updated : tensor<?x5xf32>
  }

  // CHECK-LABEL: func.func @sgd__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DP:.+]], %[[DG:.+]] = tessera.sgd_backward
  // CHECK: return %[[DP]], %[[DG]]
}
