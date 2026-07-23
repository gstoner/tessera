// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @kl(%p_log: tensor<?x17xf32>,
                %q: tensor<?x17xf32>) -> tensor<f32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.kl_divergence"(%p_log, %q)
        {axis = 1 : i64, epsilon = 1.0e-9 : f64, reduction = "mean"} :
        (tensor<?x17xf32>, tensor<?x17xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
  // CHECK-LABEL: func.func @kl__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DP:.+]], %[[DQ:.+]] = tessera.loss.distribution_backward
  // CHECK-SAME: kind = "kl"
  // CHECK: return %[[DP]], %[[DQ]]

  func.func @js(%p: tensor<?x17xf32>,
                %q: tensor<?x17xf32>) -> tensor<?xf32>
      attributes {tessera.autodiff = "reverse"} {
    %loss = "tessera.loss.js_divergence"(%p, %q)
        {axis = -1 : i64, reduction = "none"} :
        (tensor<?x17xf32>, tensor<?x17xf32>) -> tensor<?xf32>
    return %loss : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @js__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.loss.distribution_backward
  // CHECK-SAME: kind = "js"
}
