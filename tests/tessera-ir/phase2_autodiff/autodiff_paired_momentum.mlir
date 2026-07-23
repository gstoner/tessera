// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @momentum(%p: tensor<?x19xf32>, %g: tensor<?x19xf32>,
                      %v: tensor<?x19xf32>)
      -> (tensor<?x19xf32>, tensor<?x19xf32>)
      attributes {tessera.autodiff = "reverse"} {
    %new_p, %new_v = "tessera.momentum"(%p, %g, %v)
        {lr = 1.0e-2 : f64, momentum = 9.0e-1 : f64} :
        (tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>) ->
        (tensor<?x19xf32>, tensor<?x19xf32>)
    return %new_p, %new_v : tensor<?x19xf32>, tensor<?x19xf32>
  }
  // CHECK-LABEL: func.func @momentum__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[DP:.+]], %[[DG:.+]], %[[DV:.+]] = tessera.momentum_backward
  // CHECK: return %[[DP]], %[[DG]], %[[DV]]

  func.func @nesterov(%p: tensor<?x19xf32>, %g: tensor<?x19xf32>,
                      %v: tensor<?x19xf32>)
      -> (tensor<?x19xf32>, tensor<?x19xf32>)
      attributes {tessera.autodiff = "reverse"} {
    %new_p, %new_v = "tessera.nesterov"(%p, %g, %v)
        {lr = 1.0e-2 : f64, momentum = 9.0e-1 : f64} :
        (tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>) ->
        (tensor<?x19xf32>, tensor<?x19xf32>)
    return %new_p, %new_v : tensor<?x19xf32>, tensor<?x19xf32>
  }
  // CHECK-LABEL: func.func @nesterov__bwd
  // CHECK: tessera.momentum_backward
  // CHECK-SAME: nesterov = true
}
