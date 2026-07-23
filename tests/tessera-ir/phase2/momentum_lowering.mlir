// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

module {
  func.func @nesterov(%p: tensor<?x19xf32>, %g: tensor<?x19xf32>,
                      %v: tensor<?x19xf32>)
      -> (tensor<?x19xf32>, tensor<?x19xf32>) {
    %new_p, %new_v = "tessera.nesterov"(%p, %g, %v)
        {lr = 1.0e-2 : f64, momentum = 9.0e-1 : f64} :
        (tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>) ->
        (tensor<?x19xf32>, tensor<?x19xf32>)
    return %new_p, %new_v : tensor<?x19xf32>, tensor<?x19xf32>
  }
  // CHECK-LABEL: func.func @nesterov
  // CHECK-NOT: tessera.nesterov
  // CHECK: tensor.dim
  // CHECK: linalg.generic

  func.func @backward(%dp: tensor<?x19xf32>, %dv: tensor<?x19xf32>)
      -> (tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>) {
    %p, %g, %v = "tessera.momentum_backward"(%dp, %dv)
        {lr = 1.0e-2 : f64, momentum = 9.0e-1 : f64,
         nesterov = true} :
        (tensor<?x19xf32>, tensor<?x19xf32>) ->
        (tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>)
    return %p, %g, %v :
        tensor<?x19xf32>, tensor<?x19xf32>, tensor<?x19xf32>
  }
  // CHECK-LABEL: func.func @backward
  // CHECK-NOT: tessera.momentum_backward
  // CHECK: linalg.generic
}
