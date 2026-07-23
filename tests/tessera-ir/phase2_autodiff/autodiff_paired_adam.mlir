// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @adamw(%p: tensor<?x23xf32>, %g: tensor<?x23xf32>,
                   %m: tensor<?x23xf32>, %v: tensor<?x23xf32>)
      -> (tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>)
      attributes {tessera.autodiff = "reverse"} {
    %new_p, %new_m, %new_v = "tessera.adamw"(%p, %g, %m, %v)
        {lr = 2.0e-3 : f64, beta1 = 8.0e-1 : f64,
         beta2 = 9.5e-1 : f64, eps = 1.0e-7 : f64,
         weight_decay = 1.0e-2 : f64, step = 7 : i64} :
        (tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>,
         tensor<?x23xf32>) ->
        (tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>)
    return %new_p, %new_m, %new_v :
        tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>
  }
  // CHECK-LABEL: func.func @adamw__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = tessera.adam_backward
  // CHECK-SAME: adamw = true
  // CHECK-SAME: step = 7
}
