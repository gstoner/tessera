// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

module {
  func.func @adam_backward(
      %p: tensor<?x23xf32>, %g: tensor<?x23xf32>,
      %m: tensor<?x23xf32>, %v: tensor<?x23xf32>,
      %dp: tensor<?x23xf32>, %dm: tensor<?x23xf32>,
      %dv: tensor<?x23xf32>)
      -> (tensor<?x23xf32>, tensor<?x23xf32>,
          tensor<?x23xf32>, tensor<?x23xf32>) {
    %gp, %gg, %gm, %gv = "tessera.adam_backward"(
        %p, %g, %m, %v, %dp, %dm, %dv)
        {lr = 2.0e-3 : f64, beta1 = 8.0e-1 : f64,
         beta2 = 9.5e-1 : f64, eps = 1.0e-7 : f64,
         weight_decay = 1.0e-2 : f64, step = 7 : i64,
         adamw = true} :
        (tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>,
         tensor<?x23xf32>, tensor<?x23xf32>, tensor<?x23xf32>,
         tensor<?x23xf32>) ->
        (tensor<?x23xf32>, tensor<?x23xf32>,
         tensor<?x23xf32>, tensor<?x23xf32>)
    return %gp, %gg, %gm, %gv :
        tensor<?x23xf32>, tensor<?x23xf32>,
        tensor<?x23xf32>, tensor<?x23xf32>
  }
  // CHECK-LABEL: func.func @adam_backward
  // CHECK-NOT: tessera.adam_backward
  // CHECK: tensor.dim
  // CHECK: linalg.generic
  // CHECK: math.sqrt
  // CHECK: arith.cmpf
  // CHECK: arith.select
}
