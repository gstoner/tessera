// RUN: not ts-ebm-opt %s 2>&1 | FileCheck %s
//
// The temperature selects which distribution the chain samples, so it is a
// semantic key (Decision #21a): exactly one of the constant attribute and the
// runtime operand, never both and never neither. Giving both would leave the
// reader — and the lowering — to pick, and picking silently is the failure this
// rule exists to prevent.

// CHECK-DAG: requires a temperature
// CHECK-DAG: temperature is given twice
module {
  func.func private @E(%y: tensor<4xf32>) -> tensor<1xf32>
  func.func @neither(%y: tensor<4xf32>, %k: tensor<2xi64>) -> tensor<4xf32> {
    %n:2 = "tessera_ebm.langevin_step"(%y, %k) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        energy_fn = @E, eta = 1.000000e-01 : f64, manifold = "euclidean"
    } : (tensor<4xf32>, tensor<2xi64>) -> (tensor<4xf32>, tensor<2xi64>)
    return %n#0 : tensor<4xf32>
  }
  func.func @both(%y: tensor<4xf32>, %k: tensor<2xi64>, %t: f32) -> tensor<4xf32> {
    %n:2 = "tessera_ebm.langevin_step"(%y, %k, %t) {
        operandSegmentSizes = array<i32: 1, 1, 1, 0>,
        energy_fn = @E, eta = 1.000000e-01 : f64, temperature = 5.000000e-01 : f64,
        manifold = "euclidean"
    } : (tensor<4xf32>, tensor<2xi64>, f32) -> (tensor<4xf32>, tensor<2xi64>)
    return %n#0 : tensor<4xf32>
  }
}
