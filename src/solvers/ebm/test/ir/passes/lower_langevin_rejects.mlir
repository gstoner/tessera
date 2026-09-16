// RUN: ts-ebm-opt --tessera-ebm-lower-langevin %s 2>&1 | FileCheck %s
//
// Fail closed: a manifold without a native integrator, a sphere step that
// declares no status result, and a langevin_step whose energy has no
// compiler-derived gradient in the module.

module {
  func.func @E__bwd(%y: tensor<4x8xf32>, %cot: tensor<4xf32>) -> tensor<4x8xf32> {
    return %y : tensor<4x8xf32>
  }
  func.func @bivector(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "bivector", grade = 2 : i64 }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
  // The sphere integrator reports a per-row status word: a call that does not
  // declare it cannot observe the entry precondition, so it is refused.
  func.func @sphere_without_status(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "sphere" }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
  func.func @no_gradient(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { energy_fn = @Missing, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "euclidean" }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
}

// CHECK-DAG: manifold "bivector" has no native integrator yet
// CHECK-DAG: the sphere integrator reports a per-row status word
// CHECK-DAG: the compiler-derived gradient @Missing__bwd is absent
