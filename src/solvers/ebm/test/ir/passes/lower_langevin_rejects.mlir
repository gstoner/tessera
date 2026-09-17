// RUN: not ts-ebm-opt --tessera-ebm-lower-langevin %s 2>&1 | FileCheck %s
//
// `not`: since 2026-09-16 the pass FAILS when an op survives it, rather than
// exiting 0 with an error already printed and an unlowered op in the output.
//
// Fail closed: a manifold without a native integrator, a sphere step that
// declares no status result, and a langevin_step whose energy has no
// compiler-derived gradient in the module.

module {
  func.func @E__bwd(%y: tensor<4x8xf32>, %cot: tensor<4xf32>) -> tensor<4x8xf32> {
    return %y : tensor<4x8xf32>
  }
  // Every manifold outside the closed enum is refused by the op's own verifier;
  // "bivector" gained a native integrator on 2026-09-16 and is checked by
  // lower_langevin_bivector.mlir, so the manifold refused here is the one that
  // still has no integrator behind a valid enum value.
  func.func @bivector_without_status(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { operandSegmentSizes = array<i32: 1, 1, 0, 0>, energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "bivector",
          grade = 2 : i64, algebra = [3, 0, 0] }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
  // The sphere integrator reports a per-row status word: a call that does not
  // declare it cannot observe the entry precondition, so it is refused.
  func.func @sphere_without_status(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { operandSegmentSizes = array<i32: 1, 1, 0, 0>, energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "sphere" }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
  func.func @no_gradient(%y: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key)
        { operandSegmentSizes = array<i32: 1, 1, 0, 0>, energy_fn = @Missing, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "euclidean" }
        : (tensor<4x8xf32>, tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }
}

// CHECK-DAG: the bivector integrator reports a per-row status word
// CHECK-DAG: the sphere integrator reports a per-row status word
// CHECK-DAG: the compiler-derived gradient @Missing__bwd is absent
