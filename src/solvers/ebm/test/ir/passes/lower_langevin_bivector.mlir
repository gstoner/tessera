// RUN: ts-ebm-opt --tessera-ebm-lower-langevin %s | FileCheck %s
// RUN: not ts-ebm-opt --tessera-ebm-lower-langevin %s.invalid 2>&1 | FileCheck %s --check-prefix=INVALID
//
// The bivector integrator (2026-09-16, M2): the gradient and the noise are
// grade-projected with the Clifford dialect's OWN `grade` op — never a second
// grade projection in this pass (Decision #31) — the Euclidean affine step
// applies, and a final projection removes float leakage outside the subspace.
// The entry grade is checked and REPORTED in a per-row status word, never
// repaired (Decision #21a): a state carrying a blade outside the restricted
// grade must be visible to the caller. `grade` and `algebra` are semantic
// keys and are not defaulted.

module {
  func.func @E(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32> {
    %z = arith.constant dense<0.0> : tensor<4xf32>
    return %z : tensor<4xf32>
  }
  func.func @E__bwd(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %cot: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %d = arith.subf %y, %x : tensor<4x8xf32>
    return %d, %d : tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @bivector
  // CHECK: %[[G:.*]]:2 = call @E__bwd
  // entry-grade check: state minus its projection, squared row sum, status bit 0
  // CHECK: tessera_clifford.grade %arg0 grades [2] algebra [3, 0, 0]
  // CHECK: arith.subf %arg0
  // CHECK: linalg.reduce
  // CHECK: arith.cmpf ogt
  // the gradient and the noise are projected, then the affine step
  // CHECK: tessera_clifford.grade %[[G]]#0 grades [2]
  // CHECK: arith.mului_extended
  // CHECK: tessera_clifford.grade
  // CHECK: arith.addf
  // a final projection keeps the state in the subspace
  // CHECK: tessera_clifford.grade
  // CHECK: arith.select
  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>
  // CHECK-NOT: tessera_ebm.
  func.func @bivector(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key: tensor<2xi64>)
      -> (tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>) {
    %r:3 = "tessera_ebm.langevin_step"(%y, %key, %x)
        { operandSegmentSizes = array<i32: 1, 1, 0, 1>, energy_fn = @E, eta = 0.1 : f64, temperature = 0.4 : f64, manifold = "bivector",
          grade = 2 : i64, algebra = [3, 0, 0] }
        : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>)
    return %r#0, %r#1, %r#2 : tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>
  }
}

// INVALID-DAG: requires `grade` and `algebra`
// INVALID-DAG: is out of range for a 3-generator algebra
// INVALID-DAG: the feature axis must be 8
