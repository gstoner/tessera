// RUN: ts-ebm-opt --tessera-ebm-lower-langevin %s | FileCheck %s
//
// The sphere integrator (2026-09-16, M1): per row, the gradient and the noise
// are projected to the tangent plane (<v, x> x subtracted, the dot product a
// sequential f32 row sum), the affine step is taken, and the result is
// retracted by normalization. The entry precondition | |x|^2 - 1 | <= 2e-3
// and a retraction underflow |y|^2 < 1e-12 (previous state kept) are reported
// in a per-row i32 status word, never silently repaired.

module {
  func.func @E(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32> {
    %z = arith.constant dense<0.0> : tensor<4xf32>
    return %z : tensor<4xf32>
  }
  func.func @E__bwd(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %cot: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %d = arith.subf %y, %x : tensor<4x8xf32>
    return %d, %d : tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @sphere
  // CHECK: %[[G:.*]]:2 = call @E__bwd
  // CHECK: linalg.generic
  // CHECK: arith.mului_extended
  // entry norm: |x|^2 per row, deviation from 1, status bit 0
  // CHECK: arith.mulf %arg0, %arg0 : tensor<4x8xf32>
  // CHECK: linalg.reduce
  // CHECK: math.absf
  // CHECK: arith.cmpf ogt
  // tangent projection of the gradient: <g, x> x
  // CHECK: arith.mulf %[[G]]#0, %arg0 : tensor<4x8xf32>
  // CHECK: linalg.reduce
  // CHECK: tensor.expand_shape
  // CHECK: linalg.generic
  // CHECK: arith.subf %[[G]]#0
  // retraction: |y|^2, underflow guard, sqrt, divide, blend
  // CHECK: arith.cmpf olt
  // CHECK: arith.select
  // CHECK: math.sqrt {{.*}} : tensor<4xf32>
  // CHECK: arith.divf {{.*}} : tensor<4x8xf32>
  // CHECK: arith.ori {{.*}} : tensor<4xi32>
  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>
  // CHECK-NOT: tessera_ebm.
  func.func @sphere(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key: tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>) {
    %r:3 = "tessera_ebm.langevin_step"(%y, %key, %x)
        { energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "sphere" }
        : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>)
    return %r#0, %r#1, %r#2 : tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>
  }

  // T = 0 on the sphere: projected gradient descent + retraction, no Philox.
  // CHECK-LABEL: func.func @sphere_descent
  // CHECK-NOT: arith.mului_extended
  // CHECK: math.sqrt
  // CHECK: arith.divf
  // CHECK-NOT: arith.mului_extended
  func.func @sphere_descent(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key: tensor<2xi64>) -> (tensor<4x8xf32>, tensor<4xi32>) {
    %r:3 = "tessera_ebm.langevin_step"(%y, %key, %x)
        { energy_fn = @E, eta = 0.1 : f64, temperature = 0.0 : f64, manifold = "sphere" }
        : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>, tensor<4xi32>)
    return %r#0, %r#2 : tensor<4x8xf32>, tensor<4xi32>
  }
}
