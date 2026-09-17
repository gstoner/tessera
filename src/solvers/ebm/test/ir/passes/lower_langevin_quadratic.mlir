// RUN: ts-ebm-opt --tessera-ebm-lower-langevin %s | FileCheck %s
//
// The first EBM lowering (2026-09-16): energy / inner_step / langevin_step
// become arith + linalg over the compiler-derived gradient @E__bwd. The
// gradient function here is written by hand only because this driver has no
// tessera dialect; in the JIT the paired autodiff pass produces it.

module {
  // E(state, context) -> per-row energies; @E__bwd(state, context, cot) -> (dstate, dcontext)
  func.func @E(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32> {
    %z = arith.constant dense<0.0> : tensor<4xf32>
    return %z : tensor<4xf32>
  }
  func.func @E__bwd(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %cot: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %d = arith.subf %y, %x : tensor<4x8xf32>
    return %d, %d : tensor<4x8xf32>, tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @sample
  // CHECK: %[[ONES:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK: %[[G:.*]]:2 = call @E__bwd(%{{.*}}, %{{.*}}, %[[ONES]])
  // CHECK: arith.mulf %[[G]]#0, %{{.*}} : tensor<4x8xf32>
  // CHECK: arith.subf
  // CHECK: linalg.generic
  // CHECK: linalg.index
  // CHECK: arith.mului_extended
  // CHECK: math.log
  // CHECK: math.cos
  // CHECK: arith.truncf
  // CHECK: arith.addf {{.*}} : tensor<4x8xf32>
  // CHECK: arith.addi %{{.*}}, %{{.*}} : tensor<2xi64>
  // CHECK-NOT: tessera_ebm.
  func.func @sample(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key: tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>) {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key, %x)
        { operandSegmentSizes = array<i32: 1, 1, 0, 1>, energy_fn = @E, eta = 0.1 : f64, temperature = 0.5 : f64, manifold = "euclidean" }
        : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0, %r#1 : tensor<4x8xf32>, tensor<2xi64>
  }

  // Temperature 0 draws no noise at all: pure gradient descent, no linalg.
  // CHECK-LABEL: func.func @descent
  // CHECK: call @E__bwd
  // CHECK-NOT: linalg.generic
  // CHECK: arith.subf
  // CHECK-NOT: tessera_ebm.
  func.func @descent(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key: tensor<2xi64>) -> tensor<4x8xf32> {
    %r:2 = "tessera_ebm.langevin_step"(%y, %key, %x)
        { operandSegmentSizes = array<i32: 1, 1, 0, 1>, energy_fn = @E, eta = 0.1 : f64, temperature = 0.0 : f64, manifold = "euclidean" }
        : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)
    return %r#0 : tensor<4x8xf32>
  }

  // energy -> call E(state, context); inner_step -> y - eta * g.
  // CHECK-LABEL: func.func @evaluate
  // CHECK: call @E(%arg1, %arg0)
  // CHECK: arith.mulf
  // CHECK: arith.subf
  // CHECK-NOT: tessera_ebm.
  func.func @evaluate(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>, %g: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4x8xf32>) {
    %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @E } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4xf32>
    %s = "tessera_ebm.inner_step"(%y, %g) { eta = 0.25 : f64 } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %e, %s : tensor<4xf32>, tensor<4x8xf32>
  }
}
