// RUN: ts-ebm-opt --tessera-ebm-lower-langevin %s | FileCheck %s
//
// An annealing schedule: the temperature arrives as a runtime operand instead of
// a constant attribute, so one loop samples a cooling chain. With the attribute
// the whole loop runs at one temperature and a schedule would have to be
// unrolled into K differently attributed steps.
//
// The noise scale is therefore computed rather than folded: max(2*eta*T, 0) and a
// sqrt, per step. The clamp is deliberate -- a negative temperature has no
// meaning, and NaN state would propagate silently through the rest of the chain.
// The compile-time elision a constant T = 0 gets is not available here, because
// the pass cannot know whether the schedule ever reaches zero.

// CHECK-LABEL: func @annealed
// CHECK-NOT:     tessera_ebm.langevin_step
// CHECK-DAG:     arith.maximumf
// CHECK-DAG:     math.sqrt
// CHECK-DAG:     linalg.fill
// CHECK-DAG:     call @E__bwd
// CHECK-NOT:     tessera_ebm.langevin_step
module {
  func.func private @E(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
  func.func @E__bwd(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>, %cot: tensor<4xf32>)
      -> (tensor<4x8xf32>, tensor<4x8xf32>) {
    %z = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    return %z, %z : tensor<4x8xf32>, tensor<4x8xf32>
  }
  func.func @annealed(%y0: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key0: tensor<2xi64>, %t0: f32)
      -> (tensor<4x8xf32>, tensor<2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps = arith.constant 4 : index
    %ratio = arith.constant 5.000000e-01 : f32
    %r:3 = scf.for %k = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0, %t = %t0)
        -> (tensor<4x8xf32>, tensor<2xi64>, f32) {
      %n:2 = "tessera_ebm.langevin_step"(%y, %key, %t, %x) {
          operandSegmentSizes = array<i32: 1, 1, 1, 1>,
          energy_fn = @E, eta = 1.000000e-01 : f64, manifold = "euclidean"
      } : (tensor<4x8xf32>, tensor<2xi64>, f32, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)
      %cooled = arith.mulf %t, %ratio : f32
      scf.yield %n#0, %n#1, %cooled : tensor<4x8xf32>, tensor<2xi64>, f32
    }
    return %r#0, %r#1 : tensor<4x8xf32>, tensor<2xi64>
  }
}
