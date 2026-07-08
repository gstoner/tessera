// RUN: tessera-opt %s -tpp-legalize-space-time | FileCheck %s
//
// LegalizeSpaceTime normalizes stencil + temporal metadata for the rest of the
// pipeline: a stencil with no scheme/order gets scheme="central", order=2 and
// a `tpp.scheme.normalized` marker; a time.step gets its scheme validated and
// expanded into stages/order/dt.

func.func @grad_defaults(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %y = "tpp.grad"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: order = 2
  // CHECK-SAME: scheme = "central"
  // CHECK-SAME: tpp.scheme.normalized
  return %y : tensor<32x32xf32>
}

func.func @rk4_step(%s: tensor<8x8xf32>, %h: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = "tpp.time.step"(%s) ({
    %g = "tpp.grad"(%h) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  }) { scheme = "RK4", dt = 300.0 : f32 } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK: tpp.time.step
  // CHECK-SAME: tpp.time.dt = 3.000000e+02
  // CHECK-SAME: tpp.time.order = 4
  // CHECK-SAME: tpp.time.scheme = "RK4"
  // CHECK-SAME: tpp.time.stages = 4
  return %r : tensor<8x8xf32>
}
