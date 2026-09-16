// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// `tessera.sub` carried only a tangent rule until 2026-09-16, so any energy
// or loss written with a subtraction stopped reverse-mode at
// AUTODIFF_OP_NOT_DIFFERENTIABLE. The adjoint passes the cotangent to lhs and
// negates it (0 - dy on a static zero) for rhs.

module {
  func.func @quadratic(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %d = "tessera.sub"(%x, %y) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %sq = "tessera.mul"(%d, %d) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %s = "tessera.reduce"(%sq) {axis = 1 : i64, kind = "sum"} : (tensor<4x8xf32>) -> tensor<4xf32>
    %half = arith.constant dense<5.000000e-01> : tensor<4xf32>
    %e = "tessera.mul"(%s, %half) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %e : tensor<4xf32>
  }
  // CHECK-LABEL: func.func @quadratic__bwd
  // CHECK-SAME: (%[[Y:.*]]: tensor<4x8xf32>, %[[X:.*]]: tensor<4x8xf32>, %[[COT:.*]]: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
  // CHECK: %[[DY:.*]] = tessera.sub %[[ZERO]], %[[DX:.*]] : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK: return %[[DY]], %[[DX]]
}
