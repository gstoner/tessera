// score_combine is the compiler-visible CGG algebra:
//   guided = base + gamma * delta
//
// RUN: %tessera_strict_opt %s -tessera-to-linalg | FileCheck %s

func.func @score_combine_lowering(
    %base: tensor<2x4xf32>, %delta: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK-LABEL: func.func @score_combine_lowering
  // CHECK-NOT: tessera.score_combine
  // CHECK: arith.constant 7.500000e-01
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %guided = "tessera.score_combine"(%base, %delta) {gamma = 7.500000e-01 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %guided : tensor<2x4xf32>
}
