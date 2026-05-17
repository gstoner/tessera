// RUN: ts-ebm-opt --tessera-ebm-checkpoint-inner-loop="budget=2" %s | FileCheck %s
//
// The `budget` pass option overrides the default checkpoint budget (4).
// Useful for very-tight-memory backends.

module {
  func.func @small_budget_loop(
      %y0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
    %T = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %grad = arith.constant dense<0.0> : tensor<2x3xf32>
    %final = scf.for %t = %c0 to %T step %c1
        iter_args(%y = %y0) -> tensor<2x3xf32> {
      %y_new = "tessera_ebm.inner_step"(%y, %grad)
          { energy_fn = @user_E, eta = 0.1 : f64 }
          : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
      scf.yield %y_new : tensor<2x3xf32>
    }
    return %final : tensor<2x3xf32>
  }
  func.func private @user_E(
      %x : tensor<2x3xf32>, %y : tensor<2x3xf32>) -> tensor<2xf32>
}

// CHECK: scf.for
// CHECK-SAME: tessera.ebm.checkpoint_budget = 2
