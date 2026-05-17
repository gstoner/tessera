// RUN: ts-ebm-opt --tessera-ebm-checkpoint-inner-loop %s | FileCheck %s
//
// CheckpointInnerLoop walks every scf.for whose body contains an
// ebm.langevin_step or ebm.inner_step and attaches:
//   tessera.ebm.checkpoint_loop  on the for op,
//   tessera.ebm.checkpoint_budget = 4 (default) on the for op,
//   tessera.ebm.recompute_step on each inner step op.

module {
  func.func @T_step_chain(
      %y0 : tensor<8x4xf32>,
      %key0 : !ebm.rngkey) -> tensor<8x4xf32> {
    %T = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %final:2 = scf.for %t = %c0 to %T step %c1
        iter_args(%y = %y0, %key = %key0) -> (tensor<8x4xf32>, !ebm.rngkey) {
      %step:2 = "tessera_ebm.langevin_step"(%y, %key)
          { energy_fn = @user_E,
            eta = 0.05 : f64,
            temperature = 1.0 : f64,
            manifold = "euclidean" }
          : (tensor<8x4xf32>, !ebm.rngkey) -> (tensor<8x4xf32>, !ebm.rngkey)
      scf.yield %step#0, %step#1 : tensor<8x4xf32>, !ebm.rngkey
    }
    return %final#0 : tensor<8x4xf32>
  }
  func.func private @user_E(
      %x : tensor<8x4xf32>, %y : tensor<8x4xf32>) -> tensor<8xf32>
}

// CHECK: scf.for
// CHECK-SAME: tessera.ebm.checkpoint_loop
// CHECK-SAME: tessera.ebm.checkpoint_budget = 4
// CHECK: tessera_ebm.langevin_step
// CHECK-SAME: tessera.ebm.recompute_step
