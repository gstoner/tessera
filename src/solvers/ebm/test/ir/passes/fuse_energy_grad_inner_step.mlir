// RUN: ts-ebm-opt --tessera-ebm-fuse-energy-grad %s | FileCheck %s
//
// FuseEnergyGrad also works with `inner_step` (the manifold-free
// counterpart to `langevin_step`); both consume y as operand 0 and
// reference energy_fn the same way.

module {
  func.func @inner_step_chain(
      %x : tensor<32x4xf32>,
      %y : tensor<32x4xf32>,
      %grad : tensor<32x4xf32>) -> tensor<32x4xf32> {
    %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_E }
        : (tensor<32x4xf32>, tensor<32x4xf32>) -> tensor<32xf32>
    %y_new = "tessera_ebm.inner_step"(%y, %grad)
        { energy_fn = @user_E, eta = 0.1 : f64 }
        : (tensor<32x4xf32>, tensor<32x4xf32>) -> tensor<32x4xf32>
    return %y_new : tensor<32x4xf32>
  }
  func.func private @user_E(
      %x : tensor<32x4xf32>, %y : tensor<32x4xf32>) -> tensor<32xf32>
}

// CHECK: tessera_ebm.energy
// CHECK-SAME: tessera.ebm.energy_grad_fused
// CHECK: tessera_ebm.inner_step
// CHECK-SAME: tessera.ebm.energy_grad_fused
