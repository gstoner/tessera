// RUN: ts-ebm-opt --tessera-ebm-fuse-energy-grad %s | FileCheck %s
//
// FuseEnergyGrad recognizes the canonical pattern:
//   %e = ebm.energy(x, y, energy_fn = @f)
//   %step = ebm.langevin_step(y, key, energy_fn = @f, ...)
// where both reference the same y operand and the same energy_fn
// symbol, and marks both ops with `tessera.ebm.energy_grad_fused`
// plus a `fused_with_symbol` cross-link.

module {
  func.func @energy_then_step(
      %x : tensor<8x4xf32>,
      %y : tensor<8x4xf32>,
      %key : tensor<2xi64>) -> tensor<8x4xf32> {
    %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_E }
        : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8xf32>
    %step:2 = "tessera_ebm.langevin_step"(%y, %key)
        { energy_fn = @user_E,
          eta = 0.05 : f64,
          temperature = 1.0 : f64,
          manifold = "euclidean" }
        : (tensor<8x4xf32>, tensor<2xi64>) -> (tensor<8x4xf32>, tensor<2xi64>)
    return %step#0 : tensor<8x4xf32>
  }

  func.func private @user_E(
      %x : tensor<8x4xf32>, %y : tensor<8x4xf32>) -> tensor<8xf32>
}

// Both ops should carry the fusion markers.
// CHECK: tessera_ebm.energy
// CHECK-SAME: tessera.ebm.energy_grad_fused
// CHECK-SAME: tessera.ebm.fused_with_symbol = @user_E
// CHECK: tessera_ebm.langevin_step
// CHECK-SAME: tessera.ebm.energy_grad_fused
// CHECK-SAME: tessera.ebm.fused_with_symbol = @user_E
