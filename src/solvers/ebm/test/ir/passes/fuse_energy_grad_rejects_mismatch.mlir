// RUN: ts-ebm-opt --tessera-ebm-fuse-energy-grad %s | FileCheck %s
//
// FuseEnergyGrad refuses to fuse when the step's y operand differs
// from the energy's y, OR when the energy_fn symbols differ.  Neither
// op carries the `tessera.ebm.energy_grad_fused` marker in that case.

module {
  func.func @mismatched(
      %x : tensor<8x4xf32>,
      %y : tensor<8x4xf32>,
      %z : tensor<8x4xf32>,
      %key : !ebm.rngkey) -> tensor<8x4xf32> {
    // Energy is evaluated on y but the step runs on z — no fusion.
    %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_E }
        : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8xf32>
    %step:2 = "tessera_ebm.langevin_step"(%z, %key)
        { energy_fn = @user_E,
          eta = 0.05 : f64,
          temperature = 1.0 : f64,
          manifold = "euclidean" }
        : (tensor<8x4xf32>, !ebm.rngkey) -> (tensor<8x4xf32>, !ebm.rngkey)
    return %step#0 : tensor<8x4xf32>
  }
  func.func private @user_E(
      %x : tensor<8x4xf32>, %y : tensor<8x4xf32>) -> tensor<8xf32>
}

// Neither op gets the fusion marker.
// CHECK-NOT: tessera.ebm.energy_grad_fused
