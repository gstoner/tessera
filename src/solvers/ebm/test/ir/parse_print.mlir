// RUN: ts-ebm-opt %s | FileCheck %s
//
// Parse + print round-trip for the EBM dialect surface.

module {
  func.func @energy_chain(
      %x : tensor<32x4xf32>,
      %y : tensor<32x4xf32>,
      %key : !ebm.rngkey) -> tensor<32x4xf32> {

    %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_energy_fn }
        : (tensor<32x4xf32>, tensor<32x4xf32>) -> tensor<32xf32>

    %step:2 = "tessera_ebm.langevin_step"(%y, %key)
        { energy_fn = @user_energy_fn,
          eta = 0.05 : f64,
          temperature = 1.0 : f64,
          manifold = "euclidean" }
        : (tensor<32x4xf32>, !ebm.rngkey) -> (tensor<32x4xf32>, !ebm.rngkey)

    return %step#0 : tensor<32x4xf32>
  }

  func.func private @user_energy_fn(
      %x : tensor<32x4xf32>, %y : tensor<32x4xf32>) -> tensor<32xf32>
}

// CHECK: tessera_ebm.energy
// CHECK: tessera_ebm.langevin_step
// CHECK-SAME: manifold = "euclidean"
