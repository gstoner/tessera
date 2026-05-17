// RUN: ts-ebm-opt --tessera-ebm-canonicalize %s | FileCheck %s
//
// EBMCanonicalizePass tags `tessera.ebm.canonical` on every EBM op and
// mirrors the manifold attribute up onto `tessera.ebm.manifold` for
// downstream EBM6 lowering passes.

module {
  func.func @sphere_chain(
      %x : tensor<3xf32>,
      %key : !ebm.rngkey) -> tensor<3xf32> {
    %step:2 = "tessera_ebm.langevin_step"(%x, %key)
        { energy_fn = @user_energy_fn,
          eta = 0.005 : f64,
          temperature = 1.0 : f64,
          manifold = "sphere" }
        : (tensor<3xf32>, !ebm.rngkey) -> (tensor<3xf32>, !ebm.rngkey)
    return %step#0 : tensor<3xf32>
  }

  func.func private @user_energy_fn(
      %y : tensor<3xf32>) -> f32
}

// CHECK: tessera_ebm.langevin_step
// CHECK-SAME: tessera.ebm.canonical
// CHECK-SAME: tessera.ebm.manifold = "sphere"
