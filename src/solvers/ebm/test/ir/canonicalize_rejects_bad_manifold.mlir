// RUN: not ts-ebm-opt --tessera-ebm-canonicalize %s 2>&1 | FileCheck %s
//
// Decision #21a — `manifold` is a SEMANTIC key and must fail closed.
//
// W0.1: this pass previously emitted a *warning* and defaulted a missing
// `manifold` to "euclidean". That is the worst available behavior: a
// Euclidean integrator applied to spherical or bivector state does not
// diverge or error, it converges and reports a confidently wrong answer.
//
// `EBM_ManifoldAttr` in EBMOps.td now pins the legal set, so an unrecognized
// value is rejected by the op verifier before any pass runs. The pass keeps a
// matching fail-closed check for programmatically built ops that never go
// through the parser.

module {
  func.func @unknown_manifold(
      %x : tensor<3xf32>,
      %key : tensor<2xi64>) -> tensor<3xf32> {
    // CHECK: error: 'tessera_ebm.langevin_step' op attribute 'manifold' failed to satisfy constraint
    // CHECK-SAME: 'euclidean', 'sphere', or 'bivector'
    %step:2 = "tessera_ebm.langevin_step"(%x, %key)
        { energy_fn = @user_energy_fn,
          eta = 0.005 : f64,
          temperature = 1.0 : f64,
          manifold = "banana" }
        : (tensor<3xf32>, tensor<2xi64>) -> (tensor<3xf32>, tensor<2xi64>)
    return %step#0 : tensor<3xf32>
  }

  func.func private @user_energy_fn(
      %x : tensor<3xf32>,
      %y : tensor<3xf32>) -> f32
}

// The old behavior would have produced a canonicalized module with a
// silently-substituted euclidean manifold. Prove no such attribute is emitted.
// CHECK-NOT: tessera.ebm.manifold = "euclidean"
