// RUN: ts-ebm-opt --tessera-ebm-pipeline %s | FileCheck %s
//
// End-to-end EBM pipeline alias:
//   canonicalize → fuse-energy-grad → pipeline-candidates.
//
// An EBT-style chain (decode_init + inner-loop langevin + self_verify)
// emerges from this pipeline with every relevant op annotated.

module {
  func.func @ebt_chain(
      %x : tensor<2x16xf32>,
      %key0 : tensor<2xi64>) -> tensor<2x16xf32> {
    // Initialize K=4 candidates.
    %cands:2 = "tessera_ebm.decode_init"(%x, %key0)
        { K = 4 : i64, init_strategy = "noise", shape = [16] }
        : (tensor<2x16xf32>, tensor<2xi64>) -> (tensor<2x4x16xf32>, tensor<2xi64>)

    // T-step inner loop on each candidate.
    %T = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %final:2 = scf.for %t = %c0 to %T step %c1
        iter_args(%y = %cands#0, %k = %cands#1)
        -> (tensor<2x4x16xf32>, tensor<2xi64>) {
      %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_E }
          : (tensor<2x16xf32>, tensor<2x4x16xf32>) -> tensor<2x4xf32>
      %step:2 = "tessera_ebm.langevin_step"(%y, %k)
          { energy_fn = @user_E,
            eta = 0.05 : f64,
            temperature = 1.0 : f64,
            manifold = "euclidean" }
          : (tensor<2x4x16xf32>, tensor<2xi64>) -> (tensor<2x4x16xf32>, tensor<2xi64>)
      scf.yield %step#0, %step#1 : tensor<2x4x16xf32>, tensor<2xi64>
    }

    // Final energies + self_verify.
    %final_E = "tessera_ebm.energy"(%x, %final#0) { energy_fn = @user_E }
        : (tensor<2x16xf32>, tensor<2x4x16xf32>) -> tensor<2x4xf32>
    %best = "tessera_ebm.self_verify"(%final_E, %final#0)
        : (tensor<2x4xf32>, tensor<2x4x16xf32>) -> tensor<2x16xf32>
    return %best : tensor<2x16xf32>
  }
  func.func private @user_E(
      %x : tensor<2x16xf32>, %y : tensor<2x4x16xf32>) -> tensor<2x4xf32>
}

// These markers are emitted by four independent passes and appear in source
// order, not pass order -- `pipeline_K` lands on `decode_init` (the first op)
// while `checkpoint_budget` lands on the loop below it. Sequential CHECKs
// therefore cannot match them; CHECK-DAG expresses the actual contract, which
// is "all of these are present", not "in this order".
//
// Canonicalize: every ebm op gets a canonical marker.
// CHECK-DAG: tessera.ebm.canonical
// Fuse-energy-grad: the energy + langevin_step share energy_fn.
// CHECK-DAG: tessera.ebm.energy_grad_fused
// W0.2 — checkpoint-inner-loop is NOT in the default pipeline (it marks every
// step rematerializable with no demand analysis, and nothing consumes the
// attributes). Decision #10a requires an eligibility pass to ship a fixture
// whose correct output is NO annotation; this is that fixture for the default
// pipeline.
// CHECK-NOT: tessera.ebm.checkpoint_loop
// CHECK-NOT: tessera.ebm.checkpoint_budget
// CHECK-NOT: tessera.ebm.recompute_step
// Pipeline-candidates: decode_init + self_verify linked with K=4.
// CHECK-DAG: tessera.ebm.pipeline_K = 4
// CHECK-DAG: tessera.ebm.pipeline_axis = "k"
