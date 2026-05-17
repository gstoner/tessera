// RUN: ts-ebm-opt --tessera-ebm-pipeline %s | FileCheck %s
//
// End-to-end EBM pipeline alias:
//   canonicalize → fuse-energy-grad → checkpoint-inner-loop →
//   pipeline-candidates.
//
// An EBT-style chain (decode_init + inner-loop langevin + self_verify)
// emerges from this pipeline with every relevant op annotated.

module {
  func.func @ebt_chain(
      %x : tensor<2x16xf32>,
      %key0 : !ebm.rngkey) -> tensor<2x16xf32> {
    // Initialize K=4 candidates.
    %cands:2 = "tessera_ebm.decode_init"(%x, %key0)
        { K = 4 : i64, init_strategy = "noise", shape = [16] }
        : (tensor<2x16xf32>, !ebm.rngkey) -> (tensor<2x4x16xf32>, !ebm.rngkey)

    // T-step inner loop on each candidate.
    %T = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %final:2 = scf.for %t = %c0 to %T step %c1
        iter_args(%y = %cands#0, %k = %cands#1)
        -> (tensor<2x4x16xf32>, !ebm.rngkey) {
      %e = "tessera_ebm.energy"(%x, %y) { energy_fn = @user_E }
          : (tensor<2x16xf32>, tensor<2x4x16xf32>) -> tensor<2x4xf32>
      %step:2 = "tessera_ebm.langevin_step"(%y, %k)
          { energy_fn = @user_E,
            eta = 0.05 : f64,
            temperature = 1.0 : f64,
            manifold = "euclidean" }
          : (tensor<2x4x16xf32>, !ebm.rngkey) -> (tensor<2x4x16xf32>, !ebm.rngkey)
      scf.yield %step#0, %step#1 : tensor<2x4x16xf32>, !ebm.rngkey
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

// Canonicalize: every ebm op gets a canonical marker.
// CHECK: tessera.ebm.canonical
// Fuse-energy-grad: the energy + langevin_step share energy_fn.
// CHECK: tessera.ebm.energy_grad_fused
// Checkpoint-inner-loop: the scf.for got annotations.
// CHECK: tessera.ebm.checkpoint_loop
// CHECK: tessera.ebm.checkpoint_budget
// Pipeline-candidates: decode_init + self_verify linked with K=4.
// CHECK: tessera.ebm.pipeline_K = 4
// CHECK: tessera.ebm.pipeline_axis = "k"
