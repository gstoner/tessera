// RUN: ts-ebm-opt --tessera-ebm-pipeline-candidates %s | FileCheck %s
//
// When self_verify consumes externally-supplied candidates (no
// matching decode_init in the same function), the pass skips
// annotation — the backend can't infer K and the user is responsible
// for any pipelining.

module {
  func.func @external_candidates(
      %energies : tensor<2x4xf32>,
      %candidates : tensor<2x4x6xf32>) -> tensor<2x6xf32> {
    %best = "tessera_ebm.self_verify"(%energies, %candidates)
        : (tensor<2x4xf32>, tensor<2x4x6xf32>) -> tensor<2x6xf32>
    return %best : tensor<2x6xf32>
  }
}

// Without a paired decode_init, no pipeline annotations are attached.
// CHECK-NOT: tessera.ebm.pipeline_K
// CHECK-NOT: tessera.ebm.pipelined
