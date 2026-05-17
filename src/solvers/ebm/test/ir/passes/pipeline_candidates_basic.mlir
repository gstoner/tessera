// RUN: ts-ebm-opt --tessera-ebm-pipeline-candidates %s | FileCheck %s
//
// PipelineCandidates walks self_verify ops, finds the producing
// decode_init (carrying the K candidate count), and attaches matching
// pipeline annotations on both ops so a backend can map the K axis
// across streams / devices.

module {
  func.func @K8_self_verify(
      %x : tensor<3x16xf32>,
      %key : !ebm.rngkey) -> tensor<3x16xf32> {
    %cands:2 = "tessera_ebm.decode_init"(%x, %key)
        { K = 8 : i64, init_strategy = "noise", shape = [16] }
        : (tensor<3x16xf32>, !ebm.rngkey) -> (tensor<3x8x16xf32>, !ebm.rngkey)
    %energies = arith.constant dense<0.0> : tensor<3x8xf32>
    %best = "tessera_ebm.self_verify"(%energies, %cands#0)
        : (tensor<3x8xf32>, tensor<3x8x16xf32>) -> tensor<3x16xf32>
    return %best : tensor<3x16xf32>
  }
}

// Both ops carry K=8, pipeline_axis="k", and the pipelined marker.
// CHECK: tessera_ebm.decode_init
// CHECK-SAME: tessera.ebm.pipeline_K = 8
// CHECK-SAME: tessera.ebm.pipeline_axis = "k"
// CHECK-SAME: tessera.ebm.pipelined
// CHECK: tessera_ebm.self_verify
// CHECK-SAME: tessera.ebm.pipeline_K = 8
// CHECK-SAME: tessera.ebm.pipeline_axis = "k"
// CHECK-SAME: tessera.ebm.pipelined
