// RUN: tessera-opt --tessera-pipeline-partition --allow-unregistered-dialect %s | FileCheck %s
//
// Real stage partitioning (2026-06-23): PipelineStagePartitionPass assigns each
// function-body op to one of num_stages pipeline stages with a cost-balanced,
// program-order-monotonic partition (emits tessera.pp_stage). This is the "true
// stage partitioning" the insertion pass previously required an external tagger
// for. Two equal-cost matmuls over 2 stages → stage 0 then stage 1.

module attributes {
  tessera.pipeline_plan = {num_stages = 2, num_micro_batches = 2, interleaved = false}
} {
  // CHECK-LABEL: func.func @pipeline
  func.func @pipeline(%x: tensor<64x128xbf16>, %w0: tensor<128x256xbf16>,
                      %w1: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: tessera.matmul
    // CHECK-SAME: tessera.pp_stage = 0
    %a = "tessera.matmul"(%x, %w0) :
        (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
    // CHECK: tessera.matmul
    // CHECK-SAME: tessera.pp_stage = 1
    %b = "tessera.matmul"(%a, %w1) :
        (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
    return %b : tensor<64x128xbf16>
  }
}
