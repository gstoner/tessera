// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s

// ROCM-FP8-BLOCKSCALE-1. K=192 is not a whole number of 128-wide scale groups.
// A plain rank-2 e4m3 scaled_matmul in that shape is the scalar-scaled-plane
// lane (needsScalarScaledPlane): it is scheduled under the W8A8 block-scale
// contract rather than refused. The former refusal fixture for this shape lives
// in fp8_blockscale_contract_invalid.mlir as a pointer to this file.

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  // CHECK-LABEL: func.func @ragged_group
  func.func @ragged_group(%a: tensor<32x192xf8E4M3FN>, %b: tensor<192x64xf8E4M3FN>,
                          %sa: tensor<32x2xf32>, %sb: tensor<2x1xf32>) -> tensor<32x64xf32> {
    // CHECK-NOT: ROCM_FP8_BLOCKSCALE_CONTRACT
    // CHECK: schedule.matmul
    // CHECK-SAME: physical_contract = "rocm_fp8_w8a8_blockscale_v1"
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x192xf8E4M3FN>, tensor<192x64xf8E4M3FN>, tensor<32x2xf32>,
         tensor<2x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}
