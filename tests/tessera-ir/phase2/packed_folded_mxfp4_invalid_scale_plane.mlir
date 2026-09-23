// REQUIRES: tessera-rocm-backend
// RUN: not tessera-opt --tessera-graph-to-schedule %s 2>&1 | FileCheck %s

// A packed folded scale plane needs its final row-reference row. Two K32
// groups at K64 therefore require [3,N], not [2,N].
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @missing_reference(%a: tensor<65x64xui8>,
                               %b: tensor<48x32xui8>,
                               %sa: tensor<65xf32>,
                               %plane: tensor<2x48xui8>) -> tensor<65x48xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %plane) {
      physical_contract = "rocm_mxfp4_w4a8_packed_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 64], format = "e8m0_k32_plus_row_reference"}
    } : (tensor<65x64xui8>, tensor<48x32xui8>, tensor<65xf32>,
         tensor<2x48xui8>) -> tensor<65x48xbf16>
    return %0 : tensor<65x48xbf16>
  }
}

// CHECK: 'tessera.scaled_matmul' op folded prefill requires fp32 A scale [M] and versioned E8M0 row reference or K32+reference plane
