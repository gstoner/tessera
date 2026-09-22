// RUN: not tessera-opt %s 2>&1 | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_reference_shape(
      %a: tensor<65x64xui8>, %b: tensor<48x64xui8>,
      %sa: tensor<65xf32>, %ref: tensor<2x48xui8>) -> tensor<65x48xbf16> {
    // CHECK: folded prefill requires fp32 A scale [M] and E8M0 ui8 row reference [N]
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 64], format = "e8m0_row_reference"}
    } : (tensor<65x64xui8>, tensor<48x64xui8>, tensor<65xf32>,
         tensor<2x48xui8>) -> tensor<65x48xbf16>
    return %0 : tensor<65x48xbf16>
  }

  func.func @missing_approximate_consent(
      %a: tensor<65x64xui8>, %b: tensor<48x64xui8>,
      %sa: tensor<65xf32>, %ref: tensor<48xui8>) -> tensor<65x48xbf16> {
    // CHECK: folded prefill requires explicit approximate numeric_policy
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "output_column", block = [1, 64], format = "e8m0_row_reference"}
    } : (tensor<65x64xui8>, tensor<48x64xui8>, tensor<65xf32>,
         tensor<48xui8>) -> tensor<65x48xbf16>
    return %0 : tensor<65x48xbf16>
  }
}
