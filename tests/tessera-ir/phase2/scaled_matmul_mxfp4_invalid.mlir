// RUN: not tessera-opt %s 2>&1 | FileCheck %s

// The proved gfx1201 ABI is exact-per-K32. A folded-row request must not borrow
// that ABI or its device proof; the approximate route needs its own package and
// error-bearing policy carrier.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @approximate_cannot_select_exact_abi(
      %a: tensor<17x64xui8>, %b: tensor<32x19xui8>,
      %sa: tensor<17xf32>, %sb: tensor<2x19xui8>) -> tensor<17x19xbf16> {
    // CHECK: 'tessera.scaled_matmul' op rocm_mxfp4_w4a8_exact_v1 requires numeric_policy execution_mode=exact_per_block
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      physical_contract = "rocm_mxfp4_w4a8_exact_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xui8>, tensor<32x19xui8>, tensor<17xf32>,
         tensor<2x19xui8>) -> tensor<17x19xbf16>
    return %0 : tensor<17x19xbf16>
  }
}
