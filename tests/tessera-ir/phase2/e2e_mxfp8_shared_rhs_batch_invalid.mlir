// RUN: tessera-opt %s --split-input-file --verify-diagnostics
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @invalid_scale(%a: tensor<3x7x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<3x8x2xui8>, %sb: tensor<2x19xui8>) -> tensor<3x7x19xf32> {
    // expected-error @+1 {{typed shared-RHS batch scale storage/extents differ}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_rhs_rows",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<3x7x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<3x8x2xui8>,
         tensor<2x19xui8>) -> tensor<3x7x19xf32>
    return %0 : tensor<3x7x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @invalid_result(%a: tensor<3x7x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<3x7x2xui8>, %sb: tensor<2x19xui8>) -> tensor<3x8x19xf32> {
    // expected-error @+1 {{typed shared-RHS batch matrix/result extents differ or overflow}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_rhs_rows",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<3x7x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<3x7x2xui8>,
         tensor<2x19xui8>) -> tensor<3x8x19xf32>
    return %0 : tensor<3x8x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @invalid_transpose(%a: tensor<3x7x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<3x7x2xui8>, %sb: tensor<2x19xui8>) -> tensor<3x7x19xf32> {
    // expected-error @+1 {{typed shared-RHS batches require static E4M3}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_rhs_rows", transposeA = true,
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<3x7x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<3x7x2xui8>,
         tensor<2x19xui8>) -> tensor<3x7x19xf32>
    return %0 : tensor<3x7x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @invalid_signed(%a: tensor<3x7x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<3x7x2xsi8>, %sb: tensor<2x19xui8>) -> tensor<3x7x19xf32> {
    // expected-error @+1 {{typed shared-RHS batch scale storage/extents differ}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "shared_rhs_rows",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<3x7x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<3x7x2xsi8>,
         tensor<2x19xui8>) -> tensor<3x7x19xf32>
    return %0 : tensor<3x7x19xf32>
  }
}
