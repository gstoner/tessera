// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-graph-to-schedule --split-input-file --verify-diagnostics
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_storage(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xf32>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xf32>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_rhs_extent(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x18xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x18xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_scale_n(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [2, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_scale_k(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x4xi8>, %sb: tensor<4x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 16], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x4xi8>,
         tensor<4x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @approximate(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "approximate"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_granularity(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "per_tensor", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_accum(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp16", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @missing_accum(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}

// -----
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @signed_storage(%a: tensor<17x64xf8E4M3FN>, %b: tensor<64x19xf8E4M3FN>,
                     %sa: tensor<17x2xsi8>, %sb: tensor<2x19xi8>) -> tensor<17x19xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xf8E4M3FN>, tensor<64x19xf8E4M3FN>, tensor<17x2xsi8>,
         tensor<2x19xi8>) -> tensor<17x19xf32>
    return %0 : tensor<17x19xf32>
  }
}
