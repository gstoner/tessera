// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file --verify-diagnostics %s

// ROCM-FP8-BLOCKSCALE-1. An fp32-scale block-scaled fp8 matmul on gfx1201 IS
// the W8A8 contract, so a nonconforming one is refused by name -- never left
// as an unbound directive, and never scheduled as a plain fp8 GEMM. Scale
// layout, dtype and group are semantic keys (Decision #21a).

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_lhs_scale(%a: tensor<32x256xf8E4M3FN>, %b: tensor<256x64xf8E4M3FN>,
                             %sa: tensor<16x2xf32>, %sb: tensor<2x1xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: lhs_scale must use the declared fp32/E8M0 storage [M, K/scale_k] = [32, 2]}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x256xf8E4M3FN>, tensor<256x64xf8E4M3FN>, tensor<16x2xf32>,
         tensor<2x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// A per-column rhs scale under a 128-column block is the wrong block count.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_rhs_scale(%a: tensor<32x256xf8E4M3FN>, %b: tensor<256x64xf8E4M3FN>,
                             %sa: tensor<32x2xf32>, %sb: tensor<2x64xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: rhs_scale must use the declared fp32/E8M0 storage [K/scale_k, ceil(N/scale_n)] = [2, 1]}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x256xf8E4M3FN>, tensor<256x64xf8E4M3FN>, tensor<32x2xf32>,
         tensor<2x64xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// A trailing partial group has no defined scale.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @ragged_group(%a: tensor<32x192xf8E4M3FN>, %b: tensor<192x64xf8E4M3FN>,
                          %sa: tensor<32x2xf32>, %sb: tensor<2x1xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: K=192 is not a whole number of scale groups of 128}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x192xf8E4M3FN>, tensor<192x64xf8E4M3FN>, tensor<32x2xf32>,
         tensor<2x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// v1 binds e4m3 x e4m3 only; a mixed pair is refused, not rebound.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @mixed_pair(%a: tensor<32x128xf8E4M3FN>, %b: tensor<128x64xf8E5M2>,
                        %sa: tensor<32x1xf32>, %sb: tensor<1x1xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: the W8A8 block-scale contract binds e4m3 x e4m3 only}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x128xf8E4M3FN>, tensor<128x64xf8E5M2>, tensor<32x1xf32>,
         tensor<1x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// An approximate execution mode is not this contract's exact per-block scaling.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @approximate(%a: tensor<32x128xf8E4M3FN>, %b: tensor<128x64xf8E4M3FN>,
                         %sa: tensor<32x1xf32>, %sb: tensor<1x1xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: execution_mode="folded_row_reference_explicit_approximate" is not this contract's exact per-block scaling}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<32x1xf32>,
         tensor<1x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// The execution mode is a semantic key (Decision #21a): an op that states no
// numeric_policy at all is refused, never read as exact per-block scaling.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @unstated_mode(%a: tensor<32x128xf8E4M3FN>, %b: tensor<128x64xf8E4M3FN>,
                           %sa: tensor<32x1xf32>, %sb: tensor<1x1xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: numeric_policy.execution_mode must state "exact_per_block"; the scaling mode is never defaulted}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<32x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>, tensor<32x1xf32>,
         tensor<1x1xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// Transposed MXFP8 is supported, but its E8M0 scales must be encoded i8;
// floating point scale containers cannot be reinterpreted as exponent bytes.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @wrong_mx_scale_storage(%a: tensor<32x128xf8E4M3FN>, %b: tensor<64x128xf8E4M3FN>,
                           %sa: tensor<32x4xf32>, %sb: tensor<4x64xf32>) -> tensor<32x64xf32> {
    // expected-error @+2 {{ROCM_FP8_BLOCKSCALE_CONTRACT: lhs_scale must use the declared fp32/E8M0 storage [M, K/scale_k] = [32, 4]}}
    // expected-error @+1 {{E2E-REAL-2 Graph->Schedule requires}}
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"},
      transposeB = true
    } : (tensor<32x128xf8E4M3FN>, tensor<64x128xf8E4M3FN>, tensor<32x4xf32>,
         tensor<4x64xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}
