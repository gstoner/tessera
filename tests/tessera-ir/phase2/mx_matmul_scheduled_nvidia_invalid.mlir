// RUN: tessera-opt --split-input-file --verify-diagnostics %s

// E2E-REAL-6 SM120 MX (2026-11-04): the Graph contract fails closed on every
// semantic key -- scale extents, scale block/format, mixed storages, and
// any batch or transpose profile (MX has neither).

func.func @scale_extent(%a: tensor<16x64xf6E2M3FN>, %b: tensor<64x8xf6E2M3FN>,
                        %sa: tensor<16x3xui8>, %sb: tensor<2x8xui8>) -> tensor<16x8xf32> {
  // expected-error @+1 {{NVIDIA MX contract requires ui8 [M,ceil(K/32)] / [ceil(K/32),N] UE8M0 scales}}
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
    numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
    physical_contract = "nvidia_sm120_mx_blockscale_v1",
    scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"}
  } : (tensor<16x64xf6E2M3FN>, tensor<64x8xf6E2M3FN>, tensor<16x3xui8>,
       tensor<2x8xui8>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

func.func @k16_block(%a: tensor<16x64xf4E2M1FN>, %b: tensor<64x8xf4E2M1FN>,
                     %sa: tensor<16x4xui8>, %sb: tensor<4x8xui8>) -> tensor<16x8xf32> {
  // expected-error @+1 {{NVIDIA MX contract requires ui8 [M,ceil(K/32)] / [ceil(K/32),N] UE8M0 scales}}
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
    numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
    physical_contract = "nvidia_sm120_mx_blockscale_v1",
    scale_layout = {granularity = "block", block = [1, 16], format = "ue8m0"}
  } : (tensor<16x64xf4E2M1FN>, tensor<64x8xf4E2M1FN>, tensor<16x4xui8>,
       tensor<4x8xui8>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

func.func @ue4m3_format(%a: tensor<16x64xf6E3M2FN>, %b: tensor<64x8xf6E3M2FN>,
                        %sa: tensor<16x2xui8>, %sb: tensor<2x8xui8>) -> tensor<16x8xf32> {
  // expected-error @+1 {{NVIDIA MX contract requires ui8 [M,ceil(K/32)] / [ceil(K/32),N] UE8M0 scales}}
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
    numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
    physical_contract = "nvidia_sm120_mx_blockscale_v1",
    scale_layout = {granularity = "block", block = [1, 32], format = "ue4m3"}
  } : (tensor<16x64xf6E3M2FN>, tensor<64x8xf6E3M2FN>, tensor<16x2xui8>,
       tensor<2x8xui8>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

func.func @mixed_storage(%a: tensor<16x64xf6E2M3FN>, %b: tensor<64x8xf6E3M2FN>,
                         %sa: tensor<16x2xui8>, %sb: tensor<2x8xui8>) -> tensor<16x8xf32> {
  // expected-error @+1 {{NVIDIA MX contract requires static rank-2 row-major FP6/MXFP4 A/B of one element type and f32 output}}
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
    numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
    physical_contract = "nvidia_sm120_mx_blockscale_v1",
    scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"}
  } : (tensor<16x64xf6E2M3FN>, tensor<64x8xf6E3M2FN>, tensor<16x2xui8>,
       tensor<2x8xui8>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

func.func @transposed(%a: tensor<64x16xf6E2M3FN>, %b: tensor<64x8xf6E2M3FN>,
                      %sa: tensor<2x16xui8>, %sb: tensor<2x8xui8>) -> tensor<16x8xf32> {
  // expected-error @+1 {{NVIDIA MX contract requires static rank-2 row-major FP6/MXFP4 A/B of one element type and f32 output}}
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
    numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
    physical_contract = "nvidia_sm120_mx_blockscale_v1",
    scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"},
    transposeA = true
  } : (tensor<64x16xf6E2M3FN>, tensor<64x8xf6E2M3FN>, tensor<2x16xui8>,
       tensor<2x8xui8>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
