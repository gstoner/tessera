// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=GEN

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @logical_w8a8(%a: tensor<64x128xf8E4M3FN>,
                           %b: tensor<128x64xf8E4M3FN>,
                           %sa: tensor<64x4xi8>,
                           %sb: tensor<4x64xi8>) -> tensor<64x64xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>,
         tensor<64x4xi8>, tensor<4x64xi8>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  func.func @packed_w4a8(%a: tensor<17x64xui8>,
                          %b: tensor<32x19xui8>,
                          %sa: tensor<17xf32>,
                          %sb: tensor<2x19xui8>) -> tensor<17x19xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      physical_contract = "rocm_mxfp4_w4a8_exact_v1",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<17x64xui8>, tensor<32x19xui8>, tensor<17xf32>,
         tensor<2x19xui8>) -> tensor<17x19xbf16>
    return %0 : tensor<17x19xbf16>
  }
}

// The Target directive carries the exact W4A8 and MXFP8 contracts.
// TARGET-LABEL: func.func @logical_w8a8
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1"
// TARGET-LABEL: func.func @packed_w4a8
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: k = 64
// TARGET-SAME: m = 17
// TARGET-SAME: n = 19
// TARGET-SAME: numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3_raw_u8"}
// TARGET-SAME: output = "bf16"
// TARGET-SAME: package_abi = "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k.e4m3_e2m1_e8m0_bf16.wmma_exact.v1"
// TARGET-SAME: physical_contract = "rocm_mxfp4_w4a8_exact_v1"
// TARGET-SAME: scale_k = 32

// The kernel generator expands both directives (exact W4A8 since 51d52a938,
// 2026-10-14); the runtime launches the generated kernels by descriptor.
// GEN-NOT: tessera_rocm.scaled_wmma_gemm
// GEN-LABEL: gpu.func @packed_w4a8
// GEN-SAME: tessera.rocm.block_scale_contract = "rocm_mxfp4_w4a8_exact_v1"
// GEN-SAME: tessera.rocm.exact_mxfp4
// GEN: tessera_rocm.wmma
// GEN-NOT: tessera_rocm.scaled_wmma_gemm
// GEN-LABEL: gpu.func @logical_w8a8
// GEN-SAME: memref<?xi8>
// GEN-SAME: tessera.rocm.block_scale_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1"
// GEN: llvm.intr.ldexp
