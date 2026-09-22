// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s

module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @logical_w8a8(%a: tensor<64x128xf8E4M3FN>,
                           %b: tensor<128x64xf8E4M3FN>,
                           %sa: tensor<64x4xf32>,
                           %sb: tensor<4x64xf32>) -> tensor<64x64xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [64, 32], format = "e8m0"}
    } : (tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>,
         tensor<64x4xf32>, tensor<4x64xf32>) -> tensor<64x64xf32>
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

// CHECK-LABEL: func.func @logical_w8a8
// CHECK-NOT: tile.scaled_matmul_kernel
// CHECK: tessera_rocm.scaled_wmma_gemm
// CHECK-SAME: abi = "a_b_lhs_scale_rhs_scale_d_m_n_k"
// CHECK-SAME: instruction_k = 16
// CHECK-SAME: macro_k = 32
// CHECK-SAME: package_abi = "unbound"
// CHECK-SAME: partial_combine = "scale_outer_product_then_add"
// CHECK-SAME: physical_contract = "logical_block_scaled"
// CHECK-SAME: scale_format = "e8m0"
// CHECK-SAME: scale_k = 32
// CHECK-LABEL: func.func @packed_w4a8
// CHECK: tessera_rocm.scaled_wmma_gemm
// CHECK-SAME: output = "bf16"
// CHECK-SAME: package_abi = "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k.e4m3_e2m1_e8m0_bf16.wmma_exact.v1"
// CHECK-SAME: physical_contract = "rocm_mxfp4_w4a8_exact_v1"
// CHECK-SAME: scale_k = 32
