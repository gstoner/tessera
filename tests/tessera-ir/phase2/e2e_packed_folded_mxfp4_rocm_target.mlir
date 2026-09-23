// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET

// Artifact-only physical contract: fragment-order packed E2M1 B [N,K/2],
// original K32 E8M0 scales followed by one row-reference [K/32+1,N].
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @packed_folded_w4a8(%a: tensor<65x64xui8>,
                                %b: tensor<48x32xui8>,
                                %sa: tensor<65xf32>,
                                %plane: tensor<3x48xui8>) -> tensor<65x48xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %plane) {
      physical_contract = "rocm_mxfp4_w4a8_packed_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 64], format = "e8m0_k32_plus_row_reference"}
    } : (tensor<65x64xui8>, tensor<48x32xui8>, tensor<65xf32>,
         tensor<3x48xui8>) -> tensor<65x48xbf16>
    return %0 : tensor<65x48xbf16>
  }
}

// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: physical_contract = "rocm_mxfp4_w4a8_packed_folded_prefill_v1"
// SCHEDULE-SAME: scale_format = "e8m0_k32_plus_row_reference"
// SCHEDULE-SAME: storage_b = "e2m1_fragment_nk2_u8"
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: physical_contract = "rocm_mxfp4_w4a8_packed_folded_prefill_v1"
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: abi = "a_bpacked_sa_scaleplane_d_m_n_k"
// TARGET-SAME: package_abi = "tessera.rocm.mxfp4_w4a8.a_bpacked_sa_scaleplane_o_m_n_k.e4m3_e2m1_e8m0_bf16.approx_bm256_tm4.v1"
// TARGET-SAME: physical_contract = "rocm_mxfp4_w4a8_packed_folded_prefill_v1"
