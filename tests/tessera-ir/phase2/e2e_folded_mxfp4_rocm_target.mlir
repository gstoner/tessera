// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET

// The folded physical containers are not the exact K32 packed ABI. B is
// already load-time-converted E4M3 [N,K], and one E8M0 Ref[N] is applied
// after the full FP32 reduction. K64 is the physical LDS producer step.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @folded_w4a8(%a: tensor<65x64xui8>,
                         %b: tensor<48x64xui8>,
                         %sa: tensor<65xf32>,
                         %ref: tensor<48xui8>) -> tensor<65x48xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 64], format = "e8m0_row_reference"}
    } : (tensor<65x64xui8>, tensor<48x64xui8>, tensor<65xf32>,
         tensor<48xui8>) -> tensor<65x48xbf16>
    return %0 : tensor<65x48xbf16>
  }
}

// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: block_k = 64
// SCHEDULE-SAME: physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1"
// SCHEDULE-SAME: scale_format = "e8m0_row_reference"
// SCHEDULE-SAME: scale_k = 64
// SCHEDULE-SAME: storage_b = "e4m3_folded_nk_u8"
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: partial_accumulator = {combine = "row_reference_after_full_k"
// TILE-SAME: physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1"
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: abi = "a_bfold_sa_rowref_d_m_n_k"
// TARGET-SAME: block_m = 256
// TARGET-SAME: block_n = 64
// TARGET-SAME: k_step_schedule = "isolated_k_stage"
// TARGET-SAME: package_abi = "tessera.rocm.mxfp4_w4a8.a_bfold_sa_rowref_o_m_n_k.e4m3_e4m3_e8m0_bf16.approx_bm256_tm4.v1"
// TARGET-SAME: partial_combine = "row_reference_after_full_k"
// TARGET-SAME: physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1"
// TARGET-SAME: scale_k = 64
// TARGET-SAME: stage_k = 64
