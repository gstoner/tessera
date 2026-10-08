// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-graph-to-schedule | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt %s --tessera-graph-to-schedule --tessera-schedule-to-tile | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-rocm-executable{family=matmul input=tile output=target arch=gfx1201})' | FileCheck %s --check-prefix=TARGET
// RUN: %if rocm-device-libs %{ tessera-opt %s --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201})' | FileCheck %s --check-prefix=BINARY %}
// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: block_k = 32
// SCHEDULE-SAME: physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1"
// SCHEDULE-SAME: scale_format = "e8m0"
// SCHEDULE-SAME: scale_k = 32
// SCHEDULE-SAME: scale_n = 1
// SCHEDULE: shape_key = "M=7;N=19;K=64;dtype=e4m3"
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: batch_count = 3
// TILE-SAME: batching = "independent_rhs"
// TILE-SAME: physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_v1"
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: batch_count = 3
// TARGET-SAME: batching = "independent_rhs"
// TARGET-SAME: package_abi = "tessera.rocm.mxfp8_e4m3_e8m0_k32.a_b_sa_sb_o_m_n_k.f32.wide_scale.v1"
// TARGET-SAME: scale_format = "e8m0"
// BINARY: gpu.binary
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @mxfp8_independent_rhs(%a: tensor<3x7x64xf8E4M3FN>, %b: tensor<3x64x19xf8E4M3FN>,
                     %sa: tensor<3x7x2xui8>, %sb: tensor<3x2x19xui8>) -> tensor<3x7x19xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      batching = "independent_rhs",
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"}
    } : (tensor<3x7x64xf8E4M3FN>, tensor<3x64x19xf8E4M3FN>, tensor<3x7x2xui8>,
         tensor<3x2x19xui8>) -> tensor<3x7x19xf32>
    return %0 : tensor<3x7x19xf32>
  }
}
