// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-graph-to-schedule | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt %s --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' | FileCheck %s --check-prefix=GEN
// RUN: tessera-opt %s --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' | FileCheck %s --check-prefix=LOWER
// SCHED: schedule.matmul
// SCHED-SAME: macro_tile_m = 128
// SCHED-SAME: macro_tile_n = 64
// SCHED-SAME: physical_contract = "rocm_mxfp8_e4m3_e8m0_k32_nk_v1"
// SCHED-SAME: scale_format = "e8m0"
// SCHED-SAME: staging = "lds"
// GEN: memref<?xi8>
// GEN: arith.minui
// GEN: vector.load
// GEN-SAME: vector<16xf8E4M3FN>
// GEN: gpu.barrier
// GEN: tile.fragment_scaled_accumulate
// GEN-SAME: scale_format = "e8m0"
// LOWER: llvm.intr.ldexp
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @mxfp8_lds(%a: tensor<200x1024xf8E4M3FN>, %b: tensor<2048x1024xf8E4M3FN>,
                       %sa: tensor<200x32xi8>, %sb: tensor<32x2048xi8>) -> tensor<200x2048xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "e8m0"},
      transposeB = true
    } : (tensor<200x1024xf8E4M3FN>, tensor<2048x1024xf8E4M3FN>,
         tensor<200x32xi8>, tensor<32x2048xi8>) -> tensor<200x2048xbf16>
    return %0 : tensor<200x2048xbf16>
  }
}
