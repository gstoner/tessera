// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=CARRIER
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE

// The semantic K=32 scale group and the independently selected K=32 macro
// block are content-addressed Schedule decisions. Tile carries both scale
// planes and the exact partial-accumulator order; it must never silently become
// an unscaled FP8 GEMM.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @scaled_fp8(%a: tensor<64x128xf8E4M3FN>,
                        %b: tensor<128x64xf8E4M3FN>,
                        %sa: tensor<64x4xf32>,
                        %sb: tensor<4x64xf32>) -> tensor<64x64xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [64, 32], format = "e8m0"}
    } : (tensor<64x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>,
         tensor<64x4xf32>, tensor<4x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  // A larger, valid Graph scale group must enlarge macro K before Schedule
  // verification. Leaving the measured unscaled block_k=32 here constructs an
  // invalid carrier because 32 cannot contain one complete K128 scale group.
  func.func @scaled_fp8_k128(%a: tensor<128x512xf8E4M3FN>,
                             %b: tensor<512x128xf8E4M3FN>,
                             %sa: tensor<128x4xf32>,
                             %sb: tensor<4x128xf32>) -> tensor<128x128xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>,
         tensor<128x4xf32>, tensor<4x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }

  // The packed W4A8 form stays in ui8 physical containers below the public
  // dtype layer. Its named contract is what permits the schedule to interpret
  // A as raw E4M3 bytes and B as two low-even/high-odd E2M1 codes per byte.
  func.func @scaled_mxfp4_w4a8(%a: tensor<17x64xui8>,
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

// CARRIER: tessera.scaled_matmul
// CARRIER-SAME: schedule.artifact_hash = "[[HASH:[0-9a-f]{64}]]"
// CARRIER: schedule.matmul
// CARRIER-SAME: artifact_hash = "[[HASH]]"
// CARRIER-SAME: block_k = 32
// CARRIER-SAME: scale_format = "e8m0"
// CARRIER-SAME: scale_k = 32
// CARRIER-SAME: storage = "e4m3"
// CARRIER-LABEL: func.func @scaled_fp8_k128
// CARRIER: schedule.matmul
// CARRIER-SAME: block_k = 128
// CARRIER-SAME: scale_format = "fp32"
// CARRIER-SAME: scale_k = 128
// CARRIER-LABEL: func.func @scaled_mxfp4_w4a8
// CARRIER: schedule.matmul
// CARRIER-SAME: physical_contract = "rocm_mxfp4_w4a8_exact_v1"
// CARRIER-SAME: storage = "e4m3_raw_u8"
// CARRIER-SAME: storage_b = "e2m1_packed_u8"
// TILE-LABEL: func.func @scaled_fp8
// TILE-NOT: tessera.scaled_matmul
// TILE-NOT: schedule.
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: scale_k = 32
// TILE-SAME: scale_fmt = "e8m0"
// TILE-SAME: partial_accumulator = {combine = "scale_outer_product_then_add", init = "zero", instruction_steps = 2 : i64, scope = "scale_group"}
// TILE-LABEL: func.func @scaled_fp8_k128
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: scale_k = 128
// TILE-SAME: scale_fmt = "fp32"
// TILE-SAME: instruction_steps = 8 : i64
// TILE-LABEL: func.func @scaled_mxfp4_w4a8
// TILE: tile.scaled_matmul_kernel
// TILE-SAME: epilogue = #tile.epilogue<bias = false, activation = "none", output = "bf16">
// TILE-SAME: scale_k = 32
// TILE-SAME: physical_contract = "rocm_mxfp4_w4a8_exact_v1"
