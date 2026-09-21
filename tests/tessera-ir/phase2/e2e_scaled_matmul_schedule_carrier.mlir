// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=CARRIER
// RUN: not tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s 2>&1 | FileCheck %s --check-prefix=BOUNDARY

// The semantic K=32 scale group and the independently selected K=32 macro
// block are content-addressed Schedule decisions. This fixture stays at the
// fail-closed boundary until the following slice gives Tile/ROCm a real scale
// operand ABI; it must never silently become an unscaled FP8 GEMM.
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
// BOUNDARY: ROCM_MXFP4_SCALE_ABI_UNIMPLEMENTED:
