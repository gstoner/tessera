// REQUIRES: tessera-nvidia-backend
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

// E2E-REAL-6 SM120 MX (2026-11-04): OCP MX FP6/MXFP4 with UE8M0 K32 block
// scales lowers Graph -> Schedule -> Tile natively. The Schedule derives the
// storage and the instruction K from the logical element type, and the Tile
// launch op carries the scale block as its own stated contract.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @tessera_scheduled_mx(%a: tensor<19x69xf6E2M3FN>, %b: tensor<69x13xf6E2M3FN>,
                                  %sa: tensor<19x3xui8>, %sb: tensor<3x13xui8>) -> tensor<19x13xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      physical_contract = "nvidia_sm120_mx_blockscale_v1",
      scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"}
    } : (tensor<19x69xf6E2M3FN>, tensor<69x13xf6E2M3FN>, tensor<19x3xui8>,
         tensor<3x13xui8>) -> tensor<19x13xf32>
    return %0 : tensor<19x13xf32>
  }
}
// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: physical_contract = "nvidia_sm120_mx_blockscale_v1"
// SCHEDULE-SAME: scale_format = "ue8m0"
// SCHEDULE-SAME: scale_k = 32
// SCHEDULE-SAME: storage = "e2m3"
// SCHEDULE-SAME: storage_b = "e2m3"
// SCHEDULE-SAME: tile_k = 32
// SCHEDULE: shape_key = "M=19;N=13;K=69;dtype=e2m3"
// TILE-LABEL: llvm.func @tessera_tile_matmul_mx_e2m3(
// TILE: tile.matmul_kernel
// TILE-SAME: k = 32, a = "e2m3", b = "e2m3", acc = "f32"
// TILE-SAME: physical_contract = "nvidia_sm120_mx_blockscale_v1"
// TILE-SAME: tessera.scale_format = "ue8m0"
// TILE-SAME: tessera.scale_vector_size = 32 : i64
// TILE-SAME: tessera.storage_pack = #tile.packed_format<logical = "fp6_e2m3", container = "int8", logical_bits = 6, elements_per_container = 1, signedness = "format_defined", encoding = "e2m3", lane_order = "scalar_lsb">
// TILE-NOT: schedule.
// TILE-NOT: tessera.scaled_matmul

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @tessera_scheduled_mx(%a: tensor<16x64xf6E3M2FN>, %b: tensor<64x8xf6E3M2FN>,
                                  %sa: tensor<16x2xui8>, %sb: tensor<2x8xui8>) -> tensor<16x8xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      physical_contract = "nvidia_sm120_mx_blockscale_v1",
      scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"}
    } : (tensor<16x64xf6E3M2FN>, tensor<64x8xf6E3M2FN>, tensor<16x2xui8>,
         tensor<2x8xui8>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}
// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: storage = "e3m2"
// TILE-LABEL: llvm.func @tessera_tile_matmul_mx_e3m2(
// TILE-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
// TILE: tile.matmul_kernel
// TILE-SAME: a = "e3m2", b = "e3m2"

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @tessera_scheduled_mx(%a: tensor<7x33xf4E2M1FN>, %b: tensor<33x5xf4E2M1FN>,
                                  %sa: tensor<7x2xui8>, %sb: tensor<2x5xui8>) -> tensor<7x5xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      physical_contract = "nvidia_sm120_mx_blockscale_v1",
      scale_layout = {granularity = "block", block = [1, 32], format = "ue8m0"}
    } : (tensor<7x33xf4E2M1FN>, tensor<33x5xf4E2M1FN>, tensor<7x2xui8>,
         tensor<2x5xui8>) -> tensor<7x5xf32>
    return %0 : tensor<7x5xf32>
  }
}
// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: storage = "fp4_e2m1"
// SCHEDULE-SAME: tile_k = 64
// TILE-LABEL: llvm.func @tessera_tile_matmul_mx_fp4_e2m1(
// TILE: tile.matmul_kernel
// TILE-SAME: k = 64, a = "fp4_e2m1", b = "fp4_e2m1", acc = "f32"
// TILE-SAME: tessera.storage_pack = #tile.packed_format<logical = "fp4_e2m1", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "format_defined", encoding = "e2m1", lane_order = "low_to_high">
