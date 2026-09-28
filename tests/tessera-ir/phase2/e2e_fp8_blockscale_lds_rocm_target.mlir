// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' %s | FileCheck %s --check-prefix=GEN
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=LOWER
// RUN: not tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true blockscale-lds-pad-bytes=8' %s 2>&1 | FileCheck %s --check-prefix=BADPAD
// RUN: not tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true blockscale-stage-k=48' %s 2>&1 | FileCheck %s --check-prefix=BADSTAGE

// ROCM-FP8-BLOCKSCALE-1, large-M body (sync GFX1201-PERF-2026-09-27). With
// enough workgroups to cover the part's 64 CUs, the Schedule selects the
// LDS-staged multi-wave body for the [N, K] weight: eight waves of 32 rows
// share one staged K slab. It computes exactly what the register panel
// computes -- every group a zero partial, one scaled join per fragment -- and
// the store may round the fp32 accumulator once to bf16.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  // 256 x 4096: 2 x 32 = 64 workgroups at 128x128.
  func.func @w8a8_lds(%a: tensor<256x256xf8E4M3FN>, %b: tensor<4096x256xf8E4M3FN>,
                      %sa: tensor<256x2xf32>, %sb: tensor<2x32xf32>) -> tensor<256x4096xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"},
      transposeB = true
    } : (tensor<256x256xf8E4M3FN>, tensor<4096x256xf8E4M3FN>, tensor<256x2xf32>,
         tensor<2x32xf32>) -> tensor<256x4096xf32>
    return %0 : tensor<256x4096xf32>
  }

  // Ragged M = 200 (the second 128-row block is partial) with a bf16 output:
  // a ragged M follows the whole-M rule, 2 x 32 = 64 workgroups at 128x128.
  func.func @w8a8_lds_bf16(%a: tensor<200x256xf8E4M3FN>, %b: tensor<4096x256xf8E4M3FN>,
                           %sa: tensor<200x2xf32>, %sb: tensor<2x32xf32>) -> tensor<200x4096xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"},
      transposeB = true
    } : (tensor<200x256xf8E4M3FN>, tensor<4096x256xf8E4M3FN>, tensor<200x2xf32>,
         tensor<2x32xf32>) -> tensor<200x4096xbf16>
    return %0 : tensor<200x4096xbf16>
  }

  // 128 x 4096: 32 workgroups at 128x128, 64 at 128x64.
  func.func @w8a8_lds_narrow(%a: tensor<128x256xf8E4M3FN>, %b: tensor<4096x256xf8E4M3FN>,
                             %sa: tensor<128x2xf32>, %sb: tensor<2x32xf32>) -> tensor<128x4096xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"},
      transposeB = true
    } : (tensor<128x256xf8E4M3FN>, tensor<4096x256xf8E4M3FN>, tensor<128x2xf32>,
         tensor<2x32xf32>) -> tensor<128x4096xf32>
    return %0 : tensor<128x4096xf32>
  }
}

// Schedule IR states the staging, and it enters the digest.
// SCHED-LABEL: func.func @w8a8_lds(
// SCHED: schedule.matmul
// SCHED-SAME: macro_tile_m = 128 : i64, macro_tile_n = 128 : i64
// SCHED-SAME: output = "f32"
// SCHED-SAME: physical_contract = "rocm_fp8_w8a8_blockscale_nk_v1"
// SCHED-SAME: pipeline_depth = 1 : i64
// SCHED-SAME: staging = "lds"
// SCHED-SAME: warps = 8 : i64
// SCHED-LABEL: func.func @w8a8_lds_bf16(
// SCHED: schedule.matmul
// SCHED-SAME: macro_tile_m = 128 : i64, macro_tile_n = 128 : i64
// SCHED-SAME: output = "bf16"
// SCHED-SAME: staging = "lds"
// SCHED-LABEL: func.func @w8a8_lds_narrow(
// SCHED: schedule.matmul
// SCHED-SAME: macro_tile_m = 128 : i64, macro_tile_n = 64 : i64
// SCHED-SAME: staging = "lds"

// Target IR binds the workgroup the package launches.
// TARGET-LABEL: func.func @w8a8_lds(
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: block_m = 128 : i64, block_n = 128 : i64
// TARGET-SAME: output = "f32"
// TARGET-SAME: package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_bnk_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1"
// TARGET-SAME: pipeline_depth = 1 : i64
// TARGET-SAME: staging = "lds"
// TARGET-SAME: warps = 8 : i64
// TARGET-LABEL: func.func @w8a8_lds_bf16(
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: output = "bf16"
// TARGET-SAME: package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_bnk_sa_sb_o_m_n_k.e4m3_e4m3_f32_bf16.wmma_exact.v1"
// TARGET-SAME: staging = "lds"

// The generated body: a workgroup of eight waves with two LDS slabs (A and
// the weight, 144-byte rows), LDS-only barriers, fragment packs read from
// LDS, and one scaled join per fragment per group -- the register body's
// semantics on a different staging.
// GEN-LABEL: gpu.func @w8a8_lds(
// GEN-SAME: %arg4: memref<?xf32>
// GEN-SAME: workgroup(%{{.*}} : memref<18432xf8E4M3FN, #gpu.address_space<workgroup>>, %{{.*}} : memref<18432xf8E4M3FN, #gpu.address_space<workgroup>>)
// GEN-SAME: known_block_size = array<i32: 256, 1, 1>
// GEN-SAME: tessera.rocm.block_scale_contract = "rocm_fp8_w8a8_blockscale_nk_v1"
// GEN-SAME: tessera.rocm.blockscale_lds_pad_bytes = 16
// GEN-SAME: tessera.rocm.blockscale_prefetch = 0
// GEN-SAME: tessera.rocm.blockscale_stage_k = 128
// GEN-SAME: tessera.rocm.lds_bytes = 36864
// GEN-SAME: tessera.rocm.lds_waves = array<i64: 4, 2>
// GEN: scf.for
// GEN: gpu.barrier memfence [#gpu.address_space<workgroup>]
// GEN: vector.store {{.*}} {alignment = 16 : i64}
// GEN: gpu.barrier memfence [#gpu.address_space<workgroup>]
// GEN: tile.view {{.*}}space = "lds", order = "row_major"
// GEN: tile.view {{.*}}space = "lds", order = "col_major"
// GEN: tile.mma
// GEN-COUNT-8: tile.fragment_scaled_accumulate
// GEN-NOT: tile.fragment_scaled_accumulate
// GEN: tile.store
// GEN-LABEL: gpu.func @w8a8_lds_bf16(
// GEN-SAME: %arg4: memref<?xbf16>
// Only M is partial, so the column bound is each fragment's own far edge
// (origin + 16), which folds: nothing column-wise is held for the store.
// GEN: tile.fragment_unpack
// GEN-NEXT: %[[COLEND:.*]] = arith.addi %[[COL:[0-9]+]], %c16{{[_0-9]*}} : index
// GEN-NEXT: tile.store %{{.*}}, %arg4, %{{.*}}, %[[COL]], %arg5, %[[COLEND]], %arg6 {{.*}}tile.epilogue = #tile.epilogue<bias = false, activation = "none", output = "bf16">
// GEN-LABEL: gpu.func @w8a8_lds_narrow(
// GEN-SAME: tessera.rocm.lds_waves = array<i64: 4, 2>
// GEN-COUNT-4: tile.fragment_scaled_accumulate
// GEN-NOT: tile.fragment_scaled_accumulate

// The bf16 store rounds the fp32 accumulator once.
// LOWER-LABEL: gpu.func @w8a8_lds_bf16(
// LOWER-NOT: {{[[:space:]]tile\.[a-z_]+ }}
// LOWER: arith.truncf {{.*}} : f32 to bf16
// LOWER: memref.store {{.*}} : memref<?xbf16>

// BADPAD: ROCM_FP8_BLOCKSCALE_CONTRACT: blockscale-lds-pad-bytes must be a non-negative multiple of 16
// BADSTAGE: ROCM_FP8_BLOCKSCALE_CONTRACT: stage K=48 must be whole 16-byte vectors dividing scale_k=128
