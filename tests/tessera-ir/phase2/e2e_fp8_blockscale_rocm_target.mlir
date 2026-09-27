// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' %s | FileCheck %s --check-prefix=GEN
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=LOWER

// ROCM-FP8-BLOCKSCALE-1. A logical W8A8 block-scaled matmul (e4m3 operands,
// fp32 scales, scale_layout.block = [scale_n, scale_k]) binds to one named
// contract per weight layout. Graph->Schedule derives the contract -- it is
// never authored -- and every consumer below it either honours the isolated
// scale-group partial or refuses; none of them answers with an unscaled GEMM.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  // Weight [K, N]; ragged M and N; two K128 groups; one scale per 128 columns.
  func.func @w8a8_kn(%a: tensor<40x256xf8E4M3FN>, %b: tensor<256x72xf8E4M3FN>,
                     %sa: tensor<40x2xf32>, %sb: tensor<2x1xf32>) -> tensor<40x72xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}
    } : (tensor<40x256xf8E4M3FN>, tensor<256x72xf8E4M3FN>, tensor<40x2xf32>,
         tensor<2x1xf32>) -> tensor<40x72xf32>
    return %0 : tensor<40x72xf32>
  }

  // Weight [N, K] (transposeB): the checkpoint layout, K contiguous.
  func.func @w8a8_nk(%a: tensor<64x256xf8E4M3FN>, %b: tensor<96x256xf8E4M3FN>,
                     %sa: tensor<64x2xf32>, %sb: tensor<2x1xf32>) -> tensor<64x96xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [128, 128], format = "fp32"},
      transposeB = true
    } : (tensor<64x256xf8E4M3FN>, tensor<96x256xf8E4M3FN>, tensor<64x2xf32>,
         tensor<2x1xf32>) -> tensor<64x96xf32>
    return %0 : tensor<64x96xf32>
  }
}

// TARGET-LABEL: func.func @w8a8_kn
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: abi = "a_b_lhs_scale_rhs_scale_d_m_n_k"
// TARGET-SAME: block_m = 16 : i64, block_n = 16 : i64
// TARGET-SAME: instruction_k = 16 : i64, k = 256 : i64
// TARGET-SAME: macro_k = 128 : i64
// TARGET-SAME: numeric_policy = {accum = "f32", execution_mode = "exact_per_block", storage = "e4m3"}
// TARGET-SAME: output = "f32"
// TARGET-SAME: package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1"
// TARGET-SAME: partial_combine = "scale_outer_product_then_add"
// TARGET-SAME: physical_contract = "rocm_fp8_w8a8_blockscale_v1"
// TARGET-SAME: scale_format = "fp32", scale_k = 128 : i64, scale_n = 128 : i64
// TARGET-LABEL: func.func @w8a8_nk
// TARGET: tessera_rocm.scaled_wmma_gemm
// TARGET-SAME: abi = "a_bnk_lhs_scale_rhs_scale_d_m_n_k"
// TARGET-SAME: block_m = 16 : i64, block_n = 32 : i64
// TARGET-SAME: package_abi = "tessera.rocm.fp8_w8a8_blockscale.a_bnk_sa_sb_o_m_n_k.e4m3_e4m3_f32_f32.wmma_exact.v1"
// TARGET-SAME: physical_contract = "rocm_fp8_w8a8_blockscale_nk_v1"

// The generator's typed body: every group starts its own zero partial, walks
// its panels in an inner loop, and joins the accumulator through one scaled
// accumulate per fragment. The [N, K] weight is read column-major.
// GEN-LABEL: gpu.func @w8a8_kn
// GEN-SAME: tessera.rocm.block_scale_contract = "rocm_fp8_w8a8_blockscale_v1"
// GEN-SAME: tessera.rocm.scale_group_panels = 2
// GEN-SAME: tessera.rocm.scale_k = 128
// GEN-SAME: tessera.rocm.scale_n = 128
// GEN: scf.for
// GEN: tile.fragment_zero
// GEN: scf.for
// GEN: tile.mma
// GEN: tile.mma
// GEN: tile.fragment_scaled_accumulate {{.*}} {scale_n = 128 : i64}
// GEN-LABEL: gpu.func @w8a8_nk
// GEN-SAME: tessera.rocm.block_scale_contract = "rocm_fp8_w8a8_blockscale_nk_v1"
// GEN: tile.view {{.*}}order = "col_major"
// GEN: tile.fragment_scaled_accumulate

// After the architecture consumer, no Tile op survives, and each join has
// become scale loads, a product and an add on the accumulator registers.
// LOWER-LABEL: gpu.func @w8a8_kn
// LOWER-NOT: {{[[:space:]]tile\.[a-z_]+ }}
// LOWER: memref.load {{.*}} : memref<?xf32>
// LOWER: memref.load {{.*}} : memref<?xf32>
// LOWER: arith.mulf
// LOWER: arith.addf
// LOWER: gpu.return
