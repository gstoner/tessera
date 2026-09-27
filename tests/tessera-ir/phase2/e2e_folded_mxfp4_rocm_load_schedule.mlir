// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s

// The folded prefill lowering selects its physical load schedule from the
// problem: CU workgroup mode only once M spans two or more BM256 row blocks
// (sync GFX1201-LANES-2026-09-27). M = 256 is one row block; M = 257 is two.
// The per-wave M guard only where a row block is partial
// (GFX1201-PERF-2026-09-27): M = 256 keeps the CTA guard, M = 257 does not.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @one_row_block(%a: tensor<256x128xui8>,
                           %b: tensor<80x128xui8>,
                           %sa: tensor<256xf32>,
                           %ref: tensor<80xui8>) -> tensor<256x80xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 128], format = "e8m0_row_reference"}
    } : (tensor<256x128xui8>, tensor<80x128xui8>, tensor<256xf32>,
         tensor<80xui8>) -> tensor<256x80xbf16>
    return %0 : tensor<256x80xbf16>
  }

  func.func @two_row_blocks(%a: tensor<257x128xui8>,
                            %b: tensor<80x128xui8>,
                            %sa: tensor<257xf32>,
                            %ref: tensor<80xui8>) -> tensor<257x80xbf16> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %ref) {
      physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1",
      numeric_policy = {accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"},
      scale_layout = {granularity = "output_column", block = [1, 128], format = "e8m0_row_reference"}
    } : (tensor<257x128xui8>, tensor<80x128xui8>, tensor<257xf32>,
         tensor<80xui8>) -> tensor<257x80xbf16>
    return %0 : tensor<257x80xbf16>
  }
}

// CHECK: tessera_rocm.scaled_wmma_gemm
// CHECK-SAME: epilogue_schedule = "complete_tile_vector_scales"
// CHECK-SAME: m = 256
// CHECK-SAME: raster_group_m = 4
// CHECK-SAME: row_guard = "cta"
// CHECK-SAME: staging_prefetch = "register_next_slab"
// CHECK-SAME: workgroup_mode = "wgp"
// CHECK: tessera_rocm.scaled_wmma_gemm
// CHECK-SAME: m = 257
// CHECK-SAME: raster_group_m = 4
// CHECK-SAME: row_guard = "wave"
// CHECK-SAME: workgroup_mode = "cu"
