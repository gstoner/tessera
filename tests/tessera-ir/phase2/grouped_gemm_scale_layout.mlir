// RUN: tessera-opt %s | FileCheck %s

// DeepGEMM-inspired keystone (2026-06): the FP8/FP4 scale tensor is promoted
// from a `quant` dtype-name string to first-class IR — an optional
// `scales(%x_scale, %w_scale)` operand clause whose layout is described by the
// `scale_layout` attribute, plus a `numeric_policy` recording the storage/accum
// coupling (two-level FP8->FP32 accumulation).  This round-trips and is verified
// at the IR level so LayoutLegalityPass can later act on real scale operands.

// CHECK-LABEL: func.func @grouped_gemm_fp8_block_scaled
// CHECK:       tessera.grouped_gemm
// CHECK-SAME:  scales(%arg3, %arg4)
// CHECK-SAME:  accum = "fp32"
// CHECK-SAME:  storage = "fp8_e4m3"
// CHECK-SAME:  granularity = "block"
// CHECK-SAME:  packing = "ue8m0"
func.func @grouped_gemm_fp8_block_scaled(
    %x: tensor<256x64xf8E4M3FN>, %w: tensor<3x64x16xf8E4M3FN>,
    %gs: tensor<3xi64>, %xs: tensor<256x1xf32>, %ws: tensor<3x1xf32>)
    -> tensor<256x16xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs scales(%xs, %ws) {
         grouped_kind = "contiguous", grouped_alignment = 128 : i64,
         scale_layout = {granularity = "block", block = [1, 128],
                         packing = "ue8m0", vector_size = 128 : i64,
                         alignment = 128 : i64, transposed = false},
         numeric_policy = {storage = "fp8_e4m3", accum = "fp32",
                           math_mode = "default"}
       } : (tensor<256x64xf8E4M3FN>, tensor<3x64x16xf8E4M3FN>, tensor<3xi64>,
            tensor<256x1xf32>, tensor<3x1xf32>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}

// NVFP4 micro-block (1x16, e4m3 packed scale) on the SwiGLU MoE block — all four
// scale operands present as an all-or-nothing clause.
// CHECK-LABEL: func.func @moe_swiglu_nvfp4_scaled
// CHECK:       tessera.moe_swiglu_block
// CHECK-SAME:  scales(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:  block = [1, 16]
func.func @moe_swiglu_nvfp4_scaled(
    %x: tensor<128x64xf32>, %wg: tensor<3x64x128xf32>, %wu: tensor<3x64x128xf32>,
    %wd: tensor<3x128x64xf32>, %gs: tensor<3xi64>,
    %xs: tensor<128x1xf32>, %wgs: tensor<3x1xf32>, %wus: tensor<3x1xf32>,
    %wds: tensor<3x1xf32>) -> tensor<128x64xf32> {
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
         scales(%xs, %wgs, %wus, %wds) {
         scale_layout = {granularity = "block", block = [1, 16], packing = "e4m3",
                         vector_size = 16 : i64, alignment = 16 : i64,
                         transposed = false},
         numeric_policy = {storage = "nvfp4", accum = "fp32"}
       } : (tensor<128x64xf32>, tensor<3x64x128xf32>, tensor<3x64x128xf32>,
            tensor<3x128x64xf32>, tensor<3xi64>, tensor<128x1xf32>,
            tensor<3x1xf32>, tensor<3x1xf32>, tensor<3x1xf32>)
            -> tensor<128x64xf32>
  return %o : tensor<128x64xf32>
}

// Bare (unscaled) forms still parse unchanged — back-compat for the optional
// scale clause.
// CHECK-LABEL: func.func @grouped_gemm_bare
// CHECK:       tessera.grouped_gemm %arg0, %arg1, %arg2
// CHECK-NOT:   scales(
func.func @grouped_gemm_bare(%x: tensor<12x8xbf16>, %w: tensor<3x8x6xbf16>,
                             %gs: tensor<3xi64>) -> tensor<12x6xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs
       : (tensor<12x8xbf16>, tensor<3x8x6xbf16>, tensor<3xi64>) -> tensor<12x6xf32>
  return %o : tensor<12x6xf32>
}
