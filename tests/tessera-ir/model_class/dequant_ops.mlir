// RUN: tessera-opt %s | FileCheck %s
//
// M5.1 perf follow-up — the fused dequantize-into-GEMM ops are now first-class
// `tessera` dialect MLIR ops (TesseraOps.td), verified at the IR level: packed
// low-precision weight codes + a separate per-group scale operand, fp32 accum.

// CHECK-LABEL: func.func @dequant_matmul_int4
// CHECK: tessera.dequant_matmul %arg0, %arg1 scale(%arg2)
// CHECK-SAME: quant_group_size = 64
// CHECK-SAME: weight_dtype = "int4"
func.func @dequant_matmul_int4(
    %x: tensor<12x64xf32>, %w_codes: tensor<64x48xi8>, %w_scale: tensor<1x48xf32>)
    -> tensor<12x48xf32> {
  %o = tessera.dequant_matmul %x, %w_codes scale(%w_scale) {
         weight_dtype = "int4", quant_group_size = 64 : i64,
         numeric_policy = {storage = "int4", accum = "fp32"}
       } : (tensor<12x64xf32>, tensor<64x48xi8>, tensor<1x48xf32>)
           -> tensor<12x48xf32>
  return %o : tensor<12x48xf32>
}

// Bare (per-channel, no scale operand) FP8 form still parses.
// CHECK-LABEL: func.func @dequant_matmul_fp8_bare
// CHECK: tessera.dequant_matmul %arg0, %arg1
// CHECK-NOT: scale(
func.func @dequant_matmul_fp8_bare(
    %x: tensor<8x32xf32>, %w_codes: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %o = tessera.dequant_matmul %x, %w_codes {weight_dtype = "fp8_e4m3"}
       : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}

// Quantized grouped GEMM (the MoE expert-FFN core) at production-ish dims.
// CHECK-LABEL: func.func @dequant_grouped_gemm_fp8
// CHECK: tessera.dequant_grouped_gemm %arg0, %arg1, %arg2 scale(%arg3)
// CHECK-SAME: granularity = "block"
// CHECK-SAME: weight_dtype = "fp8_e4m3"
func.func @dequant_grouped_gemm_fp8(
    %x: tensor<128x7168xf32>, %w_codes: tensor<256x7168x2048xf32>,
    %gs: tensor<256xi64>, %w_scale: tensor<256x1xf32>) -> tensor<128x2048xf32> {
  %o = tessera.dequant_grouped_gemm %x, %w_codes, %gs scale(%w_scale) {
         grouped_kind = "contiguous", weight_dtype = "fp8_e4m3",
         quant_group_size = 128 : i64,
         scale_layout = {granularity = "block", block = [1, 128],
                         packing = "ue8m0", vector_size = 128 : i64,
                         alignment = 128 : i64, transposed = false},
         numeric_policy = {storage = "fp8_e4m3", accum = "fp32"}
       } : (tensor<128x7168xf32>, tensor<256x7168x2048xf32>, tensor<256xi64>,
            tensor<256x1xf32>) -> tensor<128x2048xf32>
  return %o : tensor<128x2048xf32>
}
