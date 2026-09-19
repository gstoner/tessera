// RUN: tessera-opt %s | FileCheck %s

// ROCM-FP8-BLOCKSCALE-1. A block-scaled matmul is its own op because it is a
// different program shape, not a matmul with extra operands: the operands keep
// their low-precision storage into the matrix instruction and the scale lands
// on the fp32 accumulator per K block. Folding it into `tessera.matmul` would
// force AttrSizedOperandSegments and break every generic-form fixture; widening
// the operands first (what `dequant_matmul` does) caps the GEMM at the wide
// instruction's ceiling, which is why MI355 lands on bf16 for NVFP4.

// K = 512 over blocks of 128 is four blocks, so each scale carries an axis of 4.
// CHECK-LABEL: func.func @block_scaled_fp8
// CHECK: tessera.scaled_matmul
// CHECK-SAME: scales(
// CHECK-SAME: block = [128, 128]
func.func @block_scaled_fp8(%a: tensor<128x512xf8E4M3FN>, %b: tensor<512x128xf8E4M3FN>,
                            %sa: tensor<128x4xf32>, %sb: tensor<4x128xf32>) -> tensor<128x128xf32> {
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb)
       {scale_layout = {granularity = "block", block = [128, 128], format = "fp32"}}
       : (tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>, tensor<128x4xf32>, tensor<4x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// An MX form: e8m0 scales every 32 elements along K. 512/32 = 16 blocks.
// CHECK-LABEL: func.func @mx_block_32
// CHECK: tessera.scaled_matmul
func.func @mx_block_32(%a: tensor<64x512xf8E4M3FN>, %b: tensor<512x64xf8E4M3FN>,
                       %sa: tensor<64x16xf32>, %sb: tensor<16x64xf32>) -> tensor<64x64xf32> {
  %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb)
       {scale_layout = {granularity = "block", block = [64, 32], format = "e8m0"}}
       : (tensor<64x512xf8E4M3FN>, tensor<512x64xf8E4M3FN>, tensor<64x16xf32>, tensor<16x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
