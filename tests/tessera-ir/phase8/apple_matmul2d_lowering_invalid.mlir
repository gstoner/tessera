// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --split-input-file --tessera-tiling \
// RUN:   --tessera-apple-canonical-gemm-matmul2d --allow-unregistered-dialect 2>&1 \
// RUN:   | FileCheck %s
//
// APPLE-MATMUL2D-1: the lowering refuses a packed FP8 GEMM whose K is off
// Apple's 128-byte stride quantum instead of emitting a view Metal would
// reject at descriptor creation -- or, on 4-bit data, bind to the wrong bytes.
// The reason is the verifier's own wording, so a reader sees one contract.

// CHECK: APPLE_MATMUL2D_LAYOUT_{{UNSUPPORTED}}: operand a cannot be bound as an MTLTensor view: 8/4-bit MTLTensor views need a row stride that is a multiple of 128 bytes
func.func @packed_fp8_gemm(%a: tensor<64x64xf8E4M3FN>, %b: tensor<64x128xf8E4M3FN>) -> tensor<64x128xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x64xf8E4M3FN>, tensor<64x128xf8E4M3FN>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// -----

// A sub-block origin that Apple cannot bind: an FP8 window starting at column
// 64 of a 512-wide parent lands at byte 64 -- off the 128-byte data-plane
// alignment -- so the lowering refuses instead of binding a view Metal would
// reject at descriptor creation.
// CHECK: APPLE_MATMUL2D_LAYOUT_{{UNSUPPORTED}}: operand a cannot be bound as an MTLTensor view: MTLTensor data-plane offsets must be 128-byte aligned
func.func @fp8_sub_block_misaligned(%pa: tensor<200x512xf8E4M3FN>, %pb: tensor<512x512xf8E4M3FN>) -> tensor<64x128xf32> {
  %a = tensor.extract_slice %pa[8, 64] [64, 256] [1, 1] : tensor<200x512xf8E4M3FN> to tensor<64x256xf8E4M3FN>
  %b = tensor.extract_slice %pb[0, 128] [256, 128] [1, 1] : tensor<512x512xf8E4M3FN> to tensor<256x128xf8E4M3FN>
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
