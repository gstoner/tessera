// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d \
// RUN:   --allow-unregistered-dialect 2>&1 | FileCheck %s
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
