// RUN: tessera-opt %s | FileCheck %s

// The local SwiGLU-fused MoE expert-FFN block is a first-class Graph-IR op
// (tessera.moe_swiglu_block) carrying the same grouped-layout contract as
// tessera.grouped_gemm — round-trips + is verifier-checked at the IR level.

// CHECK-LABEL: func.func @moe_block
// CHECK:       tessera.moe_swiglu_block
// CHECK-SAME:  grouped_kind = "masked"
// CHECK-SAME:  quant = "nvfp4"
func.func @moe_block(%x: tensor<256x64xbf16>, %wg: tensor<3x64x32xbf16>,
                     %wu: tensor<3x64x32xbf16>, %wd: tensor<3x32x16xbf16>,
                     %gs: tensor<3xi64>) -> tensor<256x16xf32> {
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs {
         grouped_kind = "masked", quant = "nvfp4"
       } : (tensor<256x64xbf16>, tensor<3x64x32xbf16>, tensor<3x64x32xbf16>,
            tensor<3x32x16xbf16>, tensor<3xi64>) -> tensor<256x16xf32>
  return %o : tensor<256x16xf32>
}
