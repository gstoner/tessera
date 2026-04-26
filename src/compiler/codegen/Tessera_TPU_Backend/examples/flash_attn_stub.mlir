
// RUN: tessera-tpu-opt %s -tessera-lower-attention-to-stablehlo | FileCheck %s
module {
  // Q: [B,H,S,D], K: [B,H,S,D], V: [B,H,S,D]
  func.func @flash(%q: tensor<1x12x128x64xbf16>, %k: tensor<1x12x128x64xbf16>,
                   %v: tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16> {
    %0 = "tessera.flash_attn"(%q, %k, %v) {scale = 0.125 : f32, dropout_p = 0.0 : f32, causal = false} : 
          (tensor<1x12x128x64xbf16>, tensor<1x12x128x64xbf16>, tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16>
    return %0 : tensor<1x12x128x64xbf16>
  }
}
// CHECK: stablehlo.custom_call
