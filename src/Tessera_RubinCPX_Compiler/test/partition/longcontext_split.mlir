
// RUN: %cpx_opt %s -tessera-partition-longcontext | FileCheck %s

// Demonstrates that attn.prefill_fused triggers a kv.export insertion.
module {
  func.func @prefill(%q: tensor<1x128x64xf16>, %k: tensor<1x128x64xf16>, %v: tensor<1x128x64xf16>,
                     %kv: memref<1x128x64xf16>) -> tensor<1x128x64xf16> {
    %seq = arith.constant 131072 : i64
    %o = "tessera.target.cpx.attn.prefill_fused"(%q, %k, %v, %kv, %seq)
         : (tensor<1x128x64xf16>, tensor<1x128x64xf16>, tensor<1x128x64xf16>, memref<1x128x64xf16>, i64)
           -> tensor<1x128x64xf16>
    // CHECK: tessera.target.cpx.kv.export
    return %o : tensor<1x128x64xf16>
  }
}
