
// RUN: tessera-opt -tessera-canonicalize %s | FileCheck %s
module {
  func.func @attn(%q: tensor<?x?x?xf32>, %k: tensor<?x?x?xf32>, %v: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %o = "tessera.flash_attn"(%q, %k, %v) {head_dim = 64, dropout_p = 0.0, causal = true}
         : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %o : tensor<?x?x?xf32>
  }
}
// CHECK: "tessera.flash_attn"
// CHECK-NOT: dropout_p =
// CHECK-SAME: has_dropout = false
