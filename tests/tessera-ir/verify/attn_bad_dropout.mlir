
// RUN: not tessera-opt -tessera-verify %s 2>&1 | FileCheck %s
module {
  func.func @bad_p(%q: tensor<?x?x?xf32>, %k: tensor<?x?x?xf32>, %v: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %o = "tessera.flash_attn"(%q, %k, %v) {head_dim = 64, dropout_p = 1.0, causal = false}
         : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %o : tensor<?x?x?xf32>
  }
}
// CHECK: error: [TESSERA_VFY_ATTN_DROPOUT] dropout_p must be in [0,1)
