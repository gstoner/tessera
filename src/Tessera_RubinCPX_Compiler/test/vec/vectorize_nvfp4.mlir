
// RUN: %cpx_opt %s -tessera-vectorize-nvfp4 | FileCheck %s
// CHECK: module
module {
  // skeleton matmul/attn would go here; the pass currently no-ops.
}
