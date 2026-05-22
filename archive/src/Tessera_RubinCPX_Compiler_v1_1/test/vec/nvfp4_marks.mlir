
# RUN: %cpx_opt %s -tessera-vectorize-nvfp4 | FileCheck %s

module {
  // A dummy matmul op in a made-up dialect name for demonstration.
  "tessera.linalg.matmul"() : () -> ()
  // CHECK: "tessera.linalg.matmul"
  // CHECK-SAME: tessera.nvfp4.enabled
}
