// RUN: not tessera-opt %s 2>&1 | FileCheck %s
module {
  func.func @invalid_lse(%scores: tensor<32x64xf32>,
                         %checkpoint: memref<32x64xf32>) {
    %row = arith.constant 0 : index
    "tessera_attn.lse.save"(%scores, %checkpoint, %row) {
      identity = "attention.lse",
      memory_space = "global",
      scope = "program_launch"
    } : (tensor<32x64xf32>, memref<32x64xf32>, index) -> ()
    return
  }
}
// CHECK: lse must have rank <= 1
