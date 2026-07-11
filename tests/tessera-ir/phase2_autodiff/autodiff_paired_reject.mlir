// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s
//
// An op ON the gradient path that does not implement AdjointInterface must
// fail loudly (Decision #21) — its operand cotangents would otherwise be
// silently dropped, producing wrong gradients. `arith.negf` is a real,
// registered op with no Tessera adjoint, so it exercises the rejection.

module {
  func.func @bad(%x: tensor<4x16xf32>) -> tensor<4x16xf32>
      attributes {tessera.autodiff = "reverse"} {
    %d = arith.negf %x : tensor<4x16xf32>
    // CHECK: [AUTODIFF_OP_NOT_DIFFERENTIABLE]
    return %d : tensor<4x16xf32>
  }
}
