// RUN: tessera-opt -tessera-cleanup %s | FileCheck %s
// 2026-06: un-XFAIL'd — added an EraseIdentityCast canonicalization pattern so
// the -tessera-cleanup pipeline folds a no-op (same-type) tessera.cast.
module {
  func.func @noop(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c = "tessera.cast"(%x) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %c : tensor<2x2xf32>
  }
}
// CHECK-NOT: tessera.cast
