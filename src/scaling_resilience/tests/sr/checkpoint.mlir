// RUN: tessera-opt -tessera-insert-recompute %s | FileCheck %s --check-prefix=INS
tessera_sr.checkpoint {
  %0 = "tessera.tile.gemm"() : () -> tensor<128x128xf16>
  "tessera.tile.relu"(%0) : (tensor<128x128xf16>) -> tensor<128x128xf16>
}
// INS: sr.instrumented