// RUN: mlir-opt %s | FileCheck %s
// Tile→Target (ROCm) — smoke sample (MFMA-like op name for check)
module {
  func.func @ebt_rocm_energy_tile_lowered(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<32x32xf32> {
    %acc = "tessera.target.rocm.mfma"(%a, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xf32>
    return %acc : tensor<32x32xf32>
  }
}
// CHECK-LABEL: func @ebt_rocm_energy_tile_lowered
// CHECK: tessera.target.rocm.mfma
