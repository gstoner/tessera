// RUN: mlir-opt %s | FileCheck %s
// Tile→Target (NVIDIA) — smoke sample (WGMMA-like op name for check)
module {
  func.func @ebt_nv_energy_tile_lowered(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    %acc = "tessera.target.nv.wgmma"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %acc : tensor<64x64xf32>
  }
}
// CHECK-LABEL: func @ebt_nv_energy_tile_lowered
// CHECK: tessera.target.nv.wgmma
