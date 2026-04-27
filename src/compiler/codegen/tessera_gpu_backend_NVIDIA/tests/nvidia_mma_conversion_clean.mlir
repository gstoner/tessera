// RUN: tessera-opt -tessera-nvwgmma-lowering %s | FileCheck %s

module {
  func.func @mma(%a: tensor<64x16xbf16>, %b: tensor<16x64xbf16>) -> tensor<64x64xf32> {
    %0 = "tile.mma"(%a, %b) {sm = 90 : i64} : (tensor<64x16xbf16>, tensor<16x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}

// CHECK-LABEL: func.func @mma
// CHECK-NOT: tile.mma
// CHECK-NOT: tessera.nvgpu
// CHECK: func.call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16
