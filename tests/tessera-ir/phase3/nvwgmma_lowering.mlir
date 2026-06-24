// Phase 3 NVWGMMALoweringPass lit tests.
// Verifies tile.mma → a tessera_nvidia_wgmma_mma_async runtime call carrying
// the WGMMA shape / dtype metadata.
//
// 2026-06: un-XFAIL'd.  The pass lowers tile.mma to a `call` to a declared
// runtime WGMMA fn (with tessera.nvidia.* shape/dtype attrs), not the old
// tessera.nvgpu.wgmma.* dialect ops; the tile-shape (m64n64k16) is driven by
// the tile.mma op's own sm attr, so both pass invocations emit the same call.
//
// tile.mma is a registered Tile op (Phase A0), so this fixture needs no
// --allow-unregistered-dialect — it runs through the strict driver.
//
// RUN: %tessera_strict_opt --tessera-nvwgmma-lowering='sm=90' %s \
// RUN:   | FileCheck %s --check-prefix=SM90
//
// RUN: %tessera_strict_opt --tessera-nvwgmma-lowering='sm=80' %s \
// RUN:   | FileCheck %s --check-prefix=SM80

// SM90-LABEL:  func.func @wgmma_kernel
// SM90:        call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16
// SM90-SAME:   dtype_ab = "bf16"
// SM90-SAME:   dtype_c = "f32"
// SM90-SAME:   shape = "m64n64k16"
// SM90-NOT:    tile.mma

// SM80-LABEL:  func.func @wgmma_kernel
// SM80:        call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16
// SM80-NOT:    tile.mma

module attributes {tessera.ir.version = "1.0"} {
  func.func @wgmma_kernel(
      %A: tensor<64x64xbf16>,
      %B: tensor<64x64xbf16>
  ) -> tensor<64x64xf32> {
    %C = "tile.mma"(%A, %B) {sm = 90 : i32}
           : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %C : tensor<64x64xf32>
  }
}
