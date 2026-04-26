// Phase 3 NVWGMMALoweringPass lit tests.
// Verifies tile.mma → wgmma.mma_async PTX for SM_90, WMMA fallback for SM<90.
//
// RUN: tessera-opt --tessera-nvwgmma-lowering='sm=90' %s \
// RUN:   | FileCheck %s --check-prefix=SM90
//
// RUN: tessera-opt --tessera-nvwgmma-lowering='sm=80' %s \
// RUN:   | FileCheck %s --check-prefix=SM80

// SM90-LABEL:  func.func @wgmma_kernel
// SM90:        tessera.nvgpu.wgmma.mma_async
// SM90-SAME:   shape = "m64n64k16"
// SM90-SAME:   dtype_ab = "bf16"
// SM90-SAME:   dtype_c = "f32"
// SM90:        tessera.nvgpu.wgmma.commit_group
// SM90:        tessera.nvgpu.wgmma.wait_group
// SM90-NOT:    tile.mma

// SM80-LABEL:  func.func @wgmma_kernel
// SM80:        tessera.nvgpu.mma.sync
// SM80-SAME:   shape = "m16n16k16"
// SM80-NOT:    tessera.nvgpu.wgmma.mma_async

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
