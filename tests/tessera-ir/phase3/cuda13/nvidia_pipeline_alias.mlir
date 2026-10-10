// RUN: tessera-opt --tessera-nvidia-pipeline %s | FileCheck %s --check-prefix=PIPE
// RUN: tessera-opt --tessera-nvidia-pipeline-sm90 %s | FileCheck %s --check-prefix=SM90
// RUN: tessera-opt --tessera-nvidia-pipeline-sm100 %s | FileCheck %s --check-prefix=SM100
// RUN: tessera-opt --tessera-nvidia-pipeline-sm120 %s | FileCheck %s --check-prefix=SM120
//
// Sprint G-5 (2026-05-11) — NVIDIATargetPipeline.  Validates the four
// pipeline aliases registered in `src/transforms/lib/Passes.cpp`:
//
//   * tessera-nvidia-pipeline         (default = SM_90)
//   * tessera-nvidia-pipeline-sm90    (Hopper WGMMA + TMA)
//   * tessera-nvidia-pipeline-sm100   (Blackwell tcgen05 + TMEM)
//   * tessera-nvidia-pipeline-sm120   (consumer Blackwell warp MMA)
//
// The SM120 route projects registered matmuls through Graph -> Schedule ->
// Tile before residual generic lowering. SM90 and SM100 retain their existing
// TileIRLowering chains; SM90 additionally consumes WGMMA and Hopper FA.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @entry(%A : tensor<64x16xbf16>,
                   %B : tensor<16x256xbf16>) -> tensor<64x256xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<64x16xbf16>,
                                     tensor<16x256xbf16>) -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }
}

// PIPE: tessera.effect
// SM90: tessera.effect
// SM90: tile.mbarrier.wait
// SM90-SAME: !tile.async_token
// SM90: call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16
// SM100: tessera.effect
// SM100: tile.mbarrier.wait
// SM100-SAME: !tile.async_token
// SM100: tile.mma
// SM100-SAME: sm = 100
// SM100-SAME: !tile.async_token
// SM120-NOT: tessera.matmul
// SM120-NOT: tile.async_copy
// SM120: tile.fragment_pack {{.*}} : (!tile.tile) -> !tile.fragment
// SM120: tile.fragment_pack {{.*}} : (!tile.tile) -> !tile.fragment
// SM120: tile.mma {{.*}} -> !tile.fragment
// SM120: tile.fragment_unpack
// SM120-NOT: tessera.matmul
// SM120-NOT: tile.async_copy
