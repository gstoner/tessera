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
// Each pipeline runs: EffectAnnotation → Canonicalize → SwigluFusion →
// MLAFusion → NSAFusion → HybridAttnExpand → LightningAttnFusion →
// DeltaAttnChunking → DistributionLowering → TileIRLowering →
// WarpSpec → exact-SM AsyncCopy → TMA. SM90 additionally consumes the proven
// WGMMA and Hopper FlashAttention marker passes; SM100/SM120 retain typed MMA
// and attention carriers for their exact backend pipelines.

module {
  func.func @entry(%A : tensor<64x16xbf16>,
                   %B : tensor<16x256xbf16>) -> tensor<64x256xf32> {
    %C = "tessera.matmul"(%A, %B) : (tensor<64x16xbf16>,
                                     tensor<16x256xbf16>) -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }
}

// PIPE: tessera.effect
// SM90: tessera.effect
// SM90: "tile.mbarrier.wait"
// SM90-SAME: !tile.async_token
// SM90: call @tessera_nvidia_wgmma_mma_async_bf16_m64n64k16
// SM100: tessera.effect
// SM100: "tile.mbarrier.wait"
// SM100-SAME: !tile.async_token
// SM100: tile.mma
// SM100-SAME: sm = 100
// SM100-SAME: !tile.async_token
// SM120: tessera.effect
// SM120: "tile.mbarrier.wait"
// SM120-SAME: !tile.async_token
// SM120: tile.mma
// SM120-SAME: sm = 120
// SM120-SAME: !tile.async_token
