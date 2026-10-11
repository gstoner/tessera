// RUN: tessera-opt --tessera-nvidia-pipeline %s | FileCheck %s --check-prefix=PIPE
// RUN: tessera-opt --tessera-nvidia-pipeline-sm90 %s | FileCheck %s --check-prefix=SM90
// RUN: tessera-opt --tessera-nvidia-pipeline-sm100 %s | FileCheck %s --check-prefix=SM100
//
// Sprint G-5 (2026-05-11) — NVIDIATargetPipeline.  Validates the
// pipeline aliases registered in `src/transforms/lib/Passes.cpp`:
//
//   * tessera-nvidia-pipeline         (default = SM_90)
//   * tessera-nvidia-pipeline-sm90    (Hopper WGMMA + TMA)
//   * tessera-nvidia-pipeline-sm100   (Blackwell tcgen05 + TMEM)
//   * tessera-nvidia-pipeline-sm120   (consumer Blackwell warp MMA; see below)
//
// SM90 and SM100 retain their existing TileIRLowering chains; SM90
// additionally consumes WGMMA and Hopper FA. These core aliases are registered
// in the default tessera-opt, so this fixture deliberately carries no backend
// gate (test_core_pipeline_alias_fixture_runs_without_backend_gate). The SM120
// route needs the registered tessera_nvidia dialect and lives in
// nvidia_pipeline_alias_sm120.mlir.

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
