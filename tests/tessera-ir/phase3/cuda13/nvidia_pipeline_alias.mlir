// RUN: tessera-opt --tessera-nvidia-pipeline %s | FileCheck %s --check-prefix=PIPE
// RUN: tessera-opt --tessera-nvidia-pipeline-sm90 %s | FileCheck %s --check-prefix=SM90
// RUN: tessera-opt --tessera-nvidia-pipeline-sm100 %s | FileCheck %s --check-prefix=SM100
// RUN: tessera-opt --tessera-nvidia-pipeline-sm120 %s | FileCheck %s --check-prefix=SM120
// REQUIRES: tessera_opt_built
//
// Sprint G-5 (2026-05-11) — NVIDIATargetPipeline.  Validates the four
// pipeline aliases registered in `src/transforms/lib/Passes.cpp`:
//
//   * tessera-nvidia-pipeline         (default = SM_90)
//   * tessera-nvidia-pipeline-sm90    (Hopper WGMMA + TMA)
//   * tessera-nvidia-pipeline-sm100   (Blackwell tcgen05 + TMEM)
//   * tessera-nvidia-pipeline-sm120   (Rubin, preliminary)
//
// Each pipeline runs: EffectAnnotation → Canonicalize → SwigluFusion →
// MLAFusion → NSAFusion → HybridAttnExpand → LightningAttnFusion →
// DeltaAttnChunking → DistributionLowering → TileIRLowering →
// WarpSpec → AsyncCopy → WGMMA → TMA → FlashAttnEmitter.

module attributes {tessera.target = "nvidia_sm90"} {
  func.func @entry(%A : memref<64x16xbf16, 3>,
                   %B : memref<16x256xbf16, 3>,
                   %C : memref<64x256xf32, 3>) {
    "tessera.matmul"(%A, %B, %C) : (memref<64x16xbf16, 3>,
                                    memref<16x256xbf16, 3>,
                                    memref<64x256xf32, 3>) -> ()
    return
  }
}

// All four aliases produce the same final IR shape — the pipeline runs
// EffectAnnotation as its first pass, so every output has the effect
// attribute set on `func.func`.
//
// PIPE: tessera.effect
// SM90: tessera.effect
// SM100: tessera.effect
// SM120: tessera.effect
