// RUN: ts-spectral-opt --tessera-legalize-spectral --tessera-spectral-autotune %s | FileCheck %s --check-prefix=AUTO
//
// SpectralAutotunePass attaches a deterministic FNV-1a cache key plus the
// default knob dictionary that the runtime autotuner will overwrite from
// SQLite at first invocation.

module attributes {tessera.target = "nvidia"} {
  func.func @fft_auto(%plan : !tessera_spectral.plan,
                      %src : memref<1024xcomplex<f32>>,
                      %dst : memref<1024xcomplex<f32>>) {
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<1024xcomplex<f32>>,
         memref<1024xcomplex<f32>>) -> ()
    return
  }
}

// AUTO: tessera_spectral.fft
// AUTO-SAME: tessera.autotune.cache_key = "ts-fft-
// AUTO-SAME: tessera.autotune.cached = false
// AUTO-SAME: tessera.autotune.knobs
// AUTO-SAME: pipeline_stages = 2
// AUTO-SAME: warp_specialized = true
