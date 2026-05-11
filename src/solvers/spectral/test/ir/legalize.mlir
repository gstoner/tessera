// RUN: ts-spectral-opt --tessera-legalize-spectral %s | FileCheck %s --check-prefix=LEGAL
//
// LegalizeSpectralPass annotates fft ops with the resolved radix sequence,
// per-axis lengths, direction, and a `legalized` marker that downstream
// passes (mxp, autotune, lower-to-target-ir) gate on.

module {
  func.func @fft1d(%plan : !tessera_spectral.plan,
                   %src : memref<256xcomplex<f32>>,
                   %dst : memref<256xcomplex<f32>>) {
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<256xcomplex<f32>>,
         memref<256xcomplex<f32>>) -> ()
    return
  }
}

// LEGAL: tessera_spectral.fft
// LEGAL-SAME: tessera.spectral.direction = "forward"
// LEGAL-SAME: tessera.spectral.legalized
// LEGAL-SAME: tessera.spectral.per_axis_len
// LEGAL-SAME: tessera.spectral.stages
