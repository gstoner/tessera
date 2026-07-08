// RUN: ts-spectral-opt --tessera-legalize-spectral %s | FileCheck %s --check-prefix=LEGAL
//
// LegalizeSpectralPass annotates fft ops with the resolved radix sequence,
// per-axis lengths, direction, and a `legalized` marker that downstream
// passes (mxp, autotune, lower-to-target-ir) gate on.  N=256 = 4^4, so the
// stage list is four radix-4 stages (application order) then the axis
// separator (-1).  The plan must be a real `tessera_spectral.plan` op — the
// pass resolves the radix sequence from the plan's axes + the src shape.

module {
  func.func @fft1d(%src : memref<256xcomplex<f32>>,
                   %dst : memref<256xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
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
// LEGAL-SAME: tessera.spectral.per_axis_len = [256]
// LEGAL-SAME: tessera.spectral.stages = [4, 4, 4, 4, -1]
