// RUN: ts-spectral-opt --tessera-legalize-spectral %s | FileCheck %s --check-prefix=DYN
//
// A dynamic FFT length must NOT be assigned a fabricated compile-time radix
// stage.  LegalizeSpectralPass leaves the stage list empty for the dynamic
// axis and flags the op with `tessera.spectral.dynamic_shape` +
// `tessera.spectral.dynamic_axes`, so downstream lowering routes to the
// runtime driver instead of a bogus single radix-4 transform.

module {
  func.func @fft_dyn(%src : memref<?xcomplex<f32>>,
                     %dst : memref<?xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<?xcomplex<f32>>,
         memref<?xcomplex<f32>>) -> ()
    return
  }
}

// DYN: tessera_spectral.fft
// DYN-SAME: tessera.spectral.dynamic_axes = [true]
// DYN-SAME: tessera.spectral.dynamic_shape
// DYN-SAME: tessera.spectral.legalized
// The per-axis length is recorded as -1 (dynamic) and the stage list holds
// only the axis separator (-1), i.e. zero fabricated radix stages.
// DYN-SAME: tessera.spectral.per_axis_len = [-1]
// DYN-SAME: tessera.spectral.stages = [-1]
