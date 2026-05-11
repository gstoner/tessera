// RUN: ts-spectral-opt --tessera-legalize-spectral --lower-spectral-to-target-ir %s | FileCheck %s --check-prefix=LOWER
//
// LowerSpectralToTargetIRPass picks the C ABI symbol for each radix stage
// based on the module-level `tessera.target` attribute and exposes both
// a top-level `call` (first stage) and a `stage_calls` ArrayAttr for codegen.

module attributes {tessera.target = "cpu"} {
  func.func @fft_cpu(%plan : !tessera_spectral.plan,
                     %src : memref<256xcomplex<f32>>,
                     %dst : memref<256xcomplex<f32>>) {
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<256xcomplex<f32>>,
         memref<256xcomplex<f32>>) -> ()
    return
  }
}

// LOWER: tessera_spectral.fft
// LOWER-SAME: tessera.target_ir.backend = "cpu"
// LOWER-SAME: tessera.target_ir.call = "ts_stockham_radix4_scalar"
// LOWER-SAME: tessera.target_ir.lowered
// LOWER-SAME: tessera.target_ir.stage_calls
