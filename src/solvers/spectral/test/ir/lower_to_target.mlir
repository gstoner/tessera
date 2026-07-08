// RUN: ts-spectral-opt --tessera-legalize-spectral --lower-spectral-to-target-ir %s | FileCheck %s --check-prefix=LOWER
//
// LowerSpectralToTargetIRPass picks the C ABI symbol for each radix stage
// based on the module-level `tessera.target` attribute and exposes both a
// top-level `call` (first stage) and a `stage_calls` ArrayAttr for codegen.
// N=256 = 4^4 → four radix-4 CPU stage kernels.

module attributes {tessera.target = "cpu"} {
  func.func @fft_cpu(%src : memref<256xcomplex<f32>>,
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

// LOWER: tessera_spectral.fft
// LOWER-SAME: tessera.target_ir.arbiter_op = "spectral_fft"
// LOWER-SAME: tessera.target_ir.backend = "cpu"
// LOWER-SAME: tessera.target_ir.call = "ts_stockham_r4_cpu"
// LOWER-SAME: tessera.target_ir.lowered
// LOWER-SAME: tessera.target_ir.stage_calls = ["ts_stockham_r4_cpu", "ts_stockham_r4_cpu", "ts_stockham_r4_cpu", "ts_stockham_r4_cpu"]
