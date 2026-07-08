// RUN: ts-spectral-opt --tessera-legalize-spectral --lower-spectral-to-target-ir %s | FileCheck %s --check-prefix=DYN
//
// A dynamic-length FFT lowers to the runtime Stockham driver symbol (which
// factors the runtime N and launches the per-stage kernels), NOT to a
// fabricated compile-time radix-4 stage call.

module attributes {tessera.target = "amd"} {
  func.func @fft_dyn_amd(%src : memref<?xcomplex<f32>>,
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
// DYN-SAME: tessera.target_ir.arbiter_op = "spectral_fft"
// DYN-SAME: tessera.target_ir.backend = "amd"
// DYN-SAME: tessera.target_ir.call = "ts_fft_stockham_amd"
// DYN-SAME: tessera.target_ir.dynamic
// DYN-SAME: tessera.target_ir.lowered
// DYN-SAME: tessera.target_ir.stage_calls = ["ts_fft_stockham_amd"]
