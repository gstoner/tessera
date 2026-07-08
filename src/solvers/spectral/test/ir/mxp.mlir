// RUN: ts-spectral-opt --tessera-legalize-spectral --tessera-spectral-mxp %s | FileCheck %s --check-prefix=MXP
//
// SpectralMXPPass annotates each exec op with the block-floating-point
// decision (block_size = 32 for fp8, 64 for fp16/bf16) plus a guard epsilon.

module {
  func.func @fft_fp16(%src : memref<512xcomplex<f32>>,
                      %dst : memref<512xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0], elem_precision = "fp16", acc_precision = "f32",
      scaling = "blockfp_per_stage", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<512xcomplex<f32>>,
         memref<512xcomplex<f32>>) -> ()
    return
  }
}

// MXP: tessera_spectral.fft
// MXP-SAME: tessera.mxp.acc_dtype
// MXP-SAME: tessera.mxp.block_size
// MXP-SAME: tessera.mxp.elem_dtype
// MXP-SAME: tessera.mxp.guard_eps
// MXP-SAME: tessera.mxp.legalized
// MXP-SAME: tessera.mxp.scale_blocks
// MXP-SAME: tessera.mxp.scaling
