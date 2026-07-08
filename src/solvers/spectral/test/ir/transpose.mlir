// RUN: ts-spectral-opt --tessera-legalize-spectral --tessera-spectral-transpose-plan %s | FileCheck %s --check-prefix=TP
//
// TransposePlanPass picks tile_shape + bank-conflict pad for the intra-stage
// transposes a multi-axis FFT needs.  1D FFTs get
// `tessera.transpose.required = false` (no transpose).

module {
  // 2D FFT — needs a transpose between axis-0 and axis-1.
  func.func @fft2d(%src : memref<128x128xcomplex<f32>>,
                   %dst : memref<128x128xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0, 1], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<128x128xcomplex<f32>>,
         memref<128x128xcomplex<f32>>) -> ()
    return
  }
}

// TP: tessera_spectral.fft
// TP-SAME: tessera.transpose.pad = 1
// TP-SAME: tessera.transpose.required = true
// TP-SAME: tessera.transpose.tile_shapes
// TP-SAME: tessera.transpose.vector_w
