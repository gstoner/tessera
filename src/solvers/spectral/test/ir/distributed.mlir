// RUN: ts-spectral-opt --tessera-legalize-spectral --tessera-spectral-distributed %s | FileCheck %s --check-prefix=DIST
//
// SpectralDistributedFFTPass: 2D FFT on a 2-axis mesh → pencil decomposition
// with one all-to-all between FFT axes.  Confirms axis_split + transposes +
// overlap_token annotations land.

module attributes {tessera.mesh.axes = ["dp", "tp"]} {
  func.func @fft2d_dist(%src : memref<256x256xcomplex<f32>>,
                        %dst : memref<256x256xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0, 1], elem_precision = "fp16", acc_precision = "f32",
      scaling = "none", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<256x256xcomplex<f32>>,
         memref<256x256xcomplex<f32>>) -> ()
    return
  }
}

// DIST: tessera_spectral.fft
// DIST-SAME: tessera.dist.axis_split
// DIST-SAME: tessera.dist.local_only = false
// DIST-SAME: tessera.dist.overlap_token = "comm_q_default"
// DIST-SAME: tessera.dist.transposes
