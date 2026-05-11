// RUN: ts-spectral-opt --tessera-legalize-spectral --tessera-spectral-distributed %s | FileCheck %s --check-prefix=DIST
//
// SpectralDistributedFFTPass: 2D FFT on a 2-axis mesh → pencil decomposition
// with one all-to-all between FFT axes.  Confirms axis_split + transposes +
// overlap_token annotations land.

module attributes {tessera.mesh.axes = ["dp", "tp"]} {
  func.func @fft2d_dist(%plan : !tessera_spectral.plan,
                        %src : memref<256x256xcomplex<f32>>,
                        %dst : memref<256x256xcomplex<f32>>) {
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
