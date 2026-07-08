// RUN: ts-spectral-opt --tessera-spectral-pipeline %s | FileCheck %s --check-prefix=PIPE
//
// End-to-end: legalize → mxp → transpose-plan → autotune → distributed →
// lower-to-target-ir, run as a single named pipeline alias on a 2D FFT
// with a 2-axis mesh.  Every annotation any prior pass attaches must
// survive subsequent passes (no overwrites).

module attributes {tessera.target = "nvidia", tessera.mesh.axes = ["dp", "tp"]} {
  func.func @fft2d_full(%src : memref<512x512xcomplex<f32>>,
                        %dst : memref<512x512xcomplex<f32>>) {
    %plan = "tessera_spectral.plan"() {
      axes = [0, 1], elem_precision = "fp16", acc_precision = "f32",
      scaling = "blockfp_per_stage", inplace = false, is_real_input = false,
      norm_policy = "backward"} : () -> !tessera_spectral.plan
    "tessera_spectral.fft"(%plan, %src, %dst)
      : (!tessera_spectral.plan,
         memref<512x512xcomplex<f32>>,
         memref<512x512xcomplex<f32>>) -> ()
    return
  }
}

// PIPE: tessera_spectral.fft
// Each pass's signature attribute must coexist:
// PIPE-SAME: tessera.autotune.cache_key
// PIPE-SAME: tessera.dist.axis_split
// PIPE-SAME: tessera.mxp.block_size
// PIPE-SAME: tessera.spectral.legalized
// PIPE-SAME: tessera.spectral.stages
// PIPE-SAME: tessera.target_ir.backend = "nvidia"
// PIPE-SAME: tessera.target_ir.call = "ts_stockham_r4_nvidia"
// PIPE-SAME: tessera.transpose.required = true
