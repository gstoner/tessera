// guided_denoise_region is a metadata marker for CGG orchestration. It is not
// a fused model kernel; the executable math remains denoiser outputs plus
// tessera.score_combine.
//
// RUN: %tessera_strict_opt %s | FileCheck %s

func.func @guided_denoise_region_visible(
    %ref: tensor<12x64xf32>, %fav: tensor<12x64xf32>,
    %unfav: tensor<12x64xf32>) -> tensor<12x64xf32> {
  // CHECK-LABEL: func.func @guided_denoise_region_visible
  // CHECK: tessera.guided_denoise_region
  // CHECK-SAME: gamma = 7.500000e-01 : f64
  // CHECK-SAME: preference = "quality"
  // CHECK-SAME: schedule = "vp_alpha_bar"
  // CHECK-SAME: timestep = 4 : i64
  %guided = "tessera.guided_denoise_region"(%ref, %fav, %unfav)
      {timestep = 4 : i64, schedule = "vp_alpha_bar",
       gamma = 7.500000e-01 : f64, preference = "quality"}
      : (tensor<12x64xf32>, tensor<12x64xf32>, tensor<12x64xf32>)
        -> tensor<12x64xf32>
  return %guided : tensor<12x64xf32>
}
