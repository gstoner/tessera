// RUN: %tessera_strict_opt %s -split-input-file -verify-diagnostics -o /dev/null

func.func @guided_denoise_region_bad_timestep(
    %ref: tensor<12x64xf32>, %fav: tensor<12x64xf32>,
    %unfav: tensor<12x64xf32>) -> tensor<12x64xf32> {
  // expected-error @+1 {{timestep must be non-negative}}
  %guided = "tessera.guided_denoise_region"(%ref, %fav, %unfav)
      {timestep = -1 : i64, schedule = "vp_alpha_bar", gamma = 1.0 : f64}
      : (tensor<12x64xf32>, tensor<12x64xf32>, tensor<12x64xf32>)
        -> tensor<12x64xf32>
  return %guided : tensor<12x64xf32>
}

// -----

func.func @guided_denoise_region_bad_shape(
    %ref: tensor<12x64xf32>, %fav: tensor<11x64xf32>,
    %unfav: tensor<12x64xf32>) -> tensor<12x64xf32> {
  // expected-error @+1 {{guided_denoise_region favored shapes must match}}
  %guided = "tessera.guided_denoise_region"(%ref, %fav, %unfav)
      {timestep = 4 : i64, schedule = "vp_alpha_bar", gamma = 1.0 : f64}
      : (tensor<12x64xf32>, tensor<11x64xf32>, tensor<12x64xf32>)
        -> tensor<12x64xf32>
  return %guided : tensor<12x64xf32>
}
