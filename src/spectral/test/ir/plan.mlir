\
// RUN: ts-spectral-opt %s | FileCheck %s
tosa.module {
  // Create a plan
  %plan = "tessera_spectral.plan"() {axes = [0], elem_precision = "fp8_e4m3",
                                     acc_precision = "f32", scaling = "blockfp_per_stage",
                                     inplace = false, is_real_input = false, norm_policy = "backward"} : () -> !any
  // CHECK: tessera_spectral.plan
}
