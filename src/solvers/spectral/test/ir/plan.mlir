// RUN: ts-spectral-opt %s | FileCheck %s
//
// Round-trips a tessera_spectral.plan op so we know the dialect is
// registered and the assembly format accepts our policy attributes.

module {
  func.func @plan_roundtrip() {
    %plan = "tessera_spectral.plan"() {
      axes = [0 : i64],
      elem_precision = "fp8_e4m3",
      acc_precision = "f32",
      scaling = "blockfp_per_stage",
      inplace = false,
      is_real_input = false,
      norm_policy = "backward"
    } : () -> (!tessera_spectral.plan)
    return
  }
}

// CHECK: tessera_spectral.plan
// CHECK-SAME: elem_precision = "fp8_e4m3"
// CHECK-SAME: norm_policy = "backward"
