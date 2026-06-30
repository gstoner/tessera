// SD1-2 — tessera.spec_accept_sample op verifier coverage. Positive: draft D i32,
// target_probs (D+1)×V f32, draft_probs D×V f32, accept_u D f32, resid_u 1 f32,
// result (D+2) i32. Negatives pin each emitOpError.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @ok
func.func @ok(%d: tensor<3xi32>, %tp: tensor<4x4xf32>, %dp: tensor<3x4xf32>,
    %au: tensor<3xf32>, %ru: tensor<1xf32>) -> tensor<5xi32> {
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<3xi32>, tensor<4x4xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<1xf32>)
       -> tensor<5xi32>
  return %r : tensor<5xi32>
}

// -----
func.func @bad_target_rows(%d: tensor<3xi32>, %tp: tensor<3x4xf32>,
    %dp: tensor<3x4xf32>, %au: tensor<3xf32>, %ru: tensor<1xf32>) -> tensor<5xi32> {
  // expected-error @+1 {{target_probs must be (D+1) x V f32 (one extra bonus row)}}
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<3xi32>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<1xf32>)
       -> tensor<5xi32>
  return %r : tensor<5xi32>
}

// -----
func.func @bad_resid_u(%d: tensor<3xi32>, %tp: tensor<4x4xf32>,
    %dp: tensor<3x4xf32>, %au: tensor<3xf32>, %ru: tensor<2xf32>) -> tensor<5xi32> {
  // expected-error @+1 {{resid_u must be tensor<1xf32>}}
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<3xi32>, tensor<4x4xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2xf32>)
       -> tensor<5xi32>
  return %r : tensor<5xi32>
}

// -----
func.func @bad_result(%d: tensor<3xi32>, %tp: tensor<4x4xf32>,
    %dp: tensor<3x4xf32>, %au: tensor<3xf32>, %ru: tensor<1xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{result must be tensor<(D+2)xi32>}}
  %r = "tessera.spec_accept_sample"(%d, %tp, %dp, %au, %ru)
       : (tensor<3xi32>, tensor<4x4xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<1xf32>)
       -> tensor<4xi32>
  return %r : tensor<4xi32>
}
