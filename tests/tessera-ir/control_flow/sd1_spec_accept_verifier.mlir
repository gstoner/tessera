// SD1 — tessera.spec_accept op verifier coverage. Positive: draft P×D i32,
// target P×(D+1) i32, result tensor<3xi32>. Negative cases pin each emitOpError.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @ok
func.func @ok(%d: tensor<3x4xi32>, %t: tensor<3x5xi32>) -> tensor<3xi32> {
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xi32>, tensor<3x5xi32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}

// -----
func.func @bad_target_depth(%d: tensor<3x4xi32>, %t: tensor<3x4xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{target depth must be draft depth + 1 (the bonus column)}}
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}

// -----
func.func @bad_paths(%d: tensor<3x4xi32>, %t: tensor<2x5xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{draft and target must have the same num_paths}}
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xi32>, tensor<2x5xi32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}

// -----
func.func @bad_dtype(%d: tensor<3x4xf32>, %t: tensor<3x5xf32>) -> tensor<3xi32> {
  // expected-error @+1 {{must be i32 (token ids)}}
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xf32>, tensor<3x5xf32>) -> tensor<3xi32>
  return %r : tensor<3xi32>
}

// -----
func.func @bad_result(%d: tensor<3x4xi32>, %t: tensor<3x5xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{result must be tensor<3xi32>}}
  %r = "tessera.spec_accept"(%d, %t) : (tensor<3x4xi32>, tensor<3x5xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
