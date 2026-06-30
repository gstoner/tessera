// P1a fix (review): the 0-view structural ops carry SameOperandsAndResultElementType,
// so a malformed IR that makes a pure shape/stride view silently change dtype is
// rejected at verification (a view is never a dtype conversion — that is CastOp).
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics

func.func @squeeze_dtype_change(%x: tensor<1x4xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %a = "tessera.squeeze"(%x) : (tensor<1x4xf32>) -> tensor<4xi32>
  return %a : tensor<4xi32>
}

// -----

func.func @expand_dtype_change(%x: tensor<1x4xf32>) -> tensor<3x4xf16> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %a = "tessera.expand"(%x) : (tensor<1x4xf32>) -> tensor<3x4xf16>
  return %a : tensor<3x4xf16>
}

// -----

func.func @permute_dtype_change(%x: tensor<2x3xf32>) -> tensor<3x2xi8> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %a = "tessera.permute"(%x) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xi8>
  return %a : tensor<3x2xi8>
}

// -----

func.func @squeeze_non_unit_axis(%x: tensor<2x3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{squeeze axes must select size-1 dimensions}}
  %a = "tessera.squeeze"(%x) {axes = [0]} : (tensor<2x3xf32>) -> tensor<3xf32>
  return %a : tensor<3xf32>
}

// -----

func.func @permute_bad_axis(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{perm axis 2 out of range for rank 2}}
  %a = "tessera.permute"(%x) {perm = [2, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %a : tensor<3x2xf32>
}

// -----

func.func @permute_bad_attr_type(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{perm must be an array of integer axes}}
  %a = "tessera.permute"(%x) {perm = "bad"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %a : tensor<3x2xf32>
}

// -----

func.func @permute_wrong_result_shape(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{permute result dimension 0 mismatch}}
  %a = "tessera.permute"(%x) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %a : tensor<2x3xf32>
}

// -----

func.func @expand_incompatible_dim(%x: tensor<2x3xf32>) -> tensor<4x3xf32> {
  // expected-error @+1 {{expand cannot broadcast input dimension 0}}
  %a = "tessera.expand"(%x) : (tensor<2x3xf32>) -> tensor<4x3xf32>
  return %a : tensor<4x3xf32>
}

// -----

func.func @flatten_wrong_result_shape(%x: tensor<2x3x4xf32>) -> tensor<5x4xf32> {
  // expected-error @+1 {{flatten result dimension 0 mismatch}}
  %a = "tessera.flatten"(%x) {start = 0 : i64, end = 1 : i64} : (tensor<2x3x4xf32>) -> tensor<5x4xf32>
  return %a : tensor<5x4xf32>
}
