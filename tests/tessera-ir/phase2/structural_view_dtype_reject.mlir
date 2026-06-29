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
