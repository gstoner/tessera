// SD1-4 — tessera.target_verify op verifier coverage. Positive: tokens S i32,
// logits S×V f32, result S×V f32 (the (prefix_len+1)×V target-scoring contract).
// Negatives pin each emitOpError.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @ok
func.func @ok(%t: tensor<3xi32>, %l: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %r = "tessera.target_verify"(%t, %l) : (tensor<3xi32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}

// -----
func.func @bad_logits_rows(%t: tensor<3xi32>, %l: tensor<2x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{logits must be S x V f32}}
  %r = "tessera.target_verify"(%t, %l) : (tensor<3xi32>, tensor<2x4xf32>) -> tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}

// -----
func.func @bad_result(%t: tensor<3xi32>, %l: tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{target_probs must match logits shape (S x V f32)}}
  %r = "tessera.target_verify"(%t, %l) : (tensor<3xi32>, tensor<3x4xf32>) -> tensor<3x5xf32>
  return %r : tensor<3x5xf32>
}

// -----
func.func @bad_tokens(%t: tensor<3x1xi32>, %l: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{tokens must be rank-1 i32}}
  %r = "tessera.target_verify"(%t, %l) : (tensor<3x1xi32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}
