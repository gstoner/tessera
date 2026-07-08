// RUN: tessera-opt %s -tpp-vectorize | FileCheck %s
//
// VectorizeTPP derives a concrete vector width + tile shape from the field
// shape and element type.  For f32 (256-bit SIMD -> 8 lanes) with a 64-wide
// innermost dim, vector_width = 8 and the innermost tile is a multiple of 8.
func.func @grad(%x: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %y = "tpp.grad"(%x) : (tensor<128x64xf32>) -> tensor<128x64xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.tile_shape = [8, 64]
  // CHECK-SAME: tpp.vector_width = 8
  // CHECK-SAME: tpp.vectorized
  return %y : tensor<128x64xf32>
}

// A short innermost row (5) must not be over-vectorised: vector_width clamps
// to the largest power of two <= 5, i.e. 4.
func.func @grad_short(%x: tensor<16x5xf32>) -> tensor<16x5xf32> {
  %y = "tpp.grad"(%x) : (tensor<16x5xf32>) -> tensor<16x5xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.vector_width = 4
  return %y : tensor<16x5xf32>
}
