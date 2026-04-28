// RUN: tessera-opt %s -tessera-shape-inference | FileCheck %s
// RUN: tessera-opt %s -tessera-shape-inference --fail-on-unknown=false | FileCheck %s

// CHECK-LABEL: func @test_matmul_shape_inference
func.func @test_matmul_shape_inference(
    %x : tensor<4x512xf32>,
    %w : tensor<512x256xf32>) -> tensor<4x256xf32> {

  // CHECK: tessera.inferred_shape = [4, 256]
  %out = "tessera.matmul"(%x, %w) : (tensor<4x512xf32>, tensor<512x256xf32>) -> tensor<4x256xf32>

  return %out : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: func @test_elementwise_shape
func.func @test_elementwise_shape(
    %a : tensor<8x16xf32>,
    %b : tensor<8x16xf32>) -> tensor<8x16xf32> {

  // CHECK: tessera.inferred_shape = [8, 16]
  %out = "tessera.elementwise_add"(%a, %b) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>

  return %out : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func @test_flash_attn_shape
func.func @test_flash_attn_shape(
    %q : tensor<2x8x512x64xf32>,
    %k : tensor<2x8x512x64xf32>,
    %v : tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf32> {

  // CHECK: tessera.inferred_shape = [2, 8, 512, 64]
  %out = "tessera.flash_attention"(%q, %k, %v) : (
      tensor<2x8x512x64xf32>,
      tensor<2x8x512x64xf32>,
      tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf32>

  return %out : tensor<2x8x512x64xf32>
}

// -----

// CHECK-LABEL: func @test_reshape_with_target_shape
func.func @test_reshape_with_target_shape(
    %x : tensor<4x8xf32>) -> tensor<32xf32> {

  // CHECK: tessera.inferred_shape = [32]
  %out = "tessera.reshape"(%x) {
      "tessera.target_shape" = [32 : i64]
  } : (tensor<4x8xf32>) -> tensor<32xf32>

  return %out : tensor<32xf32>
}

// -----

// CHECK-LABEL: func @test_matmul_chain_propagation
func.func @test_matmul_chain_propagation(
    %x  : tensor<1x64xf32>,
    %w1 : tensor<64x32xf32>,
    %w2 : tensor<32x16xf32>) -> tensor<1x16xf32> {

  // CHECK: tessera.inferred_shape = [1, 32]
  %h1 = "tessera.matmul"(%x, %w1) : (tensor<1x64xf32>, tensor<64x32xf32>) -> tensor<1x32xf32>

  // CHECK: tessera.inferred_shape = [1, 16]
  %h2 = "tessera.matmul"(%h1, %w2) : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>

  return %h2 : tensor<1x16xf32>
}

// -----

// Shape mismatch: expected != inferred → actual_shape is set for ErrorReporter
// CHECK-LABEL: func @test_shape_mismatch_annotation
func.func @test_shape_mismatch_annotation(
    %x : tensor<4x8xf32>,
    %w : tensor<8x16xf32>) -> tensor<4x16xf32> {

  // CHECK: tessera.actual_shape
  %out = "tessera.matmul"(%x, %w) {
      "tessera.expected_shape" = [4 : i64, 99 : i64]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

  return %out : tensor<4x16xf32>
}
