// RUN: tessera-opt %s -tessera-shape-inference | FileCheck %s
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-shape-inference{fail-on-unknown=false})' | FileCheck %s

// Sprint V8 (2026-05-22) — un-XFAIL'd.  The fixture was previously
// blocked because it referenced `tessera.elementwise_add` and
// `tessera.flash_attention` op names that the Graph IR dialect
// doesn't register.  V8 rewrote the cases to use the canonical
// registered ops (`tessera.matmul`, `tessera.flash_attn`,
// `tessera.reshape`, `tessera.transpose`) and updated
// ShapeInferencePass to also dispatch on the canonical
// `tessera.flash_attn` (kept the legacy `flash_attention` spelling
// as a soft alias for backward-compat).
//
// Each case checks that ShapeInferencePass annotates the op with
// `tessera.inferred_shape = [...]` matching the result tensor's
// static shape.

// ─────────────────────────────────────────────────────────────────────────
// MATMUL — static (M, K) × (K, N) → (M, N)
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @test_matmul_shape_inference
func.func @test_matmul_shape_inference(
    %x : tensor<4x512xf32>,
    %w : tensor<512x256xf32>) -> tensor<4x256xf32> {

  // CHECK: tessera.inferred_shape = [4, 256]
  %out = "tessera.matmul"(%x, %w) {transposeA = false, transposeB = false}
      : (tensor<4x512xf32>, tensor<512x256xf32>) -> tensor<4x256xf32>

  return %out : tensor<4x256xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// TRANSPOSE — rank-preserving permutation; inferred shape is the
// permutation of the input's static dims.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @test_transpose_shape
func.func @test_transpose_shape(%a : tensor<8x16xf32>) -> tensor<16x8xf32> {

  // CHECK: tessera.inferred_shape = [16, 8]
  %out = "tessera.transpose"(%a) {
      "tessera.perm" = [1 : i64, 0 : i64]
  } : (tensor<8x16xf32>) -> tensor<16x8xf32>

  return %out : tensor<16x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// FLASH ATTENTION — output shape = Q's shape (B, H, S, D).
// Canonical op name is tessera.flash_attn; V8 ShapeInferencePass also
// accepts the legacy `tessera.flash_attention` spelling.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @test_flash_attn_shape
func.func @test_flash_attn_shape(
    %q : tensor<2x8x512x64xf32>,
    %k : tensor<2x8x512x64xf32>,
    %v : tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf32> {

  // CHECK: tessera.inferred_shape = [2, 8, 512, 64]
  %out = "tessera.flash_attn"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 64 : i64}
      : (tensor<2x8x512x64xf32>,
         tensor<2x8x512x64xf32>,
         tensor<2x8x512x64xf32>) -> tensor<2x8x512x64xf32>

  return %out : tensor<2x8x512x64xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// RESHAPE with explicit target shape attribute.
// ─────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────
// MATMUL CHAIN — shape flows from one matmul into the next.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @test_matmul_chain_propagation
func.func @test_matmul_chain_propagation(
    %x  : tensor<1x64xf32>,
    %w1 : tensor<64x32xf32>,
    %w2 : tensor<32x16xf32>) -> tensor<1x16xf32> {

  // CHECK: tessera.inferred_shape = [1, 32]
  %h1 = "tessera.matmul"(%x, %w1) {transposeA = false, transposeB = false}
      : (tensor<1x64xf32>, tensor<64x32xf32>) -> tensor<1x32xf32>

  // CHECK: tessera.inferred_shape = [1, 16]
  %h2 = "tessera.matmul"(%h1, %w2) {transposeA = false, transposeB = false}
      : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>

  return %h2 : tensor<1x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// Shape mismatch: tessera.expected_shape ≠ inferred → pass sets
// tessera.actual_shape so the downstream ErrorReporterPass can attribute
// the mismatch to a Python source span.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @test_shape_mismatch_annotation
func.func @test_shape_mismatch_annotation(
    %x : tensor<4x8xf32>,
    %w : tensor<8x16xf32>) -> tensor<4x16xf32> {

  // CHECK: tessera.actual_shape
  %out = "tessera.matmul"(%x, %w) {
      transposeA = false, transposeB = false,
      "tessera.expected_shape" = [4 : i64, 99 : i64]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

  return %out : tensor<4x16xf32>
}
