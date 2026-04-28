// RUN: tessera-opt %s -tessera-shape-inference -tessera-error-reporter 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: tessera-opt %s -tessera-shape-inference -tessera-error-reporter --warn-shape=true 2>&1 | FileCheck %s --check-prefix=WARN

// ---- Shape error is reported with op name --------------------------------

// ERR-LABEL: @test_error_reporter_shape_mismatch
// ERR: error: shape mismatch on op 'tessera.matmul'
// ERR: expected [4, 99]
// ERR: but got [4, 16]

// WARN-LABEL: @test_error_reporter_shape_mismatch
// WARN: warning: shape mismatch on op 'tessera.matmul'

func.func @test_error_reporter_shape_mismatch(
    %x : tensor<4x8xf32>,
    %w : tensor<8x16xf32>) -> tensor<4x16xf32> {

  %out = "tessera.matmul"(%x, %w) {
      "tessera.expected_shape" = [4 : i64, 99 : i64]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

  return %out : tensor<4x16xf32>
}

// -----

// Clean IR — no errors expected.
// RUN: tessera-opt %s -tessera-shape-inference -tessera-error-reporter | FileCheck %s --check-prefix=CLEAN

// CLEAN-LABEL: @test_no_errors
// CLEAN-NOT: error:
func.func @test_no_errors(
    %a : tensor<2x4xf32>,
    %b : tensor<2x4xf32>) -> tensor<2x4xf32> {

  %out = "tessera.elementwise_add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

  return %out : tensor<2x4xf32>
}

// -----

// Python location note when PyLoc is present in fused loc.
// RUN: tessera-opt %s -tessera-shape-inference -tessera-error-reporter 2>&1 | FileCheck %s --check-prefix=PYLOC

// PYLOC: originated at Python
// PYLOC: model.py:42

func.func @test_python_loc_note(
    %x : tensor<3x3xf32>,
    %w : tensor<3x5xf32>) -> tensor<3x5xf32> {

  %out = "tessera.matmul"(%x, %w) {
      "tessera.expected_shape" = [3 : i64, 99 : i64]
  } loc(fused["model.py":42:0, "model.py":42:0]) :
      (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>

  return %out : tensor<3x5xf32>
}

// -----

// Error limit: only first N errors emitted.
// RUN: tessera-opt %s -tessera-shape-inference '-tessera-error-reporter=error-limit=1' 2>&1 | FileCheck %s --check-prefix=LIMIT

// LIMIT: error-limit reached

func.func @test_error_limit_1(
    %x : tensor<1x8xf32>, %w1 : tensor<8x4xf32>,
    %y : tensor<1x8xf32>, %w2 : tensor<8x4xf32>) -> tensor<1x4xf32> {

  // Two mismatches — only the first should appear before the limit note.
  %o1 = "tessera.matmul"(%x, %w1) {
      "tessera.expected_shape" = [1 : i64, 99 : i64]
  } : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>

  %o2 = "tessera.matmul"(%y, %w2) {
      "tessera.expected_shape" = [1 : i64, 88 : i64]
  } : (tensor<1x8xf32>, tensor<8x4xf32>) -> tensor<1x4xf32>

  return %o1 : tensor<1x4xf32>
}
