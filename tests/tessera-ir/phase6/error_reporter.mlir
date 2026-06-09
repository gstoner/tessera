// RUN: tessera-opt -split-input-file --allow-unregistered-dialect -verify-diagnostics \
// RUN:   %s -tessera-shape-inference -tessera-error-reporter

// ErrorReporterPass diagnostics, verified the MLIR-canonical way with
// -verify-diagnostics + inline expected-* annotations (robust to stderr/stdout
// ordering and generic vs pretty printing).
//
// 2026-06: un-XFAIL'd.  Previously fragile: a misplaced loc(...) before the
// type signature (parse error), FileCheck -LABEL anchors that don't survive
// generic printing, and per-section flag divergence under one input file.
// Rewritten to -verify-diagnostics so each section self-checks its diagnostic.

// == Shape error is reported with op name + expected/got shapes ==

func.func @test_error_reporter_shape_mismatch(
    %x : tensor<4x8xf32>,
    %w : tensor<8x16xf32>) -> tensor<4x16xf32> {

  // expected-error @+1 {{shape mismatch on op 'tessera.matmul': expected [4, 99] but got [4, 16]}}
  %out = "tessera.matmul"(%x, %w) {
      "tessera.expected_shape" = [4 : i64, 99 : i64]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

  return %out : tensor<4x16xf32>
}

// -----

// Clean IR — no diagnostics expected (verified by -verify-diagnostics: any
// unexpected error would fail the run).

func.func @test_no_errors(
    %a : tensor<2x4xf32>,
    %w : tensor<4x4xf32>) -> tensor<2x4xf32> {

  // No tessera.expected_shape attr → shape inference matches → no diagnostic.
  %out = "tessera.matmul"(%a, %w) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>

  return %out : tensor<2x4xf32>
}

// Note: the Python-origin loc note (PyLoc in a fused loc) and the error-limit
// cap are reported at the *Python* source location (e.g. model.py:42), not at
// this .mlir buffer, so -verify-diagnostics cannot anchor them by line.  Those
// two behaviours are covered by the Python-level diagnostics unit tests
// (tests/unit/test_diagnostics*.py / the ErrorReporter API tests).
