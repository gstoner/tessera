// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V3a (2026-05-22) — affine non-product bindings.
//
// V5's parser only accepted bindings of the form `D = H * Dh`
// (single product term).  V3a extends to sum-of-products:
// `D = H * Dh + K` (two terms, second is a single-symbol product).
// All previously-accepted V5 forms keep working unchanged.
//
// Diagnostic format split (Sprint V3a):
//   single-term:   "product of RHS = N"   (V5 wording, unchanged)
//   multi-term:    "value of RHS (sum of products) = N"

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: sum-of-products binding evaluates correctly.
// D = H * Dh + K = 4*16 + 7 = 71.  dim_sizes claims D = 71. ✓
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @affine_sum_holds
func.func @affine_sum_holds(%x: tensor<2x71xf32>) -> tensor<2x71xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh + K"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, K = 7 : i64, D = 71 : i64 }
    } {
  return %x : tensor<2x71xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: SYMDIM_BINDING_VIOLATION on multi-term binding.
// D = H * Dh + K = 4*16 + 7 = 71 but dim_sizes claims D = 72.
// Diagnostic uses the V3a sum-of-products wording.
// ─────────────────────────────────────────────────────────────────────────

// expected-error @+1 {{SYMDIM_BINDING_VIOLATION: binding 'D = H * Dh + K' violated: D = 72 but value of RHS (sum of products) = 71}}
func.func @affine_sum_broken(%x: tensor<2x72xf32>) -> tensor<2x72xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh + K"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, K = 7 : i64, D = 72 : i64 }
    } {
  return %x : tensor<2x72xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: V5 single-product binding still works (backward compat).
// D = H * Dh = 4*16 = 64.  dim_sizes claims D = 64. ✓
// Diagnostic wording (if it had fired) would have been V5's
// "product of RHS = ...".
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @v5_single_product_still_works
func.func @v5_single_product_still_works(%x: tensor<2x64xf32>) -> tensor<2x64xf32>
    attributes {
      tessera.dim_bindings = ["D = H * Dh"],
      tessera.dim_sizes = { H = 4 : i64, Dh = 16 : i64, D = 64 : i64 }
    } {
  return %x : tensor<2x64xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: sum of three singletons (A + B + C).  Tests that the
// parser handles terms that are bare symbols (1-symbol products).
// Total = A + B + C = 4 + 5 + 6 = 15.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @affine_sum_of_singletons
func.func @affine_sum_of_singletons(%x: tensor<15xf32>) -> tensor<15xf32>
    attributes {
      tessera.dim_bindings = ["Total = A + B + C"],
      tessera.dim_sizes = { A = 4 : i64, B = 5 : i64, C = 6 : i64, Total = 15 : i64 }
    } {
  return %x : tensor<15xf32>
}
