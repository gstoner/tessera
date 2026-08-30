// P2 code review (2026-08-29) — SymbolicDimEqualityPass fail-open holes and a
// false positive, all in a pass whose purpose is fail-closed verification.
//
// RUN: tessera-opt --tessera-symdim-equality -split-input-file \
// RUN:   -verify-diagnostics %s

// transposeB stores B as NxK, moving the contracting symbol to the other end of
// the rhs name list. Reading a fixed position reported correct IR as a contract
// violation.
func.func @transposed_contract_is_legal(%a: tensor<4x8xf32>,
                                        %b: tensor<16x8xf32>)
    -> tensor<4x16xf32> {
  %0 = "tessera.matmul"(%a, %b) {
    transposeB = true,
    tessera.dim_names_lhs = ["M", "K"],
    tessera.dim_names_rhs = ["N", "K"]
  } : (tensor<4x8xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// A genuinely mismatched contraction still fails, so the transpose handling did
// not weaken the check.
func.func @mismatched_contract_still_fails(%a: tensor<4x8xf32>,
                                           %b: tensor<8x16xf32>)
    -> tensor<4x16xf32> {
  // expected-error @+1 {{SYMDIM_MATMUL_CONTRACT_VIOLATION}}
  %0 = "tessera.matmul"(%a, %b) {
    tessera.dim_names_lhs = ["M", "K"],
    tessera.dim_names_rhs = ["Q", "N"]
  } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// A binding that does not parse used to be dropped, disabling exactly the
// equation it was written to verify while the pass reported success. These
// sizes violate D = H * Dh (512 != 256), and the typo must not hide that.
// expected-error @below {{SYMDIM_BINDING_MALFORMED}}
func.func @malformed_binding_is_an_error() attributes {
  tessera.dim_bindings = ["D = H ** Dh"],
  tessera.dim_sizes = {D = 512 : i64, H = 8 : i64, Dh = 32 : i64}
} {
  return
}

// -----

// A non-integer dim_sizes value was likewise dropped, silently removing that
// symbol's witness from every binding check.
// expected-error @below {{SYMDIM_DIM_SIZES_MALFORMED}}
func.func @malformed_dim_size_is_an_error() attributes {
  tessera.dim_bindings = ["D = H * Dh"],
  tessera.dim_sizes = {D = "512", H = 8 : i64, Dh = 32 : i64}
} {
  return
}

// -----

// The well-formed spelling of the same violation still fires, which is what
// makes the two errors above worth having.
// expected-error @below {{SYMDIM_BINDING_VIOLATION}}
func.func @well_formed_violation_still_fires() attributes {
  tessera.dim_bindings = ["D = H * Dh"],
  tessera.dim_sizes = {D = 512 : i64, H = 8 : i64, Dh = 32 : i64}
} {
  return
}
