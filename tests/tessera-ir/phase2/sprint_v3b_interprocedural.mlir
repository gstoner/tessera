// RUN: tessera-opt --tessera-symdim-equality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V3b (2026-05-22) — interprocedural dim-name tracking.
//
// SymbolicDimEqualityPass now resolves direct `func.call @callee(...)`
// sites via a module-level `SymbolTable`.  When the callee declares
// `tessera.arg_dim_names`, the pass cross-checks the caller's
// propagated dim-names against the callee's declared names
// position-by-position.  Mismatch ⇒ SYMDIM_CALL_ARG_MISMATCH.
//
// V3b also reads `tessera.ret_dim_names` on the callee and seeds the
// call's result values, so dim-names flow ACROSS the call boundary
// into subsequent ops in the caller.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: caller's arg dim-names match callee's. ✓
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @callee_ok
func.func @callee_ok(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]],
      tessera.ret_dim_names = [["B", "D"]]
    } {
  return %x : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @caller_ok
func.func @caller_ok(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  %y = func.call @callee_ok(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: caller propagates ["B", "D"] but callee declares ["B", "S"].
// SYMDIM_CALL_ARG_MISMATCH on arg 0.
// ─────────────────────────────────────────────────────────────────────────

func.func @callee_expects_bs(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "S"]]
    } {
  return %x : tensor<4x8xf32>
}

func.func @caller_mismatch(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  // expected-error @+1 {{SYMDIM_CALL_ARG_MISMATCH: call to '@callee_expects_bs' arg 0 propagated dim-names disagree with callee's tessera.arg_dim_names}}
  %y = func.call @callee_expects_bs(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: ret_dim_names propagate through the call boundary.
// Caller's downstream transpose checks the inferred names.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @callee_returns_bd
func.func @callee_returns_bd(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]],
      tessera.ret_dim_names = [["B", "D"]]
    } {
  return %x : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @caller_propagates_ret
func.func @caller_propagates_ret(%x: tensor<4x8xf32>) -> tensor<8x4xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  %y = func.call @callee_returns_bd(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // y now carries ["B", "D"].  The transpose declares the same;
  // V2-flow cross-check is silent (no mismatch).
  %z = "tessera.transpose"(%y) {
        tessera.dim_names_in = ["B", "D"],
        tessera.dim_names_out = ["D", "B"]
       } : (tensor<4x8xf32>) -> tensor<8x4xf32>
  return %z : tensor<8x4xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE backward-compat: callee has no arg_dim_names → V3b skips
// the cross-check (nothing to compare against).
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @callee_no_decl
func.func @callee_no_decl(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %x : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @caller_no_callee_decl
func.func @caller_no_callee_decl(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {
      tessera.arg_dim_names = [["B", "D"]]
    } {
  %y = func.call @callee_no_decl(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}
