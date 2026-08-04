// RUN: tessera-opt %s --split-input-file --tessera-verify-metadata-obligation \
// RUN:   | FileCheck %s
// W1.3 / Decision #32 — the drops the verifier ACCEPTS.
//
// The snapshot is hand-written here rather than produced by
// `--tessera-record-metadata`. That is deliberate: a controlled drop needs a
// lowering that loses an attribute, and after the `LowerMatmulToTileMMA` fix
// (see metadata_obligation.mlir) no pass in the pipeline loses one. Writing the
// "before" state directly is what lets these cases test the verifier's decision
// rather than some other pass's behaviour.
//
// This is the accepting half. A verifier exercised only by its rejections
// proves it says no, never that it says yes to the right things -- and an
// over-strict boundary verifier would be worse than none, because the first
// team to hit a false positive would delete it.

// A drop whose information moved into the level's TYPE. This is the reason
// `numeric_policy` will eventually carry at the Tile boundary, once W1.1
// parameterizes `!tile.fragment` on the accumulator.
// CHECK-LABEL: func.func @represented_in_type
module attributes {
  tessera.metadata_snapshot = {represented_in_type = ["numeric_policy"]}
} {
  func.func @represented_in_type(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {
        tessera.lowering.dropped = {numeric_policy = "represented_in_type"}
      } {
    // CHECK: return
    return %a : tensor<8xf32>
  }
}

// -----

// Declared debt. The `:W1.1` suffix is mandatory -- see the invalid fixture for
// the bare `not_yet_carried` case. This is the escape hatch that keeps the
// verifier adoptable: a boundary that cannot yet carry an attribute records who
// will fix it instead of being exempted or having the gate switched off.
// CHECK-LABEL: func.func @declared_debt
module attributes {
  tessera.metadata_snapshot = {declared_debt = ["distribution"]}
} {
  func.func @declared_debt(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {
        tessera.lowering.dropped = {distribution = "not_yet_carried:W1.1"}
      } {
    // CHECK: return
    return %a : tensor<8xf32>
  }
}

// -----

// The declaration may live on the MODULE, for whole-module lowerings where no
// single function owns the decision.
// CHECK-LABEL: func.func @module_level_declaration
module attributes {
  tessera.metadata_snapshot = {module_level_declaration = ["target"]},
  tessera.lowering.dropped = {target = "target_invariant"}
} {
  func.func @module_level_declaration(%a: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: return
    return %a : tensor<8xf32>
  }
}

// -----

// Re-spelling is NOT a drop. `tessera.layout` at one level and `tile.layout` at
// the next are the same fact, and #32 requires the INFORMATION to survive, not
// the string. A verifier keyed on the exact attribute name would report a false
// drop every time a level renamed one -- and false positives are how a gate
// gets disabled.
// CHECK-LABEL: func.func @respelled_is_not_dropped
module attributes {
  tessera.metadata_snapshot = {respelled_is_not_dropped = ["layout"]}
} {
  func.func @respelled_is_not_dropped(%a: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: return
    %0 = "tessera.cast"(%a) {tile.layout = "swizzled"}
      : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
