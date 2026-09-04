// RUN: tessera-opt %s --split-input-file --tessera-verify-metadata-obligation \
// RUN:   | FileCheck %s
//
// W1.3 / Decision #32 — the drops the verifier ACCEPTS.
//
// The snapshot is hand-written here rather than produced by
// `--tessera-record-metadata`. That is deliberate: a controlled drop needs a
// lowering that loses an attribute, and after the `LowerMatmulToTileMMA` fix
// (see metadata_obligation.mlir) no pass in the pipeline loses one. Writing the
// "before" state directly is what lets these cases test the verifier's decision
// rather than some other pass's behaviour.
//
// Encoding: {scope: {attr_name: ["<printed value>", <count>, ...]}}. Flat pairs
// rather than a nested dictionary because a printed attribute is not a legal
// MLIR identifier and so cannot be a dictionary key.
//
// This is the accepting half. A verifier exercised only by its rejections
// proves it says no, never that it says yes to the right things — and an
// over-strict boundary verifier would be worse than none, because the first
// team to hit a false positive would delete it.

// A drop whose information moved into the level's TYPE. This is the reason
// `numeric_policy` will eventually carry at the Tile boundary, once W1.1
// parameterizes `!tile.fragment` on the accumulator.
// CHECK-LABEL: func.func @represented_in_type
module attributes {
  tessera.metadata_snapshot = {
    represented_in_type = {numeric_policy = ["{accum = \22fp32\22}", 1]}}
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

// Declared debt. The `:W1.1` suffix is mandatory — see the invalid fixture for
// the bare `not_yet_carried` case. This is the escape hatch that keeps the
// verifier adoptable: a boundary that cannot yet carry an attribute records who
// will fix it instead of being exempted or having the gate switched off.
// CHECK-LABEL: func.func @declared_debt
module attributes {
  tessera.metadata_snapshot = {
    declared_debt = {distribution = ["\22block\22", 1]}}
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
  tessera.metadata_snapshot = {
    module_level_declaration = {target = ["\22sm90\22", 1]}},
  tessera.lowering.dropped = {target = "target_invariant"}
} {
  func.func @module_level_declaration(%a: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: return
    return %a : tensor<8xf32>
  }
}

// -----

// Re-spelling the NAME is not a drop. `tessera.layout` at one level and
// `tile.layout` at the next are the same fact, and #32 requires the INFORMATION
// to survive, not the string. A verifier keyed on the exact attribute name
// would report a false drop every time a level renamed one — and false
// positives are how a gate gets disabled.
// CHECK-LABEL: func.func @respelled_name_is_not_dropped
module attributes {
  tessera.metadata_snapshot = {
    respelled_name_is_not_dropped = {layout = ["\22row_major\22", 1]}}
} {
  func.func @respelled_name_is_not_dropped(%a: tensor<8xf32>) -> tensor<8xf32> {
    // CHECK: return
    %0 = "tessera.cast"(%a) {tile.layout = "row_major"}
      : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// Re-expressing the VALUE is a declared drop, not a free one.
//
// This is the case the production pipelines actually hit: Graph IR states
// `tessera.layout = "row_major"` and `TileIRLoweringPass` restates it as
// `#tile.layout<shard = ...>`. The name survives, the value does not, and the
// verifier cannot tell a re-encoding from a replaced accumulator policy — so
// the lowering says which it is. `TileIRLoweringPass` now stamps exactly this
// attribute, per function and only where a re-expression really happened.
// CHECK-LABEL: func.func @re_expressed_value
module attributes {
  tessera.metadata_snapshot = {
    re_expressed_value = {layout = ["\22row_major\22", 1]}}
} {
  func.func @re_expressed_value(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.lowering.dropped = {layout = "re_expressed"}} {
    // CHECK: return
    %0 = "tessera.cast"(%a) {
      tile.layout = #tile.layout<shard = [8] : [1] on ["reg"],
                                 replica = [] : [] on [], offset = 0>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// The PYTHON->MLIR FRONTIER (FRONTEND-IR-MEDIUM-1 item iii).
//
// Region privileges (Decision #2), the ConstraintSolver's facts (Decision #4)
// and symbolic dimension names never enter MLIR at all, so
// `--tessera-record-metadata` can never record them: it only sees what is
// already in the IR. The frontier is the one boundary whose "before" lives in
// Python, so the frontend writes this snapshot itself
// (`graph_ir.declare_frontier_debt`) and the verifier is untouched -- it
// decodes the snapshot unfiltered, finds each fact absent after the boundary,
// and demands an attributed reason exactly as it does for `numeric_policy`.
//
// This is the shape `@tessera.jit` now emits; the paired Python test is
// tests/unit/test_frontier_metadata_obligation.py.
// CHECK-LABEL: func.func @frontier
module attributes {
  tessera.metadata_snapshot = {
    frontier = {constraints = ["Divisible('K', 8)", 1],
                dim_names = ["[M, K]", 1],
                region_privilege = ["read", 2]}}
} {
  func.func @frontier(%a: tensor<8x16xf32>) -> tensor<8x16xf32>
      attributes {
        tessera.lowering.dropped = {
          constraints = "not_yet_carried:FRONTEND-IR-MEDIUM-1",
          dim_names = "not_yet_carried:FRONTEND-IR-MEDIUM-1",
          region_privilege = "not_yet_carried:FRONTEND-IR-MEDIUM-1"
        }
      } {
    // CHECK: return
    return %a : tensor<8x16xf32>
  }
}
