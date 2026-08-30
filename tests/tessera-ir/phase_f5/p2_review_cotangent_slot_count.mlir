// P2 code review (2026-08-29) — AdjointCollectiveInsertionPass indexed
// tessera.autodiff.arg_cotangents by argument number without checking its size.
//
// ArrayAttr indexes its underlying ArrayRef unchecked, so on the fleet's NDEBUG
// builds a short array reads adjacent attribute storage; whatever bits come
// back can dyn_cast to a non-empty StringAttr and splice a collective onto an
// unrelated return operand. AutodiffPass always writes one slot per argument,
// so a short array means hand-written or drifted IR — the same input class the
// existing arity bail already anticipates.
//
// RUN: tessera-opt --tessera-adjoint-collective-insertion -split-input-file \
// RUN:   -verify-diagnostics %s

// expected-error @below {{ADJOINT_COLLECTIVE_COTANGENT_SLOT_COUNT}}
func.func @short_cotangent_array(%a: tensor<4xf32>, %b: tensor<4xf32>,
                                 %c: tensor<4xf32>, %d: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {tessera.autodiff = "reverse",
                tessera.autodiff.arg_cotangents = ["d_a", "d_b"],
                tessera.weight_sharding = {arg_0 = "row"}} {
  return %a : tensor<4xf32>
}

// -----

// One slot per argument — including empty strings for unpopulated slots — is
// the shape AutodiffPass emits and must keep working.
func.func @one_slot_per_argument(%a: tensor<4xf32>, %b: tensor<4xf32>)
    -> tensor<4xf32>
    attributes {tessera.autodiff = "reverse",
                tessera.autodiff.arg_cotangents = ["d_a", ""],
                tessera.weight_sharding = {arg_0 = "row"}} {
  return %a : tensor<4xf32>
}
