// P2 code review (2026-08-29) — a saved scf.while sized its residual tape from
// the loop INIT's dynamic extents while insertState sliced with the CURRENT
// iteration's extent, with nothing requiring the two to agree.
//
// scf.while lets a `tensor<?xf32>` state yield any extent, so a state that grows
// across iterations wrote an out-of-bounds insert_slice — undefined behaviour
// rather than a diagnostic. The sibling generic_for path already demands a
// declared shape envelope for exactly this (requireEveryDynamic, with its own
// "exceeds its slot envelope" assert); until while carries one, the dynamic case
// is refused (Decision #21a).
//
// RUN: tessera-opt --tessera-autodiff-paired -verify-diagnostics %s

module {
  func.func @saved_while_dynamic_state(%x: tensor<?xf32>) -> tensor<?xf32>
      attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    // expected-error @+1 {{AUTODIFF_WHILE_DYNAMIC_STATE}}
    %count, %out = "scf.while"(%c0, %x) ({
    ^bb0(%i: index, %carry: tensor<?xf32>):
      %continue = arith.cmpi slt, %i, %c3 : index
      scf.condition(%continue) %i, %carry : index, tensor<?xf32>
    }, {
    ^bb0(%i: index, %carry: tensor<?xf32>):
      %next = "tessera.tanh"(%carry) : (tensor<?xf32>) -> tensor<?xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<?xf32>
    }) {tessera.autodiff.checkpoint_indices = array<i64: 1, 2>,
        tessera.autodiff.checkpoint_policy = "save",
        tessera.autodiff.max_iters = 3 : i64,
        tessera.autodiff.residual_digest = "1111111111111111111111111111111111111111111111111111111111111111",
        tessera.autodiff.residual_schema = "tessera.region_residual_abi.v1",
        tessera.structured_cfg.digest = "0000000000000000000000000000000000000000000000000000000000000000"} :
        (index, tensor<?xf32>) -> (index, tensor<?xf32>)
    return %out : tensor<?xf32>
  }
}
