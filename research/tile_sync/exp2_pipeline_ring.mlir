// Experiment 2: does the !tile.pipeline_state ring survive the scf.for
// back-edge, and does anything derive facts from it?
//
// (a) @well_formed_ring — the intended shape: init before the loop, state
//     carried via iter_args, pipeline_advance in the body, yield the advanced
//     state. Should parse + verify + pass legality.
func.func @well_formed_ring() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %ps0 = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  %psN = scf.for %i = %c0 to %c8 step %c1 iter_args(%ps = %ps0) -> (!tile.pipeline_state) {
    %next = tile.pipeline_advance %ps : (!tile.pipeline_state) -> !tile.pipeline_state
    scf.yield %next : !tile.pipeline_state
  }
  return
}

// (b) @stale_ring — the ring is BROKEN across the back-edge: the loop yields
//     the ORIGINAL state %ps, not the advanced %next. Dynamically the pipeline
//     never advances (every iteration re-advances the same stage-0 state) —
//     the classic stalled-ring bug. Question: does any verifier or legality
//     pass object, i.e. does anyone actually walk the ring's def-use chain?
func.func @stale_ring() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %ps0 = tile.pipeline_init {depth = 3 : i64, stage = 0 : i64, phase = 1 : i64, role = "producer"} : !tile.pipeline_state
  %psN = scf.for %i = %c0 to %c8 step %c1 iter_args(%ps = %ps0) -> (!tile.pipeline_state) {
    %next = tile.pipeline_advance %ps : (!tile.pipeline_state) -> !tile.pipeline_state
    scf.yield %ps : !tile.pipeline_state   // <-- yields the stale state
  }
  return
}
