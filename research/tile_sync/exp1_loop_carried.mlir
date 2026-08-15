// Experiment 1 TREATMENT: the canonical staged-pipeline shape.
//
// (a) @staged_tile_loop_carried — the mma's staged data arrives as an scf.for
//     block argument (iter_args), exactly as in any double-buffered pipeline.
//     Same missing-token defect as the control. Question: does
//     WARPSPEC_MMA_NOT_TOKEN_SYNCED fire, or does the pass silently skip the
//     operand because getDefiningOp() is null on a block argument?
func.func @staged_tile_loop_carried(%src: tensor<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t0, %tok0 = tile.async_copy %src : (tensor<64x64xf16>) -> (tensor<64x64xf16>, !tile.async_token)
  %res = scf.for %i = %c0 to %c8 step %c1 iter_args(%staged = %t0) -> (tensor<64x64xf16>) {
    // mma consumes the loop-carried staged tile; no token edge anywhere.
    %acc = tile.mma %staged, %staged : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf32>
    // stage the next tile for the following iteration (token dropped on the floor)
    %next, %ntok = tile.async_copy %src : (tensor<64x64xf16>) -> (tensor<64x64xf16>, !tile.async_token)
    scf.yield %next : tensor<64x64xf16>
  }
  return
}

// (b) @mbarrier_loop_carried — the !tile.mbarrier itself is carried as
//     iter_args (the phase bit flips per iteration, so this is the natural
//     shape). UPDATED after §5.2 landed: the original probe (wait with an
//     `index` "dependency" and no arrive anywhere) is now REJECTED by
//     TILE_WAIT_UNTYPED_DEPENDENCY / TILE_WAIT_GATES_ON_NOTHING — that hole is
//     closed. The remaining §5.3 gap this probe now measures: the arrive and
//     wait use DIFFERENT slots (arrive slot 1, wait slot 0) and the barrier is
//     loop-carried, so the pairing is wrong on hardware — and no pass derives
//     arrive→wait pairing or crosses the block-argument edge, so it still
//     passes silently.
func.func @mbarrier_loop_carried(%src: tensor<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %bar0 = tile.mbarrier.init {slots = 3 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %barN = scf.for %i = %c0 to %c8 step %c1 iter_args(%bar = %bar0) -> (!tile.mbarrier) {
    %desc = tile.tma.descriptor %src {tile_rows = 64 : i64, tile_cols = 64 : i64} : tensor<64x64xf16> -> !tile.tma_descriptor
    tile.tma.copy_async %desc, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 8192 : i64} : (!tile.tma_descriptor, !tile.mbarrier) -> ()
    %tok = tile.mbarrier.arrive_expect_tx %bar {slot = 1 : i64, bytes = 8192 : i64} : !tile.mbarrier -> !tile.mbarrier_token
    tile.mbarrier.wait %bar, %tok {operandSegmentSizes = array<i32: 1, 1, 0>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
    scf.yield %bar : !tile.mbarrier
  }
  return
}

// (c) @same_barrier_conflicting_ids — SSA identity vs string identity: BOTH
//     #tile.barrier annotations sit on the SAME SSA barrier value but carry
//     different tile.barrier_id strings and conflicting expect counts. The
//     arrival-count check keys on the string, not the SSA value. Question:
//     silent?
func.func @same_barrier_conflicting_ids(%src: tensor<64x64xf16>) {
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %desc = tile.tma.descriptor %src {tile_rows = 64 : i64, tile_cols = 64 : i64, tile.barrier_id = "a", tile.barrier = #tile.barrier<kind = "tma", expect = 8192>} : tensor<64x64xf16> -> !tile.tma_descriptor
  tile.tma.copy_async %desc, %bar {operandSegmentSizes = array<i32: 1, 1, 0>, mbarrier_slot = 0 : i64, expect_tx = 4096 : i64, tile.barrier_id = "b", tile.barrier = #tile.barrier<kind = "tma", expect = 4096>} : (!tile.tma_descriptor, !tile.mbarrier) -> ()
  return
}
