// RUN: tessera-opt -split-input-file -verify-diagnostics %s
//
// CAKE Phase 1 §5.2 negative fixtures (2026-08-15) — each case used to verify
// silently (measured in compiler_enhancement.md §5.1.1 / research/tile_sync/)
// and is now rejected by ODS constraints or fail-closed op verifiers. A typing
// change that only ever accepts proves nothing (Decision #10a).

// A wait whose "dependency" is an arbitrary SSA value. An i32 is not a
// completion edge; the measured hole let a loop induction variable satisfy
// the old shape-only check.
func.func @wait_dependency_untyped() {
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c = arith.constant 42 : i32
  // expected-error @+1 {{TILE_WAIT_UNTYPED_DEPENDENCY}}
  tile.mbarrier.wait %bar, %c {operandSegmentSizes = array<i32: 1, 0, 1>, slot = 0 : i64} : !tile.mbarrier, i32
  return
}

// -----

// A wait that gates on nothing: no mbarrier token, no async-token dependency.
// The bare form used to verify because the old dependency requirement was
// guarded behind the (optional) barrier operand.
func.func @wait_gates_on_nothing() {
  // expected-error @+1 {{TILE_WAIT_GATES_ON_NOTHING}}
  "tile.mbarrier.wait"() {operandSegmentSizes = array<i32: 0, 0, 0>, slot = 0 : i64} : () -> ()
  return
}

// -----

// A TMA copy that nothing can ever retire: no SSA mbarrier, no async-token
// result, no legacy grouping key.
func.func @tma_copy_gates_on_nothing(%src: tensor<64x64xf16>) {
  %desc = tile.tma.descriptor %src {tile_rows = 64 : i64, tile_cols = 64 : i64} : tensor<64x64xf16> -> !tile.tma_descriptor
  // expected-error @+1 {{TILE_TMA_COPY_GATES_ON_NOTHING}}
  "tile.tma.copy_async"(%desc) {operandSegmentSizes = array<i32: 1, 0, 0>, mbarrier_slot = 0 : i64} : (!tile.tma_descriptor) -> ()
  return
}

// -----

// A TMA descriptor whose source is a scalar — there is no memory to describe.
func.func @tma_descriptor_scalar_source(%x: i32) {
  // expected-error @+1 {{op operand #0 must be}}
  %desc = tile.tma.descriptor %x {tile_rows = 64 : i64, tile_cols = 64 : i64} : i32 -> !tile.tma_descriptor
  return
}

// -----

// TMEM indices must be index-typed.
func.func @tmem_load_float_index(%i: f32) {
  %handle = tile.tmem.allocate {bytes = 64 : i64, alignment = 16 : i64} : !tile.tmem
  // expected-error @+1 {{op operand #1 must be}}
  %v = "tile.tmem.load"(%handle, %i) : (!tile.tmem, f32) -> f32
  return
}

// -----

// tile.alloc's layout is a typed #tile.layout, rejected at parse when it is
// not (Decision #21a hoist — the check used to live only in the verifier).
func.func @alloc_untyped_layout() {
  // expected-error @+1 {{attribute 'layout' failed to satisfy constraint}}
  %buffer = tile.alloc {space = "smem", bytes = 256 : i64, layout = "row_major"} : !tile.buffer
  return
}
