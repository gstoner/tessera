// §3.2 claim probes on tile.mbarrier.wait (registered path, op verifier only).
//
// (a) a wait with NO barrier at all — MBarrierWaitOp::verify's dependency
//     requirement is guarded by getBarrier(), so absence of the barrier
//     bypasses BOTH checks. Expect: verifies.
func.func @wait_no_barrier_at_all() {
  "tile.mbarrier.wait"() {operandSegmentSizes = array<i32: 0, 0>, slot = 0 : i64} : () -> ()
  return
}

// (b) the "asynchronous dependency" can be ANY value — here an i32 constant.
//     Expect: verifies (shape check only, no type constraint).
func.func @wait_dependency_is_an_i32() {
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %c = arith.constant 42 : i32
  tile.mbarrier.wait %bar, %c {operandSegmentSizes = array<i32: 1, 1>, slot = 0 : i64} : !tile.mbarrier, i32
  return
}

// (c) the arrive→wait edge: can wait consume the !tile.mbarrier_token that
//     arrive_expect_tx produces? Syntactically yes (Variadic<AnyType>) — but
//     nothing verifies the pairing; this is indistinguishable from (b) to the
//     type system. Expect: verifies identically.
func.func @wait_consumes_mbarrier_token() {
  %bar = tile.mbarrier.init {slots = 1 : i64, phase_bits = 1 : i64} : !tile.mbarrier
  %tok = tile.mbarrier.arrive_expect_tx %bar {slot = 0 : i64, bytes = 8192 : i64} : !tile.mbarrier -> !tile.mbarrier_token
  tile.mbarrier.wait %bar, %tok {operandSegmentSizes = array<i32: 1, 1>, slot = 0 : i64} : !tile.mbarrier, !tile.mbarrier_token
  return
}
