// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1 — the typed async-token form must REJECT, not merely accept.
//
// A type that is only ever accepted proves nothing: it would satisfy
// Decision #29's letter (it has a consumer) while carrying no contract. These
// are the cases the verifier exists to catch.

// A wait whose token comes from something that is not a tile.async_copy is a
// dependency no backend can honor — nothing ever signals that token.
func.func @wait_on_a_token_from_the_wrong_producer() {
  // CHECK: error: 'tile.wait_async' op !tile.async_token operand must be produced by a tile.async_copy
  %bogus = "tile.fake_token_source"() : () -> !tile.async_token
  tile.wait_async %bogus : (!tile.async_token) -> ()
  return
}

// -----

// A block argument has no producer at all, so the copy it supposedly waits on
// cannot be identified.
func.func @wait_on_a_block_argument_token(%tok: !tile.async_token) {
  // CHECK: error: 'tile.wait_async' op !tile.async_token operand must be produced by a tile.async_copy, not a block argument
  tile.wait_async %tok : (!tile.async_token) -> ()
  return
}

// -----

// The typed form is a copy: it needs a source to copy from.
func.func @typed_copy_without_operands() {
  // CHECK: error: 'tile.async_copy' op typed form expects at least a source operand
  %tok = tile.async_copy : () -> !tile.async_token
  tile.wait_async %tok : (!tile.async_token) -> ()
  return
}

// -----

// `stage` is an OPTIONAL legacy-form grouping key (declared contract,
// reconciled 2026-08-10) — but a present key must be well formed. A negative
// stage would silently break the legacy lane's copy/wait matching
// (TileBufferReusePass keys lifetimes on it), so it fails closed here.
func.func @negative_stage_on_async_copy(%a: memref<128xf16>) {
  // CHECK: error: 'tile.async_copy' op TILE_ASYNC_STAGE_NEGATIVE
  "tile.async_copy"(%a) {stage = -1 : i64} : (memref<128xf16>) -> ()
  return
}

// -----

// Same rule on the wait side.
func.func @negative_stage_on_wait_async() {
  // CHECK: error: 'tile.wait_async' op TILE_ASYNC_STAGE_NEGATIVE
  "tile.wait_async"() {stage = -2 : i64} : () -> ()
  return
}
