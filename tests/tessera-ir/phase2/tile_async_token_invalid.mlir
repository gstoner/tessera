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
