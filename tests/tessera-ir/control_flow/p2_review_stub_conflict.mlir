// P2 code review (2026-08-29) — MaterializeControlPayloadPass overwrote a body
// stub shared by two control ops.
//
// emitOpList rewrites the stub in place. The intra-op case (thenStub ==
// elseStub) was guarded; two different control ops resolving to one symbol were
// not, so the second materialization replaced the first op's body and both
// loops then executed the second payload — with the first op's payload
// attributes already stripped, leaving the wrong body as the only body.
//
// RUN: tessera-opt --tessera-materialize-control-payload -split-input-file \
// RUN:   -verify-diagnostics %s

func.func private @loop_body(%a: tensor<4xf32>) -> tensor<4xf32>

func.func @differing_payloads_conflict(%x: tensor<4xf32>) -> tensor<4xf32> {
  %r1 = "tessera.control_for"(%x) {
    body = @loop_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64, body_opcodes = array<i32: 11>,
    body_in0 = array<i32: 1>, body_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  // expected-error @+1 {{CONTROL_PAYLOAD_STUB_CONFLICT}}
  %r2 = "tessera.control_for"(%r1) {
    body = @loop_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64, body_opcodes = array<i32: 20>,
    body_in0 = array<i32: 1>, body_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r2 : tensor<4xf32>
}

// -----

// Two ops may legitimately share one symbol when the payload is identical — the
// stub already holds the right body, so this must not be refused.
func.func private @loop_body(%a: tensor<4xf32>) -> tensor<4xf32>

func.func @identical_payloads_share_one_stub(%x: tensor<4xf32>)
    -> tensor<4xf32> {
  %r1 = "tessera.control_for"(%x) {
    body = @loop_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64, body_opcodes = array<i32: 11>,
    body_in0 = array<i32: 1>, body_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  %r2 = "tessera.control_for"(%r1) {
    body = @loop_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64, body_opcodes = array<i32: 11>,
    body_in0 = array<i32: 1>, body_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r2 : tensor<4xf32>
}
