// RUN: tessera-opt --tessera-pm-verify %s | FileCheck %s
//
// Regression net for the PMV11 half of the async-copy contract reconciliation
// (2026-08-10). PMV11VerifierPass used to REQUIRE a `stage` attribute on every
// `tile.async_copy` — a copy of the deleted ScheduleOps.cpp stage-model
// verifier — which rejected the production TileIRLoweringPass output: the
// typed form carries its dependency as a `!tile.async_token` SSA edge and has
// no stage at all. Under the declared contract (TileOps.td) `stage` is an
// optional grouping key: absent is legal, present must be >= 0.
//
// If this fixture starts failing with "'stage' must be >= 0", someone has
// reintroduced the required-stage model. Don't.

// CHECK-LABEL: func.func @pm_verify_accepts_typed_token_form
func.func @pm_verify_accepts_typed_token_form(%A: tensor<16x16xf16>) {
  // CHECK: tile.async_copy
  // CHECK-SAME: !tile.async_token
  %tile, %tok = "tile.async_copy"(%A)
      : (tensor<16x16xf16>) -> (tensor<16x16xf16>, !tile.async_token)
  "tile.wait_async"(%tok) : (!tile.async_token) -> ()
  return
}

// CHECK-LABEL: func.func @pm_verify_accepts_legacy_stage_form
func.func @pm_verify_accepts_legacy_stage_form(%a: memref<128xf16>) {
  // CHECK: tile.async_copy
  "tile.async_copy"(%a) {stage = 0 : i64} : (memref<128xf16>) -> ()
  "tile.wait_async"() {stage = 0 : i64} : () -> ()
  return
}
