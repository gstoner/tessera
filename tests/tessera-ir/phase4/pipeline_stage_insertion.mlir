// RUN: tessera-opt --tessera-pipeline-stage-insertion --allow-unregistered-dialect --verify-each=false %s | FileCheck %s

// Test: PipelineStageInsertionPass inserts tessera.pipeline.send /
// tessera.pipeline.recv at cross-stage boundaries for a 1F1B schedule.
//
// 2026-06: un-XFAIL'd.  Two corrections vs the old fixture:
//   (1) value-semantics tensor matmul (MLIR-23 verifier requires one result);
//   (2) the staged ops live in ONE function with real SSA dataflow across the
//       stage boundary — the pass keys off a stage-k result being *consumed by*
//       a stage-(k+1) op, which separate per-stage functions never expressed.
// The pass emits lightweight tessera.pipeline.* markers (consumed downstream),
// so --allow-unregistered-dialect + --verify-each=false round-trip them.

module attributes {
  tessera.schedule_digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  tessera.pipeline_schedule_schema = "tessera.pipeline_schedule.v1",
  tessera.pipeline_steps = [{action_id = "fixture", clock = 0, depends_on = [], micro_batch = 0, phase = "F", rank = 0, stage = 0}],
  tessera.pp_num_stages = 2,
  tessera.pp_num_micro_batches = 2,
  tessera.pp_interleaved = false
} {

  // The unregistered pipeline.* marker ops force generic module printing, so
  // match on the generic sym_name rather than the pretty `func.func @pipeline`.
  // CHECK: sym_name = "pipeline"
  func.func @pipeline(%x: tensor<64x128xbf16>, %w0: tensor<128x256xbf16>,
                      %w1: tensor<256x128xbf16>) -> tensor<64x128xbf16> {

    // Stage 0 produces %a, consumed by the stage-1 matmul → boundary.
    %a = "tessera.matmul"(%x, %w0) {tessera.layer = {stage = 0}} :
        (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>

    // CHECK: tessera.pipeline.send
    // CHECK-SAME: from_stage = 0
    // CHECK-SAME: micro_batch
    // CHECK-SAME: tessera.schedule_digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    // CHECK: tessera.pipeline.recv
    // CHECK-SAME: micro_batch
    // CHECK-SAME: to_stage = 1
    // CHECK: }) {tessera.schedule_digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    %b = "tessera.matmul"(%a, %w1) {tessera.layer = {stage = 1}} :
        (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>

    return %b : tensor<64x128xbf16>
  }
}
