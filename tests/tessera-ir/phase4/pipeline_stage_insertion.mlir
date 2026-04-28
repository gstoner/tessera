// RUN: tessera-opt --tessera-pipeline-stage-insertion %s | FileCheck %s

// Test: PipelineStageInsertionPass correctly inserts tessera.pipeline.send
// and tessera.pipeline.recv ops at cross-stage boundaries, implementing a
// 1F1B micro-batch schedule across pipeline stages.

module attributes {
  tessera.pipeline_plan = {
    num_stages = 2,
    num_micro_batches = 2,
    interleaved = false,
    num_chunks = 1
  }
} {

  // Stage 0: produces activations consumed by Stage 1.
  // The pass should insert pipeline.send at the end of stage 0
  // and pipeline.recv at the start of stage 1.

  // CHECK-LABEL: func.func @stage_0
  func.func @stage_0(%x: memref<64x128xbf16>) -> memref<64x256xbf16> {
    %out = memref.alloc() : memref<64x256xbf16>
    "tessera.matmul"(%x, %out) {
      tessera.layer = {stage = 0},
      tessera.weight_sharding = "col_parallel",
      tessera.tp_axis = "tp"
    } : (memref<64x128xbf16>, memref<64x256xbf16>) -> ()

    // CHECK: tessera.pipeline.send
    // CHECK-SAME: stage = 0
    // CHECK-SAME: micro_batch
    return %out : memref<64x256xbf16>
  }

  // Stage 1: receives activations from Stage 0, applies row-parallel matmul.
  // CHECK-LABEL: func.func @stage_1
  func.func @stage_1(%y: memref<64x256xbf16>) -> memref<64x128xbf16> {
    // CHECK: tessera.pipeline.recv
    // CHECK-SAME: stage = 1
    // CHECK-SAME: micro_batch
    %out = memref.alloc() : memref<64x128xbf16>
    "tessera.matmul"(%y, %out) {
      tessera.layer = {stage = 1},
      tessera.weight_sharding = "row_parallel",
      tessera.tp_axis = "tp"
    } : (memref<64x256xbf16>, memref<64x128xbf16>) -> ()

    return %out : memref<64x128xbf16>
  }

  // Backward stage 1 → backward stage 0 for 1F1B: pass should emit
  // send/recv for gradient tensors too.
  // CHECK-LABEL: func.func @stage_1_bwd
  func.func @stage_1_bwd(%grad_out: memref<64x128xbf16>) -> memref<64x256xbf16> {
    %grad_in = memref.alloc() : memref<64x256xbf16>
    "tessera.matmul"(%grad_out, %grad_in) {
      tessera.layer = {stage = 1},
      tessera.effect = "backward"
    } : (memref<64x128xbf16>, memref<64x256xbf16>) -> ()

    // CHECK: tessera.pipeline.send
    // CHECK-SAME: stage = 1
    return %grad_in : memref<64x256xbf16>
  }

  // CHECK-LABEL: func.func @stage_0_bwd
  func.func @stage_0_bwd(%grad_act: memref<64x256xbf16>) -> memref<64x128xbf16> {
    // CHECK: tessera.pipeline.recv
    // CHECK-SAME: stage = 0
    %grad_x = memref.alloc() : memref<64x128xbf16>
    "tessera.matmul"(%grad_act, %grad_x) {
      tessera.layer = {stage = 0},
      tessera.effect = "backward"
    } : (memref<64x256xbf16>, memref<64x128xbf16>) -> ()

    return %grad_x : memref<64x128xbf16>
  }
}
