// RUN: tessera-opt --tessera-pipeline --allow-unregistered-dialect --verify-each=false -split-input-file -verify-diagnostics %s
//
// The 1F1B schedule proof (2026-06-23): the full tessera-pipeline flow
// (partition -> send/recv insertion -> schedule legality) over a well-formed
// pipeline is clean; the four malformed cases each trip a stable PP_* code.

// Chunk 0 — well-formed: two dependent matmuls partition to stages 0/1, the
// 0->1 boundary gets a paired send/recv, and the 1F1B schedule verifies.
// expected-remark@+1 {{pipeline-stage-insertion}}
module attributes {
  tessera.pipeline_plan = {num_stages = 2, num_micro_batches = 2, interleaved = false}
} {
  func.func @ok(%x: tensor<64x128xbf16>, %w0: tensor<128x256xbf16>,
                %w1: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
    %a = "tessera.matmul"(%x, %w0) : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
    %b = "tessera.matmul"(%a, %w1) : (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
    return %b : tensor<64x128xbf16>
  }
}

// -----

// Too few micro-batches: 2 stages need >= 2 micro-batches (Decision #17); the
// two independent matmuls have no 0->1 dataflow, so no comm / remark is emitted.
// expected-error@+1 {{PP_MICRO_BATCHES_TOO_FEW}}
module attributes {
  tessera.pipeline_plan = {num_stages = 2, num_micro_batches = 1, interleaved = false}
} {
  func.func @few(%x: tensor<4x4xbf16>, %w: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
    %a = "tessera.matmul"(%x, %w) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    %b = "tessera.matmul"(%x, %w) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    return %b : tensor<4x4xbf16>
  }
}

// -----

// Empty stage: 2 stages declared but only one op, so the cost partition fills
// stage 0 and leaves stage 1 empty — a hole in the send/recv chain.
// expected-error@+1 {{PP_EMPTY_STAGE}}
module attributes {
  tessera.pipeline_plan = {num_stages = 2, num_micro_batches = 2, interleaved = false}
} {
  func.func @empty(%x: tensor<4x4xbf16>, %w: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
    %a = "tessera.matmul"(%x, %w) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    return %a : tensor<4x4xbf16>
  }
}

// -----

// Stage-skipping value: %a (stage 0) is consumed directly by a stage-2 op,
// skipping stage 1 — the adjacent-only insertion never routes it, so the value
// crosses a stage boundary with no send/recv (pre-tagged to force the skip).
module attributes {
  tessera.pipeline_plan = {num_stages = 3, num_micro_batches = 3, interleaved = false}
} {
  func.func @skip(%x: tensor<4x4xbf16>, %w: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
    // expected-error@+1 {{PP_UNROUTED_CROSS_STAGE_VALUE}}
    %a = "tessera.matmul"(%x, %w) {tessera.pp_stage = 0 : i64} : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    %s1 = "tessera.matmul"(%w, %w) {tessera.pp_stage = 1 : i64} : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    %c = "tessera.matmul"(%a, %w) {tessera.pp_stage = 2 : i64} : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    return %c : tensor<4x4xbf16>
  }
}
