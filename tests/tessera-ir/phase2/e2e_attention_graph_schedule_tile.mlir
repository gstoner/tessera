// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefixes=X86,APPLE
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE

// E2E-REAL-5A: the shared rank-4 forward attention boundary is target-neutral —
// the same canonical tile.attention_kernel is produced for each admitted target,
// keyed by that architecture's own schedule digest and LSE policy.

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_attention(%q: tensor<1x2x8x16xf32>, %k: tensor<1x2x8x16xf32>,
                           %v: tensor<1x2x8x16xf32>) -> tensor<1x2x8x16xf32> {
    %0 = tessera.flash_attn %q, %k, %v {
        scale = 0.25, causal = false, window_left = -1, window_right = -1,
        softcap = 0.0, dropout_p = 0.0, dropout_seed = 0, head_dim = 16,
        operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : (tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>)
        -> tensor<1x2x8x16xf32>
    return %0 : tensor<1x2x8x16xf32>
  }
}

// X86-LABEL: func.func @x86_attention
// X86-NOT: tessera.flash_attn
// X86-NOT: schedule.
// X86-COUNT-1: tile.attention_kernel
// X86-SAME: accum = "f32"
// X86-SAME: gqa = false
// X86-SAME: storage = "f32"
// X86-SAME: tessera.backward_lse_policy = "save_lse"
// X86-SAME: tessera.backward_lse_selection = "saved"
// X86-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"

// SCHEDULE-LABEL: func.func @x86_attention
// SCHEDULE: %[[GRAPH:.*]] = tessera.flash_attn{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: %[[SCHEDULED:.*]] = schedule.attention %[[GRAPH]]
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: backward_lse_policy = "save_lse"
// SCHEDULE: schedule.artifact
// SCHEDULE-SAME: hash = "[[HASH]]"

// -----

// Apple GPU slice: f32 storage, GQA (4 query heads over 2 KV heads), shared
// head/value dim, no window, no dropout — the envelope of the status-returning
// GQA MSL ABI.  Apple owns an explicit `apple7_recompute` backward LSE policy
// (its backward recomputes m/l per query row and its ABI takes no LSE buffer);
// it does not inherit x86's save_lse.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_attention(%q: tensor<1x4x8x16xf32>, %k: tensor<1x2x8x16xf32>,
                             %v: tensor<1x2x8x16xf32>) -> tensor<1x4x8x16xf32> {
    %0 = tessera.flash_attn %q, %k, %v {
        scale = 0.25, causal = false, window_left = -1, window_right = -1,
        softcap = 0.0, dropout_p = 0.0, dropout_seed = 0, head_dim = 16,
        operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : (tensor<1x4x8x16xf32>, tensor<1x2x8x16xf32>, tensor<1x2x8x16xf32>)
        -> tensor<1x4x8x16xf32>
    return %0 : tensor<1x4x8x16xf32>
  }
}

// APPLE-LABEL: func.func @apple_attention
// APPLE-NOT: tessera.flash_attn
// APPLE-NOT: schedule.
// APPLE-COUNT-1: tile.attention_kernel
// APPLE-SAME: accum = "f32"
// APPLE-SAME: gqa = true
// APPLE-SAME: storage = "f32"
// APPLE-SAME: tessera.backward_lse_policy = "apple7_recompute"
// APPLE-SAME: tessera.backward_lse_selection = "recompute"
// APPLE-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"

// SCHEDULE-LABEL: func.func @apple_attention
// SCHEDULE: %[[AGRAPH:.*]] = tessera.flash_attn{{.*}}schedule.artifact_hash = "[[AHASH:[0-9a-f]+]]"
// SCHEDULE: %[[ASCHED:.*]] = schedule.attention %[[AGRAPH]]
// SCHEDULE-SAME: artifact_hash = "[[AHASH]]"
// SCHEDULE-SAME: backward_lse_policy = "apple7_recompute"
// SCHEDULE-SAME: backward_lse_selection = "recompute"
