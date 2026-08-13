// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefixes=X86,ROCM
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_depth_attention(
      %query: tensor<8xf32>, %sources: tensor<7x3x8xf32>)
      -> tensor<3x8xf32> {
    %0 = "tessera.depth_attn"(%query, %sources) {
      eps = 1.000000e-06 : f64,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<8xf32>, tensor<7x3x8xf32>) -> tensor<3x8xf32>
    return %0 : tensor<3x8xf32>
  }
}

// X86-LABEL: func.func @x86_depth_attention
// X86-NOT: tessera.depth_attn
// X86-NOT: schedule.
// X86-COUNT-1: tile.depth_attention_kernel
// X86-SAME: accum = "f32"
// X86-SAME: eps = {{[^,]+}} : f64
// X86-SAME: merge_recurrence = "max_shifted_pairwise_merge_v1"
// X86-SAME: reassociable = true
// X86-SAME: softmax = "f32"
// X86-SAME: source_tile = 7
// X86-SAME: statistics_recurrence = "rms_key_online_softmax_stats_v1"
// X86-SAME: storage = "f32"
// X86-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// X86-SAME: tessera.workgroup_size = 1

// SCHEDULE-LABEL: func.func @x86_depth_attention
// SCHEDULE: %[[GRAPH:.*]] = tessera.depth_attn{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: %[[SCHEDULED:.*]] = schedule.depth_attention %[[GRAPH]]
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: merge_recurrence = "max_shifted_pairwise_merge_v1"
// SCHEDULE-SAME: source_count = 7
// SCHEDULE-SAME: statistics_recurrence = "rms_key_online_softmax_stats_v1"
// SCHEDULE: schedule.artifact
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE: return %[[SCHEDULED]] : tensor<3x8xf32>

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @rocm_depth_attention(
      %query: tensor<8xf32>, %sources: tensor<9x2x8xf32>)
      -> tensor<2x8xf32> {
    %0 = "tessera.depth_attn"(%query, %sources) {
      eps = 1.000000e-06 : f64,
      numeric_policy = {accum = "fp32", softmax = "fp32", storage = "fp32"}
    } : (tensor<8xf32>, tensor<9x2x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// ROCM-LABEL: func.func @rocm_depth_attention
// ROCM-NOT: tessera.depth_attn
// ROCM-NOT: schedule.
// ROCM-COUNT-1: tile.depth_attention_kernel
// ROCM-SAME: source_tile = 9
// ROCM-SAME: tessera.workgroup_size = 256
