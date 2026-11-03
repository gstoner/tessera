// E2E-REAL-6 Apple GPU GELU (2026-11-03): an isolated static rank-two GELU
// on apple7 is a native scheduled activation, lowered to one Tile launch.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_gelu_f16(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.gelu %x : (tensor<5x11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_gelu_f16
// SCHEDULE: %[[GRAPH:.*]] = tessera.gelu{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.activation %[[GRAPH]]
// SCHEDULE-SAME: approximation = "tanh"
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: kind = "gelu"
// SCHEDULE-SAME: schedule = "serial"
// SCHEDULE-SAME: storage = "f16"
// SCHEDULE-SAME: workgroup_size = 1 : i64
// SCHEDULE-SAME: zero_mode = "preserve"
// SCHEDULE: schedule.artifact

// TILE-LABEL: func.func @apple_gelu_f16
// TILE-NOT: tessera.gelu
// TILE-NOT: schedule.
// TILE: tile.activation_kernel
// TILE-SAME: approximation = "tanh"
// TILE-SAME: kind = "gelu"
// TILE-SAME: storage = "f16"
// TILE-SAME: tessera.workgroup_size = 1 : i64

// -----

// A GELU that is not the whole function body keeps its existing route: it is
// not claimed by name alone.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_gelu_chain(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.gelu %x : (tensor<5x11xf32>) -> tensor<5x11xf32>
    %1 = tessera.gelu %0 : (tensor<5x11xf32>) -> tensor<5x11xf32>
    return %1 : tensor<5x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_gelu_chain
// SCHEDULE-NOT: schedule.activation
// SCHEDULE: return

// -----

// Evidence never transfers: GELU is not a scheduled activation on SM120.
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @sm120_gelu(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.gelu %x : (tensor<5x11xf32>) -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @sm120_gelu
// SCHEDULE-NOT: schedule.activation
// SCHEDULE: return
