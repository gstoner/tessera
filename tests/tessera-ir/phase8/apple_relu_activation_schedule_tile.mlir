// APPLE-NTBI-2 slice 6 (2026-11-24): an isolated static rank-two f16 ReLU on
// apple7 is a native scheduled activation lowered to one Tile launch, carrying
// the SM120 numerical contract (propagated NaNs, f16 zeros to +0) for the
// compiler-synthesized MSL consumer.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_relu_f16(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.relu %x : (tensor<5x11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_relu_f16
// SCHEDULE: %[[GRAPH:.*]] = tessera.relu{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.activation %[[GRAPH]]
// SCHEDULE-NOT: approximation
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: kind = "relu"
// SCHEDULE-SAME: nan_mode = "propagate"
// SCHEDULE-SAME: schedule = "serial"
// SCHEDULE-SAME: storage = "f16"
// SCHEDULE-SAME: workgroup_size = 1 : i64
// SCHEDULE-SAME: zero_mode = "positive"
// SCHEDULE: schedule.artifact

// TILE-LABEL: func.func @apple_relu_f16
// TILE-NOT: tessera.relu
// TILE-NOT: schedule.
// TILE: tile.activation_kernel
// TILE-SAME: kind = "relu"
// TILE-SAME: nan_mode = "propagate"
// TILE-SAME: storage = "f16"
// TILE-SAME: tessera.workgroup_size = 1 : i64
// TILE-SAME: zero_mode = "positive"

// -----

// f32 storage has no apple7 ReLU consumer: the op is not claimed and keeps
// its existing route.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_relu_f32(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.relu %x : (tensor<5x11xf32>) -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_relu_f32
// SCHEDULE-NOT: schedule.activation
// SCHEDULE: return

// -----

// A ReLU that is not the whole function body is not claimed by name alone.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_relu_chain(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.relu %x : (tensor<5x11xf16>) -> tensor<5x11xf16>
    %1 = tessera.relu %0 : (tensor<5x11xf16>) -> tensor<5x11xf16>
    return %1 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_relu_chain
// SCHEDULE-NOT: schedule.activation
// SCHEDULE: return

// -----

// APPLE-NTBI-2 slice 8: bf16 takes the same synthesized route with the SM120
// bf16 zero contract (signed zeros preserved).
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_relu_bf16(%x: tensor<5x11xbf16>) -> tensor<5x11xbf16> {
    %0 = tessera.relu %x : (tensor<5x11xbf16>) -> tensor<5x11xbf16>
    return %0 : tensor<5x11xbf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_relu_bf16
// SCHEDULE: schedule.activation
// SCHEDULE-SAME: kind = "relu"
// SCHEDULE-SAME: storage = "bf16"
// SCHEDULE-SAME: zero_mode = "preserve"

// TILE-LABEL: func.func @apple_relu_bf16
// TILE: tile.activation_kernel
// TILE-SAME: storage = "bf16"
// TILE-SAME: zero_mode = "preserve"
