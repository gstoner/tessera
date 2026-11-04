// E2E-REAL-6 Apple CPU (2026-11-03): static f32 rank-2 matmul and last-axis
// softmax on apple_silicon_cpu are native scheduled products, each lowered to
// one Tile launch the Accelerate delegate consumes.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @apple_cpu_matmul(%a: tensor<7x9xf32>, %b: tensor<9x5xf32>) -> tensor<7x5xf32> {
    %0 = tessera.matmul %a, %b : (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
    return %0 : tensor<7x5xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_cpu_matmul
// SCHEDULE: %[[GRAPH:.*]] = tessera.matmul{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.matmul %[[GRAPH]]
// SCHEDULE-SAME: accum = "f32"
// SCHEDULE-SAME: activation = "none"
// SCHEDULE-SAME: arch = "apple_silicon_cpu"
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: storage = "f32"
// SCHEDULE: schedule.artifact
// SCHEDULE-SAME: shape_key = "M=7;N=5;K=9;dtype=f32"

// TILE-LABEL: func.func @apple_cpu_matmul
// TILE-NOT: tessera.matmul
// TILE-NOT: schedule.
// TILE: tile.matmul_kernel
// TILE-SAME: numeric_policy = {accum = "f32", storage = "f32"}

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @apple_cpu_softmax(%x: tensor<6x11xf32>) -> tensor<6x11xf32> {
    %0 = tessera.softmax %x : (tensor<6x11xf32>) -> tensor<6x11xf32>
    return %0 : tensor<6x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_cpu_softmax
// SCHEDULE: schedule.softmax
// SCHEDULE-SAME: accum = "f32"
// SCHEDULE-SAME: arch = "apple_silicon_cpu"
// SCHEDULE-SAME: exp_mode = "accurate"
// SCHEDULE-SAME: storage = "f32"
// SCHEDULE-SAME: workgroup_size = 1 : i64

// TILE-LABEL: func.func @apple_cpu_softmax
// TILE-NOT: tessera.softmax
// TILE: tile.softmax_kernel
// TILE-SAME: storage = "f32"
// TILE-SAME: tessera.workgroup_size = 1 : i64
