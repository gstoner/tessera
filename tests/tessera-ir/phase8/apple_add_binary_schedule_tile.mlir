// APPLE-NTBI-2 slice 7 (2026-11-24): an isolated static rank-two f16 add on
// apple7 is a native scheduled binary kernel lowered to one Tile launch for
// the compiler-synthesized MSL consumer: f32 arithmetic rounded once to f16,
// NaNs propagated, one serial element walk at workgroup 1.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_add_f16(%a: tensor<5x11xf16>, %b: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.add %a, %b : (tensor<5x11xf16>, tensor<5x11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_add_f16
// SCHEDULE: %[[GRAPH:.*]] = tessera.add{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.binary_kernel %[[GRAPH]]
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: kind = "add"
// SCHEDULE-SAME: nan_mode = "propagate"
// SCHEDULE-SAME: output_storage = "f16"
// SCHEDULE-SAME: schedule = "serial"
// SCHEDULE-SAME: storage = "f16"
// SCHEDULE-SAME: workgroup_size = 1 : i64
// SCHEDULE: schedule.artifact

// TILE-LABEL: func.func @apple_add_f16
// TILE-NOT: tessera.add
// TILE-NOT: schedule.
// TILE: %[[N:.*]] = arith.constant 55 : i64
// TILE: tile.elementwise_kernel %{{.*}}, %{{.*}}, %{{.*}}, %[[N]]
// TILE-SAME: family = "binary"
// TILE-SAME: kind = "add"
// TILE-SAME: schedule = "serial"
// TILE-SAME: storage = "f16"
// TILE-SAME: tessera.workgroup_size = 1 : i64

// -----

// f32 storage has no apple7 add consumer: the op is not claimed.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_add_f32(%a: tensor<5x11xf32>, %b: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.add %a, %b : (tensor<5x11xf32>, tensor<5x11xf32>) -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_add_f32
// SCHEDULE-NOT: schedule.binary_kernel
// SCHEDULE: return

// -----

// An add that is not the whole function body is not claimed by name alone.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_add_chain(%a: tensor<5x11xf16>, %b: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.add %a, %b : (tensor<5x11xf16>, tensor<5x11xf16>) -> tensor<5x11xf16>
    %1 = tessera.add %0, %b : (tensor<5x11xf16>, tensor<5x11xf16>) -> tensor<5x11xf16>
    return %1 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_add_chain
// SCHEDULE-NOT: schedule.binary_kernel
// SCHEDULE: return
