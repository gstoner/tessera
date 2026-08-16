// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefixes=X86,ROCM,APPLE
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_softmax(%x: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %0 = "tessera.softmax"(%x) {axis = -1 : i64}
        : (tensor<3x17xf32>) -> tensor<3x17xf32>
    return %0 : tensor<3x17xf32>
  }
}

// X86-LABEL: func.func @x86_softmax
// X86-NOT: tessera.softmax
// X86-NOT: schedule.
// X86-COUNT-1: tile.softmax_kernel
// X86-SAME: accum = "f32"
// X86-SAME: axis = -1
// X86-SAME: exp_mode = "accurate"
// X86-SAME: ftz = false
// X86-SAME: storage = "f32"
// X86-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// X86-SAME: tessera.workgroup_size = 1

// SCHEDULE-LABEL: func.func @x86_softmax
// SCHEDULE: %[[GRAPH:.*]] = tessera.softmax{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: %[[SCHEDULED:.*]] = schedule.softmax %[[GRAPH]]
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: storage = "f32"
// SCHEDULE-SAME: workgroup_size = 1
// SCHEDULE: schedule.artifact
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE: return %[[SCHEDULED]] : tensor<3x17xf32>

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @rocm_reduce(%x: tensor<2x3x5xf32>) -> tensor<2x5xf32> {
    %0 = "tessera.reduce"(%x) {axis = 1 : i64, kind = "mean"}
        : (tensor<2x3x5xf32>) -> tensor<2x5xf32>
    return %0 : tensor<2x5xf32>
  }
}

// ROCM-LABEL: func.func @rocm_reduce
// ROCM-NOT: tessera.reduce
// ROCM-NOT: schedule.
// ROCM-COUNT-1: tile.reduce_kernel
// ROCM-SAME: accum = "f32"
// ROCM-SAME: axis = 1
// ROCM-SAME: inner_is_one = false
// ROCM-SAME: keepdims = false
// ROCM-SAME: kind = "mean"
// ROCM-SAME: nan_mode = "propagate"
// ROCM-SAME: schedule = "serial"
// ROCM-SAME: storage = "f32"
// ROCM-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// ROCM-SAME: tessera.workgroup_size = 256

// -----

// E2E-REAL-5 Apple slice: f32 softmax is admitted for Apple GPU (delegated to
// the native MSL softmax route); reduction is not (fail-closed, see the invalid
// fixture).  The launch tile does not prescribe a workgroup for Apple.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_softmax(%x: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %0 = "tessera.softmax"(%x) {axis = -1 : i64}
        : (tensor<3x17xf32>) -> tensor<3x17xf32>
    return %0 : tensor<3x17xf32>
  }
}

// APPLE-LABEL: func.func @apple_softmax
// APPLE-NOT: tessera.softmax
// APPLE-NOT: schedule.
// APPLE-COUNT-1: tile.softmax_kernel
// APPLE-SAME: accum = "f32"
// APPLE-SAME: axis = -1
// APPLE-SAME: storage = "f32"
// APPLE-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// APPLE-SAME: tessera.workgroup_size = 1
