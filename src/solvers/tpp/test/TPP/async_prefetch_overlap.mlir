// tpp-async-prefetch software-pipelines schedule.prefetch ops: rotating
// double-buffer stages + dependency-safe hoist of overlap-policy prefetches
// above preceding compute. An into="host"/overlap="none" prefetch (the LSA
// cold-pool staging space) is recorded but NOT overlapped/hoisted.
//
// RUN: tessera-opt %s -tpp-async-prefetch -allow-unregistered-dialect | FileCheck %s

func.func @pipe(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @pipe

  // Both overlap="compute" prefetches hoist above the preceding compute (their
  // operand %b is independent of it) and take distinct double-buffer stages.
  // CHECK: schedule.prefetch
  // CHECK-SAME: tpp.prefetch.hoisted = true
  // CHECK-SAME: tpp.prefetch.overlapped = true
  // CHECK-SAME: tpp.prefetch.stage = 0
  // CHECK: schedule.prefetch
  // CHECK-SAME: tpp.prefetch.hoisted = true
  // CHECK-SAME: tpp.prefetch.overlapped = true
  // CHECK-SAME: tpp.prefetch.stage = 1
  // CHECK: comp.matmul
  %c = "comp.matmul"(%a) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %p0 = "schedule.prefetch"(%b) {into = "shared", overlap = "compute"}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %p1 = "schedule.prefetch"(%b) {into = "shared", overlap = "compute"}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>

  // into="host" + overlap="none": recorded, but not overlapped and not hoisted.
  // CHECK: schedule.prefetch
  // CHECK-SAME: into = "host"
  // CHECK-SAME: tpp.prefetch.hoisted = false
  // CHECK-SAME: tpp.prefetch.overlapped = false
  %p2 = "schedule.prefetch"(%c) {into = "host", overlap = "none"}
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %p1 : tensor<4x4xf32>
}
