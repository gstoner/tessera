// RUN: tessera-opt --tessera-pipeline --allow-unregistered-dialect --verify-each=false %s | FileCheck %s
//
// The verified 1F1B plan is materialized as an explicit dependency order.
// Each action owns a unique logical clock, so a stage is never asked to run
// forward and backward concurrently.  The target runtime may overlap
// independent actions while preserving this serialized semantic order.

module attributes {
  tessera.pipeline_plan = {
    num_stages = 2,
    num_micro_batches = 3,
    interleaved = false
  }
} {
  func.func @pipeline(%x: tensor<64x128xbf16>,
                      %w0: tensor<128x256xbf16>,
                      %w1: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
    %a = "tessera.matmul"(%x, %w0) :
      (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
    %b = "tessera.matmul"(%a, %w1) :
      (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
    return %b : tensor<64x128xbf16>
  }
}

// CHECK: tessera.pipeline_schedule_kind = "1f1b.serialized_dependency_order.v1"
// CHECK-SAME: tessera.pipeline_steps = [
// CHECK-SAME: {clock = 0 : i64, micro_batch = 0 : i64, phase = "forward", region = "warmup", stage = 0 : i64}
// CHECK-SAME: {clock = 1 : i64, micro_batch = 0 : i64, phase = "forward", region = "warmup", stage = 1 : i64}
// CHECK-SAME: {clock = 2 : i64, micro_batch = 1 : i64, phase = "forward", region = "steady", stage = 0 : i64}
// CHECK-SAME: {clock = 5 : i64, micro_batch = 0 : i64, phase = "backward", region = "steady", stage = 0 : i64}
// CHECK-SAME: {clock = 10 : i64, micro_batch = 2 : i64, phase = "backward", region = "cooldown", stage = 1 : i64}
// CHECK-SAME: {clock = 11 : i64, micro_batch = 2 : i64, phase = "backward", region = "cooldown", stage = 0 : i64}
