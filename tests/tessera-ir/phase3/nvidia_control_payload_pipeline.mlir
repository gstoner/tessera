// RUN: tessera-opt --tessera-nvidia-pipeline-sm120 \
// RUN:   --lower-tile-to-nvidia='sm=120' %s | FileCheck %s

func.func private @stub()

func.func @payload_for(%x: tensor<4xf32>) -> tensor<4xf32> {
  %r = "tessera.control_for"(%x) {
    body = @stub, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64, body_opcodes = array<i32: 1>,
    body_out_id = 1 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @payload_if(%flag: tensor<1xf32>) -> tensor<4xf32> {
  %r = "tessera.control_if"(%flag) {
    then_branch = @stub, else_branch = @stub, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 1>, then_out_id = 1 : i64,
    else_opcodes = array<i32: 1>, else_out_id = 1 : i64
  } : (tensor<1xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @payload_while(%x: tensor<4xf32>) -> tensor<4xf32> {
  %r = "tessera.control_while"(%x) {
    body = @stub, cond = @stub, carry_arg_index = 0 : i64,
    max_iters = 4 : i64, body_opcodes = array<i32: 1>,
    body_out_id = 1 : i64, cond_opcodes = array<i32: 20>,
    cond_out_id = 1 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

func.func @payload_scan(%init: tensor<4xf32>, %xs: tensor<6x4xf32>)
    -> (tensor<4xf32>, tensor<6x4xf32>) {
  %carry, %ys = "tessera.control_scan"(%init, %xs) {
    body = @stub, trip = 6 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<6x4xf32>)
      -> (tensor<4xf32>, tensor<6x4xf32>)
  return %carry, %ys : tensor<4xf32>, tensor<6x4xf32>
}

// CHECK: tessera_nvidia.control_for
// CHECK-SAME: body_opcodes = array<i32: 1>
// CHECK: tessera_nvidia.control_if
// CHECK-SAME: then_opcodes = array<i32: 1>
// CHECK: tessera_nvidia.control_while
// CHECK-SAME: max_iters = 4
// CHECK: tessera_nvidia.control_scan
// CHECK-SAME: trip = 6
