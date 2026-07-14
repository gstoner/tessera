// RUN: %tnv --lower-tile-to-nvidia='sm=120' %s | FileCheck %s

module {
  func.func @control_contracts(%flag: tensor<1xf32>, %init: tensor<4xf32>,
                               %xs: tensor<6x4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
          tensor<6x4xf32>) {
    %f = "tile.control_for"(%init) {
      source = "tessera.control_for", start = 0 : i64, stop = 4 : i64,
      step = 1 : i64, carry_arg_index = 0 : i64
    } : (tensor<4xf32>) -> tensor<4xf32>
    %c = "tile.control_if"(%flag, %init) {
      source = "tessera.control_if", flag_arg_index = 0 : i64
    } : (tensor<1xf32>, tensor<4xf32>) -> tensor<4xf32>
    %w = "tile.control_while"(%init) {
      source = "tessera.control_while", carry_arg_index = 0 : i64,
      max_iters = 8 : i64
    } : (tensor<4xf32>) -> tensor<4xf32>
    %s, %ys = "tile.control_scan"(%init, %xs) {
      source = "tessera.control_scan", trip = 6 : i64,
      carry_arg_index = 0 : i64
    } : (tensor<4xf32>, tensor<6x4xf32>)
        -> (tensor<4xf32>, tensor<6x4xf32>)
    return %f, %c, %w, %s, %ys : tensor<4xf32>, tensor<4xf32>,
        tensor<4xf32>, tensor<4xf32>, tensor<6x4xf32>
  }
}

// CHECK: tessera_nvidia.control_for
// CHECK-SAME: arch = "sm_120"
// CHECK-SAME: source = "tessera.control_for"
// CHECK: tessera_nvidia.control_if
// CHECK-SAME: arch = "sm_120"
// CHECK: tessera_nvidia.control_while
// CHECK-SAME: max_iters = 8
// CHECK: tessera_nvidia.control_scan
// CHECK-SAME: trip = 6
