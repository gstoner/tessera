// NVIDIA target projection of the control-flow family: tile.control_* lowered
// to the tessera_nvidia.control_* target ops. Split out of
// control_flow_tile_lowering.mlir so it is gated on the NVIDIA backend — the
// --lower-tile-to-nvidia pass is only registered when tessera-opt is built with
// TESSERA_HAVE_NVIDIA_BACKEND (the default CPU+Apple / CI build omits it).
//
// REQUIRES: tessera-nvidia-backend
// RUN: tessera-opt --tessera-tile-ir-lowering='sm=120' \
// RUN:   --lower-tile-to-nvidia='sm=120' %s | FileCheck %s --check-prefix=NVIDIA

module {
  func.func private @for_body(%x: tensor<4xf32>) -> tensor<4xf32>
  func.func private @then_body(%x: tensor<4xf32>) -> tensor<4xf32>
  func.func private @else_body(%x: tensor<4xf32>) -> tensor<4xf32>
  func.func private @while_body(%x: tensor<4xf32>) -> tensor<4xf32>
  func.func private @while_cond(%x: tensor<4xf32>) -> tensor<1xf32>
  func.func private @scan_body(%c: tensor<4xf32>, %x: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>)

  func.func @bounded_for(%init: tensor<4xf32>) -> tensor<4xf32> {
    %out = "tessera.control_for"(%init) {
      body = @for_body, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
      carry_arg_index = 0 : i64
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }

  func.func @cond(%flag: tensor<1xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %out = "tessera.control_if"(%flag, %x) {
      then_branch = @then_body, else_branch = @else_body,
      flag_arg_index = 0 : i64
    } : (tensor<1xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }

  func.func @bounded_while(%init: tensor<4xf32>) -> tensor<4xf32> {
    %out = "tessera.control_while"(%init) {
      body = @while_body, cond = @while_cond, carry_arg_index = 0 : i64,
      max_iters = 8 : i64
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }

  func.func @scan(%init: tensor<4xf32>, %xs: tensor<6x4xf32>)
      -> (tensor<4xf32>, tensor<6x4xf32>) {
    %carry, %ys = "tessera.control_scan"(%init, %xs) {
      body = @scan_body, trip = 6 : i64, carry_arg_index = 0 : i64
    } : (tensor<4xf32>, tensor<6x4xf32>)
        -> (tensor<4xf32>, tensor<6x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<6x4xf32>
  }
}

// NVIDIA: tessera_nvidia.control_for
// NVIDIA-SAME: arch = "sm_120"
// NVIDIA-SAME: source = "tessera.control_for"
// NVIDIA: tessera_nvidia.control_if
// NVIDIA-SAME: arch = "sm_120"
// NVIDIA-SAME: source = "tessera.control_if"
// NVIDIA: tessera_nvidia.control_while
// NVIDIA-SAME: arch = "sm_120"
// NVIDIA-SAME: max_iters = 8
// NVIDIA: tessera_nvidia.control_scan
// NVIDIA-SAME: arch = "sm_120"
// NVIDIA-SAME: trip = 6
