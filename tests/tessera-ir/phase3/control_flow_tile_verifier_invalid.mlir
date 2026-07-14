// RUN: tessera-opt %s --split-input-file --verify-diagnostics

module {
  func.func @dynamic_shape(%init: tensor<?xf32>) -> tensor<?xf32> {
    // expected-error@+1 {{'tile.control_for' op NVIDIA control-flow Tile contract requires static ranked tensors}}
    %out = "tile.control_for"(%init) {
      source = "tessera.control_for", start = 0 : i64, stop = 4 : i64,
      step = 1 : i64, carry_arg_index = 0 : i64
    } : (tensor<?xf32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
}

// -----

module {
  func.func @invalid_while(%init: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error@+1 {{'tile.control_while' op requires max_iters>0 and one shape-stable carried result}}
    %out = "tile.control_while"(%init) {
      source = "tessera.control_while", carry_arg_index = 0 : i64,
      max_iters = 0 : i64
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }
}

// -----

module {
  func.func @invalid_scan(%init: tensor<4xf32>, %xs: tensor<5x4xf32>)
      -> (tensor<4xf32>, tensor<5x4xf32>) {
    // expected-error@+1 {{'tile.control_scan' op xs/ys leading dimension must equal trip}}
    %carry, %ys = "tile.control_scan"(%init, %xs) {
      source = "tessera.control_scan", trip = 6 : i64,
      carry_arg_index = 0 : i64
    } : (tensor<4xf32>, tensor<5x4xf32>)
        -> (tensor<4xf32>, tensor<5x4xf32>)
    return %carry, %ys : tensor<4xf32>, tensor<5x4xf32>
  }
}
