// RUN: tessera-opt --split-input-file --verify-diagnostics %s

module {
  func.func @wrong_gather_rank(%source: tensor<2x3xf32>, %indices: tensor<4xi64>) -> tensor<4xf32> {
    // expected-error @+1 {{gather result rank must match the source rank}}
    %result = tessera.gather %source, %indices {axis = 1 : i64} : (tensor<2x3xf32>, tensor<4xi64>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// -----

module {
  func.func @wrong_gather_extent(%source: tensor<2x3xf32>, %indices: tensor<4xi64>) -> tensor<2x5xf32> {
    // expected-error @+1 {{gather result shape must replace the indexed axis with the index extent}}
    %result = tessera.gather %source, %indices {axis = -1 : i64} : (tensor<2x3xf32>, tensor<4xi64>) -> tensor<2x5xf32>
    return %result : tensor<2x5xf32>
  }
}

// -----

module {
  func.func @scalar_indices(%source: tensor<2x3xf32>, %indices: tensor<i64>) -> tensor<2x3xf32> {
    // expected-error @+1 {{gather indices must have at least one dimension}}
    %result = tessera.gather %source, %indices {axis = 1 : i64} : (tensor<2x3xf32>, tensor<i64>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}
