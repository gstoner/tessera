// RUN: tessera-opt %s -verify-diagnostics

module {
  func.func @bad_compare_mask(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    // expected-error @+1 {{mask must have i1 element type}}
    %m = "tessera.compare_scalar"(%x) {predicate = "gt", rhs = 0.0 : f64} :
        (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %m : tensor<2x3xf32>
  }

  func.func @bad_compare_predicate(%x: tensor<2x3xf32>) -> tensor<2x3xi1> {
    // expected-error @+1 {{predicate must be one of eq/ne/lt/le/gt/ge}}
    %m = "tessera.compare_scalar"(%x) {predicate = "approximately", rhs = 0.0 : f64} :
        (tensor<2x3xf32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @bad_public_compare_shape(
      %x: tensor<2x3xf32>, %y: tensor<2x4xf32>) -> tensor<2x3xi1> {
    // expected-error @+1 {{gt shapes must match}}
    %m = "tessera.gt"(%x, %y) :
        (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @missing_integer_signedness(
      %x: tensor<2x3xi32>, %y: tensor<2x3xi32>) -> tensor<2x3xi1> {
    // expected-error @+1 {{with signless integer operands requires signedness}}
    %m = "tessera.lt"(%x, %y) :
        (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @float_signedness(
      %x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xi1> {
    // expected-error @+1 {{signedness is only valid for integer operands}}
    %m = "tessera.lt"(%x, %y) {signedness = "signed"} :
        (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @bad_integer_signedness(
      %x: tensor<2x3xi32>, %y: tensor<2x3xi32>) -> tensor<2x3xi1> {
    // expected-error @+1 {{signedness must be signed or unsigned}}
    %m = "tessera.lt"(%x, %y) {signedness = "ambiguous"} :
        (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @bad_stats_rank(%x: tensor<2x3xf32>)
      -> (tensor<2x1xf32>, tensor<2x1xf32>) {
    // expected-error @+1 {{statistics rank must be input rank minus one}}
    %center, %inv = "tessera.normalization_stats"(%x) :
        (tensor<2x3xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>)
    return %center, %inv : tensor<2x1xf32>, tensor<2x1xf32>
  }

  func.func @bad_stats_eps(%x: tensor<2x3xf32>)
      -> (tensor<2xf32>, tensor<2xf32>) {
    // expected-error @+1 {{eps must be positive for stable rsqrt}}
    %center, %inv = "tessera.normalization_stats"(%x) {eps = 0.0 : f64} :
        (tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %center, %inv : tensor<2xf32>, tensor<2xf32>
  }

  func.func @bad_stats_axis(%x: tensor<2x3xf32>)
      -> (tensor<2xf32>, tensor<2xf32>) {
    // expected-error @+1 {{axis is out of range for input rank}}
    %center, %inv = "tessera.normalization_stats"(%x) {axis = 2 : i64} :
        (tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %center, %inv : tensor<2xf32>, tensor<2xf32>
  }

  func.func @bad_broadcast_mapping(%x: tensor<2xf32>) -> tensor<2x3xf32> {
    // expected-error @+1 {{broadcast_dimensions length must equal input rank}}
    %y = "tessera.broadcast_in_dim"(%x) {broadcast_dimensions = []} :
        (tensor<2xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}
