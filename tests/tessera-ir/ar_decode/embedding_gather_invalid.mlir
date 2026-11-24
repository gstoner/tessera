// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule -verify-diagnostics %s
//
// AR-DECODE-PROGRAM-1 S1c: inside an sm_120 module, every gather outside the
// native embedding envelope fails closed at Schedule (Decision #21a) instead
// of passing through to a Target that has no consumer for it.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @columns(%t: tensor<8x4xf32>, %i: tensor<2xi32>) -> tensor<8x2xf32> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather gathers only table rows (axis 0)}}
    %x = tessera.gather %t, %i {axis = 1 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<8x2xf32>
    return %x : tensor<8x2xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @half(%t: tensor<8x4xf16>, %i: tensor<2xi32>) -> tensor<2x4xf16> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather requires a static f32[V, H] table}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf16>, tensor<2xi32>) -> tensor<2x4xf16>
    return %x : tensor<2x4xf16>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @wide_tokens(%t: tensor<8x4xf32>, %i: tensor<2xi64>) -> tensor<2x4xf32> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather requires static i32[B] tokens}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @vector_table(%t: tensor<8xf32>, %i: tensor<2xi32>) -> tensor<2xf32> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather requires a static f32[V, H] table}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8xf32>, tensor<2xi32>) -> tensor<2xf32>
    return %x : tensor<2xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @two(%t: tensor<8x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> attributes {tessera.bindings = ["t", "i"]} {
    // expected-error @+1 {{sm_120 embedding gather requires three binding names}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @reserved(%t: tensor<8x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> attributes {tessera.bindings = ["t", "i", "status"]} {
    // expected-error @+1 {{sm_120 embedding gather binding names must be unique and not 'status'}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @policy(%t: tensor<8x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather has an unsupported policy attribute 'mode'}}
    %x = tessera.gather %t, %i {axis = 0 : i64, mode = "clip"} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120"} {
  func.func @no_arch(%t: tensor<8x4xf32>, %i: tensor<2xi32>) -> tensor<2x4xf32> attributes {tessera.bindings = ["t", "i", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather requires tessera.arch = "sm_120"}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @swapped(%i: tensor<2xi32>, %t: tensor<8x4xf32>) -> tensor<2x4xf32> attributes {tessera.bindings = ["i", "t", "o"]} {
    // expected-error @+1 {{sm_120 embedding gather requires an isolated (table, tokens) entry}}
    %x = tessera.gather %t, %i {axis = 0 : i64} : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    return %x : tensor<2x4xf32>
  }
}
