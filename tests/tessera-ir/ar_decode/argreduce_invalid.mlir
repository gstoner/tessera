// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule -verify-diagnostics %s
//
// AR-DECODE-PROGRAM-1 S1b: inside an sm_120 module, every argreduce outside
// the native envelope fails closed at Schedule (Decision #21a) instead of
// passing through to a Target that has no consumer for it.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @inner_axis(%x: tensor<3x17xf32>) -> tensor<17xi32> attributes {tessera.bindings = ["x", "i"]} {
    // expected-error @+1 {{sm_120 argreduce reduces only the last axis}}
    %i = tessera.argmax %x {axis = 0 : i64} : (tensor<3x17xf32>) -> tensor<17xi32>
    return %i : tensor<17xi32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @half(%x: tensor<3x17xf16>) -> tensor<3xi32> attributes {tessera.bindings = ["x", "i"]} {
    // expected-error @+1 {{sm_120 argreduce requires a static f32 operand of rank >= 1}}
    %i = tessera.argmax %x {axis = -1 : i64} : (tensor<3x17xf16>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @wide_index(%x: tensor<3x17xf32>) -> tensor<3xi64> attributes {tessera.bindings = ["x", "i"]} {
    // expected-error @+1 {{sm_120 argreduce result must be a static i32 index tensor}}
    %i = tessera.argmax %x {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3xi64>
    return %i : tensor<3xi64>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @no_bindings(%x: tensor<3x17xf32>) -> tensor<3xi32> {
    // expected-error @+1 {{sm_120 argreduce requires two binding names}}
    %i = tessera.argmax %x {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @policy(%x: tensor<3x17xf32>) -> tensor<3xi32> attributes {tessera.bindings = ["x", "i"]} {
    // expected-error @+1 {{sm_120 argreduce has an unsupported policy attribute 'tie_break'}}
    %i = tessera.argmax %x {axis = -1 : i64, tie_break = "last"} : (tensor<3x17xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_90"} {
  func.func @arch(%x: tensor<3x17xf32>) -> tensor<3xi32> attributes {tessera.bindings = ["x", "i"]} {
    // expected-error @+1 {{sm_120 argreduce requires tessera.arch = "sm_120"}}
    %i = tessera.argmax %x {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}

// -----

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @not_isolated(%x: tensor<3x17xf32>) -> tensor<3xi32> attributes {tessera.bindings = ["x", "i"]} {
    %y = tessera.relu %x : (tensor<3x17xf32>) -> tensor<3x17xf32>
    // expected-error @+1 {{sm_120 argreduce requires an isolated one-argument tensor entry}}
    %i = tessera.argmax %y {axis = -1 : i64} : (tensor<3x17xf32>) -> tensor<3xi32>
    return %i : tensor<3xi32>
  }
}
