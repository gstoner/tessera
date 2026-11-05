// E2E-REAL-6 Apple GPU runtime-descriptor families (2026-11-04): an opted-in
// Apple GPU module fails closed on any semantic the runtime ABI does not
// implement, instead of letting the delegate compute a different result.
// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y"]} {
  func.func @first_axis_count(%x: tensor<?x?xf32>) -> tensor<?xi32> {
    %0 = tessera.count_nonzero %x {axis = 0} : (tensor<?x?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// CHECK: error: Apple GPU native descriptor: dynamic count_nonzero requires

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y"]} {
  func.func @partly_dynamic_softmax(%x: tensor<?x7xf32>) -> tensor<?x7xf32> {
    %0 = tessera.softmax %x {axis = -1} : (tensor<?x7xf32>) -> tensor<?x7xf32>
    return %0 : tensor<?x7xf32>
  }
}

// CHECK: error: Apple GPU native descriptor: dynamic softmax requires ?x? f32

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "v", "i"]} {
  func.func @i64_indices(%x: tensor<?x?xf32>) -> (tensor<?x2xf32>, tensor<?x2xi64>) {
    %v, %i = tessera.top_k %x {k = 2} : (tensor<?x?xf32>) -> (tensor<?x2xf32>, tensor<?x2xi64>)
    return %v, %i : tensor<?x2xf32>, tensor<?x2xi64>
  }
}

// CHECK: error: Apple GPU native descriptor: requires positive int32-extent f32/f16/bf16/i32 results

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y"]} {
  func.func @policy(%x: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tessera.popcount %x {numeric_policy = "i32"} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// CHECK: error: Apple GPU native descriptor: has an unsupported policy attribute 'numeric_policy'

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @dynamic_bmm(%a: tensor<?x4x8xf32>, %b: tensor<?x8x16xf32>) -> tensor<?x4x16xf32> {
    %0 = tessera.batched_gemm %a, %b : (tensor<?x4x8xf32>, tensor<?x8x16xf32>) -> tensor<?x4x16xf32>
    return %0 : tensor<?x4x16xf32>
  }
}

// CHECK: error: Apple GPU native descriptor: batched_gemm requires static rank-3
