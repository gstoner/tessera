// E2E-REAL-6 Apple CPU: the scheduled contracts are f32-only and name their
// arch; anything else is refused rather than inheriting the f32 schedule.
// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_cpu"} {
  func.func @no_arch(%a: tensor<4x8xf32>, %b: tensor<8x5xf32>) -> tensor<4x5xf32> {
    %0 = tessera.matmul %a, %b : (tensor<4x8xf32>, tensor<8x5xf32>) -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}

// CHECK: error: E2E-REAL-2 Graph->Schedule requires static rank-2

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @f16_matmul(%a: tensor<4x8xf16>, %b: tensor<8x5xf16>) -> tensor<4x5xf16> {
    %0 = tessera.matmul %a, %b : (tensor<4x8xf16>, tensor<8x5xf16>) -> tensor<4x5xf16>
    return %0 : tensor<4x5xf16>
  }
}

// CHECK: error: E2E-REAL-2 Graph->Schedule requires static rank-2

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @rank3_softmax(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    %0 = tessera.softmax %x : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %0 : tensor<2x4x8xf32>
  }
}

// CHECK: error: E2E-REAL-6 Graph->Schedule requires a supported static semantic-kernel contract

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @reduce(%x: tensor<4x8xf32>) -> tensor<4xf32> {
    %0 = tessera.reduce %x {kind = "sum", axis = -1 : i64} : (tensor<4x8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: error: E2E-REAL-6 Graph->Schedule requires a supported static semantic-kernel contract

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @transposed(%a: tensor<8x4xf32>, %b: tensor<8x5xf32>) -> tensor<4x5xf32> {
    %0 = tessera.matmul %a, %b {transposeA = true} : (tensor<8x4xf32>, tensor<8x5xf32>) -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}

// CHECK: error: E2E-REAL-2 Graph->Schedule requires static rank-2
