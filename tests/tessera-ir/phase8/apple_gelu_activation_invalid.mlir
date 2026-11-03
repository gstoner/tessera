// E2E-REAL-6 Apple GPU GELU: an erf request is a different GELU than the
// tanh kernel computes, so Graph->Schedule refuses it rather than guessing.
// RUN: not tessera-opt --tessera-graph-to-schedule %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @erf_gelu(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.gelu %x {approximate = "none"} : (tensor<5x11xf32>) -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// CHECK: error: E2E-REAL-6 Graph->Schedule requires a supported static semantic-kernel contract
