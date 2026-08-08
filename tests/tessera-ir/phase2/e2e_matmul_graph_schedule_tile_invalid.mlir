// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @dynamic(%a: tensor<?x32xf32>, %b: tensor<32x48xf32>)
      -> tensor<?x48xf32> {
    // CHECK: E2E-REAL-2 Graph->Schedule requires static rank-2
    %0 = tessera.matmul %a, %b
        : (tensor<?x32xf32>, tensor<32x48xf32>) -> tensor<?x48xf32>
    return %0 : tensor<?x48xf32>
  }
}

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @unsupported_dtype(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
      -> tensor<32x32xf32> {
    // CHECK: E2E-REAL-2 Graph->Schedule requires static rank-2
    %0 = tessera.matmul %a, %b
        : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
