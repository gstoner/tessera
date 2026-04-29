// RUN: tessera-opt %s --verify-only | FileCheck %s
// RUN: tessera-opt %s --alias=graph-to-schedule --print-pipeline | FileCheck %s --check-prefix=PIPE

// CHECK:       module
// PIPE:        Pipeline: -tessera-verify

module @basic {
  func.func @identity(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    return %arg0 : tensor<4x4xf32>
  }
}
