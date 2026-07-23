// RUN: tessera-opt --tessera-autodiff-paired --tessera-to-linalg %s | FileCheck %s

module {
  func.func @layer_norm(%x: tensor<4x16xf32>) -> tensor<4x16xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.layer_norm"(%x) :
        (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %y : tensor<4x16xf32>
  }

  // CHECK-LABEL: func.func @layer_norm__bwd
  // CHECK-NOT: = tessera.
  // CHECK: linalg.reduce
  // CHECK: math.sqrt
  // CHECK: return
}
