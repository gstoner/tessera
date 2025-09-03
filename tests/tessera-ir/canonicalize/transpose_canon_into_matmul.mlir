
// RUN: tessera-opt -tessera-canonicalize %s | FileCheck %s
module {
  func.func @mm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %at = "tessera.transpose"(%A) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %bt = "tessera.transpose"(%B) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %mm = "tessera.matmul"(%at, %bt) {transposeA = false, transposeB = false}
          : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}
// CHECK: "tessera.matmul"
// CHECK-SAME: transposeA = true
// CHECK-SAME: transposeB = true
