
// RUN: tessera-opt -tessera-canonicalize %s | FileCheck %s
module {
  func.func @f(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %bias: tensor<?xf32>) -> tensor<?x?xf32> {
    %mm = "tessera.matmul"(%A, %B) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %add = "tessera.add"(%mm, %bias) : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    %gelu = "tessera.gelu"(%add) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %gelu : tensor<?x?xf32>
  }
}
// CHECK: "tessera.fused_epilogue"
// CHECK-SAME: epilogue = "gelu"
// CHECK-SAME: has_bias = true
