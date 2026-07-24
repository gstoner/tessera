// RUN: tessera-opt --tessera-x86-materialize-layout-casts %s | FileCheck %s

// Column-major is an executable x86 binding contract. Apple deliberately
// rejects it, so keep this architecture-owned positive case out of the shared
// Apple/NVIDIA/x86 materializer fixture.

// CHECK-LABEL: func.func @x86_column_major
// CHECK-NOT: "tessera.cast"
// CHECK: tessera.matmul %arg0, %arg1
// CHECK-SAME: tessera.x86.operand_layout_0 = "col_major"
func.func @x86_column_major(
    %arg0: tensor<4x8xf32>, %arg1: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %a = "tessera.cast"(%arg0) {tessera.layout = "col_major"} :
      (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%a, %arg1) :
      (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}
