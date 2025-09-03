
// RUN: not tessera-opt -tessera-verify %s 2>&1 | FileCheck %s
module {
  func.func @bad(%A: tensor<?x?xf32>, %B: tensor<?x?x?xf16>) -> tensor<?x?xf32> {
    %mm = "tessera.matmul"(%A, %B) : (tensor<?x?xf32>, tensor<?x?x?xf16>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}
// CHECK: error: [TESSERA_VFY_MATMUL_ELEM_TYPE_MISMATCH]
