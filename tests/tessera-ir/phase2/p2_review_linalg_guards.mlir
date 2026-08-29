// P2 code review (2026-08-29) — guards in TesseraToLinalgPass and
// SymbolicDimEqualityPass that previously failed silently or fired falsely.
//
// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

// An integer matmul with transposeA reaches the transpose emission before the
// float-only check. Creating IR and then returning notifyMatchFailure leaves
// those ops behind — the greedy driver never rolls a failed pattern back — so
// the pass used to keep re-running on the changed IR and abort with no
// diagnostic and no output at all. The pattern must now decline from the types
// alone and leave the op untouched.
// CHECK-LABEL: func.func @int_matmul_transpose_a
// CHECK: tessera.matmul
// CHECK-NOT: linalg.transpose
func.func @int_matmul_transpose_a(%a: tensor<8x4xi32>, %b: tensor<8x16xi32>)
    -> tensor<4x16xi32> {
  %0 = "tessera.matmul"(%a, %b) {transposeA = true}
      : (tensor<8x4xi32>, tensor<8x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// A float matmul with transposeA still lowers, so the reordered checks did not
// cost the supported path.
// CHECK-LABEL: func.func @float_matmul_transpose_a
// CHECK: linalg.transpose
// CHECK: linalg.matmul
func.func @float_matmul_transpose_a(%a: tensor<8x4xf32>, %b: tensor<8x16xf32>)
    -> tensor<4x16xf32> {
  %0 = "tessera.matmul"(%a, %b) {transposeA = true}
      : (tensor<8x4xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
