// P1b (S_SERIES_GAP_CLOSURE_PLAN §6.A) — identity folds for the 0-view structural
// ops. A shape-only op whose result type equals its operand type is a genuine
// no-op and folds to its input. `permute` is the exception: a square tensor keeps
// its type under a real axis swap, so it folds only when `perm` is identity.
//
// RUN: tessera-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @identity_views
func.func @identity_views(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK-NOT: tessera.squeeze
  // CHECK-NOT: tessera.unsqueeze
  // CHECK-NOT: tessera.expand
  // CHECK-NOT: tessera.broadcast
  // CHECK-NOT: tessera.flatten
  %a = "tessera.squeeze"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %b = "tessera.expand"(%a) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %c = "tessera.flatten"(%b) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %d = "tessera.broadcast"(%c) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK: return %arg0
  return %d : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @identity_permute
func.func @identity_permute(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // an identity permutation [0, 1] folds away
  // CHECK-NOT: tessera.permute
  %a = "tessera.permute"(%x) {perm = [0, 1]} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK: return %arg0
  return %a : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @real_permute_preserved
func.func @real_permute_preserved(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // a real transpose on a SQUARE tensor keeps its type but is NOT a no-op
  // CHECK: tessera.permute %{{.*}} {perm = [1, 0]}
  %a = "tessera.permute"(%x) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %a : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @dynamic_view_preserved
func.func @dynamic_view_preserved(%x: tensor<?xf32>) -> tensor<?xf32> {
  // a dynamic expand can carry a runtime extent change (1->N) in the attr-dict
  // while both sides stay tensor<?xf32> — type equality is NOT runtime identity,
  // so it must NOT fold.
  // CHECK: tessera.expand
  %a = "tessera.expand"(%x) {shape = [8]} : (tensor<?xf32>) -> tensor<?xf32>
  return %a : tensor<?xf32>
}

// CHECK-LABEL: func.func @truncated_perm_preserved
func.func @truncated_perm_preserved(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // a truncated perm ([0] on a rank-2 tensor) is malformed/non-identity — it must
  // NOT fold by matching an identity prefix.
  // CHECK: tessera.permute %{{.*}} {perm = [0]}
  %a = "tessera.permute"(%x) {perm = [0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %a : tensor<4x4xf32>
}
