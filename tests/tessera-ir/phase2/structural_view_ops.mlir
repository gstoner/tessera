// P1a (S_SERIES_GAP_CLOSURE_PLAN §6.A) — the 0-view structural Graph IR ops
// (squeeze / unsqueeze / expand / broadcast / permute / flatten) parse, verify,
// and round-trip. These give the structural family a Graph IR identity so a @jit
// body can enter the compiled pipeline + the view/copy analysis (§6.B) instead
// of dropping to the numpy reference. Pure (no FLOP, no memory effect).
//
// RUN: tessera-opt %s | tessera-opt | FileCheck %s

// CHECK-LABEL: func.func @structural_views
func.func @structural_views(%x: tensor<1x3x1x4xf32>) -> tensor<20x3xf32> {
  // CHECK: tessera.squeeze %{{.*}} {axes = [0, 2]}
  %a = "tessera.squeeze"(%x) {axes = [0, 2]}
      : (tensor<1x3x1x4xf32>) -> tensor<3x4xf32>
  // CHECK: tessera.unsqueeze %{{.*}} {axes = [0]}
  %b = "tessera.unsqueeze"(%a) {axes = [0]}
      : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
  // CHECK: tessera.expand
  %c = "tessera.expand"(%b) : (tensor<1x3x4xf32>) -> tensor<5x3x4xf32>
  // CHECK: tessera.permute %{{.*}} {perm = [2, 0, 1]}
  %d = "tessera.permute"(%c) {perm = [2, 0, 1]}
      : (tensor<5x3x4xf32>) -> tensor<4x5x3xf32>
  // CHECK: tessera.flatten
  %e = "tessera.flatten"(%d) {start = 0 : i64, end = 1 : i64}
      : (tensor<4x5x3xf32>) -> tensor<20x3xf32>
  // CHECK: tessera.broadcast
  %f = "tessera.broadcast"(%e) : (tensor<20x3xf32>) -> tensor<20x3xf32>
  return %f : tensor<20x3xf32>
}
