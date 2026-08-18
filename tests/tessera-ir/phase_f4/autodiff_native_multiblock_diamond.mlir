// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4-PRODUCT-1: consume an upstream native multi-block region directly.  A
// verified acyclic diamond is structurized without Python/AST re-entry, retains
// its CFG/Presburger identity, and then uses the ordinary saved-if pullback.

module {
  func.func @native_diamond(%pred: i1, %x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<4xf32> {
      cf.cond_br %pred, ^bb1(%x : tensor<4xf32>),
                          ^bb2(%x : tensor<4xf32>)
    ^bb1(%then_x: tensor<4xf32>):
      %a = "tessera.tanh"(%then_x) :
          (tensor<4xf32>) -> tensor<4xf32>
      %square = "tessera.mul"(%a, %a) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      cf.br ^bb3(%square : tensor<4xf32>)
    ^bb2(%else_x: tensor<4xf32>):
      %b = "tessera.sigmoid"(%else_x) :
          (tensor<4xf32>) -> tensor<4xf32>
      %square_else = "tessera.mul"(%b, %b) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      cf.br ^bb3(%square_else : tensor<4xf32>)
    ^bb3(%merged: tensor<4xf32>):
      %final = "tessera.tanh"(%merged) :
          (tensor<4xf32>) -> tensor<4xf32>
      scf.yield %final : tensor<4xf32>
    } {tessera.autodiff.checkpoint_policy = "save",
       tessera.presburger_digest = "2222222222222222222222222222222222222222222222222222222222222222",
       tessera.structured_cfg.digest = "1111111111111111111111111111111111111111111111111111111111111111"}
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @native_diamond(
// CHECK-NOT: scf.execute_region
// CHECK: %[[BRANCH:.+]]:5 = scf.if %arg0
// CHECK: tessera.autodiff.native_multiblock_structurized = true
// CHECK: tessera.presburger_digest = "2222222222222222222222222222222222222222222222222222222222222222"
// CHECK: tessera.structured_cfg.digest = "1111111111111111111111111111111111111111111111111111111111111111"
// CHECK: tessera.tanh %[[BRANCH]]#0
// CHECK-LABEL: func.func @native_diamond__bwd(
// CHECK: scf.if %arg3
// CHECK: return
