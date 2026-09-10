// RUN: tessera-opt %s --tessera-schedule-to-tile | FileCheck %s
// CHECK-NOT: schedule.ssd
// CHECK: func.func @ssd
// CHECK: tensor.empty() : tensor<3x1x1xf32>
// CHECK: tensor.empty() : tensor<2x1x2x1xf32>
// CHECK: scf.for
// CHECK: arith.divui
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.insert
// CHECK: tensor.insert
// CHECK: return
module {
  func.func @ssd(%x: tensor<3x1x1xf32>, %a: tensor<3x1xf32>,
      %b: tensor<3x1x2xf32>, %c: tensor<3x1x2xf32>, %s: tensor<1x2x1xf32>)
      -> (tensor<3x1x1xf32>, tensor<1x2x1xf32>, tensor<2x1x2x1xf32>) {
    %r:3 = "schedule.ssd"(%x,%a,%b,%c,%s) {chunk_size = 2 : i64}
      : (tensor<3x1x1xf32>, tensor<3x1xf32>, tensor<3x1x2xf32>, tensor<3x1x2xf32>, tensor<1x2x1xf32>)
      -> (tensor<3x1x1xf32>, tensor<1x2x1xf32>, tensor<2x1x2x1xf32>)
    return %r#0,%r#1,%r#2 : tensor<3x1x1xf32>, tensor<1x2x1xf32>, tensor<2x1x2x1xf32>
  }
}
