// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

module {
  func.func @public_gt(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>)
      -> tensor<2x3xi1> {
    // CHECK-LABEL: func.func @public_gt
    // CHECK-NOT: tessera.gt
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    %m = "tessera.gt"(%x, %y) :
        (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @signed_lt(%x: tensor<2x3xi32>, %y: tensor<2x3xi32>)
      -> tensor<2x3xi1> {
    // CHECK-LABEL: func.func @signed_lt
    // CHECK: arith.cmpi slt
    %m = "tessera.lt"(%x, %y) {signedness = "signed"} :
        (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @unsigned_lt(%x: tensor<2x3xi32>, %y: tensor<2x3xi32>)
      -> tensor<2x3xi1> {
    // CHECK-LABEL: func.func @unsigned_lt
    // CHECK: arith.cmpi ult
    %m = "tessera.lt"(%x, %y) {signedness = "unsigned"} :
        (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    return %m : tensor<2x3xi1>
  }

  func.func @threshold_fill(%x: tensor<2x3xf32>, %dy: tensor<2x3xf32>)
      -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @threshold_fill
    // CHECK-NOT: tessera.compare_scalar
    // CHECK-NOT: tessera.masked_fill
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %m = "tessera.compare_scalar"(%x) {predicate = "gt", rhs = 0.0 : f64} :
        (tensor<2x3xf32>) -> tensor<2x3xi1>
    %r = "tessera.masked_fill"(%dy, %m) {value = 0.0 : f64} :
        (tensor<2x3xf32>, tensor<2x3xi1>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }

  func.func @stats(%x: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    // CHECK-LABEL: func.func @stats
    // CHECK-NOT: tessera.normalization_stats
    // CHECK: linalg.reduce
    // CHECK: math.sqrt
    // CHECK: arith.divf
    %center, %inv = "tessera.normalization_stats"(%x) :
        (tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %center, %inv : tensor<2xf32>, tensor<2xf32>
  }


  func.func @broadcast_stat(%x: tensor<2xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @broadcast_stat
    // CHECK-NOT: tessera.broadcast_in_dim
    // CHECK: linalg.generic
    %y = "tessera.broadcast_in_dim"(%x) {broadcast_dimensions = [0]} :
        (tensor<2xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}
