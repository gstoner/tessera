// RUN: tessera-opt --tessera-autodiff-forward %s | FileCheck %s
//
// Forward mode carries primal and tangent state through structured regions in
// one execution. It must not clone a region into independent primal/tangent
// control paths.

module {
  func.func @branch(%pred: i1, %x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "forward",
                  tessera.autodiff.wrt_indices = [1]} {
    %out = scf.if %pred -> tensor<4xf32> {
      %square = "tessera.mul"(%x, %x) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %square : tensor<4xf32>
    } else {
      %double = "tessera.add"(%x, %x) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %double : tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }

  func.func @counted(%x: tensor<4xf32>, %w: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "forward"} {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %c3 step %c1
        iter_args(%carry = %x) -> tensor<4xf32> {
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }

  func.func @bounded_while(%x: tensor<4xf32>, %w: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "forward"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %count, %out = scf.while (%i = %c0, %carry = %x)
        : (index, tensor<4xf32>) -> (index, tensor<4xf32>) {
      %continue = arith.cmpi slt, %i, %c3 : index
      scf.condition(%continue) %i, %carry : index, tensor<4xf32>
    } do {
    ^bb0(%i: index, %carry: tensor<4xf32>):
      %next = "tessera.mul"(%carry, %w) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %next_i = arith.addi %i, %c1 : index
      scf.yield %next_i, %next : index, tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func private @branch__jvp
// CHECK-COUNT-1: scf.if
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<4xf32>, tensor<4xf32>
// CHECK: return

// CHECK-LABEL: func.func private @counted__jvp
// CHECK-COUNT-1: scf.for
// CHECK-SAME: iter_args({{.*}}, {{.*}})
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<4xf32>, tensor<4xf32>
// CHECK: return

// CHECK-LABEL: func.func private @bounded_while__jvp
// CHECK-COUNT-1: scf.while
// CHECK: scf.condition({{.*}}) {{.*}}, {{.*}}, {{.*}}, {{.*}}
// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}
// CHECK: return
