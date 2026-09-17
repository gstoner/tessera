// RUN: tessera-opt --tessera-autodiff-paired=normalize-data-while=true %s | FileCheck %s
//
// A data-dependent while whose tensor state shrinks each iteration is carried
// inside its declared shape envelope, never as a dynamic iter_arg. A
// `tensor<?xf32>` iter_arg whose extent changes between iterations type-checks
// and does not survive bufferization (the forward crashed inside JIT-compiled
// code, AUTODIFF-SHAPE-WHILE-FORWARD-2026-09-17). The normalizer now carries
// the static envelope plus one index per dynamic dim, reads the dynamic view
// back with extract_slice, and packs the next state in behind an assert.

module {
  func.func @shrink(%x: tensor<?xf32>) -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %count, %y = "scf.while"(%zero, %x) ({
    ^bb0(%i: index, %state: tensor<?xf32>):
      %bounded = arith.cmpi slt, %i, %three : index
      %n = tensor.dim %state, %zero : tensor<?xf32>
      %large = arith.cmpi sgt, %n, %three : index
      %active = arith.andi %bounded, %large : i1
      scf.condition(%active) %i, %state : index, tensor<?xf32>
    }, {
    ^bb0(%i: index, %state: tensor<?xf32>):
      %n = tensor.dim %state, %zero : tensor<?xf32>
      %m = arith.subi %n, %one : index
      %slice = tensor.extract_slice %state[0][%m][1] : tensor<?xf32> to tensor<?xf32>
      %next = "tessera.mul"(%slice, %slice) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      %j = arith.addi %i, %one : index
      scf.yield %j, %next : index, tensor<?xf32>
    }) {tessera.autodiff.max_iters = 3 : i64, tessera.autodiff.checkpoint_policy = "save",
       tessera.autodiff.saved_slot_shape_envelope_indices = array<i64: 1>,
       tessera.autodiff.saved_slot_shape_envelope_ranks = array<i64: 1>,
       tessera.autodiff.saved_slot_shape_envelope_bounds = array<i64: 16>} :
       (index, tensor<?xf32>) -> (index, tensor<?xf32>)
    return %y : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @shrink(
// CHECK-SAME:    %[[X:.*]]: tensor<?xf32>
// The input is packed into a fresh envelope once, behind the bound check.
// CHECK:   %[[EMPTY:.*]] = tensor.empty() {{.*}} : tensor<16xf32>
// CHECK:   %[[LEN0:.*]] = tensor.dim {{.*}} %[[X]]
// CHECK:   cf.assert %{{.*}}, "shape-varying loop state exceeds its carry envelope"
// CHECK:   %[[ENV0:.*]] = tensor.insert_slice %[[X]] into %[[EMPTY]][0] [%[[LEN0]]] [1] {{.*}} : tensor<?xf32> into tensor<16xf32>
// The loop carries the static envelope and the length, not a dynamic tensor.
// CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %c0{{.*}}, %[[ENV:.*]] = %[[ENV0]], %[[LEN:.*]] = %[[LEN0]]
// CHECK-SAME:  -> (index, tensor<16xf32>, index
// CHECK-NOT:   iter_args({{.*}}tensor<?xf32>
// The body sees the state as the while did: a dynamic view of the envelope.
// CHECK:     %[[VIEW:.*]] = tensor.extract_slice %[[ENV]][0] [%[[LEN]]] [1] : tensor<16xf32> to tensor<?xf32>
// CHECK:     scf.if
// CHECK:       tensor.extract_slice %[[VIEW]]
// CHECK:       tessera.mul
// The next state is packed back into the envelope behind the same guard.
// CHECK:     cf.assert %{{.*}}, "shape-varying loop state exceeds its carry envelope" {tessera.autodiff.replay_safe_guard = true}
// CHECK:     tensor.insert_slice %{{.*}} into %[[ENV]][0] [%{{.*}}] [1] : tensor<?xf32> into tensor<16xf32>
// CHECK:     scf.yield
// The envelope attributes were consumed: nothing dynamic remains for the tape.
// CHECK-NOT:   saved_slot_shape_envelope
// Users of the while keep their dynamic result; the tape results follow it.
// CHECK:   %[[OUT:.*]] = tensor.extract_slice %{{.*}}[0] [%{{.*}}] [1] {{.*}}: tensor<16xf32> to tensor<?xf32>
// CHECK:   return %[[OUT]], %{{.*}} : tensor<?xf32>, tensor<2xindex>, tensor<2x16xf32>, tensor<2xindex>
// The reverse scan carries the same envelope, so the backward never holds a
// shrinking iter_arg either.
// CHECK-LABEL: func.func @shrink__bwd(
// CHECK:   scf.for {{.*}} iter_args(%{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, tensor<16xf32>, index) {
// CHECK-NOT:   iter_args({{.*}}tensor<?xf32>
// CHECK:   return %{{.*}} : tensor<?xf32>
