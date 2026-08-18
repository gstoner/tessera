// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// W4-PRODUCT-1: bounded-dynamic carried tensors use a bounded SAVE data tape
// plus a companion logical-shape tape. No padded bound may become the logical
// extent consumed by the backward slice.

module {
  func.func @dynamic_state(%x: tensor<?xf32>, %w: tensor<?xf32>)
      -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %result = scf.for %i = %c0 to %c3 step %c1
        iter_args(%state = %x) -> tensor<?xf32> {
      %next = "tessera.mul"(%state, %w) :
          (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      scf.yield %next : tensor<?xf32>
    } {tessera.autodiff.checkpoint_policy = "save",
       tessera.autodiff.checkpoint_indices = array<i64: 1, 2>,
       tessera.autodiff.saved_slot_shape_envelope_indices = array<i64: 0>,
       tessera.autodiff.saved_slot_shape_envelope_ranks = array<i64: 1>,
       tessera.autodiff.saved_slot_shape_envelope_bounds = array<i64: 16>}
    return %result : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @dynamic_state(
// CHECK-SAME: -> (tensor<?xf32>, tensor<2x16xf32>, tensor<2x1xindex>)
// CHECK: %[[DIM:.+]] = tensor.dim
// CHECK: cf.assert {{.*}}, "saved dynamic state exceeds its slot envelope"
// CHECK: tensor.insert_slice
// CHECK-LABEL: func.func @dynamic_state__bwd(
// CHECK-SAME: tensor<2x16xf32>
// CHECK-SAME: tensor<2x1xindex>
// CHECK: tensor.extract
// CHECK: tensor.extract_slice
// CHECK: return
