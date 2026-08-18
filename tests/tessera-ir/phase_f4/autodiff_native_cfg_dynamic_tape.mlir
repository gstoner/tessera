// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// A bounded native CFG may retain dynamic tensor state only when every
// dynamic state-machine slot has an explicit maximum envelope. The compiler
// stores actual extents in companion shape tapes; backward extraction must not
// treat the padded allocation bound as the logical tensor shape.

module {
  func.func @saved_dynamic_cfg(%pred: i1, %x: tensor<?xf32>)
      -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<?xf32> {
      cf.br ^loop(%x : tensor<?xf32>)
    ^loop(%state: tensor<?xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<?xf32>) -> tensor<?xf32>
      cf.cond_br %pred, ^loop(%next : tensor<?xf32>),
                          ^exit(%next : tensor<?xf32>)
    ^exit(%result: tensor<?xf32>):
      scf.yield %result : tensor<?xf32>
    } {tessera.autodiff.checkpoint_indices = array<i64: 2>,
       tessera.autodiff.checkpoint_policy = "hybrid",
       tessera.autodiff.saved_slot_shape_envelope_bounds = array<i64: 16, 16, 16>,
       tessera.autodiff.saved_slot_shape_envelope_indices = array<i64: 2, 3, 4>,
       tessera.autodiff.saved_slot_shape_envelope_ranks = array<i64: 1, 1, 1>,
       tessera.structured_cfg.digest = "7777777777777777777777777777777777777777777777777777777777777777",
       tessera.structured_cfg.max_steps = 4 : i64}
    return %out : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @saved_dynamic_cfg(
// CHECK-DAG: scf.for
// CHECK-DAG: tessera.autodiff.residual_owner = "generic_for"
// CHECK-DAG: tessera.autodiff.residual_shape_tape_ordinals = array<i64:
// CHECK-DAG: tensor<1x16xf32>
// CHECK-DAG: tensor<1x1xindex>
// CHECK-DAG: cf.assert {{.*}}, "saved dynamic state exceeds its slot envelope"
// CHECK-LABEL: func.func @saved_dynamic_cfg__bwd(
// CHECK: tensor.extract
// CHECK: tensor.extract_slice
