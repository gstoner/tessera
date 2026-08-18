// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

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
       tessera.structured_cfg.digest = "6666666666666666666666666666666666666666666666666666666666666666",
       tessera.structured_cfg.max_steps = 4 : i64}
    return %out : tensor<?xf32>
  }
}

// CHECK: saved native CFG dynamic state requires total, positive per-slot
// CHECK-SAME: shape-envelope indices/ranks/bounds matching the state ABI
