// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

// A user attribute cannot make mutation/RNG/I/O/collective work replayable.
// Effectful region products require an operation-owned recorded-product ABI;
// until one exists, native CFG structurization fails closed.
module {
  func.func @effectful_cfg(%pred: i1, %x: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<4xf32> {
      cf.br ^body(%x : tensor<4xf32>)
    ^body(%state: tensor<4xf32>):
      %next = arith.addf %state, %state
          {tessera.autodiff.replay_safe_guard = true,
           tessera.effect_kind = "memory"} : tensor<4xf32>
      cf.cond_br %pred, ^body(%next : tensor<4xf32>),
                          ^exit(%next : tensor<4xf32>)
    ^exit(%result: tensor<4xf32>):
      scf.yield %result : tensor<4xf32>
    } {tessera.structured_cfg.digest = "8888888888888888888888888888888888888888888888888888888888888888",
       tessera.structured_cfg.max_steps = 4 : i64}
    return %out : tensor<4xf32>
  }
}

// CHECK: general native CFG state-machine replay requires pure,
