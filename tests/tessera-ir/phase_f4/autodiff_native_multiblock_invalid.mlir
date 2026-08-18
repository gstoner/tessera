// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

module {
  func.func @native_cycle_rejected(%pred: i1, %x: tensor<4xf32>)
      -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<4xf32> {
      cf.br ^bb1(%x : tensor<4xf32>)
    ^bb1(%state: tensor<4xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<4xf32>) -> tensor<4xf32>
      cf.cond_br %pred, ^bb1(%next : tensor<4xf32>),
                          ^bb2(%next : tensor<4xf32>)
    ^bb2(%result: tensor<4xf32>):
      scf.yield %result : tensor<4xf32>
    }
    return %out : tensor<4xf32>
  }
}

// CHECK: general native CFG requires a positive bounded
// CHECK-SAME: tessera.structured_cfg.max_steps and SHA-256 CFG identity
