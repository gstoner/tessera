// RUN: not tessera-opt %s --split-input-file --tessera-graph-to-schedule 2>&1 | FileCheck %s

// The MoE token-gather owner admits only archs with moe_dispatch device proof
// (gfx1151, gfx1201). A sibling RDNA4 part, the generic name, and a target/arch
// pair that disagree all fail closed; none inherits a proof.

// CHECK: MoE token gather needs isolated gfx1151 or gfx1201 Graph entry
module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1200", tessera.arch = "gfx1200"} {
  func.func @gfx1200(%x: tensor<7x13xf32>, %token: tensor<9xi32>) -> tensor<9x13xf32> attributes {tessera.bindings = ["x", "token", "o"]} {
    %o = tessera.moe_dispatch %x, %token {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32>
    return %o : tensor<9x13xf32>
  }
}

// -----

// CHECK: MoE token gather needs isolated gfx1151 or gfx1201 Graph entry
module attributes {tessera.ir.version = "1.0", tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @generic(%x: tensor<7x13xf32>, %token: tensor<9xi32>) -> tensor<9x13xf32> attributes {tessera.bindings = ["x", "token", "o"]} {
    %o = tessera.moe_dispatch %x, %token {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32>
    return %o : tensor<9x13xf32>
  }
}

// -----

// CHECK: MoE token gather needs isolated gfx1151 or gfx1201 Graph entry
module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1151"} {
  func.func @mismatch(%x: tensor<7x13xf32>, %token: tensor<9xi32>) -> tensor<9x13xf32> attributes {tessera.bindings = ["x", "token", "o"]} {
    %o = tessera.moe_dispatch %x, %token {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32>
    return %o : tensor<9x13xf32>
  }
}
