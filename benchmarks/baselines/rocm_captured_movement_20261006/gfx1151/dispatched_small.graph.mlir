module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @dispatched(%a0: tensor<9xi32>, %a1: tensor<7x13xf32>) -> (tensor<9x13xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "02d0b3f23c08adffa1f954e3b57d80b1d0ac5ee914fce183a28554f3e2dda28e", tessera.structured_cfg.blocks = 2, tessera.bindings = ["a1", "a0", "v0"]} {
    %v0 = tessera.moe_dispatch %a1, %a0 {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32> loc("tests/unit/test_public_movement_frontend.py":20:12)
    return %v0 : tensor<9x13xf32>
  }
}