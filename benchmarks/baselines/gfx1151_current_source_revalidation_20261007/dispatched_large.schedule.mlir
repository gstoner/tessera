module attributes {tessera.arch = "gfx1151", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151"} {
  func.func @dispatched(%arg0: tensor<128xi32>, %arg1: tensor<64x256xf32>) -> tensor<128x256xf32> attributes {tessera.bindings = ["a1", "a0", "v0"], tessera.frontend.authority = "tracer", tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "02d0b3f23c08adffa1f954e3b57d80b1d0ac5ee914fce183a28554f3e2dda28e", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %0 = tessera.moe_dispatch %arg1, %arg0 {schedule.artifact_hash = "e86bac8e31588c53aac4d4cc44b1d7a7dc58af1d6272eed8a46269b2c0cb44c6", tessera.effect_kind = "collective"} : (tensor<64x256xf32>, tensor<128xi32>) -> tensor<128x256xf32>
    %1 = schedule.moe_dispatch %0 {artifact_hash = "e86bac8e31588c53aac4d4cc44b1d7a7dc58af1d6272eed8a46269b2c0cb44c6", contract = {arch = "gfx1151", bindings = ["a1", "a0", "v0"], layout = "row_major", route = "direct_gather", shape = array<i64: 64, 128, 256>, target = "rocm_gfx1151"}} : tensor<128x256xf32> -> tensor<128x256xf32>
    return %1 : tensor<128x256xf32>
  }
}

