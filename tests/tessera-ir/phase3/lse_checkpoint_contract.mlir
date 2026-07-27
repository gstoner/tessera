// RUN: tessera-opt %s | FileCheck %s

module {
  func.func @checkpoint_roundtrip(
      %lse: tensor<16xf32>,
      %checkpoint: memref<4x128xf32>,
      %row: index) -> f32 {
    "tessera_attn.lse.save"(%lse, %checkpoint, %row) {
      identity = "attention.forward.0.lse",
      memory_space = "global",
      scope = "program_launch",
      cache_policy = "streaming"
    } : (tensor<16xf32>, memref<4x128xf32>, index) -> ()
    %loaded = "tessera_attn.lse.load"(%checkpoint, %row) {
      identity = "attention.forward.0.lse",
      memory_space = "global",
      scope = "program_launch",
      cache_policy = "streaming"
    } : (memref<4x128xf32>, index) -> f32
    return %loaded : f32
  }
}

// CHECK: tessera_attn.lse.save
// CHECK-SAME: identity = "attention.forward.0.lse"
// CHECK-SAME: memory_space = "global"
// CHECK-SAME: scope = "program_launch"
// CHECK: tessera_attn.lse.load
// CHECK-SAME: identity = "attention.forward.0.lse"
