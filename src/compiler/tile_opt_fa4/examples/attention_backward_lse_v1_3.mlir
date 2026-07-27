// Persistent backward LSE checkpoint example.
module {
  func.func @load_lse(%checkpoint: memref<128xf32>, %row: index) -> f32 {
    %lse = "tessera_attn.lse.load"(%checkpoint, %row) {
      identity = "attention.lse",
      memory_space = "global",
      scope = "program_launch",
      cache_policy = "streaming"
    } : (memref<128xf32>, index) -> f32
    return %lse : f32
  }
}
