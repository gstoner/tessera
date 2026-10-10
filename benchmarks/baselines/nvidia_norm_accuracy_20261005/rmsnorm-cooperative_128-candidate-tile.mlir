module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_rmsnorm_bf16_cooperative_128_acd55b7e68(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    tile.norm_kernel %arg0, %arg1, %arg2, %arg3, %cst {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "cooperative_128", storage = "bf16", tessera.norm_epsilon = 9.99999974E-6 : f32, tessera.schedule_hash = "acd55b7e682b20e3de3b925663dc343e29d8fdfa4231d05fdc5c66436e701ab7", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

