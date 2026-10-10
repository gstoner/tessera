module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_rmsnorm_bf16_0124369db0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    tile.norm_kernel %arg0, %arg1, %arg2, %arg3, %cst {accum = "f32", affine = false, axis = -1 : i64, kind = "rmsnorm", schedule = "serial", storage = "bf16", tessera.norm_epsilon = 9.99999974E-6 : f32, tessera.schedule_hash = "0124369db05e93c7bb8d304cdfbdae177c94d87c4192a8b5d4aa1edbd3400694", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

