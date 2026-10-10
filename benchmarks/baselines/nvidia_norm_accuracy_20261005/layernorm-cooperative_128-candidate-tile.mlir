module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_layernorm_bf16_cooperative_128_960acb2a23(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    tile.norm_kernel %arg0, %arg1, %arg2, %arg3, %cst {accum = "f32", affine = false, axis = -1 : i64, kind = "layernorm", schedule = "cooperative_128", storage = "bf16", tessera.norm_epsilon = 9.99999974E-6 : f32, tessera.schedule_hash = "960acb2a23e1329a8ca2b8e985ed95a8402228ac86689f463081d9f5f8a68d74", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

