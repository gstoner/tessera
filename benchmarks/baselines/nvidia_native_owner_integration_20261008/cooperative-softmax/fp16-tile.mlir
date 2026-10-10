module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_softmax_f16_cooperative_128(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    tile.softmax_kernel %arg0, %arg1, %arg2, %arg3 {accum = "f32", axis = -1 : i64, exp_mode = "approx_exp2", ftz = false, schedule = "cooperative_128", storage = "f16", tessera.schedule_hash = "d1b6b72f8e4afd0047197bb9d7adabf028c281afac00aef1f8b162dbaa2cc5ca", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

