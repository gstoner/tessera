module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_softmax_f32_cooperative_128(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    tile.softmax_kernel %arg0, %arg1, %arg2, %arg3 {accum = "f32", axis = -1 : i64, exp_mode = "approx_exp2", ftz = false, schedule = "cooperative_128", storage = "f32", tessera.schedule_hash = "9d3b4bf250bf3f9199314ab3950b38ef15c37cb05f576417bdcf98b0f374c29d", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }
}

