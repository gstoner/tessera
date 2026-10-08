module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_layernorm_f16_cooperative_128_91af0be746(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    tile.norm_kernel %arg0, %arg1, %arg2, %arg3, %cst {accum = "f32", affine = false, axis = -1 : i64, kind = "layernorm", schedule = "cooperative_128", storage = "f16", tessera.norm_epsilon = 9.99999974E-6 : f32, tessera.schedule_hash = "91af0be74644b9fc6b94866c4092781592a17de22b292ae6f303c18643c15859", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

