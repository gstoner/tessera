module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_layernorm_f16_63e41f1195(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    tile.norm_kernel %arg0, %arg1, %arg2, %arg3, %cst {accum = "f32", affine = false, axis = -1 : i64, kind = "layernorm", schedule = "serial", storage = "f16", tessera.norm_epsilon = 9.99999974E-6 : f32, tessera.schedule_hash = "63e41f119592248ac8a931eedc5560562eb1dd5ad666d121282b2a0cd9de4c1e", tessera.workgroup_size = 128 : i64} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }
}

