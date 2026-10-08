module attributes {tessera.arch = "gfx1151", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151"} {
  llvm.func @tessera_tile_moe_dispatch_f32_direct(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64) attributes {tessera.native_contract = {arch = "gfx1151", bindings = ["a1", "a0", "v0"], layout = "row_major", route = "direct_gather", shape = array<i64: 7, 9, 13>, target = "rocm_gfx1151"}, tessera.schedule_hash = "c1c500b208823e3bdb1bcf7ecf275261aeaba25876b43d23a31784f78cab2178"} {
    tile.moe_dispatch_kernel %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {index_storage = "i32", storage = "f32"} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

