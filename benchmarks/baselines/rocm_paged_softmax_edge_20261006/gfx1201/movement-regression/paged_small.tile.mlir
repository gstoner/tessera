module attributes {tessera.arch = "gfx1201", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201"} {
  llvm.func @tessera_tile_paged_kv_read_f32_direct(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64) attributes {tessera.native_contract = {arch = "gfx1201", bindings = ["a0", "a1", "v0"], layout = "row_major", page_ownership = "read_only_borrow", shape = array<i64: 4, 4, 4, 3, 8, 1, 5>, table_bounds = "runtime_checked_physical_page_indices", target = "rocm_gfx1201"}, tessera.schedule_hash = "c5fb6219e7e53afc0c69391546762fec5fa9e1087f81c20fe7251bd08e09b38f"} {
    tile.paged_kv_read_kernel %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 {route = "direct", storage = "f32", table_storage = "i32"} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

