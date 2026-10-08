// Native Tile-only numerical fixture; not an MXFP8 Graph package.
// Two K16 FP8 WMMA panels form one K32 partial before E8M0 scaling.
module attributes {tessera.arch = "gfx1201", tessera.target = "rocm"} {
  gpu.module @e8m0_consumer_fixture_mod {
    gpu.func @e8m0_consumer_fixture(%arg0: memref<?xf8E4M3FN>, %arg1: memref<?xf8E4M3FN>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xf32>, %arg5: index, %arg6: index, %arg7: index) kernel attributes {tessera.rocm.scale_group_panels = 2 : i64, tessera.rocm.scale_k = 32 : i64, tessera.rocm.scale_n = 1 : i64, tessera.rocm.schedule_raster_group = 1 : i64, tessera.rocm.schedule_raster_order = "row_major"} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f8E4M3FN
      %cst_0 = arith.constant dense<0.000000e+00> : vector<16xf8E4M3FN>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %thread_id_x = gpu.thread_id x
      %0 = arith.andi %thread_id_x, %c15 : index
      %1 = arith.shrui %thread_id_x, %c4 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %c16_2 = arith.constant 16 : index
      %c16_3 = arith.constant 16 : index
      %2 = arith.muli %block_id_y, %c16_2 : index
      %3 = arith.muli %block_id_x, %c16_3 : index
      %c0_4 = arith.constant 0 : index
      %4 = arith.addi %2, %c0_4 : index
      %5 = arith.addi %4, %0 : index
      %6 = arith.muli %5, %arg7 : index
      %7 = arith.cmpi slt, %5, %arg5 : index
      %8 = arith.select %7, %5, %c0 : index
      %9 = arith.muli %8, %arg7 : index
      %c0_5 = arith.constant 0 : index
      %10 = arith.addi %3, %c0_5 : index
      %11 = arith.addi %10, %0 : index
      %12 = arith.cmpi slt, %11, %arg6 : index
      %13 = arith.select %12, %11, %c0 : index
      %14 = tile.fragment_zero : <m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
      %c16_6 = arith.constant 16 : index
      %15 = arith.remui %arg7, %c16_6 : index
      %16 = arith.subi %arg7, %15 : index
      %17 = arith.cmpi ne, %15, %c0 : index
      %18 = arith.addi %2, %c16_2 : index
      %19 = arith.addi %3, %c16_3 : index
      %20 = arith.cmpi sle, %18, %arg5 : index
      %21 = arith.cmpi sle, %19, %arg6 : index
      %22 = arith.andi %20, %21 : i1
      scf.if %22 {
        %c32 = arith.constant 32 : index
        %23 = arith.divui %arg7, %c32 : index
        %24 = arith.muli %23, %c32 : index
        %c32_7 = arith.constant 32 : index
        %25 = scf.for %arg8 = %c0 to %24 step %c32_7 iter_args(%arg9 = %14) -> (!tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) {
          %27 = tile.fragment_zero : <m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %31 = tile.view %arg0, %4, %arg8, %arg7 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
          %32 = tile.fragment_pack %31 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">
          %36 = tile.view %arg1, %arg8, %10, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
          %37 = tile.fragment_pack %36 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">
          %38 = tile.mma %32, %37, %27 : (!tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %c16_8 = arith.constant 16 : index
          %39 = arith.addi %arg8, %c16_8 : index
          %43 = tile.view %arg0, %4, %39, %arg7 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
          %44 = tile.fragment_pack %43 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">
          %48 = tile.view %arg1, %39, %10, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index) -> !tile.tile
          %49 = tile.fragment_pack %48 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">
          %50 = tile.mma %44, %49, %38 : (!tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %c32_9 = arith.constant 32 : index
          %51 = arith.divui %arg8, %c32_9 : index
          %52 = tile.fragment_scaled_accumulate %arg9, %50 scales(%arg2, %arg3) at(%4, %10) group(%51, %23) bounds(%arg5, %arg6) {scale_format = "e8m0", scale_n = 1 : i64} : <m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">, memref<?xi8>, memref<?xi8>
          scf.yield %52 : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
        }
        %26 = tile.fragment_unpack %25 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>} : (!tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.tile
        tile.store %26, %arg4, %4, %10, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : !tile.tile, memref<?xf32>, index, index, index
      } else {
        %c32 = arith.constant 32 : index
        %23 = arith.divui %arg7, %c32 : index
        %24 = arith.muli %23, %c32 : index
        %c32_7 = arith.constant 32 : index
        %25 = scf.for %arg8 = %c0 to %24 step %c32_7 iter_args(%arg9 = %14) -> (!tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) {
          %27 = tile.fragment_zero : <m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %31 = tile.view %arg0, %4, %arg8, %arg5, %arg7, %arg7 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index, index, index) -> !tile.tile
          %32 = tile.fragment_pack %31 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">
          %36 = tile.view %arg1, %arg8, %10, %arg7, %arg6, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index, index, index) -> !tile.tile
          %37 = tile.fragment_pack %36 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">
          %38 = tile.mma %32, %37, %27 : (!tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %c16_8 = arith.constant 16 : index
          %39 = arith.addi %arg8, %c16_8 : index
          %43 = tile.view %arg0, %4, %39, %arg5, %arg7, %arg7 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index, index, index) -> !tile.tile
          %44 = tile.fragment_pack %43 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">
          %48 = tile.view %arg1, %39, %10, %arg7, %arg6, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : (memref<?xf8E4M3FN>, index, index, index, index, index) -> !tile.tile
          %49 = tile.fragment_pack %48 : (!tile.tile) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">
          %50 = tile.mma %44, %49, %38 : (!tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "a", layout = "row_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "e4m3", acc = "f32", role = "b", layout = "col_major", family = "wmma">, !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
          %c32_9 = arith.constant 32 : index
          %51 = arith.divui %arg8, %c32_9 : index
          %52 = tile.fragment_scaled_accumulate %arg9, %50 scales(%arg2, %arg3) at(%4, %10) group(%51, %23) bounds(%arg5, %arg6) {scale_format = "e8m0", scale_n = 1 : i64} : <m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">, memref<?xi8>, memref<?xi8>
          scf.yield %52 : !tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">
        }
        %26 = tile.fragment_unpack %25 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>} : (!tile.fragment<m = 16, n = 16, k = 16, elem = "f32", acc = "f32", role = "acc", layout = "row_major", family = "wmma">) -> !tile.tile
        tile.store %26, %arg4, %4, %10, %arg5, %arg6, %arg6 {tile.layout = #tile.layout<shard = [16, 16] : [16, 1] on ["tlane", "reg"], replica = [] : [] on [], offset = 0>, tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 0>} : !tile.tile, memref<?xf32>, index, index, index, index, index
      }
      gpu.return
    }
  }
}
