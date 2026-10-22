module attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program = 3 : i64, tessera.native.tensor_program_capacity_types = [memref<65x64xf16>, memref<64x47xf16>, memref<65x64xf16>, memref<65x47xf32>], tessera.native.tensor_program_shape_bindings = [array<i64: 0, 1>, array<i64: 1, 2>, array<i64: 0, 1>, array<i64: 0, 2>], tessera.native.tensor_program_shape_variables = array<i64: 0, 0, 65, 0, 1, 64, 1, 1, 47>, tessera.native.tensor_program_source = "module attributes {tessera.arch = \22sm_120\22, tessera.frontend.authority = \22tracer\22, tessera.ir.version = \221.0\22, tessera.native.tensor_program_shape_bounds_request = {K = 64 : i64, M = 65 : i64, N = 47 : i64}, tessera.target = \22nvidia_sm120\22} {\0A  func.func @norm_lhs(%arg0: tensor<1x1xf16>, %arg1: tensor<1x1xf16>) -> tensor<1x1xf32> attributes {tessera.frontend.authority = \22tracer\22, tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = \2295c9e304d0a75dc18cf890b24c90c8a8e9799dd275a50c29e378c8c24c575401\22, tessera.structured_cfg.schema = \22tessera.structured_cfg.v1\22} {\0A    %0 = tessera.rmsnorm %arg0 {tessera.effect_kind = \22pure\22} : (tensor<1x1xf16>) -> tensor<1x1xf16>\0A    %1 = tessera.matmul %0, %arg1 {output_dtype = \22fp32\22, tessera.effect_kind = \22pure\22} : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf32>\0A    return %1 : tensor<1x1xf32>\0A  }\0A}", tessera.native.tensor_program_source_sha256 = "06c6c1dd475eceeabc6b5f2afd9e79c0f5436332c1f4b3a438b895f80ecf3960", tessera.target = "nvidia_sm120"} {
  func.func @norm_lhs(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) -> memref<?x?xf32> attributes {tessera.frontend.authority = "tracer", tessera.native.tensor_program = 3 : i64, tessera.structured_cfg.blocks = 2 : i64, tessera.structured_cfg.digest = "95c9e304d0a75dc18cf890b24c90c8a8e9799dd275a50c29e378c8c24c575401", tessera.structured_cfg.schema = "tessera.structured_cfg.v1"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %c0_0 = arith.constant 0 : index
    %c65 = arith.constant 65 : index
    %0 = arith.cmpi ugt, %dim, %c0_0 : index
    %1 = arith.cmpi ule, %dim, %c65 : index
    %2 = arith.andi %0, %1 : i1
    cf.assert %2, "native tensor shape extent violates positive capacity"
    %c1 = arith.constant 1 : index
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf16>
    %c0_2 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %3 = arith.cmpi ugt, %dim_1, %c0_2 : index
    %4 = arith.cmpi ule, %dim_1, %c64 : index
    %5 = arith.andi %3, %4 : i1
    cf.assert %5, "native tensor shape extent violates positive capacity"
    %c1_3 = arith.constant 1 : index
    %dim_4 = memref.dim %arg1, %c1_3 : memref<?x?xf16>
    %c0_5 = arith.constant 0 : index
    %c47 = arith.constant 47 : index
    %6 = arith.cmpi ugt, %dim_4, %c0_5 : index
    %7 = arith.cmpi ule, %dim_4, %c47 : index
    %8 = arith.andi %6, %7 : i1
    cf.assert %8, "native tensor shape extent violates positive capacity"
    %c0_6 = arith.constant 0 : index
    %dim_7 = memref.dim %arg1, %c0_6 : memref<?x?xf16>
    %9 = arith.cmpi eq, %dim_7, %dim_1 : index
    cf.assert %9, "native tensor semantic dimension equality violated"
    %alloc = memref.alloc() : memref<65x64xf16>
    %10 = arith.muli %dim, %dim_1 : index
    %c4160 = arith.constant 4160 : index
    %11 = arith.cmpi ule, %10, %c4160 : index
    cf.assert %11, "native tensor active view exceeds allocation capacity"
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [%dim, %dim_1], strides: [%dim_1, 1] : memref<65x64xf16> to memref<?x?xf16>
    tile.program_invoke %arg0, %reinterpret_cast {callee = @tensor_stage_0::@source, input_count = 1 : i64, source_attributes = {tessera.effect_kind = "pure"}} : memref<?x?xf16>, memref<?x?xf16>
    %alloc_8 = memref.alloc() : memref<65x47xf32>
    %12 = arith.muli %dim, %dim_4 : index
    %c3055 = arith.constant 3055 : index
    %13 = arith.cmpi ule, %12, %c3055 : index
    cf.assert %13, "native tensor active view exceeds allocation capacity"
    %reinterpret_cast_9 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [%dim, %dim_4], strides: [%dim_4, 1] : memref<65x47xf32> to memref<?x?xf32>
    tile.program_invoke %reinterpret_cast, %arg1, %reinterpret_cast_9 {callee = @tensor_stage_1::@source, input_count = 2 : i64, source_attributes = {output_dtype = "fp32", shape_bounds = [65, 47, 64], tessera.effect_kind = "pure", transposeA = false, transposeB = false}} : memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf32>
    memref.dealloc %alloc : memref<65x64xf16>
    return %reinterpret_cast_9 : memref<?x?xf32>
  }
  module @tensor_stage_0 attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_capacity_types = [memref<65x64xf16>, memref<65x64xf16>], tessera.native.tensor_program_dimension_bindings = array<i64: 0, 1, 0, 1>, tessera.native.tensor_program_member, tessera.native.tensor_program_shape_profile, tessera.target = "nvidia_sm120"} {
    func.func @source(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {tessera.native.tensor_dimension_bindings = array<i64: 0, 1, 0, 1>, tessera.native.tensor_shape_capacity = array<i64: 65, 64>} {
      %0 = tessera.rmsnorm %arg0 {schedule.artifact_hash = "1d50a728af809d7c8c182cd8a1f2d8f988cf95bec7052e9a814648ad41a3778e", tessera.effect_kind = "pure"} : (tensor<?x?xf16>) -> tensor<?x?xf16>
      %1 = schedule.norm %0 {accum = "f32", arch = "sm_120", artifact_hash = "1d50a728af809d7c8c182cd8a1f2d8f988cf95bec7052e9a814648ad41a3778e", axis = -1 : i64, epsilon = 9.99999974E-6 : f32, kind = "rmsnorm", shape_bounds = [65, 64], storage = "f16", workgroup_size = 128 : i64} : tensor<?x?xf16> -> tensor<?x?xf16>
      schedule.artifact {arch = "sm_120", hash = "1d50a728af809d7c8c182cd8a1f2d8f988cf95bec7052e9a814648ad41a3778e", numeric_policy = "f16->f32", shape_key = "family=norm;storage=f16;axis=-1", tile = {workgroup_size = 128 : i64}}
      return %1 : tensor<?x?xf16>
    }
  }
  module @tensor_stage_1 attributes {tessera.arch = "sm_120", tessera.frontend.authority = "tracer", tessera.ir.version = "1.0", tessera.native.tensor_program_capacity_types = [memref<65x64xf16>, memref<64x47xf16>, memref<65x47xf32>], tessera.native.tensor_program_dimension_bindings = array<i64: 0, 1, 1, 2, 0, 2>, tessera.native.tensor_program_member, tessera.native.tensor_program_shape_profile, tessera.target = "nvidia_sm120"} {
    func.func @source(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>) -> tensor<?x?xf32> attributes {tessera.native.tensor_dimension_bindings = array<i64: 0, 1, 1, 2, 0, 2>, tessera.native.tensor_shape_capacity = array<i64: 65, 47>} {
      %0 = tessera.matmul %arg0, %arg1 {output_dtype = "fp32", rhs_storage_order = "row_major", schedule.artifact_binding = "matmul-0", schedule.artifact_hash = "098e926015411aa63fe5ad8c56d7a6219f7e3ea4d314a32289631f7255b4b825", shape_bounds = [65, 47, 64], tessera.effect_kind = "pure"} : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf32>
      %1 = schedule.matmul %0 {a_layout = "row_major", accum = "f32", activation = "none", arch = "sm_120", artifact_binding = "matmul-0", artifact_hash = "098e926015411aa63fe5ad8c56d7a6219f7e3ea4d314a32289631f7255b4b825", b_layout = "row_major", bias = false, macro_tile_m = 128 : i64, macro_tile_n = 128 : i64, output = "f32", pipeline_depth = 1 : i64, raster_group = 1 : i64, raster_order = "row_major", residual = false, storage = "f16", tile_k = 16 : i64, tile_m = 16 : i64, tile_n = 8 : i64, warps = 1 : i64} : tensor<?x?xf32> -> tensor<?x?xf32>
      schedule.artifact {arch = "sm_120", binding = "matmul-0", hash = "098e926015411aa63fe5ad8c56d7a6219f7e3ea4d314a32289631f7255b4b825", numeric_policy = "f16->f32", shape_key = "M=65;N=47;K=64;dtype=f16", tile = {k = 16 : i64, m = 16 : i64, macro_m = 128 : i64, macro_n = 128 : i64, n = 8 : i64, pipeline_depth = 1 : i64, warps = 1 : i64}}
      return %1 : tensor<?x?xf32>
    }
  }
}

