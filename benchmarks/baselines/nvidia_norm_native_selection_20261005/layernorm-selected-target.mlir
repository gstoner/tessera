module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.nvidia.arch = "sm_120", tessera.target = "nvidia_sm120"} {
  llvm.mlir.global internal @__tessera_sm120_reduce_scratch_f32() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<128 x f32>
  llvm.func @tessera_tile_norm_layernorm_f16_cooperative_128_91af0be746(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = nvvm.read.ptx.sreg.ctaid.x : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = arith.extui %2 : i32 to i64
    %4 = arith.cmpi ult, %1, %arg2 : i64
    %5 = llvm.mlir.addressof @__tessera_sm120_reduce_scratch_f32 : !llvm.ptr<3>
    scf.if %4 {
      %c128_i64 = arith.constant 128 : i64
      %6 = arith.muli %1, %arg3 : i64
      %cst_0 = arith.constant 0.000000e+00 : f32
      %7 = scf.for %arg4 = %3 to %arg3 step %c128_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %34 = arith.addi %6, %arg4 : i64
        %35 = llvm.getelementptr %arg0[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %36 = llvm.load %35 {alignment = 2 : i64} : !llvm.ptr -> f16
        %37 = llvm.fpext %36 : f16 to f32
        %38 = arith.addf %arg5, %37 : f32
        scf.yield %38 : f32
      }
      %8 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %7, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      nvvm.barrier
      %c64_i64 = arith.constant 64 : i64
      %9 = arith.cmpi ult, %3, %c64_i64 : i64
      scf.if %9 {
        %c64_i64_9 = arith.constant 64 : i64
        %34 = arith.addi %3, %c64_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c32_i64 = arith.constant 32 : i64
      %10 = arith.cmpi ult, %3, %c32_i64 : i64
      scf.if %10 {
        %c32_i64_9 = arith.constant 32 : i64
        %34 = arith.addi %3, %c32_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c16_i64 = arith.constant 16 : i64
      %11 = arith.cmpi ult, %3, %c16_i64 : i64
      scf.if %11 {
        %c16_i64_9 = arith.constant 16 : i64
        %34 = arith.addi %3, %c16_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c8_i64 = arith.constant 8 : i64
      %12 = arith.cmpi ult, %3, %c8_i64 : i64
      scf.if %12 {
        %c8_i64_9 = arith.constant 8 : i64
        %34 = arith.addi %3, %c8_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c4_i64 = arith.constant 4 : i64
      %13 = arith.cmpi ult, %3, %c4_i64 : i64
      scf.if %13 {
        %c4_i64_9 = arith.constant 4 : i64
        %34 = arith.addi %3, %c4_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c2_i64 = arith.constant 2 : i64
      %14 = arith.cmpi ult, %3, %c2_i64 : i64
      scf.if %14 {
        %c2_i64_9 = arith.constant 2 : i64
        %34 = arith.addi %3, %c2_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c1_i64 = arith.constant 1 : i64
      %15 = arith.cmpi ult, %3, %c1_i64 : i64
      scf.if %15 {
        %c1_i64_9 = arith.constant 1 : i64
        %34 = arith.addi %3, %c1_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c0_i64 = arith.constant 0 : i64
      %16 = llvm.getelementptr %5[%c0_i64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %17 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
      nvvm.barrier
      %18 = arith.uitofp %arg3 : i64 to f32
      %19 = arith.divf %17, %18 : f32
      %20 = scf.for %arg4 = %3 to %arg3 step %c128_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %34 = arith.addi %6, %arg4 : i64
        %35 = llvm.getelementptr %arg0[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %36 = llvm.load %35 {alignment = 2 : i64} : !llvm.ptr -> f16
        %37 = llvm.fpext %36 : f16 to f32
        %38 = arith.subf %37, %19 : f32
        %39 = arith.mulf %38, %38 : f32
        %40 = arith.addf %arg5, %39 : f32
        scf.yield %40 : f32
      }
      %21 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %20, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      nvvm.barrier
      %c64_i64_1 = arith.constant 64 : i64
      %22 = arith.cmpi ult, %3, %c64_i64_1 : i64
      scf.if %22 {
        %c64_i64_9 = arith.constant 64 : i64
        %34 = arith.addi %3, %c64_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c32_i64_2 = arith.constant 32 : i64
      %23 = arith.cmpi ult, %3, %c32_i64_2 : i64
      scf.if %23 {
        %c32_i64_9 = arith.constant 32 : i64
        %34 = arith.addi %3, %c32_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c16_i64_3 = arith.constant 16 : i64
      %24 = arith.cmpi ult, %3, %c16_i64_3 : i64
      scf.if %24 {
        %c16_i64_9 = arith.constant 16 : i64
        %34 = arith.addi %3, %c16_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c8_i64_4 = arith.constant 8 : i64
      %25 = arith.cmpi ult, %3, %c8_i64_4 : i64
      scf.if %25 {
        %c8_i64_9 = arith.constant 8 : i64
        %34 = arith.addi %3, %c8_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c4_i64_5 = arith.constant 4 : i64
      %26 = arith.cmpi ult, %3, %c4_i64_5 : i64
      scf.if %26 {
        %c4_i64_9 = arith.constant 4 : i64
        %34 = arith.addi %3, %c4_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c2_i64_6 = arith.constant 2 : i64
      %27 = arith.cmpi ult, %3, %c2_i64_6 : i64
      scf.if %27 {
        %c2_i64_9 = arith.constant 2 : i64
        %34 = arith.addi %3, %c2_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c1_i64_7 = arith.constant 1 : i64
      %28 = arith.cmpi ult, %3, %c1_i64_7 : i64
      scf.if %28 {
        %c1_i64_9 = arith.constant 1 : i64
        %34 = arith.addi %3, %c1_i64_9 : i64
        %35 = llvm.getelementptr %5[%34] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %36 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %37 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %38 = arith.addf %36, %37 : f32
        llvm.store %38, %21 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c0_i64_8 = arith.constant 0 : i64
      %29 = llvm.getelementptr %5[%c0_i64_8] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %30 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
      nvvm.barrier
      %31 = arith.divf %30, %18 : f32
      %32 = arith.addf %31, %cst : f32
      %33 = nvvm.rsqrt %32 : f32
      scf.for %arg4 = %3 to %arg3 step %c128_i64  : i64 {
        %34 = arith.addi %6, %arg4 : i64
        %35 = llvm.getelementptr %arg0[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %36 = llvm.load %35 {alignment = 2 : i64} : !llvm.ptr -> f16
        %37 = llvm.fpext %36 : f16 to f32
        %38 = arith.subf %37, %19 : f32
        %39 = arith.mulf %38, %33 : f32
        %40 = llvm.fptrunc %39 : f32 to f16
        %41 = llvm.getelementptr %arg1[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        llvm.store %40, %41 {alignment = 2 : i64} : f16, !llvm.ptr
      }
    }
    llvm.return
  }
}

