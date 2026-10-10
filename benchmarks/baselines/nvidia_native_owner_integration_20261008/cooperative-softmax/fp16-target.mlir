module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.nvidia.arch = "sm_120", tessera.target = "nvidia_sm120"} {
  llvm.mlir.global internal @__tessera_sm120_reduce_scratch_f32() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<128 x f32>
  llvm.func @tessera_tile_softmax_f16_cooperative_128(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %0 = nvvm.read.ptx.sreg.ctaid.x : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = arith.extui %2 : i32 to i64
    %4 = arith.cmpi ult, %1, %arg2 : i64
    %5 = llvm.mlir.addressof @__tessera_sm120_reduce_scratch_f32 : !llvm.ptr<3>
    scf.if %4 {
      %c128_i64 = arith.constant 128 : i64
      %6 = arith.muli %1, %arg3 : i64
      %cst = arith.constant 0xFF800000 : f32
      %7 = scf.for %arg4 = %3 to %arg3 step %c128_i64 iter_args(%arg5 = %cst) -> (f32)  : i64 {
        %29 = arith.addi %6, %arg4 : i64
        %30 = llvm.getelementptr %arg0[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %31 = llvm.load %30 {alignment = 2 : i64} : !llvm.ptr -> f16
        %32 = llvm.fpext %31 : f16 to f32
        %33 = arith.maximumf %arg5, %32 : f32
        scf.yield %33 : f32
      }
      %8 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %7, %8 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      nvvm.barrier
      %c64_i64 = arith.constant 64 : i64
      %9 = arith.cmpi ult, %3, %c64_i64 : i64
      scf.if %9 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c64_i64_9 = arith.constant 64 : i64
        %30 = arith.addi %3, %c64_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c32_i64 = arith.constant 32 : i64
      %10 = arith.cmpi ult, %3, %c32_i64 : i64
      scf.if %10 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c32_i64_9 = arith.constant 32 : i64
        %30 = arith.addi %3, %c32_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c16_i64 = arith.constant 16 : i64
      %11 = arith.cmpi ult, %3, %c16_i64 : i64
      scf.if %11 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c16_i64_9 = arith.constant 16 : i64
        %30 = arith.addi %3, %c16_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c8_i64 = arith.constant 8 : i64
      %12 = arith.cmpi ult, %3, %c8_i64 : i64
      scf.if %12 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c8_i64_9 = arith.constant 8 : i64
        %30 = arith.addi %3, %c8_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c4_i64 = arith.constant 4 : i64
      %13 = arith.cmpi ult, %3, %c4_i64 : i64
      scf.if %13 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c4_i64_9 = arith.constant 4 : i64
        %30 = arith.addi %3, %c4_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c2_i64 = arith.constant 2 : i64
      %14 = arith.cmpi ult, %3, %c2_i64 : i64
      scf.if %14 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c2_i64_9 = arith.constant 2 : i64
        %30 = arith.addi %3, %c2_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c1_i64 = arith.constant 1 : i64
      %15 = arith.cmpi ult, %3, %c1_i64 : i64
      scf.if %15 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c1_i64_9 = arith.constant 1 : i64
        %30 = arith.addi %3, %c1_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.maximumf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c0_i64 = arith.constant 0 : i64
      %16 = llvm.getelementptr %5[%c0_i64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %17 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
      nvvm.barrier
      %cst_0 = arith.constant 0.000000e+00 : f32
      %18 = scf.for %arg4 = %3 to %arg3 step %c128_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %29 = arith.addi %6, %arg4 : i64
        %30 = llvm.getelementptr %arg0[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %31 = llvm.load %30 {alignment = 2 : i64} : !llvm.ptr -> f16
        %32 = llvm.fpext %31 : f16 to f32
        %33 = arith.subf %32, %17 : f32
        %cst_9 = arith.constant 1.44269502 : f32
        %34 = arith.mulf %33, %cst_9 : f32
        %35 = nvvm.ex2 %34 : f32
        %36 = arith.addf %arg5, %35 : f32
        scf.yield %36 : f32
      }
      %19 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      llvm.store %18, %19 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      nvvm.barrier
      %c64_i64_1 = arith.constant 64 : i64
      %20 = arith.cmpi ult, %3, %c64_i64_1 : i64
      scf.if %20 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c64_i64_9 = arith.constant 64 : i64
        %30 = arith.addi %3, %c64_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c32_i64_2 = arith.constant 32 : i64
      %21 = arith.cmpi ult, %3, %c32_i64_2 : i64
      scf.if %21 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c32_i64_9 = arith.constant 32 : i64
        %30 = arith.addi %3, %c32_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c16_i64_3 = arith.constant 16 : i64
      %22 = arith.cmpi ult, %3, %c16_i64_3 : i64
      scf.if %22 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c16_i64_9 = arith.constant 16 : i64
        %30 = arith.addi %3, %c16_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c8_i64_4 = arith.constant 8 : i64
      %23 = arith.cmpi ult, %3, %c8_i64_4 : i64
      scf.if %23 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c8_i64_9 = arith.constant 8 : i64
        %30 = arith.addi %3, %c8_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c4_i64_5 = arith.constant 4 : i64
      %24 = arith.cmpi ult, %3, %c4_i64_5 : i64
      scf.if %24 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c4_i64_9 = arith.constant 4 : i64
        %30 = arith.addi %3, %c4_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c2_i64_6 = arith.constant 2 : i64
      %25 = arith.cmpi ult, %3, %c2_i64_6 : i64
      scf.if %25 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c2_i64_9 = arith.constant 2 : i64
        %30 = arith.addi %3, %c2_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c1_i64_7 = arith.constant 1 : i64
      %26 = arith.cmpi ult, %3, %c1_i64_7 : i64
      scf.if %26 {
        %29 = llvm.getelementptr %5[%3] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %c1_i64_9 = arith.constant 1 : i64
        %30 = arith.addi %3, %c1_i64_9 : i64
        %31 = llvm.getelementptr %5[%30] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %33 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
        %34 = arith.addf %32, %33 : f32
        llvm.store %34, %29 {alignment = 4 : i64} : f32, !llvm.ptr<3>
      }
      nvvm.barrier
      %c0_i64_8 = arith.constant 0 : i64
      %27 = llvm.getelementptr %5[%c0_i64_8] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %28 = llvm.load %27 {alignment = 4 : i64} : !llvm.ptr<3> -> f32
      nvvm.barrier
      scf.for %arg4 = %3 to %arg3 step %c128_i64  : i64 {
        %29 = arith.addi %6, %arg4 : i64
        %30 = llvm.getelementptr %arg0[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %31 = llvm.load %30 {alignment = 2 : i64} : !llvm.ptr -> f16
        %32 = llvm.fpext %31 : f16 to f32
        %33 = arith.subf %32, %17 : f32
        %cst_9 = arith.constant 1.44269502 : f32
        %34 = arith.mulf %33, %cst_9 : f32
        %35 = nvvm.ex2 %34 : f32
        %36 = arith.divf %35, %28 : f32
        %37 = llvm.fptrunc %36 : f32 to f16
        %38 = llvm.getelementptr %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        llvm.store %37, %38 {alignment = 2 : i64} : f16, !llvm.ptr
      }
    }
    llvm.return
  }
}

