module attributes {tessera.arch = "sm_120", tessera.ir.version = "1.0", tessera.nvidia.arch = "sm_120", tessera.target = "nvidia_sm120"} {
  llvm.func @tessera_tile_norm_layernorm_f16_63e41f1195(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = nvvm.read.ptx.sreg.ctaid.x : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = arith.extui %2 : i32 to i64
    %c128_i64 = arith.constant 128 : i64
    %4 = arith.muli %1, %c128_i64 : i64
    %5 = arith.addi %4, %3 : i64
    %6 = arith.cmpi ult, %5, %arg2 : i64
    scf.if %6 {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %7 = arith.muli %5, %arg3 : i64
      %cst_0 = arith.constant 0.000000e+00 : f32
      %8 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %15 = arith.addi %7, %arg4 : i64
        %16 = llvm.getelementptr %arg0[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %17 = llvm.load %16 {alignment = 2 : i64} : !llvm.ptr -> f16
        %18 = llvm.fpext %17 : f16 to f32
        %19 = arith.addf %arg5, %18 : f32
        scf.yield %19 : f32
      }
      %9 = arith.uitofp %arg3 : i64 to f32
      %10 = arith.divf %8, %9 : f32
      %11 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %15 = arith.addi %7, %arg4 : i64
        %16 = llvm.getelementptr %arg0[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %17 = llvm.load %16 {alignment = 2 : i64} : !llvm.ptr -> f16
        %18 = llvm.fpext %17 : f16 to f32
        %19 = arith.subf %18, %10 : f32
        %20 = arith.mulf %19, %19 : f32
        %21 = arith.addf %arg5, %20 : f32
        scf.yield %21 : f32
      }
      %12 = arith.divf %11, %9 : f32
      %13 = arith.addf %12, %cst : f32
      %14 = nvvm.rsqrt %13 : f32
      scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64  : i64 {
        %15 = arith.addi %7, %arg4 : i64
        %16 = llvm.getelementptr %arg0[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %17 = llvm.load %16 {alignment = 2 : i64} : !llvm.ptr -> f16
        %18 = llvm.fpext %17 : f16 to f32
        %19 = arith.subf %18, %10 : f32
        %20 = arith.mulf %19, %14 : f32
        %21 = llvm.fptrunc %20 : f32 to f16
        %22 = llvm.getelementptr %arg1[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        llvm.store %21, %22 {alignment = 2 : i64} : f16, !llvm.ptr
      }
    }
    llvm.return
  }
}

