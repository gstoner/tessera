module attributes {tessera.nvidia.arch = "sm_120"} {
  llvm.func @tessera_tile_softmax_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
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
      %cst = arith.constant 0xFF800000 : f32
      %8 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %12 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> f32
        %13 = arith.maximumf %arg5, %12 : f32
        scf.yield %13 : f32
      }
      %cst_0 = arith.constant 0.000000e+00 : f32
      %9 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %12 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> f32
        %13 = arith.subf %12, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %14 = arith.mulf %13, %cst_1 : f32
        %15 = nvvm.ex2 %14 : f32
        %16 = arith.addf %arg5, %15 : f32
        scf.yield %16 : f32
      }
      scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %12 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> f32
        %13 = arith.subf %12, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %14 = arith.mulf %13, %cst_1 : f32
        %15 = nvvm.ex2 %14 : f32
        %16 = arith.divf %15, %9 : f32
        %17 = llvm.getelementptr %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %16, %17 {alignment = 4 : i64} : f32, !llvm.ptr
      }
    }
    llvm.return
  }
  llvm.func @tessera_tile_softmax_f16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
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
      %cst = arith.constant 0xFF800000 : f32
      %8 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> f16
        %13 = llvm.fpext %12 : f16 to f32
        %14 = arith.maximumf %arg5, %13 : f32
        scf.yield %14 : f32
      }
      %cst_0 = arith.constant 0.000000e+00 : f32
      %9 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> f16
        %13 = llvm.fpext %12 : f16 to f32
        %14 = arith.subf %13, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %15 = arith.mulf %14, %cst_1 : f32
        %16 = nvvm.ex2 %15 : f32
        %17 = arith.addf %arg5, %16 : f32
        scf.yield %17 : f32
      }
      scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> f16
        %13 = llvm.fpext %12 : f16 to f32
        %14 = arith.subf %13, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %15 = arith.mulf %14, %cst_1 : f32
        %16 = nvvm.ex2 %15 : f32
        %17 = arith.divf %16, %9 : f32
        %18 = llvm.fptrunc %17 : f32 to f16
        %19 = llvm.getelementptr %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f16
        llvm.store %18, %19 {alignment = 2 : i64} : f16, !llvm.ptr
      }
    }
    llvm.return
  }
  llvm.func @tessera_tile_softmax_bf16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64) attributes {nvvm.kernel} {
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
      %cst = arith.constant 0xFF800000 : f32
      %8 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> bf16
        %13 = llvm.fpext %12 : bf16 to f32
        %14 = arith.maximumf %arg5, %13 : f32
        scf.yield %14 : f32
      }
      %cst_0 = arith.constant 0.000000e+00 : f32
      %9 = scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64 iter_args(%arg5 = %cst_0) -> (f32)  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> bf16
        %13 = llvm.fpext %12 : bf16 to f32
        %14 = arith.subf %13, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %15 = arith.mulf %14, %cst_1 : f32
        %16 = nvvm.ex2 %15 : f32
        %17 = arith.addf %arg5, %16 : f32
        scf.yield %17 : f32
      }
      scf.for %arg4 = %c0_i64 to %arg3 step %c1_i64  : i64 {
        %10 = arith.addi %7, %arg4 : i64
        %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
        %12 = llvm.load %11 {alignment = 2 : i64} : !llvm.ptr -> bf16
        %13 = llvm.fpext %12 : bf16 to f32
        %14 = arith.subf %13, %8 : f32
        %cst_1 = arith.constant 1.44269502 : f32
        %15 = arith.mulf %14, %cst_1 : f32
        %16 = nvvm.ex2 %15 : f32
        %17 = arith.divf %16, %9 : f32
        %18 = llvm.fptrunc %17 : f32 to bf16
        %19 = llvm.getelementptr %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
        llvm.store %18, %19 {alignment = 2 : i64} : bf16, !llvm.ptr
      }
    }
    llvm.return
  }
}

