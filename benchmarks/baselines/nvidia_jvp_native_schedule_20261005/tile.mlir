module attributes {tessera.arch = "sm_120", tessera.attention_checkpoint_identity = "22bb30ebb7e367bb757916d36d784bac9a1488a2265466e1d39929609dc57ab4", tessera.attention_jvp_contract = {active = [true, true, true], algorithm = "cooperative_saved_lse_moments_v1", arch = "sm_120", argument_roles = array<i64: 0, 1, 2, 3, 4, 5>, causal = true, family = "attention_checkpoint_jvp", ownership = "private_saved_generation_distinct_tangent", scale = 0.353553385 : f32, shape = array<i64: 1, 2, 1, 4, 6, 8, 8>, target = "nvidia_sm120", workgroup_size = 128 : i64}, tessera.attention_jvp_schedule_hash = "32afc5fbcbf1ee4a139fba72873efe001b813b77e228d8b85228315ed31f0f1a", tessera.autodiff.attention_jvp_contract = "{\22active\22:[true,true,true],\22causal\22:true,\22dims\22:[1,2,1,4,6,8,8],\22scale\22:0.35355338454246521,\22schema\22:1}", tessera.native_tensor_contract = "{\22arguments\22:[{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22q\22,\22shape\22:[1,2,4,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22k\22,\22shape\22:[1,1,6,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22v\22,\22shape\22:[1,1,6,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22primal\22,\22shape\22:[1,2,4,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22lse\22,\22shape\22:[1,2,4],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22dq\22,\22shape\22:[1,2,4,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22dk\22,\22shape\22:[1,1,6,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22dv\22,\22shape\22:[1,1,6,8],\22writable\22:false},{\22dtype\22:\22fp32\22,\22kind\22:\22tensor\22,\22name\22:\22tangent\22,\22shape\22:[1,2,4,8],\22writable\22:true},{\22kind\22:\22index\22,\22maximum\22:128,\22minimum\22:128,\22name\22:\22scratch\22}],\22block\22:[128,1,1],\22grid\22:[8,1,1],\22schema\22:1}", tessera.target = "nvidia_sm120"} {
  gpu.module @attention_jvp {
    gpu.func @saved_lse_jvp(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: index) kernel attributes {tessera.schedule_hash = "32afc5fbcbf1ee4a139fba72873efe001b813b77e228d8b85228315ed31f0f1a"} {
      %thread_id_x = gpu.thread_id x
      %block_id_x = gpu.block_id x
      %0 = arith.index_cast %block_id_x : index to i64
      %1 = arith.index_cast %thread_id_x : index to i64
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 0.353553385 : f32
      %cst_1 = arith.constant 1.44269502 : f32
      %c4_i64 = arith.constant 4 : i64
      %2 = arith.remui %0, %c4_i64 : i64
      %c4_i64_2 = arith.constant 4 : i64
      %3 = arith.divui %0, %c4_i64_2 : i64
      %c2_i64 = arith.constant 2 : i64
      %4 = arith.remui %3, %c2_i64 : i64
      %c2_i64_3 = arith.constant 2 : i64
      %5 = arith.divui %3, %c2_i64_3 : i64
      %c2_i64_4 = arith.constant 2 : i64
      %6 = arith.divui %4, %c2_i64_4 : i64
      %c6_i64 = arith.constant 6 : i64
      %c1_i64 = arith.constant 1 : i64
      %7 = arith.muli %5, %c1_i64 : i64
      %8 = arith.addi %7, %6 : i64
      %9 = arith.muli %8, %c6_i64 : i64
      %c8_i64 = arith.constant 8 : i64
      %10 = arith.muli %0, %c8_i64 : i64
      %c8_i64_5 = arith.constant 8 : i64
      %11 = arith.muli %0, %c8_i64_5 : i64
      %c2_i64_6 = arith.constant 2 : i64
      %12 = arith.addi %2, %c2_i64_6 : i64
      %13 = llvm.getelementptr %arg4[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %14 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
      %alloca = memref.alloca(%arg9) : memref<?xf32>
      "tile.alloc_shared"(%alloca) : (memref<?xf32>) -> ()
      %alloca_7 = memref.alloca(%arg9) : memref<?xf32>
      "tile.alloc_shared"(%alloca_7) : (memref<?xf32>) -> ()
      %c8 = arith.constant 8 : index
      scf.for %arg10 = %c0 to %c8 step %c1 {
        %15 = arith.index_cast %arg10 : index to i64
        %c6 = arith.constant 6 : index
        %16:2 = scf.for %arg11 = %thread_id_x to %c6 step %c128 iter_args(%arg12 = %cst, %arg13 = %cst) -> (f32, f32) {
          %25 = arith.index_cast %arg11 : index to i64
          %26 = arith.cmpi ule, %25, %12 : i64
          %27:2 = scf.if %26 -> (f32, f32) {
            %28 = arith.addi %9, %25 : i64
            %c8_i64_10 = arith.constant 8 : i64
            %29 = arith.muli %28, %c8_i64_10 : i64
            %c8_11 = arith.constant 8 : index
            %30:2 = scf.for %arg14 = %c0 to %c8_11 step %c1 iter_args(%arg15 = %cst, %arg16 = %cst) -> (f32, f32) {
              %48 = arith.index_cast %arg14 : index to i64
              %49 = arith.addi %10, %48 : i64
              %50 = llvm.getelementptr %arg0[%49] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
              %51 = llvm.load %50 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
              %52 = arith.addi %29, %48 : i64
              %53 = llvm.getelementptr %arg1[%52] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
              %54 = llvm.load %53 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
              %55 = arith.addi %10, %48 : i64
              %56 = llvm.getelementptr %arg5[%55] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
              %57 = llvm.load %56 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
              %58 = arith.addi %29, %48 : i64
              %59 = llvm.getelementptr %arg6[%58] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
              %60 = llvm.load %59 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
              %61 = arith.mulf %51, %54 : f32
              %62 = arith.addf %arg15, %61 : f32
              %63 = arith.mulf %51, %60 : f32
              %64 = arith.mulf %57, %54 : f32
              %65 = arith.addf %64, %63 : f32
              %66 = arith.addf %arg16, %65 : f32
              scf.yield %62, %66 : f32, f32
            }
            %31 = arith.mulf %30#0, %cst_0 : f32
            %32 = arith.subf %31, %14 : f32
            %33 = arith.mulf %32, %cst_1 : f32
            %34 = math.exp2 %33 : f32
            %35 = arith.mulf %30#1, %cst_0 : f32
            %c8_i64_12 = arith.constant 8 : i64
            %36 = arith.muli %28, %c8_i64_12 : i64
            %37 = arith.addi %36, %15 : i64
            %38 = llvm.getelementptr %arg2[%37] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
            %39 = llvm.load %38 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
            %40 = llvm.getelementptr %arg7[%37] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
            %41 = llvm.load %40 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
            %42 = arith.mulf %34, %35 : f32
            %43 = arith.addf %arg12, %42 : f32
            %44 = arith.mulf %34, %41 : f32
            %45 = arith.mulf %42, %39 : f32
            %46 = arith.addf %45, %44 : f32
            %47 = arith.addf %arg13, %46 : f32
            scf.yield %43, %47 : f32, f32
          } else {
            scf.yield %arg12, %arg13 : f32, f32
          }
          scf.yield %27#0, %27#1 : f32, f32
        }
        memref.store %16#0, %alloca[%thread_id_x] : memref<?xf32>
        memref.store %16#1, %alloca_7[%thread_id_x] : memref<?xf32>
        gpu.barrier
        %c64 = arith.constant 64 : index
        %17 = arith.cmpi ult, %thread_id_x, %c64 : index
        scf.if %17 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c64 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c64 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c32 = arith.constant 32 : index
        %18 = arith.cmpi ult, %thread_id_x, %c32 : index
        scf.if %18 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c32 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c32 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c16 = arith.constant 16 : index
        %19 = arith.cmpi ult, %thread_id_x, %c16 : index
        scf.if %19 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c16 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c16 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c8_8 = arith.constant 8 : index
        %20 = arith.cmpi ult, %thread_id_x, %c8_8 : index
        scf.if %20 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c8_8 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c8_8 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c4 = arith.constant 4 : index
        %21 = arith.cmpi ult, %thread_id_x, %c4 : index
        scf.if %21 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c4 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c4 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c2 = arith.constant 2 : index
        %22 = arith.cmpi ult, %thread_id_x, %c2 : index
        scf.if %22 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c2 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c2 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c1_9 = arith.constant 1 : index
        %23 = arith.cmpi ult, %thread_id_x, %c1_9 : index
        scf.if %23 {
          %25 = memref.load %alloca[%thread_id_x] : memref<?xf32>
          %26 = arith.addi %thread_id_x, %c1_9 : index
          %27 = memref.load %alloca[%26] : memref<?xf32>
          %28 = arith.addf %25, %27 : f32
          memref.store %28, %alloca[%thread_id_x] : memref<?xf32>
          %29 = memref.load %alloca_7[%thread_id_x] : memref<?xf32>
          %30 = arith.addi %thread_id_x, %c1_9 : index
          %31 = memref.load %alloca_7[%30] : memref<?xf32>
          %32 = arith.addf %29, %31 : f32
          memref.store %32, %alloca_7[%thread_id_x] : memref<?xf32>
        }
        gpu.barrier
        %c0_i64 = arith.constant 0 : i64
        %24 = arith.cmpi eq, %1, %c0_i64 : i64
        scf.if %24 {
          %25 = memref.load %alloca[%c0] : memref<?xf32>
          %26 = memref.load %alloca_7[%c0] : memref<?xf32>
          %27 = arith.addi %11, %15 : i64
          %28 = llvm.getelementptr %arg3[%27] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
          %29 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr<1> -> f32
          %30 = llvm.getelementptr %arg8[%27] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
          %31 = arith.mulf %29, %25 : f32
          %32 = arith.subf %26, %31 : f32
          llvm.store %32, %30 {alignment = 4 : i64} : f32, !llvm.ptr<1>
        }
        gpu.barrier
      }
      gpu.return
    }
  }
}

