module {
  func.func @__tessera_tensor_shape_plan(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.getelementptr %arg0[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.getelementptr %arg0[%c1_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %3 = llvm.load %2 : !llvm.ptr -> i64
    %c2_i64 = arith.constant 2 : i64
    %4 = llvm.getelementptr %arg0[%c2_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %5 = llvm.load %4 : !llvm.ptr -> i64
    %c3_i64 = arith.constant 3 : i64
    %6 = llvm.getelementptr %arg0[%c3_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %7 = llvm.load %6 : !llvm.ptr -> i64
    %c0_i64_0 = arith.constant 0 : i64
    %8 = arith.cmpi ugt, %1, %c0_i64_0 : i64
    %c65_i64 = arith.constant 65 : i64
    %9 = arith.cmpi ule, %1, %c65_i64 : i64
    %10 = arith.andi %8, %9 : i1
    cf.cond_br %10, ^bb2, ^bb1
  ^bb1:  // 21 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21
    %c1_i32 = arith.constant 1 : i32
    return %c1_i32 : i32
  ^bb2:  // pred: ^bb0
    %c0_i64_1 = arith.constant 0 : i64
    %11 = arith.cmpi ugt, %3, %c0_i64_1 : i64
    %c64_i64 = arith.constant 64 : i64
    %12 = arith.cmpi ule, %3, %c64_i64 : i64
    %13 = arith.andi %11, %12 : i1
    cf.cond_br %13, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %c0_i64_2 = arith.constant 0 : i64
    %14 = arith.cmpi ugt, %7, %c0_i64_2 : i64
    %c47_i64 = arith.constant 47 : i64
    %15 = arith.cmpi ule, %7, %c47_i64 : i64
    %16 = arith.andi %14, %15 : i1
    cf.cond_br %16, ^bb4, ^bb1
  ^bb4:  // pred: ^bb3
    %17 = arith.cmpi eq, %1, %1 : i64
    cf.cond_br %17, ^bb5, ^bb1
  ^bb5:  // pred: ^bb4
    %18 = arith.cmpi eq, %3, %3 : i64
    cf.cond_br %18, ^bb6, ^bb1
  ^bb6:  // pred: ^bb5
    %19 = arith.cmpi eq, %5, %3 : i64
    cf.cond_br %19, ^bb7, ^bb1
  ^bb7:  // pred: ^bb6
    %20 = arith.cmpi eq, %7, %7 : i64
    cf.cond_br %20, ^bb8, ^bb1
  ^bb8:  // pred: ^bb7
    %21 = arith.muli %1, %3 : i64
    %c127_i64 = arith.constant 127 : i64
    %22 = arith.addi %1, %c127_i64 : i64
    %c128_i64 = arith.constant 128 : i64
    %23 = arith.divui %22, %c128_i64 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %c1_i64_4 = arith.constant 1 : i64
    %c0_i64_5 = arith.constant 0 : i64
    %24 = arith.cmpi ugt, %1, %c0_i64_5 : i64
    %c65_i64_6 = arith.constant 65 : i64
    %25 = arith.cmpi ule, %1, %c65_i64_6 : i64
    %26 = arith.andi %24, %25 : i1
    cf.cond_br %26, ^bb9, ^bb1
  ^bb9:  // pred: ^bb8
    %c0_i64_7 = arith.constant 0 : i64
    %27 = arith.cmpi ugt, %3, %c0_i64_7 : i64
    %c64_i64_8 = arith.constant 64 : i64
    %28 = arith.cmpi ule, %3, %c64_i64_8 : i64
    %29 = arith.andi %27, %28 : i1
    cf.cond_br %29, ^bb10, ^bb1
  ^bb10:  // pred: ^bb9
    %c0_i64_9 = arith.constant 0 : i64
    %30 = arith.cmpi ugt, %23, %c0_i64_9 : i64
    %c1_i64_10 = arith.constant 1 : i64
    %31 = arith.cmpi ule, %23, %c1_i64_10 : i64
    %32 = arith.andi %30, %31 : i1
    cf.cond_br %32, ^bb11, ^bb1
  ^bb11:  // pred: ^bb10
    %c0_i64_11 = arith.constant 0 : i64
    %33 = arith.cmpi ugt, %c1_i64_3, %c0_i64_11 : i64
    %c1_i64_12 = arith.constant 1 : i64
    %34 = arith.cmpi ule, %c1_i64_3, %c1_i64_12 : i64
    %35 = arith.andi %33, %34 : i1
    cf.cond_br %35, ^bb12, ^bb1
  ^bb12:  // pred: ^bb11
    %c0_i64_13 = arith.constant 0 : i64
    %36 = arith.cmpi ugt, %c1_i64_4, %c0_i64_13 : i64
    %c1_i64_14 = arith.constant 1 : i64
    %37 = arith.cmpi ule, %c1_i64_4, %c1_i64_14 : i64
    %38 = arith.andi %36, %37 : i1
    cf.cond_br %38, ^bb13, ^bb1
  ^bb13:  // pred: ^bb12
    %c0_i64_15 = arith.constant 0 : i64
    %c0_i64_16 = arith.constant 0 : i64
    %c0_i64_17 = arith.constant 0 : i64
    %c0_i64_18 = arith.constant 0 : i64
    %c0_i64_19 = arith.constant 0 : i64
    %c0_i64_20 = arith.constant 0 : i64
    %c7_i64 = arith.constant 7 : i64
    %39 = arith.addi %7, %c7_i64 : i64
    %c8_i64 = arith.constant 8 : i64
    %40 = arith.divui %39, %c8_i64 : i64
    %c15_i64 = arith.constant 15 : i64
    %41 = arith.addi %1, %c15_i64 : i64
    %c16_i64 = arith.constant 16 : i64
    %42 = arith.divui %41, %c16_i64 : i64
    %c1_i64_21 = arith.constant 1 : i64
    %c0_i64_22 = arith.constant 0 : i64
    %43 = arith.cmpi ugt, %1, %c0_i64_22 : i64
    %c65_i64_23 = arith.constant 65 : i64
    %44 = arith.cmpi ule, %1, %c65_i64_23 : i64
    %45 = arith.andi %43, %44 : i1
    cf.cond_br %45, ^bb14, ^bb1
  ^bb14:  // pred: ^bb13
    %c0_i64_24 = arith.constant 0 : i64
    %46 = arith.cmpi ugt, %7, %c0_i64_24 : i64
    %c47_i64_25 = arith.constant 47 : i64
    %47 = arith.cmpi ule, %7, %c47_i64_25 : i64
    %48 = arith.andi %46, %47 : i1
    cf.cond_br %48, ^bb15, ^bb1
  ^bb15:  // pred: ^bb14
    %c0_i64_26 = arith.constant 0 : i64
    %49 = arith.cmpi ugt, %3, %c0_i64_26 : i64
    %c64_i64_27 = arith.constant 64 : i64
    %50 = arith.cmpi ule, %3, %c64_i64_27 : i64
    %51 = arith.andi %49, %50 : i1
    cf.cond_br %51, ^bb16, ^bb1
  ^bb16:  // pred: ^bb15
    %c0_i64_28 = arith.constant 0 : i64
    %52 = arith.cmpi ugt, %3, %c0_i64_28 : i64
    %c64_i64_29 = arith.constant 64 : i64
    %53 = arith.cmpi ule, %3, %c64_i64_29 : i64
    %54 = arith.andi %52, %53 : i1
    cf.cond_br %54, ^bb17, ^bb1
  ^bb17:  // pred: ^bb16
    %c0_i64_30 = arith.constant 0 : i64
    %55 = arith.cmpi ugt, %7, %c0_i64_30 : i64
    %c47_i64_31 = arith.constant 47 : i64
    %56 = arith.cmpi ule, %7, %c47_i64_31 : i64
    %57 = arith.andi %55, %56 : i1
    cf.cond_br %57, ^bb18, ^bb1
  ^bb18:  // pred: ^bb17
    %c0_i64_32 = arith.constant 0 : i64
    %58 = arith.cmpi ugt, %7, %c0_i64_32 : i64
    %c47_i64_33 = arith.constant 47 : i64
    %59 = arith.cmpi ule, %7, %c47_i64_33 : i64
    %60 = arith.andi %58, %59 : i1
    cf.cond_br %60, ^bb19, ^bb1
  ^bb19:  // pred: ^bb18
    %c0_i64_34 = arith.constant 0 : i64
    %61 = arith.cmpi ugt, %40, %c0_i64_34 : i64
    %c6_i64 = arith.constant 6 : i64
    %62 = arith.cmpi ule, %40, %c6_i64 : i64
    %63 = arith.andi %61, %62 : i1
    cf.cond_br %63, ^bb20, ^bb1
  ^bb20:  // pred: ^bb19
    %c0_i64_35 = arith.constant 0 : i64
    %64 = arith.cmpi ugt, %42, %c0_i64_35 : i64
    %c5_i64 = arith.constant 5 : i64
    %65 = arith.cmpi ule, %42, %c5_i64 : i64
    %66 = arith.andi %64, %65 : i1
    cf.cond_br %66, ^bb21, ^bb1
  ^bb21:  // pred: ^bb20
    %c0_i64_36 = arith.constant 0 : i64
    %67 = arith.cmpi ugt, %c1_i64_21, %c0_i64_36 : i64
    %c1_i64_37 = arith.constant 1 : i64
    %68 = arith.cmpi ule, %c1_i64_21, %c1_i64_37 : i64
    %69 = arith.andi %67, %68 : i1
    cf.cond_br %69, ^bb22, ^bb1
  ^bb22:  // pred: ^bb21
    %c0_i64_38 = arith.constant 0 : i64
    %c0_i64_39 = arith.constant 0 : i64
    %c0_i64_40 = arith.constant 0 : i64
    %70 = llvm.getelementptr %arg1[%c0_i64_40] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %70 : i64, !llvm.ptr
    %c1_i64_41 = arith.constant 1 : i64
    %71 = llvm.getelementptr %arg1[%c1_i64_41] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %71 : i64, !llvm.ptr
    %c2_i64_42 = arith.constant 2 : i64
    %72 = llvm.getelementptr %arg1[%c2_i64_42] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %72 : i64, !llvm.ptr
    %c3_i64_43 = arith.constant 3 : i64
    %73 = llvm.getelementptr %arg1[%c3_i64_43] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %73 : i64, !llvm.ptr
    %c4_i64 = arith.constant 4 : i64
    %74 = llvm.getelementptr %arg1[%c4_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %74 : i64, !llvm.ptr
    %c5_i64_44 = arith.constant 5 : i64
    %75 = llvm.getelementptr %arg1[%c5_i64_44] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %75 : i64, !llvm.ptr
    %c6_i64_45 = arith.constant 6 : i64
    %76 = llvm.getelementptr %arg1[%c6_i64_45] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %76 : i64, !llvm.ptr
    %c7_i64_46 = arith.constant 7 : i64
    %77 = llvm.getelementptr %arg1[%c7_i64_46] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %77 : i64, !llvm.ptr
    %c8_i64_47 = arith.constant 8 : i64
    %78 = llvm.getelementptr %arg1[%c8_i64_47] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %78 : i64, !llvm.ptr
    %c9_i64 = arith.constant 9 : i64
    %79 = llvm.getelementptr %arg1[%c9_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %79 : i64, !llvm.ptr
    %c10_i64 = arith.constant 10 : i64
    %80 = llvm.getelementptr %arg1[%c10_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_15, %80 : i64, !llvm.ptr
    %c11_i64 = arith.constant 11 : i64
    %81 = llvm.getelementptr %arg1[%c11_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_16, %81 : i64, !llvm.ptr
    %c12_i64 = arith.constant 12 : i64
    %82 = llvm.getelementptr %arg1[%c12_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_17, %82 : i64, !llvm.ptr
    %c13_i64 = arith.constant 13 : i64
    %83 = llvm.getelementptr %arg1[%c13_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_18, %83 : i64, !llvm.ptr
    %c14_i64 = arith.constant 14 : i64
    %84 = llvm.getelementptr %arg1[%c14_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_19, %84 : i64, !llvm.ptr
    %c15_i64_48 = arith.constant 15 : i64
    %85 = llvm.getelementptr %arg1[%c15_i64_48] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_20, %85 : i64, !llvm.ptr
    %c16_i64_49 = arith.constant 16 : i64
    %86 = llvm.getelementptr %arg1[%c16_i64_49] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %23, %86 : i64, !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %87 = llvm.getelementptr %arg1[%c17_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_3, %87 : i64, !llvm.ptr
    %c18_i64 = arith.constant 18 : i64
    %88 = llvm.getelementptr %arg1[%c18_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_4, %88 : i64, !llvm.ptr
    %c19_i64 = arith.constant 19 : i64
    %89 = llvm.getelementptr %arg1[%c19_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %89 : i64, !llvm.ptr
    %c20_i64 = arith.constant 20 : i64
    %90 = llvm.getelementptr %arg1[%c20_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %90 : i64, !llvm.ptr
    %c21_i64 = arith.constant 21 : i64
    %91 = llvm.getelementptr %arg1[%c21_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %91 : i64, !llvm.ptr
    %c22_i64 = arith.constant 22 : i64
    %92 = llvm.getelementptr %arg1[%c22_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %92 : i64, !llvm.ptr
    %c23_i64 = arith.constant 23 : i64
    %93 = llvm.getelementptr %arg1[%c23_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %93 : i64, !llvm.ptr
    %c24_i64 = arith.constant 24 : i64
    %94 = llvm.getelementptr %arg1[%c24_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %94 : i64, !llvm.ptr
    %c25_i64 = arith.constant 25 : i64
    %95 = llvm.getelementptr %arg1[%c25_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_38, %95 : i64, !llvm.ptr
    %c26_i64 = arith.constant 26 : i64
    %96 = llvm.getelementptr %arg1[%c26_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_39, %96 : i64, !llvm.ptr
    %c27_i64 = arith.constant 27 : i64
    %97 = llvm.getelementptr %arg1[%c27_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %40, %97 : i64, !llvm.ptr
    %c28_i64 = arith.constant 28 : i64
    %98 = llvm.getelementptr %arg1[%c28_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %42, %98 : i64, !llvm.ptr
    %c29_i64 = arith.constant 29 : i64
    %99 = llvm.getelementptr %arg1[%c29_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_21, %99 : i64, !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
