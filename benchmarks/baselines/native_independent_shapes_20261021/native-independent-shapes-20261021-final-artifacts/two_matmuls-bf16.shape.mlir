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
    %c4_i64 = arith.constant 4 : i64
    %8 = llvm.getelementptr %arg0[%c4_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %9 = llvm.load %8 : !llvm.ptr -> i64
    %c5_i64 = arith.constant 5 : i64
    %10 = llvm.getelementptr %arg0[%c5_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %11 = llvm.load %10 : !llvm.ptr -> i64
    %c0_i64_0 = arith.constant 0 : i64
    %12 = arith.cmpi ugt, %1, %c0_i64_0 : i64
    %c65_i64 = arith.constant 65 : i64
    %13 = arith.cmpi ule, %1, %c65_i64 : i64
    %14 = arith.andi %12, %13 : i1
    cf.cond_br %14, ^bb2, ^bb1
  ^bb1:  // 29 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24, ^bb25, ^bb26, ^bb27, ^bb28, ^bb29
    %c1_i32 = arith.constant 1 : i32
    return %c1_i32 : i32
  ^bb2:  // pred: ^bb0
    %c0_i64_1 = arith.constant 0 : i64
    %15 = arith.cmpi ugt, %7, %c0_i64_1 : i64
    %c64_i64 = arith.constant 64 : i64
    %16 = arith.cmpi ule, %7, %c64_i64 : i64
    %17 = arith.andi %15, %16 : i1
    cf.cond_br %17, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %c0_i64_2 = arith.constant 0 : i64
    %18 = arith.cmpi ugt, %11, %c0_i64_2 : i64
    %c47_i64 = arith.constant 47 : i64
    %19 = arith.cmpi ule, %11, %c47_i64 : i64
    %20 = arith.andi %18, %19 : i1
    cf.cond_br %20, ^bb4, ^bb1
  ^bb4:  // pred: ^bb3
    %21 = arith.cmpi eq, %1, %1 : i64
    cf.cond_br %21, ^bb5, ^bb1
  ^bb5:  // pred: ^bb4
    %c32_i64 = arith.constant 32 : i64
    %22 = arith.cmpi eq, %3, %c32_i64 : i64
    cf.cond_br %22, ^bb6, ^bb1
  ^bb6:  // pred: ^bb5
    %c32_i64_3 = arith.constant 32 : i64
    %23 = arith.cmpi eq, %5, %c32_i64_3 : i64
    cf.cond_br %23, ^bb7, ^bb1
  ^bb7:  // pred: ^bb6
    %24 = arith.cmpi eq, %7, %7 : i64
    cf.cond_br %24, ^bb8, ^bb1
  ^bb8:  // pred: ^bb7
    %25 = arith.cmpi eq, %9, %7 : i64
    cf.cond_br %25, ^bb9, ^bb1
  ^bb9:  // pred: ^bb8
    %26 = arith.cmpi eq, %11, %11 : i64
    cf.cond_br %26, ^bb10, ^bb1
  ^bb10:  // pred: ^bb9
    %c32_i64_4 = arith.constant 32 : i64
    %c32_i64_5 = arith.constant 32 : i64
    %c7_i64 = arith.constant 7 : i64
    %27 = arith.addi %7, %c7_i64 : i64
    %c8_i64 = arith.constant 8 : i64
    %28 = arith.divui %27, %c8_i64 : i64
    %c15_i64 = arith.constant 15 : i64
    %29 = arith.addi %1, %c15_i64 : i64
    %c16_i64 = arith.constant 16 : i64
    %30 = arith.divui %29, %c16_i64 : i64
    %c1_i64_6 = arith.constant 1 : i64
    %c0_i64_7 = arith.constant 0 : i64
    %31 = arith.cmpi ugt, %1, %c0_i64_7 : i64
    %c65_i64_8 = arith.constant 65 : i64
    %32 = arith.cmpi ule, %1, %c65_i64_8 : i64
    %33 = arith.andi %31, %32 : i1
    cf.cond_br %33, ^bb11, ^bb1
  ^bb11:  // pred: ^bb10
    %c0_i64_9 = arith.constant 0 : i64
    %34 = arith.cmpi ugt, %7, %c0_i64_9 : i64
    %c64_i64_10 = arith.constant 64 : i64
    %35 = arith.cmpi ule, %7, %c64_i64_10 : i64
    %36 = arith.andi %34, %35 : i1
    cf.cond_br %36, ^bb12, ^bb1
  ^bb12:  // pred: ^bb11
    %c0_i64_11 = arith.constant 0 : i64
    %37 = arith.cmpi ugt, %c32_i64_4, %c0_i64_11 : i64
    %c32_i64_12 = arith.constant 32 : i64
    %38 = arith.cmpi ule, %c32_i64_4, %c32_i64_12 : i64
    %39 = arith.andi %37, %38 : i1
    cf.cond_br %39, ^bb13, ^bb1
  ^bb13:  // pred: ^bb12
    %c0_i64_13 = arith.constant 0 : i64
    %40 = arith.cmpi ugt, %c32_i64_4, %c0_i64_13 : i64
    %c32_i64_14 = arith.constant 32 : i64
    %41 = arith.cmpi ule, %c32_i64_4, %c32_i64_14 : i64
    %42 = arith.andi %40, %41 : i1
    cf.cond_br %42, ^bb14, ^bb1
  ^bb14:  // pred: ^bb13
    %c0_i64_15 = arith.constant 0 : i64
    %43 = arith.cmpi ugt, %7, %c0_i64_15 : i64
    %c64_i64_16 = arith.constant 64 : i64
    %44 = arith.cmpi ule, %7, %c64_i64_16 : i64
    %45 = arith.andi %43, %44 : i1
    cf.cond_br %45, ^bb15, ^bb1
  ^bb15:  // pred: ^bb14
    %c0_i64_17 = arith.constant 0 : i64
    %46 = arith.cmpi ugt, %7, %c0_i64_17 : i64
    %c64_i64_18 = arith.constant 64 : i64
    %47 = arith.cmpi ule, %7, %c64_i64_18 : i64
    %48 = arith.andi %46, %47 : i1
    cf.cond_br %48, ^bb16, ^bb1
  ^bb16:  // pred: ^bb15
    %c0_i64_19 = arith.constant 0 : i64
    %49 = arith.cmpi ugt, %28, %c0_i64_19 : i64
    %c8_i64_20 = arith.constant 8 : i64
    %50 = arith.cmpi ule, %28, %c8_i64_20 : i64
    %51 = arith.andi %49, %50 : i1
    cf.cond_br %51, ^bb17, ^bb1
  ^bb17:  // pred: ^bb16
    %c0_i64_21 = arith.constant 0 : i64
    %52 = arith.cmpi ugt, %30, %c0_i64_21 : i64
    %c5_i64_22 = arith.constant 5 : i64
    %53 = arith.cmpi ule, %30, %c5_i64_22 : i64
    %54 = arith.andi %52, %53 : i1
    cf.cond_br %54, ^bb18, ^bb1
  ^bb18:  // pred: ^bb17
    %c0_i64_23 = arith.constant 0 : i64
    %55 = arith.cmpi ugt, %c1_i64_6, %c0_i64_23 : i64
    %c1_i64_24 = arith.constant 1 : i64
    %56 = arith.cmpi ule, %c1_i64_6, %c1_i64_24 : i64
    %57 = arith.andi %55, %56 : i1
    cf.cond_br %57, ^bb19, ^bb1
  ^bb19:  // pred: ^bb18
    %c0_i64_25 = arith.constant 0 : i64
    %c0_i64_26 = arith.constant 0 : i64
    %58 = arith.muli %1, %7 : i64
    %c127_i64 = arith.constant 127 : i64
    %59 = arith.addi %1, %c127_i64 : i64
    %c128_i64 = arith.constant 128 : i64
    %60 = arith.divui %59, %c128_i64 : i64
    %c1_i64_27 = arith.constant 1 : i64
    %c1_i64_28 = arith.constant 1 : i64
    %c0_i64_29 = arith.constant 0 : i64
    %61 = arith.cmpi ugt, %1, %c0_i64_29 : i64
    %c65_i64_30 = arith.constant 65 : i64
    %62 = arith.cmpi ule, %1, %c65_i64_30 : i64
    %63 = arith.andi %61, %62 : i1
    cf.cond_br %63, ^bb20, ^bb1
  ^bb20:  // pred: ^bb19
    %c0_i64_31 = arith.constant 0 : i64
    %64 = arith.cmpi ugt, %7, %c0_i64_31 : i64
    %c64_i64_32 = arith.constant 64 : i64
    %65 = arith.cmpi ule, %7, %c64_i64_32 : i64
    %66 = arith.andi %64, %65 : i1
    cf.cond_br %66, ^bb21, ^bb1
  ^bb21:  // pred: ^bb20
    %c0_i64_33 = arith.constant 0 : i64
    %67 = arith.cmpi ugt, %60, %c0_i64_33 : i64
    %c1_i64_34 = arith.constant 1 : i64
    %68 = arith.cmpi ule, %60, %c1_i64_34 : i64
    %69 = arith.andi %67, %68 : i1
    cf.cond_br %69, ^bb22, ^bb1
  ^bb22:  // pred: ^bb21
    %c0_i64_35 = arith.constant 0 : i64
    %70 = arith.cmpi ugt, %c1_i64_27, %c0_i64_35 : i64
    %c1_i64_36 = arith.constant 1 : i64
    %71 = arith.cmpi ule, %c1_i64_27, %c1_i64_36 : i64
    %72 = arith.andi %70, %71 : i1
    cf.cond_br %72, ^bb23, ^bb1
  ^bb23:  // pred: ^bb22
    %c0_i64_37 = arith.constant 0 : i64
    %73 = arith.cmpi ugt, %c1_i64_28, %c0_i64_37 : i64
    %c1_i64_38 = arith.constant 1 : i64
    %74 = arith.cmpi ule, %c1_i64_28, %c1_i64_38 : i64
    %75 = arith.andi %73, %74 : i1
    cf.cond_br %75, ^bb24, ^bb1
  ^bb24:  // pred: ^bb23
    %c0_i64_39 = arith.constant 0 : i64
    %c0_i64_40 = arith.constant 0 : i64
    %c0_i64_41 = arith.constant 0 : i64
    %c0_i64_42 = arith.constant 0 : i64
    %c0_i64_43 = arith.constant 0 : i64
    %c0_i64_44 = arith.constant 0 : i64
    %c7_i64_45 = arith.constant 7 : i64
    %76 = arith.addi %11, %c7_i64_45 : i64
    %c8_i64_46 = arith.constant 8 : i64
    %77 = arith.divui %76, %c8_i64_46 : i64
    %c15_i64_47 = arith.constant 15 : i64
    %78 = arith.addi %1, %c15_i64_47 : i64
    %c16_i64_48 = arith.constant 16 : i64
    %79 = arith.divui %78, %c16_i64_48 : i64
    %c1_i64_49 = arith.constant 1 : i64
    %c0_i64_50 = arith.constant 0 : i64
    %80 = arith.cmpi ugt, %1, %c0_i64_50 : i64
    %c65_i64_51 = arith.constant 65 : i64
    %81 = arith.cmpi ule, %1, %c65_i64_51 : i64
    %82 = arith.andi %80, %81 : i1
    cf.cond_br %82, ^bb25, ^bb1
  ^bb25:  // pred: ^bb24
    %c0_i64_52 = arith.constant 0 : i64
    %83 = arith.cmpi ugt, %11, %c0_i64_52 : i64
    %c47_i64_53 = arith.constant 47 : i64
    %84 = arith.cmpi ule, %11, %c47_i64_53 : i64
    %85 = arith.andi %83, %84 : i1
    cf.cond_br %85, ^bb26, ^bb1
  ^bb26:  // pred: ^bb25
    %c0_i64_54 = arith.constant 0 : i64
    %86 = arith.cmpi ugt, %7, %c0_i64_54 : i64
    %c64_i64_55 = arith.constant 64 : i64
    %87 = arith.cmpi ule, %7, %c64_i64_55 : i64
    %88 = arith.andi %86, %87 : i1
    cf.cond_br %88, ^bb27, ^bb1
  ^bb27:  // pred: ^bb26
    %c0_i64_56 = arith.constant 0 : i64
    %89 = arith.cmpi ugt, %77, %c0_i64_56 : i64
    %c6_i64 = arith.constant 6 : i64
    %90 = arith.cmpi ule, %77, %c6_i64 : i64
    %91 = arith.andi %89, %90 : i1
    cf.cond_br %91, ^bb28, ^bb1
  ^bb28:  // pred: ^bb27
    %c0_i64_57 = arith.constant 0 : i64
    %92 = arith.cmpi ugt, %79, %c0_i64_57 : i64
    %c5_i64_58 = arith.constant 5 : i64
    %93 = arith.cmpi ule, %79, %c5_i64_58 : i64
    %94 = arith.andi %92, %93 : i1
    cf.cond_br %94, ^bb29, ^bb1
  ^bb29:  // pred: ^bb28
    %c0_i64_59 = arith.constant 0 : i64
    %95 = arith.cmpi ugt, %c1_i64_49, %c0_i64_59 : i64
    %c1_i64_60 = arith.constant 1 : i64
    %96 = arith.cmpi ule, %c1_i64_49, %c1_i64_60 : i64
    %97 = arith.andi %95, %96 : i1
    cf.cond_br %97, ^bb30, ^bb1
  ^bb30:  // pred: ^bb29
    %c0_i64_61 = arith.constant 0 : i64
    %c0_i64_62 = arith.constant 0 : i64
    %c0_i64_63 = arith.constant 0 : i64
    %c0_i64_64 = arith.constant 0 : i64
    %c0_i64_65 = arith.constant 0 : i64
    %c0_i64_66 = arith.constant 0 : i64
    %98 = llvm.getelementptr %arg1[%c0_i64_66] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %98 : i64, !llvm.ptr
    %c1_i64_67 = arith.constant 1 : i64
    %99 = llvm.getelementptr %arg1[%c1_i64_67] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c32_i64_4, %99 : i64, !llvm.ptr
    %c2_i64_68 = arith.constant 2 : i64
    %100 = llvm.getelementptr %arg1[%c2_i64_68] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c32_i64_5, %100 : i64, !llvm.ptr
    %c3_i64_69 = arith.constant 3 : i64
    %101 = llvm.getelementptr %arg1[%c3_i64_69] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %101 : i64, !llvm.ptr
    %c4_i64_70 = arith.constant 4 : i64
    %102 = llvm.getelementptr %arg1[%c4_i64_70] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %102 : i64, !llvm.ptr
    %c5_i64_71 = arith.constant 5 : i64
    %103 = llvm.getelementptr %arg1[%c5_i64_71] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %11, %103 : i64, !llvm.ptr
    %c6_i64_72 = arith.constant 6 : i64
    %104 = llvm.getelementptr %arg1[%c6_i64_72] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %104 : i64, !llvm.ptr
    %c7_i64_73 = arith.constant 7 : i64
    %105 = llvm.getelementptr %arg1[%c7_i64_73] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %105 : i64, !llvm.ptr
    %c8_i64_74 = arith.constant 8 : i64
    %106 = llvm.getelementptr %arg1[%c8_i64_74] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %106 : i64, !llvm.ptr
    %c9_i64 = arith.constant 9 : i64
    %107 = llvm.getelementptr %arg1[%c9_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %107 : i64, !llvm.ptr
    %c10_i64 = arith.constant 10 : i64
    %108 = llvm.getelementptr %arg1[%c10_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %108 : i64, !llvm.ptr
    %c11_i64 = arith.constant 11 : i64
    %109 = llvm.getelementptr %arg1[%c11_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %11, %109 : i64, !llvm.ptr
    %c12_i64 = arith.constant 12 : i64
    %110 = llvm.getelementptr %arg1[%c12_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %110 : i64, !llvm.ptr
    %c13_i64 = arith.constant 13 : i64
    %111 = llvm.getelementptr %arg1[%c13_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %111 : i64, !llvm.ptr
    %c14_i64 = arith.constant 14 : i64
    %112 = llvm.getelementptr %arg1[%c14_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c32_i64_4, %112 : i64, !llvm.ptr
    %c15_i64_75 = arith.constant 15 : i64
    %113 = llvm.getelementptr %arg1[%c15_i64_75] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c32_i64_4, %113 : i64, !llvm.ptr
    %c16_i64_76 = arith.constant 16 : i64
    %114 = llvm.getelementptr %arg1[%c16_i64_76] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %114 : i64, !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %115 = llvm.getelementptr %arg1[%c17_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %115 : i64, !llvm.ptr
    %c18_i64 = arith.constant 18 : i64
    %116 = llvm.getelementptr %arg1[%c18_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_25, %116 : i64, !llvm.ptr
    %c19_i64 = arith.constant 19 : i64
    %117 = llvm.getelementptr %arg1[%c19_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_26, %117 : i64, !llvm.ptr
    %c20_i64 = arith.constant 20 : i64
    %118 = llvm.getelementptr %arg1[%c20_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %28, %118 : i64, !llvm.ptr
    %c21_i64 = arith.constant 21 : i64
    %119 = llvm.getelementptr %arg1[%c21_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %30, %119 : i64, !llvm.ptr
    %c22_i64 = arith.constant 22 : i64
    %120 = llvm.getelementptr %arg1[%c22_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_6, %120 : i64, !llvm.ptr
    %c23_i64 = arith.constant 23 : i64
    %121 = llvm.getelementptr %arg1[%c23_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %121 : i64, !llvm.ptr
    %c24_i64 = arith.constant 24 : i64
    %122 = llvm.getelementptr %arg1[%c24_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %122 : i64, !llvm.ptr
    %c25_i64 = arith.constant 25 : i64
    %123 = llvm.getelementptr %arg1[%c25_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_39, %123 : i64, !llvm.ptr
    %c26_i64 = arith.constant 26 : i64
    %124 = llvm.getelementptr %arg1[%c26_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_40, %124 : i64, !llvm.ptr
    %c27_i64 = arith.constant 27 : i64
    %125 = llvm.getelementptr %arg1[%c27_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_41, %125 : i64, !llvm.ptr
    %c28_i64 = arith.constant 28 : i64
    %126 = llvm.getelementptr %arg1[%c28_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_42, %126 : i64, !llvm.ptr
    %c29_i64 = arith.constant 29 : i64
    %127 = llvm.getelementptr %arg1[%c29_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_43, %127 : i64, !llvm.ptr
    %c30_i64 = arith.constant 30 : i64
    %128 = llvm.getelementptr %arg1[%c30_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_44, %128 : i64, !llvm.ptr
    %c31_i64 = arith.constant 31 : i64
    %129 = llvm.getelementptr %arg1[%c31_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %60, %129 : i64, !llvm.ptr
    %c32_i64_77 = arith.constant 32 : i64
    %130 = llvm.getelementptr %arg1[%c32_i64_77] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_27, %130 : i64, !llvm.ptr
    %c33_i64 = arith.constant 33 : i64
    %131 = llvm.getelementptr %arg1[%c33_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_28, %131 : i64, !llvm.ptr
    %c34_i64 = arith.constant 34 : i64
    %132 = llvm.getelementptr %arg1[%c34_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %132 : i64, !llvm.ptr
    %c35_i64 = arith.constant 35 : i64
    %133 = llvm.getelementptr %arg1[%c35_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %11, %133 : i64, !llvm.ptr
    %c36_i64 = arith.constant 36 : i64
    %134 = llvm.getelementptr %arg1[%c36_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %134 : i64, !llvm.ptr
    %c37_i64 = arith.constant 37 : i64
    %135 = llvm.getelementptr %arg1[%c37_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_61, %135 : i64, !llvm.ptr
    %c38_i64 = arith.constant 38 : i64
    %136 = llvm.getelementptr %arg1[%c38_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_62, %136 : i64, !llvm.ptr
    %c39_i64 = arith.constant 39 : i64
    %137 = llvm.getelementptr %arg1[%c39_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_63, %137 : i64, !llvm.ptr
    %c40_i64 = arith.constant 40 : i64
    %138 = llvm.getelementptr %arg1[%c40_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_64, %138 : i64, !llvm.ptr
    %c41_i64 = arith.constant 41 : i64
    %139 = llvm.getelementptr %arg1[%c41_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c0_i64_65, %139 : i64, !llvm.ptr
    %c42_i64 = arith.constant 42 : i64
    %140 = llvm.getelementptr %arg1[%c42_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %77, %140 : i64, !llvm.ptr
    %c43_i64 = arith.constant 43 : i64
    %141 = llvm.getelementptr %arg1[%c43_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %79, %141 : i64, !llvm.ptr
    %c44_i64 = arith.constant 44 : i64
    %142 = llvm.getelementptr %arg1[%c44_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %c1_i64_49, %142 : i64, !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
