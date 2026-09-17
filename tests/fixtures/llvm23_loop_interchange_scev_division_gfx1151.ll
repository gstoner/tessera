; Minimal reproducer for an LLVM 23.1.1 assertion in LoopInterchange, reduced
; with llvm-reduce from the AD-generated `product` kernel of a rank-2 dynamic
; backward (`tensor<?x?xf32>`, y = x*x, role=backward) as serialized by
; `gpu-module-to-binary` for gfx1151 (Tajasarus, assertions-ON LLVM, 2026-09-17).
;
;   opt -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -disable-output THIS.ll
;   opt: .../ScalarEvolutionDivision.cpp:59: SCEVDivision::divide(...):
;   Assertion `Numerator->getType() == Denominator->getType() &&
;   "Numerator and Denominator must have the same type"' failed.
;   #13 llvm::LoopInterchangePass::run(llvm::LoopNest&, ...)
;
; A two-deep i64 loop nest storing through a GEP whose base is a generic
; pointer cast from address space 5 (32-bit indices in the AMDGPU data layout)
; hands LoopInterchange's dependence check SCEVs of two widths. Passes without
; incident under `-enable-loopinterchange=0`, and an NDEBUG `opt` (ROCm 10.0's)
; compiles it in silence. The real kernel this came from executes correctly on
; gfx1151 (benchmarks/baselines/runtime_shape_frames_20260908/
; rocm_gfx1151_revalidation_20260917.json, case dynamic_matrix_backward).
; Not Tessera code; owed upstream. Recorded in docs/audit/backend/rocm/todo.md
; under ROCM-HOST-RED-ZONE-FOLLOWUPS-2026-09-17.
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @product(i64 %0) {
  br label %2

2:                                                ; preds = %13, %1
  %3 = phi i64 [ %14, %13 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, %0
  br i1 %4, label %5, label %15

5:                                                ; preds = %8, %2
  %6 = phi i64 [ %12, %8 ], [ 0, %2 ]
  %7 = icmp slt i64 %6, %0
  br i1 %7, label %8, label %13

8:                                                ; preds = %5
  %9 = mul i64 %3, %0
  %10 = getelementptr [4 x i8], ptr addrspacecast (ptr addrspace(5) null to ptr), i64 %9
  %11 = getelementptr [4 x i8], ptr %10, i64 %6
  store float 0.000000e+00, ptr %11, align 4
  %12 = add i64 %6, 1
  br label %5

13:                                               ; preds = %5
  %14 = add i64 %3, 1
  br label %2

15:                                               ; preds = %2
  ret void
}
