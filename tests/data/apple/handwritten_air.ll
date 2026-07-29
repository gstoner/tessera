; Hand-written Apple AIR IR — the S0 fixture.
;
; NOT produced by Apple's MSL front end. This file exists to prove a GPU will
; execute LLVM IR that Tessera wrote, which is the precondition for an
; MLIR -> AIR emitter (Decision #26a). `xcrun metal -c` accepts it, `metallib`
; packages it, and `metal-objdump --syms` finds the entry — but only running it
; shows the numbers are right, which is what test_gpu_executes_hand_written_air_ir
; checks.
;
; The metadata below is the AIR contract any emitter must produce: !air.kernel
; naming the function, one !air.buffer descriptor per argument (location index,
; access, address space, element size/align/type/name), and the builtin
; descriptor for thread_position_in_grid. Device buffers are addrspace(1).
;
; Kernel body is `o[i] = a[i] * 3.0f` — exact in f32 so the test can demand
; bit-equality rather than a tolerance.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare i32 @air.thread_position_in_grid.i32() local_unnamed_addr

define void @tessera_handwritten(float addrspace(1)* nocapture readonly "air-buffer-no-alias" %a,
                                 float addrspace(1)* nocapture writeonly "air-buffer-no-alias" %o,
                                 i32 %tid) local_unnamed_addr #0 {
entry:
  %idx = zext i32 %tid to i64
  %pa = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %v  = load float, float addrspace(1)* %pa, align 4
  %r  = fmul float %v, 3.000000e+00
  %po = getelementptr inbounds float, float addrspace(1)* %o, i64 %idx
  store float %r, float addrspace(1)* %po, align 4
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3}
!air.kernel = !{!4}
!air.version = !{!8}
!air.language_version = !{!9}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"air.max_device_buffers", i32 31}
!2 = !{i32 7, !"air.max_constant_buffers", i32 31}
!3 = !{i32 7, !"air.max_textures", i32 128}
!4 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @tessera_handwritten, !5, !6}
!5 = !{}
!6 = !{!10, !11, !12}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!11 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"o"}
!12 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
