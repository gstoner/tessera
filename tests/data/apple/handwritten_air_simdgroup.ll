; Hand-written Apple AIR IR that CALLS the simdgroup-matrix builtins.
;
; The S0 fixture proved a GPU runs hand-written IR. This proves the harder and
; more important case: an emitter can reach Apple's MATRIX UNITS the same way.
; That is the ceiling `simdgroup_matrix` sets, and the reason SPIR-V was
; rejected as a path — it cannot express these.
;
; The finding it encodes: simdgroup ops are NOT opaque types or intrinsics
; needing backend support. `simdgroup_float8x8` is a plain <64 x float>, and the
; operations are ordinary external functions — declare + call. Threadgroup
; memory is an addrspace(3) global; barriers are @air.wg.barrier. All of it is
; expressible in MLIR's LLVM dialect without extension.
;
; Kernel: fill an 8x8 accumulator with in[0], store it to threadgroup, then have
; each of 64 lanes copy one element to device memory. Every output element must
; equal in[0] exactly.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float) local_unnamed_addr
declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, float addrspace(3)* nocapture writeonly, <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr
declare void @air.wg.barrier(i32, i32) local_unnamed_addr

@scratch = internal addrspace(3) global [64 x float] undef, align 64

define void @tessera_sg_probe(float addrspace(1)* nocapture readonly "air-buffer-no-alias" %in,
                              float addrspace(1)* nocapture writeonly "air-buffer-no-alias" %out,
                              i32 %tid) local_unnamed_addr #0 {
entry:
  %v = load float, float addrspace(1)* %in, align 4
  %acc = tail call <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float %v)
  %base = getelementptr inbounds [64 x float], [64 x float] addrspace(3)* @scratch, i64 0, i64 0
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %acc, float addrspace(3)* %base, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  tail call void @air.wg.barrier(i32 2, i32 0)
  %idx = zext i32 %tid to i64
  %sp = getelementptr inbounds [64 x float], [64 x float] addrspace(3)* @scratch, i64 0, i64 %idx
  %r = load float, float addrspace(3)* %sp, align 4
  %op = getelementptr inbounds float, float addrspace(1)* %out, i64 %idx
  store float %r, float addrspace(1)* %op, align 4
  ret void
}

attributes #0 = { mustprogress nofree norecurse nounwind }

!llvm.module.flags = !{!0, !1}
!air.kernel = !{!2}
!air.version = !{!6}
!air.language_version = !{!7}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"air.max_device_buffers", i32 31}
!2 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @tessera_sg_probe, !3, !4}
!3 = !{}
!4 = !{!8, !9, !10}
!6 = !{i32 2, i32 8, i32 0}
!7 = !{!"Metal", i32 4, i32 0, i32 0}
!8 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!9 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!10 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
