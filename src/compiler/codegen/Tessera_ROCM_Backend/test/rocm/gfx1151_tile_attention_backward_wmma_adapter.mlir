// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-wmma-flash-attn-bwd-kernel)' %s | FileCheck %s --check-prefix=KERNEL

module {
  llvm.func @attention_backward_wmma(
      %do: !llvm.ptr, %q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr,
      %bias: !llvm.ptr, %dq: !llvm.ptr, %dk: !llvm.ptr, %dv: !llvm.ptr,
      %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64,
      %d: i64, %dv_dim: i64) attributes {gpu.kernel} {
    tile.attention_backward_kernel %do, %q, %k, %v, %bias, %dq, %dk, %dv,
        %b, %hq, %hkv, %sq, %sk, %d, %dv_dim {
      storage = "f16", accum = "f32", scale = 0.125 : f32,
      causal = true, bias = true, window_left = 64 : i64,
      window_right = 0 : i64, softcap = 8.0 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64,
      head_dim = 64 : i64, value_dim = 64 : i64, gqa = true,
      route = "deterministic_direct", deterministic = true,
      lse_checkpoint = "recompute",
      workspace_bytes = 0 : i64, workspace_owner = "output_element"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.flash_attn_bwd
// TARGET-SAME: attn_bias = true
// TARGET-SAME: gqa = true
// TARGET-SAME: head_dim = 64
// TARGET-SAME: logit_softcap = true
// TARGET-SAME: schedule = "gfx1151_wmma_backward_split_reduced"
// TARGET-SAME: sliding_window = true
// TARGET-SAME: source = "tile.attention_backward_kernel"
// TARGET-SAME: split_reduced = true
// TARGET-NOT: tile.attention_backward_kernel

// KERNEL: gpu.module @attention_backward_wmma_mod
// KERNEL: gpu.func @attention_backward_wmma_pre(
// KERNEL: gpu.func @attention_backward_wmma_dkdv(
// KERNEL: gpu.func @attention_backward_wmma_dkdv_reduce(
// KERNEL: gpu.func @attention_backward_wmma_dq(
// KERNEL: tessera_rocm.wmma
// KERNEL-NOT: tessera_rocm.flash_attn_bwd
// KERNEL-NOT: tile.attention_backward_kernel
