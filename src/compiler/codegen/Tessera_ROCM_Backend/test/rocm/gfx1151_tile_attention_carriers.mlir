// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s

module {
  llvm.func @attention_forward(
      %q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr, %bias: !llvm.ptr,
      %o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64,
      %d: i64, %dv: i64) attributes {gpu.kernel} {
    tile.attention_kernel %q, %k, %v, %bias, %o, %b, %hq, %hkv, %sq, %sk,
        %d, %dv {
      storage = "f16", accum = "f32", scale = 0.5 : f32,
      causal = true, bias = true, window_left = 8 : i64,
      window_right = 0 : i64, softcap = 2.0 : f32,
      dropout_p = 0.125 : f32, dropout_seed = 17 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }

  llvm.func @attention_backward(
      %do: !llvm.ptr, %q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr,
      %dq: !llvm.ptr, %dk: !llvm.ptr, %dv: !llvm.ptr,
      %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64,
      %d: i64, %dv_dim: i64) attributes {gpu.kernel} {
    tile.attention_backward_kernel %do, %q, %k, %v, %dq, %dk, %dv,
        %b, %hq, %hkv, %sq, %sk, %d, %dv_dim {
      storage = "f32", accum = "f32", scale = 0.5 : f32,
      causal = false, bias = false, window_left = -1 : i64,
      window_right = -1 : i64, softcap = 0.0 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64,
      route = "deterministic_direct", deterministic = true,
      lse_checkpoint = "recompute",
      workspace_bytes = 0 : i64, workspace_owner = "output_element"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @attention_forward
// CHECK: gpu.block_id x
// CHECK: gpu.thread_id x
// CHECK: scf.for
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// CHECK: math.exp
// CHECK: llvm.store
// CHECK-LABEL: llvm.func @attention_backward
// CHECK: gpu.block_id x
// CHECK: gpu.thread_id x
// CHECK-COUNT-3: scf.if
// CHECK: scf.for
// CHECK: math.exp
// CHECK: llvm.store
// CHECK-NOT: tile.attention_kernel
// CHECK-NOT: tile.attention_backward_kernel
