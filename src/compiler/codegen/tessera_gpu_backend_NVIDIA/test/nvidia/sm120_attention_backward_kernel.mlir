// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @tessera_tile_attention_backward_f32_deterministic(
      %do: !llvm.ptr, %q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr,
      %dq: !llvm.ptr, %dk: !llvm.ptr, %dv: !llvm.ptr,
      %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64, %d: i64, %dv_dim: i64)
      attributes {nvvm.kernel} {
    tile.attention_backward_kernel %do, %q, %key, %v, %dq, %dk, %dv,
        %b, %hq, %hkv, %sq, %sk, %d, %dv_dim {
      storage = "f32", accum = "f32", scale = 0.5 : f32,
      causal = true, bias = false, window_left = 2 : i64,
      window_right = 1 : i64, softcap = 1.7 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64,
      route = "deterministic_direct", deterministic = true,
      workspace_bytes = 0 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        !llvm.ptr, i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @tessera_tile_attention_backward_f32_deterministic
// CHECK-SAME: attributes {nvvm.kernel
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.store
// CHECK-NOT: tile.attention_backward_kernel
