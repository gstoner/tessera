// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-wmma-flash-attn-kernel)' %s | FileCheck %s --check-prefix=KERNEL

// Bias + softcap and deterministic nonzero dropout compose on the optimized
// WMMA route. The scalar direct lowering remains the semantic reference for
// this exact counter formula.
module {
  llvm.func @attention_features(
      %q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr, %bias: !llvm.ptr,
      %o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64,
      %d: i64, %dv: i64) {
    tile.attention_kernel %q, %k, %v, %bias, %o, %b, %hq, %hkv, %sq, %sk,
        %d, %dv {
      storage = "f16", accum = "f32", scale = 0.125 : f32,
      causal = true, bias = true, window_left = 64 : i64,
      window_right = 0 : i64, softcap = 8.0 : f32,
      dropout_p = 0.125 : f32, dropout_seed = 17 : i64,
      head_dim = 64 : i64, value_dim = 64 : i64, gqa = true
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.flash_attn
// TARGET-SAME: attn_bias = true
// TARGET-SAME: dropout = true
// TARGET-SAME: gqa = true
// TARGET-SAME: logit_softcap = true
// TARGET-SAME: schedule = "gfx1151_wmma_streaming"
// TARGET-SAME: sliding_window = true

// KERNEL: gpu.func @attention_features(
// KERNEL: math.tanh
// KERNEL: arith.muli
// KERNEL: arith.andi
// KERNEL: arith.cmpi uge
// KERNEL: arith.select
// KERNEL: tessera_rocm.wmma
// KERNEL-NOT: tessera_rocm.flash_attn
