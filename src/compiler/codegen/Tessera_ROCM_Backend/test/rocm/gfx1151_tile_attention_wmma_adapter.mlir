// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-wmma-flash-attn-kernel)' %s | FileCheck %s --check-prefix=KERNEL

module {
  llvm.func @attention_wmma(
      %q: !llvm.ptr, %k: !llvm.ptr, %v: !llvm.ptr, %o: !llvm.ptr,
      %b: i64, %hq: i64, %hkv: i64, %sq: i64, %sk: i64,
      %d: i64, %dv: i64) attributes {gpu.kernel} {
    tile.attention_kernel %q, %k, %v, %o, %b, %hq, %hkv, %sq, %sk, %d, %dv {
      storage = "f16", accum = "f32", scale = 0.125 : f32,
      causal = true, bias = false, window_left = 64 : i64,
      window_right = 0 : i64, softcap = 8.0 : f32,
      dropout_p = 0.0 : f32, dropout_seed = 0 : i64,
      head_dim = 64 : i64, value_dim = 64 : i64, gqa = true
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }
}

// TARGET: tessera_rocm.flash_attn
// TARGET-SAME: gqa = true
// TARGET-SAME: head_dim = 64
// TARGET-SAME: logit_softcap = true
// TARGET-SAME: schedule = "gfx1151_wmma_streaming"
// TARGET-SAME: sliding_window = true
// TARGET-SAME: source = "tile.attention_kernel"
// TARGET-NOT: tile.attention_kernel

// KERNEL: gpu.module @attention_wmma_mod
// KERNEL: gpu.func @attention_wmma(
// KERNEL-SAME: memref<?xf16>
// KERNEL: tessera_rocm.wmma
// KERNEL: math.exp
// KERNEL-NOT: tessera_rocm.flash_attn
// KERNEL-NOT: tile.attention_kernel
