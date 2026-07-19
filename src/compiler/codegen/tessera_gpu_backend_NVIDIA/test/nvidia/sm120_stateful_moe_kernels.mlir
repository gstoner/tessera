// RUN: %tnv --tessera-lower-to-nvidia-sm120 %s | FileCheck %s

module {
  llvm.func @replay_decode(%delta: !llvm.ptr, %x: !llvm.ptr, %bcoef: !llvm.ptr,
      %s0: !llvm.ptr, %c: !llvm.ptr, %a: !llvm.ptr, %y: !llvm.ptr,
      %batch: i64, %channels: i64, %state: i64, %tokens: i64)
      attributes {nvvm.kernel} {
    tile.replay_ssm_decode_kernel %delta, %x, %bcoef, %s0, %c, %a, %y,
        %batch, %channels, %state, %tokens {
      storage = "f32", route = "output_only"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
        !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }

  llvm.func @replay_flush(%delta: !llvm.ptr, %x: !llvm.ptr, %bcoef: !llvm.ptr,
      %s0: !llvm.ptr, %a: !llvm.ptr, %batch: i64, %channels: i64,
      %state: i64, %tokens: i64) attributes {nvvm.kernel} {
    tile.replay_ssm_flush_kernel %delta, %x, %bcoef, %s0, %a, %batch,
        %channels, %state, %tokens {
      storage = "f32", route = "state_and_output", deterministic = true
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }

  llvm.func @moe_dispatch(%x: !llvm.ptr, %token: !llvm.ptr, %o: !llvm.ptr,
      %t: i64, %s: i64, %h: i64) attributes {nvvm.kernel} {
    tile.moe_dispatch_kernel %x, %token, %o, %t, %s, %h {
      storage = "f32", index_storage = "i32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @moe_combine(%partials: !llvm.ptr, %token: !llvm.ptr,
      %weights: !llvm.ptr, %o: !llvm.ptr, %t: i64, %s: i64, %h: i64)
      attributes {nvvm.kernel} {
    tile.moe_combine_kernel %partials, %token, %weights, %o, %t, %s, %h {
      storage = "f32", index_storage = "i32", deterministic = true
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }

  llvm.func @grouped_gemm(%x: !llvm.ptr, %w: !llvm.ptr, %offsets: !llvm.ptr,
      %o: !llvm.ptr, %t: i64, %k: i64, %n: i64, %e: i64)
      attributes {nvvm.kernel} {
    tile.grouped_gemm_kernel %x, %w, %offsets, %o, %t, %k, %n, %e {
      storage = "f32", accum = "f32", index_storage = "i32"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @replay_decode
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-LABEL: llvm.func @replay_flush
// CHECK: scf.for
// CHECK: llvm.store
// CHECK-LABEL: llvm.func @moe_dispatch
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK-LABEL: llvm.func @moe_combine
// CHECK: scf.for
// CHECK: llvm.store
// CHECK-LABEL: llvm.func @grouped_gemm
// CHECK: scf.for
// CHECK: llvm.store
// CHECK-NOT: tile.replay_ssm
// CHECK-NOT: tile.moe_
// CHECK-NOT: tile.grouped_gemm_kernel
