// RUN: %tnv --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

module {
  llvm.func @lion_lineage(%p: !llvm.ptr, %g: !llvm.ptr, %m: !llvm.ptr,
                          %dp: !llvm.ptr, %dm: !llvm.ptr, %outp: !llvm.ptr,
                          %outg: !llvm.ptr, %outm: !llvm.ptr, %n: i64)
      attributes {nvvm.kernel} {
    tile.training_kernel %p, %g, %m, %dp, %dm, %outp, %outg, %outm, %n {
      family = "lion_vjp", storage = "f32",
      derivative_policy = "stop_gradient_through_sign",
      mutation_mode = "functional", alias_policy = "no_input_output_alias",
      state_transition = "m@0-read-only;d_m@1-fresh",
      learning_rate = 1.0e-4 : f32, beta2 = 9.9e-1 : f32,
      weight_decay = 0.0 : f32,
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @lion_lineage
// CHECK: llvm.call @llvm.nvvm.cuda.training.kernel.contract
// CHECK-NOT: tile.training_kernel
