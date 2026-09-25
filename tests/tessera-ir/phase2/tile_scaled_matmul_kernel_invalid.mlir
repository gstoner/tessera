// RUN: not tessera-opt %s 2>&1 | FileCheck %s

// scale_k = 32 over instruction k = 16 needs instruction_steps = 2; every
// other partial_accumulator field below is valid, so this rejects the count.
// CHECK: 'tile.scaled_matmul_kernel' op partial_accumulator must state the physical contract's zero-init, combination, scope, scale_k/instruction_k steps, and isolated scheduling boundary
llvm.func @wrong_partial_steps(%a: !llvm.ptr, %b: !llvm.ptr,
                               %sa: !llvm.ptr, %sb: !llvm.ptr,
                               %d: !llvm.ptr, %m: i64, %n: i64, %k: i64) {
  tile.scaled_matmul_kernel %a, %b, %sa, %sb, %d, %m, %n, %k {
    mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16,
                          a = "e4m3", b = "e4m3", acc = "f32",
                          a_layout = "row_major", b_layout = "col_major",
                          k_blocks = 2, scale_k = 32, scale_fmt = "e8m0">,
    partial_accumulator = {scope = "scale_group", init = "zero",
      combine = "scale_outer_product_then_add", instruction_steps = 1 : i64,
      schedule_scope = "scale_group", cross_step_motion = "forbid"}
  } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
  llvm.return
}
