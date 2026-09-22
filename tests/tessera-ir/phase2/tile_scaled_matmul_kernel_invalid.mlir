// RUN: not tessera-opt %s 2>&1 | FileCheck %s

// CHECK: 'tile.scaled_matmul_kernel' op partial_accumulator must state zero-init, scale-group scope, scale_outer_product_then_add, scale_k/instruction_k steps, and an isolated scale-group scheduling boundary
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
