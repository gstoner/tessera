// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s
// RUN: %trop --allow-unregistered-dialect --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefix=TARGET
// RUN: not %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true canonical-staging=lds' %s 2>&1 | FileCheck %s --check-prefix=LDS
//
// ROCM-SPLIT-K-1: cross-workgroup split-K on the typed route. The gfx1201 MoE
// router-gate shape (M=16, N=256, K=2048) produces 16 output tiles on 32 WGPs,
// so the Schedule splits K in two. The generator must emit TWO kernels:
//
//   * the PARTIAL (the program's own entry symbol): blockIdx.z selects the
//     slice, the K loop walks [z*1024, z*1024+1024), and the fp32 partial is
//     stored into workspace plane z -- with NO bias operand and NO epilogue;
//   * the ORDERED REDUCE: sums slices 0..S-1 in fixed order, then applies the
//     program's bias + activation exactly once.
//
// The unsplit control below (@unsplit) must produce neither (Decision #10a:
// the negative case is the point). The Target IR boundary (TARGET) must carry
// the pair onto tessera_rocm.wmma_gemm, and the LDS body (LDS) has no split
// partial, so it must refuse rather than run unsplit.

module attributes {tessera.arch = "gfx1201"} {
  func.func @router(%a: !llvm.ptr, %b: !llvm.ptr, %bias: !llvm.ptr, %d: !llvm.ptr) {
    %m = arith.constant 16 : i64
    %n = arith.constant 256 : i64
    %k = arith.constant 2048 : i64
    tile.matmul_kernel %a, %b, %bias, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
      epilogue = #tile.epilogue<bias = true, activation = "gelu", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.canonical_k_loop = true,
      tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
      tessera.macro_tile_m = 16 : i64, tessera.macro_tile_n = 16 : i64,
      tessera.split_k = 2 : i64, tessera.split_k_reduction = "ordered"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }

  func.func @unsplit(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr) {
    %m = arith.constant 1024 : i64
    %n = arith.constant 1024 : i64
    %k = arith.constant 2048 : i64
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 2>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.canonical_k_loop = true,
      tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64, tessera.tile_k = 16 : i64,
      tessera.macro_tile_m = 64 : i64, tessera.macro_tile_n = 64 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// The partial: A, B, W, M, N, K -- no bias memref.
// CHECK-LABEL: gpu.module @router_mod
// CHECK: gpu.func @router(%{{.*}}: memref<?xf16>, %{{.*}}: memref<?xf16>, %[[W:.*]]: memref<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) kernel
// CHECK-SAME: tessera.rocm.split_k = 2 : i64
// CHECK-SAME: tessera.rocm.split_k_reduce_entry = "router_splitk_reduce"
// CHECK-SAME: tessera.rocm.split_k_reduction = "ordered"
// CHECK-SAME: tessera.rocm.split_k_role = "partial"
// CHECK-SAME: tessera.rocm.split_k_slice = 1024 : i64
// CHECK: gpu.block_id  z
// CHECK-NOT: arith.maximumf
// CHECK-NOT: math.tanh
// CHECK: gpu.return
// The ordered reduce: W, bias, D, M, N; a fixed-order loop over the slices,
// then bias add and the activation, once.
// CHECK: gpu.func @router_splitk_reduce(%[[RW:.*]]: memref<?xf32>, %[[RB:.*]]: memref<?xf32>, %[[RD:.*]]: memref<?xf32>, %{{.*}}: index, %{{.*}}: index) kernel
// CHECK-SAME: tessera.rocm.split_k_role = "ordered_reduce"
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C2]] step %{{.*}} iter_args
// CHECK: memref.load %[[RW]][
// CHECK: arith.addf
// CHECK: memref.load %[[RB]][
// CHECK: arith.addf
// CHECK: memref.store %{{.*}}, %[[RD]][
// CHECK: gpu.return
// The unsplit control has neither half.
// CHECK-LABEL: gpu.module @unsplit_mod
// CHECK-NOT: split_k
// CHECK-NOT: gpu.block_id  z
// CHECK-NOT: splitk_reduce

// TARGET-LABEL: func.func @router
// TARGET: tessera_rocm.wmma_gemm
// TARGET-SAME: split_k = 2 : i64
// TARGET-SAME: split_k_reduction = "ordered"
// TARGET-LABEL: func.func @unsplit
// TARGET: tessera_rocm.wmma_gemm
// TARGET-NOT: split_k

// LDS: ROCM_SPLIT_K_{{UNSUPPORTED}}: split-K is implemented on the register-staged body only
