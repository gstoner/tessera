// What generate-wmma-gemm-kernel does with the schedule knobs it is handed,
// at the level ROCm GEMM actually enters it: a Tile-level tile.matmul_kernel
// from the scheduled route, with via-tile=true (what tessera-rocm-executable's
// input=tile branch passes). executable_pipeline_graph_matmul_options.mlir (ROCm
// suite) proves that pipeline hands the generator the requested knobs; this
// file proves what the generator then does with them.
//
// Converted 2026-09-26 from graph_matmul_generator_knobs.mlir, which drove the
// same knobs through the canonical M/N/K scf.for entry (tessera-tiling ->
// tessera-tile-ir-lowering -> generator). That entry was Lane B's and was
// deleted with it (docs/audit/backend/rocm/ROCM_LANE_MAP.md); the sched-groups
// half is still live on the typed register body and moves here, the canonical
// LDS body's knob refusal went with the body.
//
// Register staging consumes sched-groups: asking for two groups must change
// the kernel.
//
// LDS staging names the typed LDS body, which exists only under via-tile=true.
// The direct lane (via-tile=false) has the register body alone, so an lds
// request there is refused rather than answered with the register kernel.

// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true sched-groups=2' | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt %s --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true sched-groups=0' | FileCheck %s --check-prefix=NOSCHED
// RUN: not tessera-opt %s --allow-unregistered-dialect --generate-wmma-gemm-kernel='canonical-staging=lds' 2>&1 | FileCheck %s --check-prefix=DIRECTLDS

module attributes {tessera.arch = "gfx1151"} {
  func.func @tile_gemm(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
                       %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.tile_m = 16 : i64, tessera.tile_n = 16 : i64,
      tessera.tile_k = 16 : i64, tessera.macro_tile_m = 32 : i64,
      tessera.macro_tile_n = 64 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// SCHED: gpu.func @tile_gemm
// SCHED: rocdl.sched.group.barrier
// SCHED-SAME: vmem_read
// SCHED: rocdl.sched.group.barrier
// SCHED-SAME: mfma_wmma

// NOSCHED: gpu.func @tile_gemm
// NOSCHED-NOT: rocdl.sched.group.barrier
// NOSCHED: gpu.return

// DIRECTLDS: error: generate-wmma-gemm-kernel: canonical-staging=lds selects the LDS-staged typed body and requires via-tile=true
// DIRECTLDS-NOT: gpu.func
