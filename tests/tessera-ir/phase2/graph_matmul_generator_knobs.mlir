// Graph-input matmul reaches generate-wmma-gemm-kernel with via-tile=false, by
// the canonical M/N/K loop matcher, never by an attribute -- so these fixtures
// drive the same tiling passes tessera-rocm-executable's graph branch runs.
// executable_pipeline_graph_matmul_options.mlir (ROCm suite) proves that
// pipeline hands the generator the requested knobs; this file proves what the
// generator then does with them (Codex review on #836).
//
// Register staging (the general body) consumes sched-groups: asking for two
// groups must change the kernel.
//
// LDS staging reaches the canonical comparison body, one fixed wave with an
// unpadded one-slab copy that implements none of the knobs. A request that
// changes one is refused by name, listing exactly what it cannot honour
// (Decision #21a), instead of being answered with that same fixed kernel.
// Knobs left at their defaults still produce the body.

// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel='sched-groups=2' | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt %s --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel='sched-groups=0' | FileCheck %s --check-prefix=NOSCHED
// RUN: not tessera-opt %s --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel='canonical-staging=lds lds-pad-dwords=4 lds-waves-m=1 lds-double-buffer=true' 2>&1 | FileCheck %s --check-prefix=REFUSE
// RUN: tessera-opt %s --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel='canonical-staging=lds lds-pad-dwords=1 lds-waves-m=2 lds-waves-n=2 k-unroll=1 sched-groups=0' | FileCheck %s --check-prefix=DEFAULTS

module attributes {tessera.arch = "gfx1151"} {
  func.func @graph_gemm(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}

// SCHED: gpu.func @graph_gemm
// SCHED-SAME: tessera.rocm.physical_staging = "register"
// SCHED: rocdl.sched.group.barrier
// SCHED-SAME: vmem_read
// SCHED: rocdl.sched.group.barrier
// SCHED-SAME: mfma_wmma

// NOSCHED: gpu.func @graph_gemm
// NOSCHED-SAME: tessera.rocm.physical_staging = "register"
// NOSCHED-NOT: rocdl.sched.group.barrier
// NOSCHED: gpu.return

// REFUSE: error: ROCM_CANONICAL_LDS_KNOB_UNSUPPORTED
// REFUSE-SAME: requested lds-waves-m=1 lds-pad-dwords=4 lds-double-buffer=true

// DEFAULTS: gpu.func @graph_gemm
// DEFAULTS-SAME: tessera.rocm.physical_staging = "lds"
