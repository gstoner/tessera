// RUN: tessera-opt --tessera-async-copy-lowering --tessera-nvtma-descriptor --tessera-tile-dataflow-legality %s | FileCheck %s
//
// TILE-SYNC-TYPED-2026-08-15 regression (PR #566 review): the LEGACY lane — a
// tokenless tile.async_copy plus a keyless tile.wait_async — must survive the
// staged NVTMA retrofit under the fail-closed sync verifiers.
//
//   * Between the two passes the copy carries AsyncCopyLowering's explicit
//     `tile.pending_mbarrier` marker (barrier assignment deferred), so it is
//     not a copy that "gates on nothing"; NVTMADescriptorPass strips the
//     marker when it binds the real SSA barrier.
//   * The keyless wait carries `tile.retire_all`; with NO completion tokens
//     minted anywhere, NVTMADescriptorPass must PRESERVE the marker (its
//     "retire everything" meaning was never realized as a token set) — a
//     barrier-only wait would be rejected as TILE_WAIT_GATES_ON_NOTHING.

func.func @gemm_kernel(%src: tensor<64x64xbf16>) {
  "schedule.mesh.region"() ({
    %producer = tile.role {name = "legacy.producer", kind = "producer", members = ["async_copy"]} : !tile.role
    %consumer = tile.role {name = "legacy.consumer", kind = "consumer", members = ["mma"]} : !tile.role
    %born = tile.mbarrier.init with %producer, %consumer {slots = 1 : i64, phase_bits = 2 : i64, tile.pipeline = "legacy.0"} : !tile.mbarrier
    "tile.async_copy"(%src) {tile_rows = 64 : i64, tile_cols = 64 : i64} : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    "tile.wait_async"() : () -> ()
    "schedule.yield"() : () -> ()
  }) {mesh = @mesh0, axis = "tp"} : () -> ()
  return
}

// CHECK: %[[MBAR:.*]] = tile.mbarrier.init
// The rebuilt copy binds the SSA barrier; the pending marker is resolved.
// CHECK: tile.tma.copy_async %{{.*}}, %[[MBAR]]
// CHECK-NOT: tile.pending_mbarrier
// The keyless wait keeps its declared retire-all semantics.
// CHECK: tile.mbarrier.wait %[[MBAR]]
// CHECK-SAME: tile.retire_all
