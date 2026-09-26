// RUN: not %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=graph output=binary arch=gfx1201 staging=lds lds-pad-dwords=4 lds-waves-m=1 lds-waves-n=4 k-unroll=2 sched-groups=3 lds-copy-width=2 lds-copy-depth=2 lds-double-buffer=true lds-b-row-major=true})' %s 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201 staging=lds lds-pad-dwords=4 lds-waves-m=1 lds-waves-n=4 k-unroll=2 sched-groups=3 lds-copy-width=2 lds-copy-depth=2 lds-double-buffer=true lds-b-row-major=true})' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=TILE
//
// The Tile level serializes the full explicit schedule request
// (ROCMExecutablePipeline.pass_pipeline()) into the generator; an explicit
// lds-pad-dwords=4 (and every option below) must reach it, not a default
// (Codex review on #834).
//
// The Graph level is REFUSED for matmul since 2026-09-26: the Graph->Tile
// shortcut that skipped Schedule IR (Lane B) was retired, and matmul enters at
// Tile level from the scheduled route (Graph -> Schedule -> Tile). This is the
// negative case: the refusal must fire, not a silent pipeline.

module {}

// GRAPH: ROCm matmul has no Graph-level pipeline entry
// GRAPH-SAME: scheduled route (Graph -> Schedule -> Tile)
// GRAPH-NOT: generate-wmma-gemm-kernel

// TILE: generate-wmma-gemm-kernel{
// TILE-SAME: canonical-staging=lds
// TILE-SAME: k-unroll=2
// TILE-SAME: lds-pad-dwords=4
// TILE-SAME: lds-waves-m=1 lds-waves-n=4
// TILE-SAME: via-tile=true
