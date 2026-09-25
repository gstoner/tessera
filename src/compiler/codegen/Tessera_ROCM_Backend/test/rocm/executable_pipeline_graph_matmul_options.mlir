// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=graph output=binary arch=gfx1201 staging=lds lds-pad-dwords=4 lds-waves-m=1 lds-waves-n=4 k-unroll=2 sched-groups=3 lds-copy-width=2 lds-copy-depth=2 lds-double-buffer=true lds-b-row-major=true})' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: %trop --pass-pipeline='builtin.module(tessera-rocm-executable{family=matmul input=tile output=binary arch=gfx1201 staging=lds lds-pad-dwords=4 lds-waves-m=1 lds-waves-n=4 k-unroll=2 sched-groups=3 lds-copy-width=2 lds-copy-depth=2 lds-double-buffer=true lds-b-row-major=true})' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=TILE
//
// Every input level serializes the same explicit schedule request
// (ROCMExecutablePipeline.pass_pipeline()). The graph level used to call the
// matmul generator with only family/staging, so an explicit lds-pad-dwords=4
// (and every other option below) silently became the helper's defaults.
// Codex review on #834. The generator must receive exactly what was asked
// for at both levels; only via-tile differs.

module {}

// GRAPH: generate-wmma-gemm-kernel{
// GRAPH-SAME: canonical-staging=lds
// GRAPH-SAME: k-unroll=2
// GRAPH-SAME: lds-b-row-major=true
// GRAPH-SAME: lds-copy-depth=2
// GRAPH-SAME: lds-copy-width=2
// GRAPH-SAME: lds-double-buffer=true
// GRAPH-SAME: lds-pad-dwords=4
// GRAPH-SAME: lds-waves-m=1 lds-waves-n=4
// GRAPH-SAME: sched-groups=3
// GRAPH-SAME: via-tile=false

// TILE: generate-wmma-gemm-kernel{
// TILE-SAME: canonical-staging=lds
// TILE-SAME: k-unroll=2
// TILE-SAME: lds-pad-dwords=4
// TILE-SAME: lds-waves-m=1 lds-waves-n=4
// TILE-SAME: via-tile=true
