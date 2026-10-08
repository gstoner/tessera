# Native typed independent-RHS and shared-LHS scaled batches

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Sync ROCM-INDEPENDENT-SCALED-BATCH-2026-10-07.

## Production route

Original typed Graph -> verified native AD where requested -> Schedule ->
Tile -> ROCm Target/LLVM -> HSACO -> checked native program -> HIP execution.
Schedule uses per-batch M; GPU grid z selects memref views for A/SA where
batched, B/SB and output. Shared LHS A/SA remain shared. No Python operand
replication, kernel arithmetic or per-batch launch loop is used.

## Owning-device proof

RX 9070 XT/gfx1201, GPU-28d9e7efbf2ef716, matching LLVM/MLIR 23.1.1 compiler.
67 primal/JVP tests pass. Twenty-four new primal cases cover both policies,
FP32/E8M0 scales, KN/NK and B3/M7/N19/K256, B2/M100/N129/K1536 and
B2/M128/N4096/K256. Four new FP32 scale-JVP cases compare independent float64
analytic and central-difference oracles and changed seeds. Changed scales in
only the second RHS batch catch batch aliasing; warm compiler subprocesses
are forbidden. Raw tests and actual owning source/compiler/runtime hashes
are recorded here. Aggregate assessment prose and a verifier comment can
differ from the recorded owning source bytes.

## Engineering loop and costs

The initial wider test had two FP32 NK failures: its LDS package projected
static image dimensions to runtime scalars, which the new batch gate rejected.
The repair retains declared policy, positive batch capacity, static logical
program validation and checked geometry while permitting projected image
dimensions. Historical failure logs remain unmodified.

The current 24-row packet checks independent float64 block numerics before
timing. Each row records 21 public-call and prepared update/invoke/read windows
and 21 native HIP sequence-event windows. Events include native enqueue gaps
and are not isolated kernel cost. No AITER/Radiance comparison or speedup
promotion is claimed. Initial packets are retained separately.

## Limits

Dynamic/nested batches, transposed lhs, encoded-scale AD, composed AD and
universal batching/transpose closure remain open. Sibling physical parity
needs independent owning-host proof. This does not close wider W8A8,
MXFP4 M256 attribution, general paged KV, other image keys or larger
programs. Full-unit and aggregate PR delivery remain open.

## Shared regression gates

649 native fixtures pass with 66 unsupported feature cases; 395 shared semantic,
package, diagnostic, pass metadata, operator/dtype audit and lifecycle checks pass.
Twenty-two package checks include consistent-witness corruption attempts for
batch planes, RHS/scale prefixes and output capacities. The initial compiler
refusal used a pre-rebuild binary; it is superseded by the matching build.

113 target capability/manifest/runtime reconciliation checks pass.
