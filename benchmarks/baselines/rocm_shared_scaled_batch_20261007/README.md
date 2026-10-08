# gfx1201 native typed shared-RHS scaled batches

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Synchronization: ROCM-SHARED-SCALED-BATCH-2026-10-07.
Owning device: Tajasaurus RX 9070 XT, gfx1201, GPU-28d9e7efbf2ef716.

## Compiler and runtime contract

Original typed Graph holds A[B,M,K], shared rank-two B (KN or NK),
A scales [B,M,G], shared B scales [G,C], and output [B,M,N].
The explicit policy is shared_rhs_rows with exact_per_block fp32 accumulation.
FP32 and encoded E8M0 [1,32] scale formats retain their independent policies.
TransposeA is outside this envelope; transposeB changes physical B orientation
while scales keep logical group/column axes. Static whole K scale groups and
compact physical storage are required.

Native Graph verification checks logical ranks, scale prefixes, result
dimensions and batch*row overflow. Schedule flattens B*M launch rows. Tile
binds the original tensors through bufferization/raw pointers without
replication, unpacking or a Python batch launch loop. Native Target codegen
recovers flat M/N/K while the SSA program retains logical ranks, ownership,
byte capacities and read/write lifetimes. Its manifest carries batching;
package validation rejects incompatible output/scale prefixes even when
a corrupted witness is consistently reencoded.

Public f32 scale JVP preserves these attributes through native differentiation.
The native owner executes actual product/sum SSA members with one completion
owner; shared RHS and scale storage are not replicated. Encoded bytes have
no implicit STE. This is not a generic batching/linear-transpose closure claim.

## Validation

- device-tests.log: 39 owning public primal/JVP cases pass. Eight new primal
  cases cover two batch envelopes, FP8/MXFP8 and KN/NK; six new scale-JVP cases
  cover left/right/both seeds, independent float64 block numerics and central
  finite differences. Warm changed inputs/seeds forbid compiler subprocesses.
- shared-tests.log: 541 host checks pass, including native program corruption,
  logical shape/scale admission, dtype/operator and metadata registries.
- target-registry-tests.log: 222 pass, one foreign/configuration case skips.
- native-core.json: 494 native core fixtures pass, 66 unsupported feature
  fixtures remain unsupported. Positive and four-case negative new batch
  fixtures are separately retained.
- source-tools.json binds actual owning source/test/recorder/compiler/runtime
  bytes. Capability prose and backend manifest/spec updates on the aggregate
  are assessments, not additional physical proof.

initial-device-package-refusal.log retains the first owning run's eight
package validation refusals: the old owner validator assumed rank-two storage.
The subsequent compiler program-policy export and exact prefix validation
supersede them. native-fixture.log retains the initial incorrect check for a
carrier M attribute; native-fixture-current.log checks the actual sealed
Schedule shape key and completes lowering. These logs are history, not gates.

## Timing scope

timings.json records 12 correctness-gated rows, with 21 public wall-clock,
21 prepared update/invoke/readback wall-clock, and 21 native HIP sequence-event
windows each. Native events repeat the complete C++ sequence 30 times and
report per-sequence cost; they include native enqueue gaps and are not isolated
kernel timings. Warm compiler subprocesses are forbidden. HSACO hashes and
live rocminfo remain in the packet.

Public primal medians range 0.93-2.00 ms; public scale-JVP medians range
1.77-2.55 ms. Prepared host costs range 0.35-1.24 ms, and native sequence-event
costs range 0.010-0.525 ms across different programs. These are separate metrics
and shape/format observations, not a speedup comparison or AITER/Radiance claim.

## Remaining work

Independent/shared-LHS typed FP8 batches, dynamic/nested maps, general
linear-transpose/composed AD, active encoded-scale differentiation policy,
broader cache/layout/performance families, sibling physical parity, fresh
green full-unit closure and aggregate PR delivery remain open. The two
generic batching/transpose closure assertions and partial/planned coverage
states are unchanged.
