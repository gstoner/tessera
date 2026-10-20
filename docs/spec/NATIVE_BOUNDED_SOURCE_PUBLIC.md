---
last_updated: 2026-10-19
audit_role: reference
---

# Bounded original-source public attention reverse

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

## Public contract

For homogeneous FP16/BF16 RMSNorm(Q) plus zero-dropout saved-LSE attention,
native_backward accepts sequence_bounds=(SqCap,SkCap). Native MLIR retains
the original static source and paired AD witness, projects the bounded program
and owns every Schedule/Tile/Target/native member. Python binds inputs and
transports contracts; it constructs neither derivatives nor backend bodies.

The initial tracer/AST certificate is structural, with zero concrete executions.
It describes the original static trace; numerical authority is the physical
package. Each execution certificate separately binds actual input/seed/gradient
signatures, capacity policy, compiled images and exact SM120 attestation.
The original certificate is not a numerical proof for every future shape.

One retained owner is keyed by source semantics, capacities, gradient selection,
fixed dimensions/storage and a real thread lifetime. Sq/Sk changes reuse images
and maximum allocations. Exceeding those capacities rejects the frame rather
than compiling another owner. Cold tracing uses abstract metadata; warm replay
requires no tracing, eager source execution or external tools. Native CUDA events
order resident producer streams; returned host gradients own independent storage.

## Portable contract

Outer schemas v3/v4 identify bounded four/five-member programs. Static v1/v2 and
integer attention schema5 (saved-LSE plus bounded) retain their meanings.
Offline validation checks source/paired static signatures, explicit bounds,
logical/trace/capacity shapes and byte counts, roles/lifetimes, module/member
capacity attributes, norm/source-row contracts, checkpoint agreement, entries,
image hashes and step ABIs. Rehashed metadata cannot bypass these checks.
Actual-frame launch geometry is separate from capacity grids.

PreparedSourceAttentionReverse.from_manifest requires no compiler/provider for
validation. Explicitly retaining that object reuses one capacity owner; its
runtime_artifact registers the live owner for runtime.launch. A bare serialized
artifact with no live owner may prepare and close per call; persistence is not
claimed for that mode.

## Evidence and limits

Evidence: benchmarks/baselines/native_bounded_source_public_20261019/README.md.
The packet covers FP16/BF16, inline/row correction, serial/cooperative norm,
argument/result permutations, selected gradients, trace at Sq=Sk=1, shrink/grow,
independent oracles, producer streams, compiler-free fresh-process replay and
invalid/tampered contracts.

Public timing separates first-frame compilation/prepare-inclusive host windows,
warm completed public calls and last-call native events. Events may include
enqueue gaps and are not isolated profiler kernel time. Capacity-selected and
static strategies may differ on tiny frames; no universal promotion follows.
Physical admission is SM120 only. General producers, mixed/quantized attention,
pitched layouts, masks/dropout, nested AD, representative model quality and ROCm
route/performance closure remain full-goal requirements.
