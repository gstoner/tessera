---
last_updated: 2026-10-19
audit_role: reference
---

# Native bounded original-source attention reverse projection

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

## Source and paired witnesses

An explicit SM120 source reverse request with positive i64 sequence capacities
accepts the existing homogeneous FP16/BF16 unweighted RMSNorm(Q) feeding saved-LSE
attention. The traced semantic source remains static. Paired differentiation
constructs its actual expressions before projection; no Python derivative or
backend Graph constructor is introduced.

The original source and static paired Graph text/hash remain immutable witnesses.
A separate capacity-scoped program module clones the actual members and root
calls. This avoids putting module bounds around the retained static checkpoint
operations, which would contradict their shape contracts.

Only sequence axis 2 becomes dynamic in each known rank-three/rank-four member
tensor. Batch, head counts and feature dimensions remain fixed. Root argument,
cotangent, requested gradient and intermediate roles determine Sq versus Sk;
equal trace lengths and tracing at one do not determine sequence identity.
Existing native row-body hashing reseals the projected source pullback as
schema8. Forward normalization uses its native query-sequence-bound marker.
Checkpoint, row-correction and compact backward consumers retain native seven
dimension attention capacities.

## Additive source-program schemas

| String schema | Meaning |
| --- | --- |
| tessera.native.source_attention_reverse.v1 | Existing static four-member program |
| tessera.native.source_attention_reverse.v2 | Existing static five-member row program |
| tessera.native.source_attention_reverse.v3 | Bounded four-member program |
| tessera.native.source_attention_reverse.v4 | Bounded five-member row program |

These string source-program schemas do not change the integer attention ABI
schemas. In particular, attention schema5 still means saved-LSE plus bounded.

Bounded plans declare shape_policy=bounded_sequences_v1, two sequence_bounds and
seven shape_bounds. Each buffer retains its logical shape, static trace_shape,
maximum capacity_shape, capacity byte span, ownership and first-write/last-read
indices. The logical dynamic value is the canonical MLIR23 sentinel used by the
existing attention shape contract, not a positive capacity relabelled as shape.
Both compiler members and the typed call program are verified before export.

The existing auto row policy uses capacity geometry for the bounded program;
the static policy and static v1/v2 serialization remain unchanged.

## Validation boundary

Native projection and transport tests cover both storage types, inline and row
programs, root/result permutations, requested gradient ordering, trace-at-one,
invalid bounds and sibling target rejection. Transport replays the native source
and forbids Python Graph construction. Static full-outline identity is checked
against the immutable normalization prerequisite compiler.

Raw projected-member device sequencing is diagnostic proof. Retained composed
CUDA capacity ownership, atomic runtime extent updates, public JIT and
compiler-free portable restore remain separate required integration gates.
The plan execution_state remains artifact_only until that route is integrated;
raw device execution does not promote a package or certify other architectures.

## Current evidence

The projection/shared cohort passes 359 tests, including 44 bounded source cases.
A separate 171-case regression cohort passes. Eighteen static source profiles
retain identical full outlines against the immutable normalization compiler.
Twenty-four owning RTX5070 cases execute projected members through five
shrink/grow pairs with fixed allocation addresses and independently checked
primal/VJP results. This is diagnostic member sequencing, not a retained public
program or portable restore. Evidence and negative runs are preserved in
benchmarks/baselines/native_bounded_source_projection_20261019/.
