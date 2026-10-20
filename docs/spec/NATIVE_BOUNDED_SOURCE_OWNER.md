---
last_updated: 2026-10-19
audit_role: reference
---

# Native retained bounded source reverse owner

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

## Compiler and runtime boundary

The native source projection retains actual paired AD SSA and exports four or
five compiled members. The retained C++ owner executes those images; it does not
construct Graph IR, derivatives or kernel bodies. Source-program string schemas
v3/v4 remain distinct from integer attention schema5 (saved-LSE plus bounded).

This increment adds exact-SM120 native C entrypoints. The ordinary public
PreparedSourceAttentionReverse adapter and offline portable validator still need
bounded integration. Raw native execution is not public or portable closure.

| New C entrypoint | Contract |
| --- | --- |
| tessera_nvidia_source_attention_reverse_bounded_prepare | Four members, maximum capacity registration |
| tessera_nvidia_source_attention_reverse_rows_bounded_prepare | Five members including explicit FP32 correction rows |
| tessera_nvidia_source_attention_reverse_bounded_invoke | Checked actual frame for four-member owner |
| tessera_nvidia_source_attention_reverse_rows_bounded_invoke | Checked actual frame for five-member owner |

Existing static signatures, last-times and close entrypoints remain compatible.
The native ABI registry is regenerated from the implementation. A bounded owner
rejects a static invoke, including when actual dimensions are missing.

## Capacity ownership and frame transaction

Prepare receives capacities in [B,Hq,Hkv,Sq,Sk,D,Dv] order. B/Hq/Hkv/D/Dv remain
fixed, while Sq/Sk must be positive and no larger than their capacities.
The named source storage is homogeneous FP16 or BF16 with FP32 normalization
working values and FP32 LSE/correction rows. Head grouping, checked byte products,
query-row geometry and launch limits are validated.

One arena and pinned staging allocation retain the maximum attention bindings.
Separate maximum-capacity buffers own normalized Q and the source gradient.
Offsets and allocations remain fixed across invocations; active byte counts
and grids change with Sq/Sk. The normalization schedule is taken from its
compiler-generated entry: serial and cooperative_128 use different CTA grids.

A pure SourceReverseFrame value computes dimensions, active spans and grids
without copying CUDA ownership. The matching source invoke clears its completed
receipt, validates the candidate geometry, context generation, argument/seed
permutation, selected gradient spans, placement/overlap and producer dependencies.
Only after these checks does it commit value geometry and enqueue snapshots.
Invalid dimensions or spans do not change active geometry or caller outputs.

Resident roots cannot alias the arena, private normalized Q or private source
gradient. Explicit producer streams are ordered using CUDA events; legacy and
per-thread stream handles are supported. Temporary dependency events may be
created per resident invocation; this is not a claim of allocation-free calls.
No returned output borrows native private storage.

The C interface accepts trusted typed pointers and image contracts. Public
dtype/layout validation and portable Graph/image consistency checks must still
be connected before exposing this bounded program through the public adapter.

## Validation and timing scopes

Owning RTX5070 tests cover repeated shrink/grow calls with both source programs,
storage formats, argument/result permutations, requested gradients, zero inputs,
readonly roots, guarded host outputs, invalid frames, producer ordering and
context recovery. An opt-in test-only provider includes the production owner
source plus an observer; it checks actual allocation addresses, capacities,
offsets and active geometry, and can attempt private-root aliasing. Observer
symbols do not belong to the shipped native provider ABI.

The paired recorder alternates inline/row calls and reverses profile, runtime
shape and initial arm order. It checks independent source-VJP oracles and root/
output guards before and after timing. Compilation, native prepare, each native
stage/program CUDA-event window and completed ctypes-host call are separate.
These event windows include enqueue effects and are not isolated profiler
kernel time. Normalization, saved-forward and source-pullback timed images are
identical between arms; only backward decomposition changes.

At full Sq17/Sk19, width257 row programs measure program-event ratios about
0.170-0.177 and completed-host windows about 0.97-1.04ms versus 4.94-5.41ms inline.
Tiny Sq=Sk=1 row programs are slower in both orders. No universal row-policy
promotion follows; actual-shape policy and public-route benchmarking remain open.

Evidence: benchmarks/baselines/native_bounded_source_owner_20261019/README.md.

## Current proof

48 distinct production-provider cases pass across 36-case and 12-case cohorts.
The final 12-case observer suite includes a different valid geometry rejected
for incompatible spans and verifies that the previous active frame is retained.
Forty static source-owner regressions pass. The ABI gates pass 28 distinct tests
after the three generic artifact lifecycle cases are enabled; the Darwin-only
Apple bridge remains skipped. These results do not substitute for public JIT,
portable replay or full delivery gates.
