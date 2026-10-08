# Public ROCm movement frontend and native JIT

Owner E2E-REAL-6; sibling FRONTEND-IR-MEDIUM-1.
Synchronization key ROCM-PUBLIC-MOVEMENT-2026-10-05.

## Integration and numerical contract

The existing public kv_cache_read now accepts f32 physical pages[P,PS,H,D]
and an explicit i32 logical page table[LP], returning one [end-start,H,D]
tensor. Cache-handle reads keep their established two-result (K,V) interface.
Repeated/reordered physical pages and default end=start+1 are covered.
The public moe_dispatch now gathers explicit i32 token-of-slot[S] rows from
x[T,H], returning [S,H]; its previous identity reference and same-as-input
shape were incorrect for unequal token/slot counts. DispatchPlan uses the
existing stdlib reference. Non-null transport remains outside this local
tensor form and cannot be silently dropped by native packaging.

Tracing records static positional cache bounds, preserves NumPy int32 index
storage, and uses catalog shapes without executing eager movement arithmetic.
The native C++ Schedule contract accepts either function-argument order
while requiring both distinct entry arguments as semantic operands.
Native Schedule/Tile still seals the exact typed Graph and semantic binding
roles. No Python GPU body, new operation, dtype, target, pass or diagnostic
is introduced. This does not admit general page layouts or distributed MoE.

The ordinary JIT binds inputs by traced SSA names, projects scalar dimensions
from the exact tensor contract and checks descriptor ABI, scalar storage and
shape agreement. Output storage follows checked static descriptor guards.
Checkpoint storage keeps its tuple result contract. Runtime artifacts are
reused per compiled specialization; per-launch index/buffer/ABI guards remain.
Changing actual input shapes creates distinct typed Graph specializations.

## Exact-device evidence and timing scope

AMD Radeon 8060S/gfx1151: public paged, default-end/reordered paged and
reordered-argument MoE, each with two shapes and repeated specialization.
RX 9070 XT/gfx1201: public paged and default-end/reordered paged.
All ordinary JIT outputs are bit-exact, including NaN payloads, infinities
and signed zero. Tests forbid the eager implementation and verify adjacent
Graph/Schedule/Tile/Target/backend digests and native Schedule producer.

Owning-device JIT/spine tests: gfx1151 23 pass, 3 other-hardware skips;
gfx1201 21 pass, 5 other-hardware skips. The dedicated gfx1151 source tree
lacks the root pytest mark registry, producing two recorded mark warnings.
Shared frontend/cache/native Schedule tests: 114 pass, 19 skips.
Operation/dtype/trace/MoE/diagnostic/pass/JIT registry tests: 503 pass, 16 skips.
Final registry/manifest/conformance gates: 637 pass, 9 skips; audit lifecycle
11 pass. Compiler-plan ownership/log links and Ruff pass.
Six gfx1201 NVFP4 checkpoint-JIT regressions pass with the shared binding change.
Explicit transport=None is normalized to the local MoE default on a copied
Graph; non-null transport remains rejected. Its ordinary gfx1151 JIT is proved.
The initial compiler rejected reversed arguments; matching native compiler
rebuilds resolved that restriction. gfx1201's first rebuild lacked the
configured LLVM dependency library path; the matched retry succeeds.

Ten balanced warm trials per row compare ordinary JIT, prebound checked
descriptor launch and retained production helper. These are full host calls,
including transfers/completion/download, with compilation excluded.
First-call wall time, every sample, profiles, compiler/source fingerprints,
native images and all adjacent IR snapshots are retained in per-device packets.
No device-kernel timing or application-wide speedup is claimed here.

| Architecture | Case | JIT ms | Descriptor ms | Retained ms |
| --- | --- | ---: | ---: | ---: |
| gfx1151 | paged small | 0.6969 | 0.5342 | 0.6359 |
| gfx1151 | paged large | 0.7205 | 0.5465 | 0.6395 |
| gfx1151 | paged_default small | 0.7035 | 0.5344 | 0.6282 |
| gfx1151 | paged_default large | 0.7104 | 0.5449 | 0.6409 |
| gfx1151 | dispatched small | 0.7045 | 0.5289 | 1.7655 |
| gfx1151 | dispatched large | 1.4204 | 1.2489 | 4.0989 |
| gfx1201 | paged small | 0.7090 | 0.5471 | 0.6418 |
| gfx1201 | paged large | 0.7889 | 0.6199 | 0.7355 |
| gfx1201 | paged_default small | 0.7426 | 0.5655 | 0.6765 |
| gfx1201 | paged_default large | 0.7857 | 0.6127 | 0.7401 |

The initial packets preserve uncached runtime-artifact reconstruction.
The profile attributed repeated artifact creation/hash/serialization to
Python wrapper overhead. Reuse removes that work; final JIT wall time is
lower, but the initial/final runs are sequential characterization rather
than an interleaved performance A/B.

MoE JIT is 2.51–2.89x faster than the retained helper for these two envelopes.
gfx1151 paged JIT is about 10–13% slower than retained calls; three of four
rows miss the existing 10% non-regression gate. gfx1201 paged JIT is about
6–10.5% slower; one of four rows narrowly misses that gate. Sequential timing
packets vary, so these measurements do not justify default-route promotion.
The prebound descriptor route remains faster than retained calls.
Do not transfer its performance certificate to the entire public JIT wrapper.

## Remaining architectural work

The next measured targets are native/prevalidated call binding and compiler
sessions, eliminating per-call textual Graph serialization while preserving
the sealed Graph, ownership and checked ABI. General layouts, asynchronous/
resident movement, controlled kernel-only timing and retained transport
retirement remain open. FP8/MXFP8/MXFP4 numerical and performance gates remain
independent requirements. Full five-slice closure is not established.

The shared frontend reference/shape change is assessed for all backends.
Apple/x86 physical packaging remains follow-up required; neither gains ROCm
execution evidence. NVIDIA's shared native paged Schedule accepts reversed
entry roles and passes host compiler tests, but this turn does not claim an
ordinary SM120 movement JIT route or exact-device performance.
Graphify is unavailable in the authoritative WSL scratch checkout.
