# Bounded native attention row owner — SM120

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-BOUNDED-ROW-OWNER-20261017.

Recorded by `benchmarks/nvidia/benchmark_bounded_attention_rows.py` and
`benchmarks/nvidia/benchmark_bounded_attention_rows_ab.py`.

## Implemented and executed envelope

Original Python frontend and textual Graph export retain bounded Sq/Sk through
native MLIR differentiation, Schedule, Tile, NVIDIA Target lowering and PTX.
FP16/BF16 roots and gradients, matching-half or FP32 O/dO, optional FP32
query/key/dense bias and saved-LSE seeds retain their physical storage.
Compiler-owned row correction remains probability-dot(dO,V) minus the row seed;
the consumer reads that residual without subtracting the seed twice.

Portable VJP v4 identifies bounded row programs specifically. Static row v3,
legacy VJP v1/v2 and all JVP schema meanings remain unchanged. Capacity guards
and paired source identity are recomputed against native contracts. Dynamic
bias stride extents remain runtime values; trace-at-one broadcast axes stay 1.
The row producer and consumer retain seven logical scalar dimensions; the
forward may independently need four physical bias extents.

The existing prepared C++ owner reuses images through compiler-free public
replay. Its bounded Python service retains native concrete-frame owners under
a bounded per-thread LRU; changing shape may allocate a new native frame, but
does not compile new images. Resident capture shares existing typed buffer,
stream, private snapshot and result-lifetime machinery. Its row allocation is
released after the consumer on the private stream. Exactly dV-only omits the
producer and its row allocation. Output generations survive later reverse calls.

## Validation

RTX5070 / SM120 device evidence is recorded, not inferred from parsing:

- 16 raw bounded native kernel cases include explicit residual perturbation,
  independent FP64 comparisons and guarded output spans.
- 55 new public/capture cases: 37 replay/capture, four pending producer/snapshot
  checks, eight original frontend exports, two fresh-process compiler-absent
  replays and four 64-thread packed/logical partial-gradient cases. Actual Sq/Sk prefixes include trace-at-one, capacity and ragged rows.
- 172 existing static-row/prepared-owner/typed-capture regressions pass.
- 70 portable/static/prepared host gates pass, including capacity mutation,
  schema downgrade/upgrade refusal, role activity and value-only elision.
- 16/16 static native Target outputs remain byte-identical to frozen PR975.
- Ruff and two three-file mypy selections pass. Broader final publication gates
  are still required. Counts from overlapping selections are not added.

Source, tool/provider hashes, raw logs and separate portable program archives
bind the results. The native provider was rebuilt into its own scratch path.
No compiler/backend validation or numerical tolerance was weakened.

## Separate native stage and public timings

Ten samples per row, three numerical warmups, one bounded program per storage,
interleaved runtime shapes and independent FP64 comparison outside every timed
call. Full public time includes validation, snapshots, native execution and
completed host readback; compilation and oracle work are excluded.
Native CUDA event windows include dispatch/enqueue gaps and are not isolated
instruction latency. The fourth window spans forward start through reverse stop.

| Storage | Sq/Sk | Public ms | Forward / row / backward / full device interval ms |
|---|---|---:|---|
| float16 | 3/5 | 0.380 | 0.047 / 0.015 / 0.020 / 0.086 |
| float16 | 17/129 | 0.841 | 0.228 / 0.158 / 0.159 / 0.546 |
| float16 | 33/129 | 0.840 | 0.203 / 0.158 / 0.167 / 0.529 |
| bfloat16 | 3/5 | 0.368 | 0.043 / 0.012 / 0.017 / 0.081 |
| bfloat16 | 17/129 | 0.817 | 0.207 / 0.156 / 0.157 / 0.521 |
| bfloat16 | 33/129 | 0.837 | 0.217 / 0.156 / 0.165 / 0.539 |

This is characterization, not an A/B speedup or default-selector promotion.
Do not compare these whole native event windows with captured graph kernel
windows in older packets. Synthetic rows do not establish model quality.

Reproduce on Super-Bear with matching compiler, CUDA13.3 and native provider:

```sh
python -m benchmarks.nvidia.benchmark_bounded_attention_rows \
  --output /home/angstorms/scratch/bounded-row-owner-measurement --samples 10
```

## Remaining

Wider masks/layouts, half JVP, composed/higher-order
AD, automatic schedule selection, broader required gates and publication remain
open. Sibling backend plans assess the shared contract but inherit no SM120
physical proof. The original five-slice objective remains open.


## Matched bounded resident inline/prepass A/B

Recorded by `benchmarks/nvidia/benchmark_bounded_attention_rows_ab.py`.
Both arms trace the same original Python attention function, carry the same
bounded Sq/Sk contract and have byte-identical forward images and checkpoint
identity. Stored roots and seeds are identical. Independent FP64 comparisons
and bit-identical arm outputs pass before and after profiling. Twelve host
samples per arm alternate AB/BA order. Each captured stage repeats exact native
functions 32 times in three event windows; a preallocated diagnostic residual
keeps allocation cost out of those dispatch windows.

| Storage | Sq/Sk | Inline / prepass completed resident reverse ms | Host ratio | Captured reverse dispatch ratio |
|---|---|---:|---:|---:|
| float16 | 3/5 | 0.272 / 0.295 | 0.92x | 2.16x |
| float16 | 17/129 | 5.774 / 0.542 | 10.65x | 17.73x |
| float16 | 33/129 | 10.983 / 0.579 | 18.97x | 33.55x |
| bfloat16 | 3/5 | 0.328 / 0.445 | 0.74x | 2.15x |
| bfloat16 | 17/129 | 5.802 / 0.623 | 9.32x | 17.79x |
| bfloat16 | 33/129 | 10.924 / 0.560 | 19.50x | 33.75x |

The host interval includes seed snapshots, fresh gradient allocations, native
reverse execution and completion. It excludes forward capture, external upload,
readback and oracle work; it is not a whole public-call interval. Dispatch
windows exclude copies/allocations and include CUDA graph dispatch overhead.
Producer, consumer and combined prepass samples are separate in the packet.
Tiny completed calls regress despite faster captured dispatch. The default
stays inline; these six profiles do not justify universal selection.

## Broad gate failure and repair in progress

The first broad host run completed with 27,347 passes, 5,905 skips and 517
failures. 516 failed x86 packaging/differential cases report the same explicit
missing-backend build diagnostic; CMakeCache confirms the selected compiler
was configured with `TESSERA_BUILD_X86_BACKEND=OFF`. The remaining naming gate
ran before this packet README was tracked. Its focused rerun passes, and the
README now explicitly names both recorder paths. No test is weakened.

The compiler rebuilt with the x86 backend enabled passes 561 owning-file
cases, with 514 existing hardware skips. Final broad validation remains
required; this is compiler coverage, not exact x86 device proof. The initial log is retained
losslessly. Rebase on merged PR975 preserves all 5,459 compared source/test
files byte-for-byte; the duplicate fixture patch was verified against merged
main before being reconciled.


## Final local publication gates

Final frozen source 13307df372 passes **27,867** host WSL unit/compiler cases,
with 5,902 skips and eleven warnings, zero failures. All 55 new bounded owning
RTX5070 cases pass again with the final matching compiler pair. That pair
emits byte-identical Target IR and PTX for all ten measured A/B members, so
adding x86 registration did not change the measured physical NVIDIA images.
Final recorder/routing/audit gates pass twenty cases, focused typing is clean,
and all 33 generated documents pass. AST graph update completes offline.

The initial 517-failure run remains preserved and is not used as a green gate.
Its x86 configuration/naming repairs pass 561 cases with 514 existing skips.
No checker, tolerance or admission rule was weakened. Final local gates do not
substitute for exact-head public CI and a fresh complete review audit before
merge. The original five-slice objective remains open.
