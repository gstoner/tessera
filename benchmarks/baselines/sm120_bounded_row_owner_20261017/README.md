# Bounded native attention row owner — SM120

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-BOUNDED-ROW-OWNER-20261017.

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
- 51 new public/capture cases: 37 replay/capture, four pending producer/snapshot
  checks, eight original frontend exports and two fresh-process compiler-absent
  replays. Actual Sq/Sk prefixes include trace-at-one, capacity and ragged rows.
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

Matched inline/prepass A/B, wider masks/layouts, half JVP, composed/higher-order
AD, automatic schedule selection, broader required gates and publication remain
open. Sibling backend plans assess the shared contract but inherit no SM120
physical proof. The original five-slice objective remains open.
