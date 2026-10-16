# SM120 optional flat ReLU Schedule/Tile

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: W11-RELU-FLAT-20261015.

Serial remains the default; flat_128 is explicit. Native Graph -> Schedule ->
Tile -> NVIDIA LLVM/NVVM -> PTX owns arithmetic. Python binds artifacts and
checked metadata. Native prepared providers negotiate mapping support before
owner allocation.

## Validation and measurements

The initial owning RTX5070 gate passes 46 tests: exhaustive FP16/BF16 payloads,
native policy/hash refusal, independent composed numerics, unchanged norm/
consumer PTX, retained outputs, static/bounded two-sided replay and stale
providers. Final affected regressions and source-bound timing remain required.

Exploratory BF16 129x256 captured batch means were about 40.7 us serial and
1.1 us flat; a small single-column regression and noisy composed event/wall
measurements prevent default promotion. Final source-bound packet supersedes
exploratory values. Event and completed wall time are separate.

## Reproduce on owning RTX5070 WSL

Build the benchmark-only driver harness (contains no kernel arithmetic):

~~~sh
nvcc -shared -Xcompiler -fPIC -O2 benchmarks/nvidia/relu_capture_probe.cpp \
  -o /path/to/scratch/libmatched_capture.so -lcuda
python benchmarks/nvidia/benchmark_relu_schedule.py \
  --harness /path/to/scratch/libmatched_capture.so \
  --output /path/to/scratch/matched-results.json
~~~

Use matching production tessera-opt/tessera-nvidia-opt, LLVM/MLIR 23.1.1, and
the rebuilt provider. Serial/flat/flat/serial order retains every sample.
Standalone graph capture amortizes enqueue but includes graph dispatch.
Prepared member/program event windows may include CPU enqueue gaps.
Completed host calls include validation, copies and synchronization.

General producer/layout/AD, model quality and five-slice obligations remain
open. No FP8/MXFP8/MXFP4 or sibling proof is inferred.

## Affected regression checkpoint

The matching final provider/compiler passes 668 affected tests with one
Darwin-only skip. This includes the 46-case activation group, native producer
partition/DAG/bounded routes, resident stream/lifetime and portable seals,
diagnostic/pass/ABI checks, public attention forward, public VJP and prepared
VJP. One existing fork deprecation warning is retained in the log. Focused
four-file mypy passes. Final matched benchmark and generated-doc/registry/
publication gates remain open.

## Final matched packet

[matched.json](matched.json) pins 21 implementation/helper sources, compiler
and provider binaries, harness and complete recorder source. Each arm passes
numerical checks before timing and after captured replay. Both normalization
and consumer PTX are byte-identical between composed serial/flat programs.

Captured standalone means (microseconds, medians over the two ABBA arms):

| Rows x Columns | Storage | Serial | Flat |
| --- | --- | ---: | ---: |
| 7 x 32 | f16 | 1.723 | 0.862 |
| 17 x 64 | f16 | 4.273 | 1.102 |
| 129 x 256 | f16 | 40.815 | 1.190 |
| 256 x 256 | f16 | 46.586 | 1.126 |
| 257 x 257 | f16 | 47.697 | 1.137 |
| 65537 x 1 | f16 | 1.124 | 0.977 |
| 7 x 32 | bf16 | 2.038 | 1.196 |
| 17 x 64 | bf16 | 4.060 | 0.924 |
| 129 x 256 | bf16 | 40.643 | 1.218 |
| 256 x 256 | bf16 | 46.722 | 1.227 |
| 257 x 257 | bf16 | 47.588 | 0.985 |
| 65537 x 1 | bf16 | 1.394 | 1.404 |


These measurements change both lane memory mapping and CTA parallelism;
they do not attribute the entire difference to one microarchitectural cause.
No counters or hardware-independent strategy claim is supplied. Captured
means include graph dispatch and are distinct from prepared event/wall values.

The packet retains all six composed cases and every ABBA sample. Program
event and completed host measurements remain variable; short cases can
regress and the standalone singleton-column difference is small and noisy.
Serial stays the default. FP8/MXFP8/MXFP4 remain required independent
evaluation points before any broader strategy/default decision.

The 668-case affected gate preceded descriptive manifest/execution wording
updates; native arithmetic, provider and checked binding files are unchanged.
The subsequent 243-case delivery gate covers those descriptions and their
regenerated execution dashboard. Later benchmark provenance-only edits and
its capacity guard are exercised by the final matched packet. Historical
packets are not relabeled as current measurements. See [validation.json](validation.json).
PR publication and broader producer/AD/five-slice closure remain open.

## Direct resident proof and final gates

[resident.json](resident.json) separately proves the actual native
direct-resident flat launcher, not the benchmark harness: both storages pass
all 65,536 bit patterns at 257x257, preserve inputs, reject an unrelated
producer stream and retain the previous output after refusal. Its complete
script, source pins and provider hash are included. This adds two probes;
they are not counted as extra pytest cases.

Final focused views pass after the benchmark provenance cleanup. The W1.1
Latest pointer and timestamp now match the owned log; its three dependent
generated views and 11 audit-document tests pass. Both claim-lint gates and
diff hygiene pass. The final AST-only Graphify update completed; its HTML
size limit has no bearing on compiler or numerical validation.


## Subsequent captured-program checkpoint

[captured.json](captured.json) is a later runtime/binding source epoch.
It retains 22 source pins, compiler/provider/harness hashes and all ABBA arms.
The older matched.json, resident.json and validation.json keep their actual
4a22a407b schedule-checkpoint identities; they are not relabeled.

The captured runtime uses the same sealed native images and checked grids.
Graphs exist only inside the locked last-host-frame profile call, are bounded
to 4096 repetitions/65536 member nodes, and are retired before returning.
The default profile remains a native enqueue-event loop. Inference submission
and serial selection are unchanged.

Captured whole-program medians (microseconds):

| M,K,N | Storage | Serial | Flat |
| --- | --- | ---: | ---: |
| 7,32,9 | f16 | 5.564 | 4.546 |
| 17,64,19 | f16 | 12.011 | 8.783 |
| 129,256,33 | f16 | 45.746 | 6.156 |
| 7,32,9 | bf16 | 5.558 | 4.542 |
| 17,64,19 | bf16 | 11.989 | 8.731 |
| 129,256,33 | bf16 | 45.705 | 6.136 |


Captured stage/program event averages exclude graph creation, priming,
retirement and output download; they include graph dispatch. Completed
profile-call wall time includes those costs plus extra calibration execution:
roughly 5–26 ms in this packet, and is not inference or compiler-package cost.
Completed prepared inference remains separately recorded around 0.24–0.32 ms
for the named cases, with variation. No public/default speedup is asserted.

The affected gate passes 681 cases with one Darwin-only skip, including 13
captured static/bounded/two-sided frame, lease/lifetime, limits and provider
compatibility cases. Focused mypy passes. The intentional red gate has eleven
failures before the API exists; the two-sided cases were added afterward.
Final ABI/document/Graphify and publication gates remain required.

Reproduce with the existing recorder and --captured-program:

~~~sh
python benchmarks/nvidia/benchmark_relu_schedule.py \
  --harness /path/to/scratch/libmatched_capture.so --captured-program \
  --output /path/to/scratch/captured.json
~~~

Use the provider exporting tessera_nvidia_matmul_profile_captured. Captured
borrowed-root resident profiling and persistent inference graphs remain open.
FP8/MXFP8/MXFP4 and sibling timing cannot be inferred from this half-storage
measurement extension.

### Captured checkpoint delivery gates

The subsequent registry/runtime-ABI/document gate passes 264 tests (one skip,
65 slow cases deselected). All 33 generated documents are synchronized;
compiler-plan ownership, both claim-lint gates, Ruff, focused mypy and the
AST-only Graphify update pass. No mathematical guard or assertion was removed.

[host-attribution.json](host-attribution.json) is a separate instrumented
200-call investigation after warmup, with subprocess creation forbidden.
All calls complete numerically and native scratch remains fixed. Most
accounted time is inside the prepared call boundary, combining marshalling,
native submission, transfers and completion; cProfile cannot separate those
native phases. It does not show recurring Python compiler/IR validation.
This motivates native transfer/submission attribution or resident execution
work before assuming that deleting Python validation will solve host latency.
Instrumented totals are not latency or throughput comparisons.

Original five-slice and wider producer/AD obligations remain open.
