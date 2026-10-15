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
