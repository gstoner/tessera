# SM120 resident source-active JVP — 2026-10-16

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1. Synchronization key:
W11-RESIDENT-SOURCE-20261016. Scope: the named static FP16/BF16
RMSNorm(explicit FP32 accumulator) -> ReLU -> plain FP32 matmul source-active JVP.

## Source, device and compiler proof

The frozen implementation is d1ba6e98df65483f921ec75ee34b08801cbfca2a.
Super-Bear WSL reports NVIDIA GeForce RTX 5070, compute capability 12.0,
UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, CUDA 13.3.
LLVM/MLIR 23.1.1 assertions builds and the native provider are recorded by raw
SHA256 in timings.json.gz and validation/corrected-validation.json.gz.
All validation phase source/tool/helper witnesses match before and after.
The full unit gate passes 26,836 cases, skips 7,678 and records 11 warnings.
The separate required compiler-route lane passes 880 cases with zero skips,
including all 121 native normalization JVP/VJP numerical precision cases.
The owning-device lane passes 348 cases with zero skips.

The ordinary typed frontend retains resident CUDA objects. Native event waits
order producer streams; context, allocation capacity and compact shape/storage
checks precede launch. All roots are host or all are resident. Native ownership
retains private edges; results are two independent completed host FP32 arrays.
Initial resident uploads occur outside timed invocations. Roots and producer
streams must remain alive until the synchronous call completes.

The eight captured programs retain actual Graph, Schedule, Tile, Target,
LLVM/NVVM and PTX. Consumer Tile IR contains tile.view, typed fragment_pack,
fragment accumulators, tile.mma and stores; PTX uses the FP16/BF16 tensor MMA
instruction. Producer Tile IR contains scalar views and a verified native row
body, including differentiated arithmetic from Graph SSA. Compiler outlines
remain marked artifact_only; the separate execution receipts establish device
execution for this envelope. Standalone replay disables compiler subprocesses,
preserves the timed images and compares two outputs to an independent FP64
oracle with unchanged tolerances.

Twelve actual CUDA-driver interposition cases verify native entries, grids
and blocks against the sealed package metadata: row producer 128 threads,
both matmul consumers 32 threads. This covers both storage types, both argument
orders and minimal/ragged shapes. Resealed consumer-128 metadata is rejected.
The native 112-byte step ABI and portable schema numbering are unchanged.
Geometry observations reference 16 unique captured packages across the measured
and driver-traced envelopes. This does not establish other backends' geometry.

## Measurements

Recorded by benchmarks/nvidia/benchmark_active_source_jvp.py with
--resident-inputs --include-projection-shapes. Seven alternating-mode windows,
five completed calls per window, same native owner and images. Values below are
medians of public completed-call wall time, including host outputs. Device CUDA
event regions and per-stage values are stored separately; they are not inferred
from wall time and do not include root transfer/wait/readback costs.

| Storage | M x N x K | Host roots ms | Resident roots ms | Resident / host |
|---|---|---|---|---|
| f16 | 17 x 19 x 35 | 0.8091 | 1.2819 | 1.5845 |
| bf16 | 17 x 19 x 35 | 0.7593 | 1.2260 | 1.6146 |
| f16 | 129 x 33 x 257 | 0.8885 | 1.3288 | 1.4955 |
| bf16 | 129 x 33 x 257 | 0.8762 | 1.2739 | 1.4539 |
| f16 | 256 x 1024 x 1536 | 3.3889 | 2.7822 | 0.8210 |
| bf16 | 256 x 1024 x 1536 | 3.3765 | 2.7048 | 0.8011 |
| f16 | 256 x 1024 x 2048 | 4.4834 | 3.0350 | 0.6769 |
| bf16 | 256 x 1024 x 2048 | 4.1339 | 3.0252 | 0.7318 |

The larger synthetic M256 projection cases reduce public latency 18–32%;
the two smaller shapes regress 45–61%. These results do not promote a default
route or attribute the difference solely to PCIe/cache behavior. The row
producer dominates the larger device regions (~0.905–1.130 ms versus
~0.120–0.162 ms per consumer). Tiny device regions are about 0.0068 ms while
public calls remain near 1 ms, leaving host overhead open. Device event regions
can include enqueue gaps. Synthetic projections are not model-quality evidence.

## Packet contents and reproduction

- timings.json.gz and timings.log.gz: corrected frozen benchmark epoch.
- source-snapshot-d1ba6e98d.json.gz: immutable source bytes; consumers compare
  the packet's own recorded hashes, never require current source to equal history.
- programs/index.json and programs/*.json.gz: eight timed programs and oracle
  receipts, captured by capture_programs.py after timing.
- geometry-observations.json.gz and geometry-programs/: actual driver records,
  saved-output oracle comparisons and additional package records.
- validation/: frozen orchestrator, collection helper, raw pytest logs/JUnit and
  final source/tool witnesses. scripts/ci_require_executed.py checks both JUnit
  lanes. The unit skips are disclosed above and are not exact-device evidence.
- compression-manifest.json: raw and compressed SHA256 and size for every gzip
  record. gzip timestamps are fixed at zero; no compiler binaries are archived.
- history/: the red frontend/resident-owner/invalid-stream/radial regressions,
  missing LLVM tools and the mypy correction. The invalid low stream token 3
  previously reached cuStreamGetCtx and crashed; it now rejects before the query.
  history/pre_geometry retains the earlier c8a1f89a2 certificate and timings,
  whose consumer block metadata was subsequently corrected. These are historical,
  not canonical current proofs. Images were conserved across the geometry fix.

Reproduce on the owning host with a matching compiler/provider build, the
NVIDIA environment script, and PYTHONPATH=python:.:
python benchmarks/nvidia/benchmark_active_source_jvp.py --resident-inputs
--include-projection-shapes --output /path/to/timings.json.
Then run python capture_programs.py --timings /path/to/timings.json
--output /path/to/programs. Compiler paths and provider are explicit in the
validation witness; reproduce from the recorded source rather than mixing epochs.
capture_programs.py is an evidence recorder, not a timed executable route.
The archived collector consumes completed driver-test records, not fresh timings.

## Remaining scope

Generic producer reconstruction and accumulator lineage, dynamic/pitched/mixed
roots, resident results, reverse composition and broader frontend/AD integration
remain open. The shared stable wider-accumulator RMS JVP/VJP formula passes native
LLVM numerical tests, but does not establish Apple/ROCm/x86 physical parity.
No FP8/MXFP8/MXFP4 schedule, selector or promotion changes occur here. The larger
five-slice program remains open. Subsequent evidence/documentation commits do
not change this frozen implementation certificate.

## PR954 optional-strides review follow-through

PR954 review follow-through at e164fd4d8 treats omitted CUDA strides as
compact, matching the optional-field contract already admitted by frontend
tracing. Twelve host regressions first reproduce KeyError; all 97 focused host
controls then pass. The expanded owning RTX5070 lane passes 396 cases without
skips, including 48 omitted-strides public/replay/profile combinations across
both storage types, argument orders and minimal/ragged shapes. Its clean
source-before/after witness and raw logs/JUnit are archived separately.
The d1ba6e98d full-unit/benchmark certificate retains its original scope; no
new timing or sibling physical parity is inferred from this metadata repair.
The five validation/optional-strides records preserve the red host regression,
97-pass host gate and 396-case device JUnit/log/source witness. The source snapshot
in the review receipt is the metadata helper at e164fd4d8; native src tree identity
remains b1dac085ee213deed38ad5eaf82707bdca0006b4. It does not replace the earlier
benchmark epoch or assert that the later repair was timed.
