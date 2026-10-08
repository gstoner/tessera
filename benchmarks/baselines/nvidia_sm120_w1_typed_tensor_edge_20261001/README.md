# SM120 typed tensor producer edge

**Owner:** W1.1 / NVIDIA fragment-producer closure  
**Target:** NVIDIA sm_120a, Super-Bear RTX 5070  
**Source:** `58b848ccbc7682db03d3b1e350a5421ded56984d` (dirty worktree; benchmark assertion and audit changes present)  
**Route:** Graph -> Schedule -> Tile -> NVIDIA Target IR -> PTX -> checked native ABI

The targeted `--require-typed-sm120-producer` benchmark asserts consumer Tile
IR contains `tile.view`, typed `tile.fragment_pack` A/B,
`tile.fragment_zero`, `tile.mma`, `tile.fragment_unpack`, and `tile.store`;
Target IR contains `nvvm.mma.sync`; PTX contains
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. It also checks in the
actual K-loop header that `iter_args` returns a fragment, that `tile.mma`
accepts and returns fragments, and that `scf.yield` returns the fragment.
The K=64 case proves four iterations. Its Schedule/Tile digests are
`77daa46194f48f366b70443dcea8593d3c3fbb79b0e7a1c90db8ac3cf5b39b39` and
`dc759ebb2ccdb11c4d4a6eb4879b03f85bb02dfbbc72cabbb2f7d6f1707ccbed`.

## Exact-device results

Both cases use fp16 RMSNorm output as the fp16 matmul LHS and produce fp32
output. Correctness is checked against NumPy before device timing.

| MxK x KxN | K iterations | RMSNorm max abs error | Matmul max abs error | Resident producer / consumer median |
|---|---:|---:|---:|---:|
| 16x16 x 16x8 | 1 | 0.0 | 2.38e-07 | 10.41 / 10.47 us |
| 16x64 x 64x8 | 4 | 0.0 | 1.43e-06 | 9.48 / 10.52 us |

Both packages report `execution_kind=native_gpu`. Each resident chain reuses
the same intermediate allocation on the same stream, uploads inputs once, and
keeps source, RHS, intermediate, and output allocations disjoint. The K=64
case exercises accumulator reuse across four MMA iterations.

A separate exact-device unit test traces both operations from the public
`from_text` frontend at K=64 with `output_dtype="fp32"`, then checks the same
loop header, fragment MMA/yield, Target IR, and PTX evidence before launch.
Three-batch timings are diagnostic only: producer/consumer CV is
0.054/0.035 at K=16 and
0.062/0.051 at K=64. They do not
support a performance or selector claim. The original seven-batch K=16 packet
is retained as `typed_edge.json`; it recorded high consumer variation.

## Scope

This proves two static shapes on the compiler-owned Schedule -> Tile route. It
does not migrate the generic `LowerMatmulToTileMMA` or
`LowerKReductionAddToTileMMA` constructors in
`src/transforms/lib/TileIRLoweringPass.cpp`. Those legacy tensor-valued
producers still need migration with bufferization and lifetime conversion.
This result does not establish dynamic-shape, alternate-storage/layout, or
cross-architecture support.

Raw packets: [single K tile](single_k_typed_edge.json),
[four K tiles](multi_k_typed_edge.json), and
[original seven-batch run](typed_edge.json).


## Second producer: scheduled softmax -> matmul

The package contract now accepts a shape-preserving last-axis fp16/bf16
softmax producer alongside RMSNorm. The public from_text route on Super-Bear
compiles separate Graph packages through Schedule and Tile, then executes them
on one CUDA stream with a resident edge allocation. The consumer Tile IR uses
tile.view and typed fragments and emits SM120 mma.sync. The CUDA launcher
has an explicit resident softmax ABI branch and launches directly against the
device pointers; it does not stage this producer through host arrays.

Exact RTX 5070 proof: 16x64 softmax -> 64x8 matmul, with fp16 and bf16
storage and fp32 output. The fp16 run records producer max absolute error 0.0
against a stable fp32 softmax reference and consumer max absolute error
7.45e-9 against an independent matmul reference. Both dtype cases report
native_gpu; intermediate, inputs, and output allocations are disjoint, and
both launches use the same stream. The timing packet covers fp16 only.

Fifteen CUDA-event samples of 100 resident launches each gave producer median
11.51 us (CV 4.95%) and consumer median 9.88 us (CV 5.34%). Resident
execute-through-sync was 777.48 ms and includes upload, launches, and
synchronization. No comparison or promotion is claimed. The packet records
image and Schedule digests: [softmax -> matmul](softmax_matmul.json).

A bounded dynamic-K extension reuses the same producer and consumer images
with K bound 16. The exact-device unit test checks active K=7, 11, and 16 for
fp16 and bf16 against independent softmax and matmul references. The diagnostic
benchmark packet covers fp16 active K=7 only: producer and consumer medians are
7.14 us and 9.99 us for 15 samples of 300 launches; CV is 24.5% and 83.9%.
One consumer sample is a large outlier, so these timings are unstable and make
no performance claim. Resident execute-through-sync is 672.4 ms and includes
upload, launch, and synchronization. See
[bounded dynamic-K packet](softmax_matmul_dynamic_k16_active7.json).

This adds one producer family and a bounded dynamic-K envelope to the checked
Schedule -> Tile route, but does not migrate the generic tensor-valued
constructors in TileIRLoweringPass.cpp; those remain open under W1.1.

### Dynamic-K timing recheck

A follow-up exact-device run increased each CUDA-event sample to 1,000 launches
and used 21 samples at active K=7 (bound K=16). Both resident packages again
matched the independent references (softmax max error 0.0; matmul max error
1.49e-8) and reported `native_gpu` on the same intermediate allocation and
stream. The producer/consumer medians were 10.37/11.87 us, but CV remained
36.8%/37.0%; several samples were near twice the median. This confirms the
small dynamic-K timings remain unstable even with longer event batches. The
packet is diagnostic only and does not support performance promotion:
[recheck packet](softmax_matmul_dynamic_k16_active7_recheck_20261001.json).

### Four-panel timing refresh

A fresh 21-sample RTX 5070 run of the K=64 RMSNorm -> matmul case again
passed both NumPy checks and the structural proof for four fragment
accumulator iterations. Independent package events measured producer median
11.26 us (CV 9.94%) and consumer median 10.69 us (CV 14.68%). Resident-chain
medians were 11.90/11.55 us; one consumer event outlier raised its CV to
146.1%. The samples confirm separate producer/consumer execution and typed
accumulator lineage, but are too variable for a performance claim. Packet:
[multi-K recheck](multi_k_recheck_20261001.json).


## Third producer: LayerNorm -> matmul

The contract now admits affine-free, last-axis fp16/bf16 LayerNorm as a
shape-preserving producer. Its existing scheduled tile.norm_kernel carries
kind=layernorm; the matmul consumer follows the canonical
Graph -> Schedule -> Tile path and consumes the same resident intermediate.

On Super-Bear's RTX 5070 (sm_120a), public from_text Graph traces executed
fp16 and bf16 LayerNorm -> matmul, with both package receipts reporting
native_gpu. The fp16 M/K/N=256/256/256 packet passed its independent
LayerNorm and matmul oracles (maximum absolute errors 9.77e-4 and 1.53e-5).
It also confirms one caller-owned intermediate allocation and no spills in
either package.

Seven samples of 500 repeated launches measured resident producer median
71.64 us (CV 0.008%) and consumer median 10.06 us (CV 1.68%). These are
stage attribution results for this exact shape; no route or performance
promotion follows. Packet: [LayerNorm M/K/N=256/256/256](layernorm_m256k256n256.json).

Generic pre-schedule tensor-valued Tile MMA constructors remain open; this
producer-family addition does not claim their migration.

## Exact SM120 typed-matmul route recheck

The exact-device test for the narrow typed-fragment producer had been defined
with a leading underscore, so pytest never collected its five shape cases. It
is now a collected test and passes all five on Super-Bear's RTX 5070. Each
package traverses Graph IR -> Schedule IR -> Tile IR -> NVIDIA Target IR,
contains pointer-backed tile.view plus typed fragment_pack and tile.mma, and
executes native NVVM MMA. The descriptor's Graph, Schedule, Schedule IR, and
Tile digests are compared directly with the returned compiler artifacts.
All five outputs passed the fp32 oracle with max absolute error at most
4.77e-7.

A separate packet records CUDA-event and end-to-end timings for the same five
shapes. Event medians ranged from 8.82 to 16.09 us and end-to-end medians from
0.293 to 0.362 ms. Several coefficients of variation are high, so this is
route attribution only and supports no speedup or selector claim. The generic
pre-schedule tensor-valued constructors remain a separate open W1.1 obligation.

[Exact-device timing and lineage packet](typed_m16n8_20261002.json).

## Fresh exact-device typed producer and matmul refresh — 2026-10-02

Super-Bear RTX 5070 (sm_120a) reran the public Graph -> Schedule -> Tile ->
NVIDIA Target IR -> PTX matmul route for five shapes. Every package passed its
fp32 numerical oracle before timing (maximum absolute error 4.77e-7). The
nine-sample CUDA-event medians were 10.83–15.81 us; end-to-end medians were
0.365–0.391 ms. Event variation was high on two shapes and end-to-end CV was
5.9–10.9%, so timings are diagnostic and do not support promotion.
[Typed-matmul packet](typed_m16n8_followup_20261002.json).

The resident RMSNorm -> matmul edge was also rerun at M/K/N=16/64/8. Tile IR
contains pointer-backed `tile.view`, typed A/B fragment packs, a zero fragment
accumulator, `tile.mma`, unpack and store; the K loop carries the accumulator
through four iterations. The intermediate is caller-owned, reused at the same
address on one stream, and disjoint from source, RHS, and final output. Exact
RTX 5070 output errors were 0 for RMSNorm and 1.43e-6 for matmul. Resident
event medians were 10.42 us producer and 11.45 us consumer with 7.4% and 8.0%
CV. These single-shape timings are attribution only; no speedup claim follows.
[Resident-edge packet](rmsnorm_matmul_followup_20261002.json).

This strengthens the compiler-owned public producer edge proof. It still does
not migrate the generic `LowerMatmulToTileMMA` and
`LowerKReductionAddToTileMMA` tensor-valued constructors in
`TileIRLoweringPass.cpp`.


## 2026-10-02 production-pipeline producer census and K-loop proof

The earlier W1.1 source trace cited `tests/tessera-ir/phase2/full_pipeline.mlir`,
which is an x86 fixture and did not establish the SM120 route. The replacement
regression, `test_sm120_graph_matmul_k_loop_carries_typed_fragment_accumulator`,
runs a 64x64x256 Graph matmul through `tessera-nvidia-pipeline-sm120`. It checks
two pointer-backed `tile.view` operands, typed A/B fragment packs, a typed
`fragment_zero` accumulator carried by `scf.for`, typed `tile.mma` consuming
that carried value, and one `fragment_unpack`. It also asserts that neither
`tile.async_copy` nor `tile.tma.copy_async` appears in this route. The test
passes on the production toolchain.

The same 64x64x256 matmul was packaged and executed on Super-Bear's RTX 5070
(sm_120). It passed the fp32 oracle with max absolute error 8.34e-7 and records
Graph, Schedule, Tile, Target IR, and image digests. Five-sample event and
end-to-end medians were 9.13 us and 0.313 ms; CVs were 15.8% and 13.1%, so
these timings are diagnostic only. The exact-device result proves the
four-panel accumulator path; no performance or selector claim follows.

The two generic constructors remain in `TileIRLoweringPass.cpp` for generic
legacy pipelines. The named SM120 pipeline lowers canonical Graph matmul via
Schedule before residual Tile lowering; `tessera-tiling` (which creates the
legacy K-reduction add marker) is not in that pipeline. This closes the
canonical SM120 producer path for the four-panel case, while broader legacy
pipeline migration and arbitrary tensor lifetimes remain separate work.

[Exact-device K=64 full-pipeline packet](typed_k64_full_pipeline_20261002.json).


## Static ragged-K typed producer — 2026-10-02

The canonical SM120 Schedule-to-Tile producer now admits positive static K values that are not multiples of 16 when M is divisible by 16 and N by 8. It emits bounded pointer-backed `tile.view` operands so the final 16-wide K panel zero-fills out-of-range lanes; the physical leading dimension remains the exact logical K. Fused epilogues and ragged M/N remain outside this producer envelope.

Super-Bear RTX 5070 exact-device fp16 and bf16 cases at M/K/N=48/67/16 passed the independent fp32 matmul oracle. Maximum absolute errors were 5.96e-7 (fp16) and 3.81e-6 (bf16). The structural regression checks both six-operand views and the K=67 leading dimension. The correctness-gated seven-row packet separates CUDA-event from end-to-end time. In the current 11-sample, 1000-event-repetition packet, fp16 K=67 measured 15.75 us event median at 33.6% CV and 0.363 ms end-to-end at 3.8% CV. Repeated packet runs varied materially, so the row remains route and correctness evidence only; no selector change follows.

[Stability packet](typed_matmul_ragged_k67_stability_20261002.json).


## W1.1 static M/N/K tails through typed fragments — 2026-10-02

Owner W1.1; synchronization key `NVIDIA-W1.1-STATIC-MNK-TAILS-2026-10-02`. The canonical
SM120 Graph -> Schedule -> Tile producer now admits every positive static M/N/K
in its existing fp16/bf16, fp32 accumulation/output, unfused envelope. Input
`tile.view` operations carry logical M/N/K bounds when any dimension has a
tail. NVIDIA lowers the existing six-operand static bounded `tile.store`
form with per-element M/N predicates; exact static leading dimensions remain.
Canonical typed packages have no tensor-valued MMA operands.

Large ragged-K shapes above the macro crossover previously inherited a
`_macro_kernel` symbol from workload alone. Their new typed producer uses a
16x8 output tile, so keeping that name would incorrectly launch the 32x32 CTA
grid. The compiler now names the actual producer geometry. Packaging projects
the actual native Tile entry; the duplicate Python entry-name and macro-policy
implementations are retired. Sixteen compiler admission/entry tests cover both
storage types and both sides of the crossover.

The full exact RTX 5070 scheduled device module passed 32 cases. This includes
17 typed numerical shape/dtype cases through M/K/N=257/513/257, six fp16/bf16
resident canary cases, and the existing macro/dynamic/epilogue cases. Canaries
prove input and output guard preservation; the tests synchronize before closing
borrowed views and verify owned-buffer cleanup. Seven compiler structural
checks and 48 resident-program/frontend checks also pass.

Shared launch validation now accepts opposite contiguous row/column labels only
when at most one dimension can vary; the physical addresses are then identical.
Genuinely different orders and strided mismatches still fail. NVIDIA-owned
buffers normalize scalar dtype classes to NumPy dtype objects to retain bf16
metadata at the checked CUDA ABI. Shared contract, diagnostic/pass, dtype and
audit gates passed 488 tests. Apple/ROCm/x86 have contract-only coverage and
require their own native singleton execution proof.

The twelve-row packet has eleven samples, 1000 CUDA-event repetitions, 20
end-to-end repetitions and 20 warmups. Source fingerprints match the measured
checkout. The new rows are:

| M/K/N | Max absolute error | Event median (us) | Event CV | E2E median (ms) | E2E CV |
| --- | --- | --- | --- | --- | --- |
| 1/1/1 | 0 | 10.93 | 36.3% | 0.298 | 92.3% |
| 17/19/23 | 1.19e-07 | 15.79 | 31.7% | 0.365 | 2.2% |
| 31/33/9 | 1.79e-07 | 15.01 | 48.3% | 0.300 | 50.2% |
| 48/67/17 | 4.77e-07 | 10.01 | 36.3% | 0.294 | 52.2% |
| 257/513/257 | 7.15e-06 | 16.57 | 7.6% | 0.526 | 36.5% |

Timing variance is high; no speedup, selection or promotion claim follows.
Compute Sanitizer 13.3 could not initialize the WSL/WDDM debugger interface and
reported unsupported-device errors. Its earlier application rerun passed 21
cases, but there is no memcheck proof. Resident numerical/canary evidence is
separate.

This closes the static-tail extension of the canonical typed producer.
The two generic direct-Tile constructors, fused epilogues and broader arbitrary
tensor-lifetime envelopes remain W1.1 obligations.

[Packet](typed_matmul_static_mnk_tails_20261002.json).
[Validation and compiler fingerprints](static_mnk_validation_20261002.json).
[Final test transcript](static_mnk_validation_20261002.txt).
<!-- entry-fields:end -->

## Positive legacy SM120 matmul entry migration — 2026-10-02

Original frontend names now enter the legacy Tile pass and delegate to registered native Graph → Schedule → Tile passes. MLIR SymbolTable owns launch-namespace normalization. Tile output is identical to canonical Schedule replay, and the recorded image is compiled from that delegated product. Exact RTX 5070 fp16/bf16 numerics pass, including static tails and bounded dynamic fused bias/ReLU/residual reuse with padded-view output canaries. The focused lane passed 386 tests. SM80/SM90 FileCheck regressions are compiler artifact evidence only.

| M/K/N | Max absolute error | Event median (us) | E2E median (ms) | Event CV | E2E CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/256/64 | 2.86e-06 | 8.91 | 0.303 | 40.9% | 1.7% |
| 17/19/23 | 1.19e-07 | 11.94 | 0.350 | 45.4% | 58.3% |
| 48/67/17 | 4.77e-07 | 8.67 | 0.294 | 46.7% | 2.9% |
| 257/513/257 | 7.15e-06 | 16.56 | 0.431 | 0.4% | 2.7% |

Seven samples, 1000 device repetitions, 20 E2E repetitions, 20 device warmups; twelve fp16 shapes are recorded. Package-build time includes canonical comparison and delegated packaging. Timing is diagnostic with no selector promotion. Canonical K-reduction steps, arbitrary tensor lifetimes and older-target execution remain open.

[Packet](legacy_scheduled_producer_20261002.json), [validation hashes](legacy_scheduled_producer_validation_20261002.json), [focused transcript](legacy_scheduled_producer_tests_20261002.txt).


## Native tiling to K-reduction Schedule — 2026-10-02

Owner W1.1; sync W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02.

For an explicit nvidia_sm120/sm_120 Graph matmul, generic tiling now emits the
registered native Schedule contract. Schedule-to-Tile owns the typed fragment
K loop and accumulator lineage. The direct Tile entry consumes an existing
Schedule without generating it again. MLIR SymbolTable normalization is shared
by both entry points before Schedule hashing; no Python emitter or Target bypass
was introduced. Generic tensor tile-size overrides are explicitly gated because
the SM120 native profile owns physical instruction geometry.

Exact RTX 5070 fp16/bf16 static-tail and multi-panel numerical checks exercise
both entries. Bounded dynamic fused bias/ReLU/residual shapes and padded output
canaries also exercise both entries. The twelve-shape fp16 packet compiles the
actual tiling-produced Tile product, proves byte-identical canonical Schedule
and Tile replay, and records separate CUDA-event and E2E timing with compiler,
runtime and source fingerprints. Timing variance prevents performance promotion.

The ordinary SM120 frontend tiling path no longer creates tensor-valued
canonical K-step MMA producers. Already-authored generic tensor K-step graphs,
arbitrary producer bufferization/lifetimes, native Schedule tuning knobs, and
older-target physical migrations remain open. Older-target FileCheck results
are compiler artifact evidence, not exact-device parity.

[Packet](native_k_scheduled_producer_20261002.json). The measurement uses seven samples, 1000 device repetitions, 20 end-to-end repetitions and 20 warmups. Package-build cost includes the canonical comparison and delegated package build.

Validation for W1.1-SM120-NATIVE-K-SCHEDULE-2026-10-02: 508 passed,
50 environment/capability skips, zero failures. Generic T16/T32 tiling,
fused epilogue tiling and SM80/SM90 Tile FileCheck fixtures passed. Packet
source, compiler and native-launch-library hashes match the measured checkout.

[Validation fingerprints](native_k_scheduled_producer_validation_20261002.json),
[focused test transcript](native_k_scheduled_producer_tests_20261002.txt).


## Resident fused epilogue integration — 2026-10-02

Owner W1.1; sync W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02.

The checked SM120 RMSNorm/LayerNorm/softmax-to-matmul tensor edge now carries
native fused bias, activation and residual operands. The Graph, native
Schedule/Tile, Target IR and image remain compiler-owned; Python sequences
compiled packages and manages allocations. The resident CUDA bridge consumes
the complete A/B/bias?/residual?/D pointer order followed by M/N/K and optional
leading dimensions. It validates buffer counts and span limits and queues the
consumer on the producer stream without host intermediate transfers.

Host execute and resident execute accept named fp32 bias/residual operands,
validate active shapes, reject missing/extraneous operands and preserve
allocation lifetime through stream completion. Program validation binds
epilogue semantics to the compiled entry and checks operand shapes/storage.
The shared bounded-axis Graph projection preserves epilogue arguments and
caller-owned Graph state, with dynamic M/N residual and N bias dimensions.
Numerics check the ordered fp32 epilogue before final fp16 output conversion.

Exact RTX 5070 evidence covers fp16/bf16, fp16/fp32 output and static/dynamic-M
reuse. This is a named producer edge, not general tensor bufferization or
arbitrary lifetime graphs. Native bias derivatives, broader AD integration,
older-target migrations and general producer graphs remain open.

The recorder reports independent CUDA-event producer and consumer samples,
plus allocation/upload/launch/synchronization/cleanup wall time. Compilation
and downloads are excluded from that E2E domain; package build is separate.
Seven samples and 1000 device repetitions do not imply selector promotion.

[Packet](resident_fused_epilogue_20261002.json).

Boundary for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: fused consumers
currently use the native tile.matmul_kernel carrier. Their explicit typed
fragment epilogue migration remains open; resident execution does not close
that producer census.

Validation for W1.1-SM120-RESIDENT-EPILOGUE-2026-10-02: 625 passed,
49 environment/capability skips, zero failures. Eight isolated exact-device
benchmark rows passed before timing and after device replay. Packet source,
compiler and native-launch-library hashes match the measured checkout.

| Input/output | Active M/K/N | Producer us | Consumer us | E2E ms | Consumer CV |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16/fp16 | 16/16/8 | 11.39 | 11.71 | 1.786 | 42.7% |
| fp16/fp16 | 7/16/8 | 9.89 | 10.61 | 1.810 | 38.4% |
| fp16/fp32 | 16/16/8 | 10.17 | 10.57 | 1.795 | 45.1% |
| fp16/fp32 | 7/16/8 | 11.18 | 10.64 | 1.770 | 44.5% |
| bf16/fp16 | 16/16/8 | 9.54 | 10.50 | 1.793 | 43.0% |
| bf16/fp16 | 7/16/8 | 15.99 | 16.35 | 1.810 | 42.4% |
| bf16/fp32 | 16/16/8 | 10.66 | 10.21 | 1.834 | 2.6% |
| bf16/fp32 | 7/16/8 | 9.72 | 10.33 | 1.812 | 39.2% |

No performance promotion follows these diagnostic timings.

[Validation](resident_fused_epilogue_validation_20261002.json), [test transcript](resident_fused_epilogue_tests_20261002.txt).


## Explicit typed fragment epilogue — 2026-10-02

Owner W1.1; sync W1.1-SM120-TYPED-EPILOGUE-2026-10-02.

Native Schedule-to-Tile now emits typed views, packs, a loop-carried fp32
fragment accumulator, unpack and store for positive f16/bf16 SM120 matmuls
with bias/activation/residual and f16/f32 output. The epilogue is on tile.store:
bias then activation then residual in fp32, followed by final output conversion.
Auxiliary loads are inside the output M/N guard. The explicit producer uses a
16x8 one-warp grid; native symbol/descriptor geometry remains matched. This
replaces the deferred tile.matmul_kernel carrier for this named envelope.

The existing shared store verifier now discounts ordered bias/residual pointers
from address arity and checks the required residual order contract. Schedule
pass metadata and the existing bias diagnostic guidance describe this
extension. ROCm explicitly rejects a residual store pending its own consumer;
its earlier bias/activation contract is retained. No Python Tile constructor,
new operation or Target bypass was added.

General producer bufferization/lifetime graphs and already-authored generic
tensor K-step graphs remain open. This does not close W1.1 for other storage
types or older targets, and it does not claim sibling exact-device parity.

The earlier resident_fused_epilogue_20261002 packet is a historical snapshot
of the deferred Tile producer. The typed packet is a new measurement with
compiler, Target compiler, source and native library fingerprints.

[Typed packet](typed_resident_fused_epilogue_20261002.json).

Validation for W1.1-SM120-TYPED-EPILOGUE-2026-10-02: 688 passed,
49 environment/capability skips, zero failures. Twenty-four correctness-gated
RTX 5070 rows cover fp16/bf16 input, fp16/fp32 output, static/dynamic M, and
bounds M/K/N=16/16/8, 17/67/23 and 256/256/512. The packet binds compiler,
Target compiler, native launch library and source hashes to this checkout.
Generic tiling, SM80/SM90 and the gfx1151 shared-store verifier fixture pass as
compiler evidence only. Timings are diagnostic; there is no selector promotion
or speedup claim. Fused macro-CTA reuse optimization remains a measured follow-up.

| Input/output | Active M/K/N | Producer us | Consumer us | E2E ms | Consumer CV |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16/fp16 | 256/256/512 | 59.49 | 19.72 | 2.162 | 25.3% |
| fp16/fp16 | 127/256/512 | 53.23 | 10.34 | 2.171 | 8.1% |
| fp16/fp32 | 256/256/512 | 59.48 | 16.22 | 2.241 | 3.2% |
| fp16/fp32 | 127/256/512 | 53.23 | 11.48 | 7.173 | 1.1% |
| bf16/fp16 | 256/256/512 | 59.49 | 16.20 | 2.253 | 0.3% |
| bf16/fp16 | 127/256/512 | 53.23 | 11.62 | 2.205 | 4.6% |
| bf16/fp32 | 256/256/512 | 59.48 | 16.22 | 2.281 | 0.4% |
| bf16/fp32 | 127/256/512 | 53.23 | 11.41 | 2.158 | 1.9% |

[Validation](typed_resident_fused_epilogue_validation_20261002.json), [test transcript](typed_resident_fused_epilogue_tests_20261002.txt).
