---
audit_role: plan
plan_state: landing
owner: Apple backend
target: apple_gpu
last_updated: 2026-07-18
---

# Apple compiler, exact-device, and performance plan

This plan brings the proof discipline established by the CUDA and ROCm work to
the Apple backend. It complements [`APPLE_AUDIT.md`](APPLE_AUDIT.md), the
generated execution inventory, and the durable architecture under
[`docs/backends/apple/`](../../../backends/apple/). Those documents remain the
status and architecture authorities; this file owns the active execution order
and completion gates.

The goal is not to transplant CUDA warps or AMD waves into Metal. Apple route
selection must be measured across MPS, MPSGraph, synthesized MSL,
`simdgroup_matrix`, Metal 4 cooperative tensors/MPP, and authored package
subgraphs. Logical fixtures, ABI contracts, numerical oracles, diagnostic
rules, and benchmark schemas should be shared with CUDA and ROCm; physical
fragments, threadgroup shapes, residency strategy, and command-buffer schedules
remain Apple-owned.

## Current state and immediate risk

- APPLE-TEST-1 now has a structural inventory at
  `tests/_support/apple_inventory.py`: its current scan records **0** direct
  Apple/Darwin/Metal capability gates. Apple device, integration, compiler-tool,
  and portable simulated-host cases are classified at their actual proof
  boundary. Offline MSL
  compiler checks are now `compiler_tool` tests with a shared `metal`-tool
  boundary, rather than device-gated tests. The first
  cohorts raise `pytest -m hardware_apple_gpu` collection from **3 to 976 of
  15,374** unit tests: the MPSGraph warmup and MegaMoE measured paths, exact
  native proofs for f32 CSR/COO SpMM, SDDMM, BSMM, scatter, optimizer, local
  MoE, MoE transport, and RNG, plus gather/concat/slice/softcap/transpose,
  mixed-program residency, TopK, projections, BMM, reduction, MPSGraph
  Tier-1, composed MHA, MPSGraph-runtime/cache, control-flow stress, and
  memory-budget residency proofs, quantized matmul, TopK, complex-runtime,
  evaluator/native-required, value-executor, fusion-synthesis, GA/EBM benchmark,
  control-flow/tracing, attention, delta, LDT, MoE, and other JIT-route proofs.
  The shared pytest boundary now supplies the Darwin/Metal skip; the marked
  proofs retain their explicit `native_gpu` and `metal_runtime` assertions
  where JIT provenance is available.
  APPLE-TEST-2 binds the first cohort's
  execution-matrix row,
  generic-envelope ownership where applicable, runtime ABI symbols, marked
  native node, and explicit fallback node in one registry; the shared native
  assertion rejects a semantic `reference_cpu` result. The f32 MPS matmul,
  MPSGraph BSMM/gather, and Philox symbols used by the cohort are now
  ABI-registered, so APPLE-REG-1 rejects an unregistered replacement.
- **2026-07-18 APPLE-TEST-1 closure:** a fresh full unit-tree collection found
  **976 of 15,374** nodes behind the centralized `hardware_apple_gpu` boundary,
  while the structural inventory still found **0** direct Apple/Darwin/Metal
  capability skips. The residency, runtime, and offline Metal-compiler cohorts
  retain their marker/provenance ratchets, so a newly added inline gate or
  misclassified compiler test fails the portable inventory suite. Ongoing
  classification enforcement is maintenance, not an open implementation rung.
  **APPLE-TEST-1 is closed.** No shared marker semantics or sibling-backend
  ownership changed; NVIDIA and ROCm are not applicable.
- APPLE-CI-2 now has an executable host-free ownership gate:
  `scripts/run_apple_host_free_compiler_tests.py`. It reads the CMake backend
  declarations, probes Apple/NVIDIA/ROCm pass registration, then selects only
  `compiler_tool` tests owned by the declared compiler capability set. On the
  current Apple-only build, Apple lowering is registered while the NVIDIA and
  ROCm probes are explicitly unregistered; the selected Apple artifact lane is
  green. The gate accepts any CMake cache type for `LLVM_DIR`, resolves and
  verifies the matching MLIR runner-utils dylib, and exports that exact path to
  the selected tests. Foreign compiler tests carry `compiler_nvidia` or
  `compiler_rocm`. This closes APPLE-CI-2. NVIDIA and ROCm are not applicable:
  their compiler ownership expressions and toolchain runners are unchanged.
- Cohort ledger: **APPLE-TEST-2-C1 / APPLE-REG-1-C1** records f32 sparse
  transport (CSR/COO SpMM and SDDMM), BSMM, scatter, optimizer, local MoE,
  MoE transport, and Philox RNG. Each row binds its execution-matrix path,
  native and fallback node, and runtime ABI symbols in
  `apple_exact_device_proofs.py`; complex/conformal remains outside this cohort
  until a hardware-marked execute/compare proof replaces its fallback-capable
  portable tests.
- Cohort ledger: **APPLE-TEST-2-C2 / APPLE-REG-1-C2** records only the fused
  interleaved-f32 complex/conformal subset (`complex_mul`, `complex_exp`,
  `mobius`, and `stereographic`). The device proof requires a traced fused ABI
  route and numerical oracle; its bridge-miss negative is explicitly
  `reference_cpu`. The long-tail complex/certificate operations remain outside
  C2 because they are intentionally host structured or lack a fused ABI route.
- Cohort ledger: **APPLE-TEST-2-C3 / APPLE-REG-1-C3** records only f32
  MPSGraph `sum`, `mse_loss`, and `mae_loss` (binary subtraction plus
  multiply/absolute-value plus reduction).
  Their exact-device nodes execute and compare on Metal; a forced missing
  MPSGraph binding must return `reference_cpu` from `runtime.launch`, rather
  than retaining the execution-matrix default. Huber, smooth-L1, log-cosh, and
  the loss-family lane remain outside C3 because their middle computations are
  host structured. NVIDIA/ROCm require no plan change: their loss/reduction
  paths have separate exact-device owners and no shared ABI changed.
- **2026-07-17 C1–C3 exact-device evidence:** all 12 distinct ledger-native
  nodes passed twice on Metal from separate freshly compiled runtime images;
  the 12 corresponding fallback-injection nodes passed and asserted
  `reference_cpu`. The two C2 rows intentionally share one fused
  complex/conformal native node and one bridge-miss negative. This closes the
  first cohort's placement, oracle, fresh-runtime, and fallback-negative
  evidence only; **APPLE-TEST-2 remains landing/open** until the same proof
  ladder covers the remaining Apple families, ordering/stress, and performance
  layers.
- **2026-07-17 broader exact-device evidence:** two independent fresh-runtime
  runs collected 853 nodes and each completed **849 passed, 4 skipped, 0
  failed** (97.4 s / 99.2 s). The four skipped legacy hand-written synthesis
  comparisons have now been explicitly reclassified as retired, non-native ABI
  contracts; their live synthesized replacements carry the Metal-placement and
  numerical-oracle evidence, and a forced missing-synthesis binding must return
  the reference route. A third fresh-runtime post-reclassification run completed
  **850 passed, 0 skipped, 0 failed** (100.5 s). The LLVM/MLIR 23 migration also
  fixed the JIT engine transformer's dangling callback and bounded the
  process-wide ExecutionEngine cache, which had previously made serial device
  validation segfault after accumulated JIT compiles.
- **2026-07-17 stateful and performance ladder evidence:** a separate fresh
  runtime passed the package/session-cache, resident block-paged KV, ReplaySSM,
  command-buffer, MPSGraph-LRU, and control-flow cohort (**63 passed**), with
  the bulk-MPSGraph/control-flow ordering stress raised to 75 iterations. Two
  independent route-characterization runs (21 rows each) and two independent
  ReplaySSM runs (12 rows each) reported native dispatch and numerical
  validation for every row. The temporary artifacts are
  `/private/tmp/apple-routes-proof-{a,b}.json` and
  `/private/tmp/apple-ssm-replay-proof-{a,b}.json`; they are evidence, not a
  committed performance ratchet. The remaining proof-ledger work is to add the
  same explicit fallback-injection record to the other native family owners;
  the closure update immediately below records the final family set and
  corrected serial performance selection.
- **2026-07-17 APPLE-TEST-2 closure:** the proof ledger now includes the C1--C3
  ABI cohort, synthesized matmul/reduction replacement, paged-KV attention, and
  fused ReplaySSM. ReplaySSM's C ABI now returns an explicit dispatch bit: its
  exact-device node requires `native_gpu` and its forced missing-binding
  negative requires `reference_cpu`, so a numerically identical host reference
  can no longer earn the native rung. The final fresh-runtime correctness lane
  passed **850/850**; the serial measured lane passed **69/69**. Two simulated
  distributed-MoE wall-clock tests were removed from the Apple hardware marker
  because they use modeled communication and do not assert an Apple route; the
  JIT-bridge benchmark fixture typo was corrected. **APPLE-TEST-2 is closed.**
  The plan remains `landing` because APPLE-REG-1, TILE, retuning, paged-KV,
  ReplaySSM serving expansion, and device-keyed performance selection are
  separate owning items.
- **2026-07-17 APPLE-REG-1 closure:** the canonical Apple ABI registry,
  runtime-header ABI, target-map, exact-device proof, and Tile-envelope drift
  gates passed against the LLVM/MLIR 23 `build-apple` compiler. The Tile status
  test now honors `$TESSERA_OPT` before the stale default build path, preventing
  an ABI-incompatible LLVM dylib from masquerading as a lowering failure.
  **APPLE-REG-1 is closed.** No dtype/op/diagnostic/target state was added in
  this slice, so NVIDIA and ROCm are not applicable.
- **2026-07-17 APPLE-TILE-1 start:** the real Tile-to-Apple status/materialized
  artifact gate passes with the LLVM 23 compiler, but it is not yet an
  exact-device fragment proof: the current fixture uses `tile.mock` and asserts
  runtime status only. TILE-1 remains open until a shared logical value fixture
  selects an Apple-owned fragment/layout from target capabilities and proves
  packing, ragged store, geometry/resource record, and native execute/compare.
- **2026-07-17 APPLE-TILE-1 value/ragged evidence:** the value-preserving
  `tile.batched_gemm` path now runs both aligned `2x4x8 @ 2x8x16` and ragged
  `2x5x7 @ 2x7x9` fixtures for f32/f16/bf16. Each exact-device case asserts
  `native_gpu` + `metal_runtime` and compares against the NumPy oracle; the
  fixture supplies only logical shapes, while Apple lowering owns BMM packing
  and route selection. **8 passed.** TILE-1 remains open for an explicit
  selected physical fragment/layout and threadgroup/resource record; the MPS
  BMM value route must not be relabeled as simdgroup-fragment materialization.
- **2026-07-17 APPLE-TILE-1 fragment-materialization landing rung:** Apple7+
  Tile selection now owns an exact `simdgroup_matrix` descriptor: fp16/bf16
  storage, fp32 accumulation, an 8x8x8 MMA fragment, 32 lanes, and a
  `(32,1,1)` threadgroup. The target-selected materializer consumes that
  descriptor to emit the existing steel-shaped MSL artifact with cooperative
  packing, bounds zero-padding, partial-edge store, and double buffering.
  Its host-free structure and target limits gates passed **85 passed, 9
  compiler-tool skips**. At that point this was artifact evidence only; the
  source-backed ABI and exact-device evidence are recorded below.
- **2026-07-17 APPLE-TILE-1 resource-contract landing rung:** each selected
  simdgroup artifact now carries a target-owned record for its `(32,1,1)`
  launch geometry, 32 lanes, staged-A/B bytes, ragged-store scratch, buffering
  mode, and total threadgroup-memory demand. Materialization rejects a tile
  that exceeds the selected target's threadgroup-memory capacity (the
  double-buffered 32x32x16 fp16/bf16 case records 4,352 bytes). The focused
  fragment/emitter/feature suite passed **67 passed, 9 compiler-tool skips**.
  This completed resource planning for the artifact path; runtime evidence is
  recorded below.
- **2026-07-17 APPLE-TILE-1 native single-fragment rung:** a distinct,
  registered `tessera_apple_gpu_tile_simdgroup_gemm_f16` C ABI now accepts the
  selected steel MSL source and entry, binds fp16 A/B and fp32 output, and
  dispatches exactly one 32-lane `(32,1,1)` threadgroup per 8x8 output tile.
  It is separate from the MPS BMM ABI and rejects any other threadgroup size;
  the non-Darwin stub returns 0. A fresh runtime image compiled and ran the
  8x8 fp16 fragment on Metal with zero fp32-oracle error; the focused proof
  test also forces the ABI binding missing and observes an explicit non-native
  result. The follow-on expanded this exact-device proof to bf16 8x8x8 and
  ragged/multi-fragment fp16 `13x16 @ 16x11`; both remain native and match the
  fp32 oracle (**46 passed, 9 compiler-tool skips**). A 30-repetition warm
  end-to-end characterization retained at
  `/private/tmp/apple-tile-simdgroup-characterization.json` reports medians of
  0.310 ms (8x8x8), 0.311 ms (13x16x11), and 0.315 ms (32x16x32); it has no
  device-event timing or MPS comparison, so it is not a selector decision.
  The C++ full pipeline now selects this ABI only for strict static rank-2
  `tile.matmul`/`tile.gemm` with fp16 or bf16; rank-3 `tile.batched_gemm`
  deliberately remains on MPS BMM. The Python value executor materializes the
  selected source and rejects a non-native result rather than using MPS/NumPy.
  The compiler/runtime ABI regression passed **18 passed**. TILE-1 remains open
  for retained runtime resource/provenance telemetry and comparative device-time
  performance selection.
- **2026-07-17 APPLE-TILE-1 telemetry/first comparison rung:** every direct
  source-backed dispatch can now return a record containing the ABI symbol,
  source SHA-256, native/reference result, execution mode, selected resource
  record, and runtime MSL pipeline-cache size. Its focused regression passed
  **17 passed**. A warm 30-repetition `32x16 @ 16x32` end-to-end comparison
  retained at `/private/tmp/apple-tile-simdgroup-vs-mps.json` recorded 0.314 ms
  median for native fp16 simdgroup and 0.229 ms for the existing f32 MPS route.
  These are not equivalent dtype paths and have no device-event timing, so they
  are explicitly **not** a selector decision. Remaining work is equal-dtype MPS
  comparison plus Metal device-time/resource telemetry and a two-run stability
  gate before any production-route change.
- **2026-07-17 APPLE-TILE-1 equal-dtype stability rung:** two independent warm
  30-repetition fp16 `32x16 @ 16x32` comparisons passed their respective fp16
  numerical oracles (the MPS route uses documented `rtol=1e-2` accumulation
  tolerance). Retained evidence at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-two-run.json` measured
  simdgroup medians of 0.336/0.293 ms versus MPS medians of 0.234/0.226 ms.
  MPS is the end-to-end winner for this one shape; no selector changed because
  the runtime presently exposes neither command-buffer GPU timestamps nor
  Metal counter sampling. The Tile record supplies selected static resource
  bytes and pipeline-cache state, but not measured occupancy/spills. The next
  required implementation is a dedicated runtime timing/counter ABI, followed
  by a broader shape/dtype corpus and an explicit promotion threshold.
- **2026-07-17 APPLE-TILE-1 kernel-time rung:** the runtime now records the
  completed command buffer's `kernelStartTime`/`kernelEndTime` (falling back to
  GPU start/end only when available) through
  `tessera_apple_gpu_tile_last_device_time_ns`. The exact-device proof requires
  a positive measured value (**17 passed**). Two 30-repetition equal-dtype fp16
  kernel-time runs retained at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-device-two-run.json` measured
  simdgroup medians 23.1/21.4 us and MPS medians 21.8/18.8 us for `32x16 @
  16x32`; MPS wins this shape in both domains. The following bounded-counter
  rung replaces the then-missing capability-gated counter path; no selector
  changed.
- **2026-07-17 APPLE-TILE-1 bounded counter/corpus rung:** the runtime now
  discovers the named `MTLCommonCounterSetTimestamp` set only when the device
  supports dispatch-boundary samples, allocates a two-sample buffer, and
  samples immediately before/after the source-backed Tile compute encoder.
  The dispatch record retains either its measured timestamp delta or explicit
  `counter_sampling_supported: false`; it never manufactures occupancy or
  spill values. This M1 Max reports the latter while retaining positive
  command-buffer kernel timing. The new
  `benchmark_tile_simdgroup.py` corpus made two independent 30-repetition
  warm runs for fp16/bf16, aligned `8x8x8`/`32x16x32`/`256x256x256`, and ragged
  `127x63x129` shapes in both end-to-end and kernel domains. All eight
  end-to-end rows retain MPS. Kernel-only microcase movement is not a
  production promotion: the selector's production domain is end-to-end, where
  MPS remains the route. The selector contract requires native placement plus
  numerical proof, retained resource/counter evidence, and a 5% win in both
  intended-domain runs; no production route changed.
- **2026-07-17 APPLE-GEMM-1 capture-telemetry landing rung:** an opt-in,
  thread-local dispatch record now spans the owned Apple command-buffer paths.
  Legacy MPS/MSL records use completed `kernelStartTime`/`kernelEndTime`
  (command-buffer time only as an explicit fallback); the shared MTL4 encoder
  uses a reusable two-entry timestamp heap and converts its raw tick delta with
  the device timestamp frequency. The same record retains the live MTL4
  threads-per-threadgroup, execution width, maximum threads, and static
  threadgroup-memory properties. Capture is disabled by default so precise
  timestamp sampling cannot perturb production dispatch. The standalone
  MPSGraph row-op path now encodes into an owned `MPSCommandBuffer`, commits its
  live `rootCommandBuffer`, and reports a whole-dispatch interval only when
  MPSGraph did not auto-flush and replace the supplied Metal command buffer;
  occupancy and spill fields remain null rather than inferred.
  `select_stable_gemm_routes.py` aggregates two or more current-schema reports
  by exact Apple GPU family and emits separate device/end-to-end decisions. A
  promotion requires native placement, numerical proof, repeated samples,
  retained resources/counters, at most 15% cross-run drift, and a 5% win in
  every run. Two fresh 30-repetition Apple7 reports at
  `/private/tmp/apple-gemm-stable-{c,d}.json` cover square, rectangular,
  ragged, fp16/f32, MPS, simdgroup, cooperative-tensor, MSL, and MPSGraph
  routes; `/private/tmp/apple-gemm-stable-ledger.json` records **0 promotions,
  13 incumbent retentions, and 9 inconclusive timing-domain rows**. MPSGraph
  device intervals are present in both reports; its three device decisions are
  inconclusive because cross-run drift exceeded the 15% bound. No production
  selector changed. NVIDIA and ROCm are not applicable: this is an Apple-only
  Metal ABI and Apple-only report extension, with no shared IR, schedule, or
  cross-backend benchmark schema change.
- **2026-07-17 APPLE-GEMM-1 paired-winner/resource evidence rung:** absolute
  cross-process latency is now diagnostic rather than a promotion veto. Each
  report runs nine alternating route blocks of 30 repetitions; a candidate
  must win at least 75% of paired blocks, clear 5% median speedup in both fresh
  processes, and keep cross-process speedup spread within five percentage
  points. The committed Apple7 ledger is
  `benchmarks/baselines/apple7_gemm_route_ledger.json`: **3 timing-domain
  promotions, 19 incumbent retentions, 0 inconclusive rows**. Only
  end-to-end winners affect production: f32 `128x257` and `256x256` softmax
  select MPSGraph instead of MSL on Apple7 (24.2--28.0% and 36.9--40.2%
  paired median wins, respectively, winning all 18 blocks). The f32
  `64x64x64` simdgroup route wins device time by 38.5--40.1% but loses end to
  end, so MPS remains production. All other measured matmul shapes retain MPS.
  The new profiling-capability ABI records what public Metal actually exposes
  on this M1 Max: compiled-pipeline limits, stage-boundary timestamp sampling,
  and the Metal 4 timestamp heap are available; statistic/stage-utilization
  counter sets and dispatch-boundary sampling are unavailable. Live MSL/MTL4
  records retain execution width, maximum threads, static threadgroup memory,
  simdgroups per threadgroup, and a clearly named threadgroup-capacity proxy.
  The runtime ABI exposes no register count, scratch bytes, spill count, or
  true occupancy metric, so those per-dispatch fields remain null rather than
  inferred from pipeline limits. A separate bounded Instruments `Metal System
  Trace` now supplies genuine compiler/spill evidence, summarized reproducibly
  by `benchmarks/apple_gpu/summarize_metal_trace.py` in
  `benchmarks/baselines/apple7_gemm_metal_trace_evidence.json`. The exact
  Apple7 process trace retained four compute-shader compile intervals (2.356 ms
  total, 1.486 ms maximum), two MTLibrary creation intervals (0.258 ms total),
  and seven named compute shaders. Exact command-buffer joins observed one
  64-byte spill event on each of ten `tessera.rowop.mpsgraph` submissions and
  zero spill events on ten f32 MPS GEMMs, ten f16 MPS GEMMs, ten explicit MSL
  softmax submissions, and twenty reusable MTL4 submissions. The MTL4 command
  buffer is intentionally reused and Instruments retains it as `Command Buffer
  0`, so its zero-event row is an aggregate MTL4 observation rather than a
  per-kernel claim. The default system-trace template recorded
  `counter-profile=0`, but the standalone `Metal GPU Counters` instrument is
  available on this M1 Max and two bounded captures enabled profile 3 with
  shader profiler 1. Its genuine `Compute Occupancy` counter (ID 24) produced
  376 command-buffer-correlated samples: f32 MPS GEMM retained 144 samples
  (one nonzero sample, 0.282% maximum), the reusable MTL4 buffer retained 12
  zero-valued samples, MPSGraph retained 216 zero-valued samples, and explicit
  MSL softmax retained four zero-valued samples. Those zeros are the measured
  counter values for this small characterization workload, not synthesized
  occupancy estimates; f16 MPS had no in-interval sample and remains null.
  The live threadgroup-capacity/concurrency proxy remains alongside the raw
  counter evidence. NVIDIA and ROCm are not applicable because no shared IR,
  schedule, or cross-backend ABI changed.
- **2026-07-17 APPLE-EPILOGUE-1 native/resource/selection rung:** synthesized
  f32, f16, and bf16 epilogues already had common-oracle coverage for bias,
  ReLU, GELU, SiLU, residual guards, ragged stores, large reductions, and a
  forced symbol-missing negative. The runtime now labels every synthesized
  command buffer and retains its live pipeline limits, actual threadgroup, and
  total pipeline-static plus encoder-requested threadgroup memory. A ragged
  `64x64x2049` tiled softmax proof records at least `2049 * sizeof(float)`
  scratch; an fp16 bias+SiLU proof records the selected cooperative-matrix
  threadgroup and both match the backend-neutral `FusedRegion` oracle.
  MPSGraph unary and binary epilogue dispatches now use an explicitly owned
  `MPSCommandBuffer` and expose status-returning ABI variants, so native
  placement is independent of numerical success. Before the later Metal 4
  envelope closure below, MPSGraph could legally call `commitAndContinue` and
  replace the supplied root command buffer, so timing remained null rather
  than reporting a partial interval.
  `benchmark_epilogue_routes.py` collected two fresh Apple7 runs with seven
  alternating trials of 15 repetitions for aligned `64x64x64`, ragged
  `65x63x67`, and `256x256x256` f32/f16 ReLU plus f32 bias+SiLU. The committed
  `benchmarks/baselines/apple7_epilogue_route_ledger.json` records a stable
  end-to-end synthesized-fusion win for all nine comparable rows (49.8--71.6%
  paired median speedup and 100% paired-block wins in both processes). Device
  decisions remain explicitly inconclusive because the unfused MPSGraph
  segments do not expose complete command-buffer intervals. Production already
  selects the synthesized fused route for these supported regions, so this
  evidence ratifies rather than changes that selector. GELU and bf16 remain
  native correctness/resource proofs but are not compared against a false
  mixed-dtype or missing-MPSGraph incumbent. NVIDIA and ROCm are not applicable:
  the new ABI and schedule evidence are Apple Metal-only and no shared IR or
  numerical contract changed.
- **2026-07-18 APPLE-TILE-1 closure:** the shared logical fixture now selects
  an Apple-owned descriptor and schedule without test-authored physical maps;
  the selected f16/bf16 fragment path has packing, ragged-edge, resource,
  provenance, native execute/compare, device-time, and counter-capability
  evidence. The two-run aligned/ragged corpus retains MPS in every end-to-end
  row. That measured non-promotion is a valid selector outcome, not unfinished
  Tile work. **APPLE-TILE-1 is closed.**
- **2026-07-18 APPLE-GEMM-1 closure:** the paired Apple7 ledger records a
  stable decision for every measured timing-domain row: three promotions and
  nineteen incumbent retentions, with no inconclusive rows. Native placement,
  numerical validation, timing-domain separation, resources, and bounded
  Instruments compiler/spill evidence are retained. New device families or
  candidate routes require a new corpus; they do not keep this Apple7 ratchet
  open. **APPLE-GEMM-1 is closed.**
- **2026-07-18 APPLE-EPILOGUE-1 closure:** every supported f32/f16/bf16
  epilogue has native placement, common-oracle, resource, ragged-store, and
  fallback-negative proof. The two-run Apple7 ledger records stable
  synthesized-fusion end-to-end wins for all nine comparable rows. Unsupported
  pairs retain an explicit non-fused route or registered diagnostic.
  **APPLE-EPILOGUE-1 is closed.**
- **2026-07-18 MPSGraph device-interval closure:** the telemetry-only Metal 4
  bracket writes a timestamp before graph execution, makes MPSGraph wait on
  that event, signals a second event at its documented completed stage, then
  writes the final timestamp after that signal. It therefore spans every
  internal `commitAndContinue` root rotation without treating a partial root
  interval as a graph interval. Unary (including the prior queue-owned
  epilogue path), binary, row-op, transpose, paged gather, and BSMM now encode
  through the owned descriptor path. The result is labeled
  `metal4_mpsgraph_envelope`, deliberately distinct from direct MTL4 encoder
  timing; without a Metal 4 timestamp heap telemetry stays unavailable rather
  than fabricated. Fresh exact-device softmax and epilogue smoke evidence has
  complete MPSGraph interval coverage. The historical Apple7 selector ledger
  is unchanged: a new two-run corpus is required before any device-domain
  selector decision can use this new timing domain. NVIDIA and ROCm are not
  applicable because this is an Apple Metal runtime telemetry path only.
- **2026-07-17 APPLE-ATTN-FWD-1 placement/resource landing rung:** the f32 and
  f16 online-softmax MSL command buffers now carry stable route labels, retain
  their actual `Sq x B`-derived threadgroup and live pipeline limits, and expose
  status-returning ABI variants. The exact-device proof covers ragged
  `B=2, Sq=17, Sk=19, D=128`, causal masking, f32/f16 storage with f32 softmax
  accumulation, positive command-buffer GPU time, and a shared NumPy oracle.
  The D=257 envelope negative returns status 0 and no device interval, so the
  legacy reference fallback cannot be mislabeled native. This is a landing
  rung, not closure: bias, softcap, window, MHA/GQA/MQA, long-context, resident
  command-buffer, cooperative-matrix, and MPSGraph candidate comparisons still
  need the full two-run measured corpus; APPLE-ATTN-BWD-1 is untouched.
- **2026-07-17 APPLE-ATTN-FWD-1 variant/selector rung:** one status-returning
  online-softmax ABI now composes additive bias, causal or sliding-window
  masking, logit soft-cap, and direct MHA/GQA/MQA KV-head indexing for native
  f32/f16 storage. It retains the actual threadgroup and pipeline limits and
  rejects invalid grouping, negative windows, and D>256 before submission. The
  exact-device matrix covers MHA, GQA, and MQA, ragged `Sq=5/Sk=37`, the
  combined bias+window+softcap contract, and MQA `Sk=1025`; every row matches
  the shared f32-accumulation oracle. The MPSGraph BSMM candidate now owns and
  labels its command buffer and returns native status. Two independent Apple7
  runs, each using seven alternating trials of 20 repetitions, compare f32/f16
  aligned `B1/H4/S64/D64`, ragged `B1/H4/Sq65/Sk67/D64`, and throughput
  `B1/H8/S128/D64` plain MHA. The retained
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all six end-to-end rows; production selection is exact-device,
  exact-shape, dtype, and timing-domain keyed. Device timing retains MSL for
  rows without a stable 5% MPSGraph win. The resident command-buffer candidate
  is measured separately in its device-resident input domain and retains live
  resources, but its shared-session command buffer exposes no complete device
  interval. No cooperative-matrix attention ABI exists, so that candidate is
  explicitly unavailable rather than assigned synthetic timing. This is not
  full APPLE-ATTN-FWD-1 closure: wider B/head/D and long-context matrices,
  variant-capable resident/cooperative candidates, and complete device timing
  remain open. bf16 continues to be labeled host-conversion plus f32 GPU
  compute, and APPLE-ATTN-BWD-1 remains separate. NVIDIA and ROCm are not
  applicable because the new ABI, selector, and physical schedule are
  Apple-only; shared attention semantics and numerical policy are unchanged.
- **2026-07-17 APPLE-ATTN-FWD-1 closure:** the forward lane now covers the
  remaining physical and evidence gaps without expanding into backward. The
  selector corpus spans `B=1/2`, 4/8/16 query heads, `D=64/128/256`, aligned
  and ragged lengths, and plain-MHA context through `Sk=1025`. The variant
  corpus adds MHA/GQA/MQA, bias+causal+window+softcap, `B=2`, ragged
  `Sq=65/Sk=67`, and decode-style MQA through `Sk=2049`. The resident scalar
  and one-SIMD-group-per-query-row candidates now accept the same variant ABI;
  the latter is named `cooperative_simdgroup` rather than being mislabeled a
  Metal cooperative-matrix route. No attention-specific cooperative-matrix ABI
  is available on this SDK/host, and that capability remains explicit rather
  than receiving synthetic measurements. f16 and bf16 keep native two-byte
  device storage; GPU-side casts surround f32 accumulation on the resident
  command buffer, with no host fp32 staging inside the attention ABI.
  `ts_enc_commit_wait` now publishes the completed owned-command-buffer Metal
  interval. Two independent Apple7 warm reports, each with five alternating
  trials of ten repetitions, retain 9 MSL variant rows and 18 resident versus
  cooperative rows; every row is native, matches the shared oracle, and every
  resident/cooperative row has 100% device-time coverage. Logical input/output
  bytes, residency, intermediate-storage policy, actual threadgroup/pipeline
  limits, GPU time, and end-to-end time are retained; unavailable occupancy,
  register, and spill counters remain null. The regenerated
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all eight plain-MHA end-to-end rows. In the distinct device-interval
  domain only f32 `B1/H16/Sq16/Sk1025/D256` has a stable two-run 5% win;
  all other device rows retain online MSL. `APPLE-ATTN-BWD-1` remains a
  separate open item and no backward implementation or policy changed.
  NVIDIA and ROCm are not applicable: this closes Apple-only runtime ABIs,
  storage handling, schedules, and evidence, with no shared IR, attention
  semantic, or numerical-policy change.
- **2026-07-17 APPLE-PAGED-KV-1 retained staged-gather rung:** the existing
  non-contiguous resident MPSGraph gather now encodes through an explicitly
  owned, labeled `MPSCommandBuffer`. `ResidentBlockPagedKVCache` retains
  `last_gather_execution` and the capture record for each gather; a framework
  pipeline that exposes no public PSO limits records the MPSGraph API and an
  explicit unavailability reason rather than synthetic resources. The
  exact-device proof interleaves two sequences to produce physical table
  `[0, 2, 4]`, gathers the correct non-identity values, and requires native
  status. Existing remap/reuse, concurrent-sequence, exhaustion, and teardown
  tests remain green. This closes provenance loss for the staged candidate but
  not APPLE-PAGED-KV-1: a direct resident page-table attention candidate,
  causal-offset/boundary stress, leak telemetry, and two-domain comparison are
  still required.
- **2026-07-17 APPLE-REPLAY-1 native block/timing landing rung:** output-only
  replay and fp32/f16 block decode now label their command buffers and retain
  live threadgroup/pipeline records. The block ABI returns native status, which
  propagates to `SSMStateHandle.last_block_execution`; N>256 returns an explicit
  reference provenance and common-oracle result. Focused rollback, forced
  binding-miss, f32/f16 block, resource, and ABI tests pass. Two independent
  Apple7 reports at 512 tokens, capacity 16, and 20 repetitions cover
  `1x128x128`, `1x256x128`, and `4x128x64`. The committed
  `benchmarks/baselines/apple7_replay_ssm_evidence.json` retains complete native,
  numerical, resource, end-to-end, and device-per-token evidence for all six
  output-only/block rows. End-to-end cross-run drift is 0.3--2.1%; device drift
  is 0.9--26.8%. The ledger deliberately makes no selector decision because the
  legacy benchmark does not interleave paired route blocks. Persistent resident
  inputs, forced flush/partial rejection/block-submit ordering, asynchronous
  ring backpressure, cleanup stress, and a paired selector corpus remain open.
  NVIDIA and ROCm are not applicable to these Apple-only runtime ABI changes;
  shared SSM state semantics and numerical policy are unchanged.
- **2026-07-18 APPLE-PAGED-KV-1 closure:** `ResidentBlockPagedKVCache` now owns
  one persistent int32 page table per live sequence. Its direct f32 MSL
  candidate forms rope-key scores and latent values by following that physical
  table in one dispatch; the staged peer performs two on-GPU non-contiguous
  gathers plus dense resident attention. Both share the same non-identity
  oracle, right-aligned or explicit causal offsets, and bounded windows. A
  failed multi-block reservation is transactional, lifecycle telemetry accounts
  for live pages/tables/calls, and teardown frees every table and pool. Thirteen
  focused tests pass on the Apple host, including exact-device direct/staged
  placement and equivalence. The committed two-run Apple7 corpus covers
  `127x64x32x1` and `512x128x64x1` with ten measured repetitions after three
  warmups. Direct wins both runs in both device and end-to-end domains and is
  promoted only for those exact f32 rows; unmeasured rows retain staged.
- **2026-07-18 APPLE-REPLAY-1 closure:** the Apple serving handle keeps scalar
  A, S0, and fixed-capacity delta/x/b/c rings in persistent `DeviceTensor`
  buffers. Block submissions encode against those buffers, commit without
  waiting, and rely on ordered Metal command-queue execution. Output slots stay
  leased until `wait()`, enforce explicit backpressure, reject flush/rollback
  while submissions are pending, and are drained during idempotent cleanup.
  Forced flush, ordered multi-block submission, rollback, partial speculative
  rejection, slot reuse, and cleanup match `SSMStateHandle` in seven new
  exact-device tests. The expanded Apple ReplaySSM and benchmark-contract
  regression set passes 52 tests. A narrow checkpoint-fold follow-up now gives
  one Metal lane to each `(batch, channel, state)` element, serially replays
  tokens without atomics, and writes resident `S0`; a second kernel clears all
  fixed-capacity rings in the same ordered command buffer. Native provenance,
  forced-boundary equivalence, repeated flush/cleanup, and a portable explicit
  fallback negative cover the
  lifecycle. The dedicated two-run Apple7 flush corpus records device and
  end-to-end timing separately at `1x128x64/T16` and `1x256x128/T16`: native
  device medians are 20.9--30.0 us and end-to-end medians are 298--318 us.
  The shared vectorized CPU fold remains faster end-to-end for these isolated
  flushes (44.6--146 us), so the native route is a residency/ordering closure,
  not a latency promotion. The paired serving two-run Apple7
  corpus compares `fused_block` with `resident_ring` at `1x128x64/T16` and
  `1x256x128/T16`, ten repetitions after three warmups. Fused block is the
  stable end-to-end winner; the smaller device-domain winner flips between
  runs and therefore earns no promotion, while the larger row stably retains
  fused block. NVIDIA and ROCm are not affected: their resident CUDA/HIP
  contexts and physical schedules remain independently proven.
- **2026-07-18 APPLE-RETUNE-1 paired-corpus foundation:**
  `benchmark_legacy_retune.py` now measures grouped GEMM, MoE SwiGLU,
  MPSGraph reduction, contiguous resident-KV reads, absorbed/explicit MLA, and
  ReplaySSM block/token-loop decode through one interleaved two-run schema.
  Every row shares a numerical oracle and records native/reference provenance,
  resource/API evidence, paired end-to-end medians, and a device interval only
  when it covers the complete route. The Apple7 corpus retains grouped fused
  GEMM and fused Replay decode, promotes single-dispatch MoE and absorbed MLA
  end-to-end on their exact small rows, and retains explicit MLA in the device
  domain. Reduction has end-to-end native evidence but no owned device interval;
  mapped KV and multi-dispatch peers remain explicitly ineligible for device
  selection. APPLE-RETUNE-1 stays active for wider shapes/dtypes, grouped
  SwiGLU/transport byte-bandwidth rows, and complete command-buffer intervals
  for the remaining composed routes.
- **2026-07-18 APPLE-ROUTE-1 strict-ingestion foundation:** production lookup
  no longer reads a literal exact-row table. The v2 ledger gate matches the live
  Apple family and physical-device model, OS, SDK, configured LLVM/compiler
  digest, runtime-source digest, expiry window, native provenance, correctness,
  and requested timing domain; admitted decisions expose their exact ledger-row
  citation. The fresh Apple7 retune ledger admits eight decisions in a clean
  host process. Older v1 GEMM/attention/backward/paged-KV/Replay ledgers lack
  this envelope and are rejected, so those operations conservatively retain
  their incumbents until fresh strict ledgers are recorded. APPLE-ROUTE-1 stays
  active until each completed family is migrated and package-subgraph selection
  is separated into its own strict ledger namespace. NVIDIA and ROCm are not
  applicable to this Apple-only corpus/selector change; their physical-device
  probes, retained ledgers, and production selectors are unchanged.
- A fallback result can prove semantics, but it cannot prove `native_gpu`, GPU
  residency, Metal ordering, resource lifetime, or performance. Device tests
  must assert their execution state and provenance explicitly.
- Apple already has broad MPS/MPSGraph/MSL execution, Metal 4 probes,
  `simdgroup_matrix` and cooperative-matrix candidates, fused GELU/SiLU
  epilogues, online-softmax attention, resident block-paged KV, ReplaySSM,
  command-buffer batching, route characterization, and a hot-path baseline.
  The work below strengthens, compares, and retunes these paths rather than
  reimplementing them blindly.
- The committed Apple hot-path ratchet is predominantly f32 and end-to-end
  wall-clock. It does not yet provide the square/rectangular/ragged/dtype matrix
  or per-candidate GPU-counter/resource evidence now required for CUDA/ROCm.
- Attention backward now has an Apple-owned native proof and stable route
  ledger; its physical schedules remain independent of CUDA and ROCm.
- **2026-07-18 APPLE-ATTN-BWD-1 native-candidate foundation:** the Apple
  runtime now exposes a status-only f32 MHA backward ABI. Two MSL encoders on
  one labeled command buffer recompute the softmax and produce dQ, dK, and dV
  with f32 accumulation; each output element owns its reduction, so the route
  is deterministic and has zero workspace/atomic traffic. Exact-device ragged
  and causal oracle tests verify all three gradients and repeated launches are
  bit-identical. The same ABI now owns a zero-workspace atomic dK/dV candidate
  using relaxed compare/exchange f32 accumulation and a deterministic two-way
  split candidate using exactly one additional f32 dK+dV partial plus a
  fixed-order reduction. The policy rejects deterministic atomic requests and
  insufficient split workspace before dispatch. Exact-device tests cover all
  three routes on ragged, batched, causal, and noncausal shapes against the same
  oracle; serial and split repeats are bit-identical, while atomic repeats are
  validated numerically under its explicitly nondeterministic contract.
  `benchmark_attention_backward.py` produces paired route rows with warmup
  separation, per-trial GPU/end-to-end medians, resources, workspace policy,
  and per-gradient error. Two Apple7 smoke collections each have twelve native,
  numerically valid rows and complete device-time coverage. Atomic wins every
  end-to-end row on this small foundation matrix; device-interval winners vary
  by row and run, so no timing domain is collapsed into another and `auto`
  remains on serial recompute. This is not yet a selector corpus: GQA/MQA,
  bias, softcap/window, f16/bf16 storage, workspace caps, wider and long-context
  shapes, and a committed stable selection corpus remain active. NVIDIA
  and ROCm are not applicable: the shared derivative semantics are unchanged
  and no CUDA/ROCm schedule is transferred.
- **2026-07-18 APPLE-ATTN-BWD-1 closure:** all three candidates now use
  query-streaming softmax/dP work rather than recomputing one softmax per output
  element. Atomic work owns one query row and confines contention to final
  dK/dV updates; serial gives one deterministic owner each KV head; split gives
  two deterministic owners one exact additional f32 dK+dV footprint and then
  reduces in fixed order. The status ABI shares forward's flattened-head
  MHA/GQA/MQA mapping, right-aligned causal and sliding-window masks, additive
  bias, and correctly differentiated logit softcap. Legacy rectangular causal
  callers retain their original zero-offset triangle. Native f16 and bf16
  inputs are read directly from two-byte Metal storage; dQ/dK/dV accumulate and
  return f32. Exact-device tests cover every route and dtype, batched/ragged
  MHA, GQA, MQA, bias, causal/noncausal windows, softcap, invalid-route
  rejection, deterministic repeats, and workspace limits.
  Two independent Apple7 reports contain 18 native, numerically valid, fully
  device-timed rows each. The committed
  `benchmarks/baselines/apple7_attention_backward_route_ledger.json` contains
  twelve timing-domain decisions. End-to-end selection promotes split-reduce
  for four rows, including causal `Sk=1025`, and atomic for two rows; paired
  median wins range from 27.8% to 67.3%, with 100% trial wins in both reports.
  Every device-interval row retains serial recompute. The legacy ledger records
  exact-device/shape/dtype/domain decisions, but strict v2 production ingestion
  now retains serial until those rows are re-recorded with current context;
  determinism and split-workspace policy remain enforced independently.
  **APPLE-ATTN-BWD-1 is closed.** NVIDIA and
  ROCm are not applicable to the Apple ABI, storage readers, schedules, or
  selector rows; shared derivative semantics remain unchanged.
- FP8/FP4/MX execution remains gated by the macOS 27 SDK/runtime surface. The
  compiler-side scale-layout and multi-plane contracts already exist; do not
  claim hardware execution until the public Metal tensor path runs natively.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus now
  carries additive compiler/resource, cold/warm, cache, and two-run stability
  evidence. Existing v1/v2 rows migrate without changing Apple selection, and
  no CUDA schedule or selector is transferred to Metal. Apple follow-up is to
  populate the same logical evidence fields from Metal-native counters during
  its own performance work; current Apple plan state is otherwise unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: not applicable to Apple
  execution. The fixes are confined to Ubuntu apt.llvm.org discovery,
  CUDA/NVVM lowering, and Linux NVIDIA/ROCm lit shell selection. No Apple IR,
  ABI, Metal schedule, numerical policy, or exact-device evidence changed.

## Completion definition

This plan reaches `closed` only when all of the following are true:

1. Host-free and compiler-artifact tests remain portable lanes. Apple promotion
   is owned by one local Metal 4 exact-device gate with two fresh-process
   correctness runs, an isolated paired-performance corpus, and a sealed packet
   pushed to the coordinating PR. Registered GitHub self-hosted runners are not
   used. Metal 3 is a non-blocking compatibility lane.
2. Every device test proves `native_gpu` placement on the intended route. A
   non-Darwin stub, NumPy fallback, symbol-presence check, or reference
   recomputation cannot earn a device pass.
3. Dtype, op, target, diagnostic, runtime-symbol, execution-state, and generated
   documentation registries are drift-gated. Every newly emitted diagnostic is
   registered and every live plan uses `open`, `landing`, or `closed`.
4. Portable Tile fixtures execute without test-authored physical fragments and
   select an Apple-owned layout/schedule from observed device capabilities.
5. Performance records use repeated medians after warmup, separate GPU/kernel
   time from end-to-end time where Metal counters permit it, and retain route,
   compiler, OS/SDK, device, residency, and resource evidence.
6. Paged KV and ReplaySSM pass the same non-identity, rollback, ordering, stress,
   and lifecycle closure used on CUDA/ROCm.
7. Production route changes consume only matching native-and-correct evidence;
   stale reports, reference rows, or records from another Apple GPU family
   cannot change selection.
8. The complete exact-device correctness lane passes twice from a fresh runtime
   image, and the isolated performance lane produces stable winner decisions.

## Apple-host preflight

Run decisive tests outside a sandbox in a fresh process. Record the exact host
before interpreting a skip or timing change:

```bash
sw_vers
system_profiler SPDisplaysDataType
xcodebuild -version
xcrun --sdk macosx --show-sdk-version
xcrun --find metal
python3 --version
git rev-parse HEAD
```

Also record Apple GPU family/capability probe output, macOS deployment target,
Metal language version, power mode, thermal state, and whether another process
is using the GPU. Metal 4 promotion requires a named Metal 4 host. Metal 3
coverage is compatibility-only and cannot promote a Metal 4 route; never
generalize a winner across Apple GPU families without a matching record.

### Use the dedicated LLVM/MLIR 23 prefix

The generic Homebrew `llvm` symlink is not this lane: it may resolve to a
different keg (currently LLVM 22) or be absent. Apple validation and
`build-apple` use the dedicated, pinned upstream `release/23.x` build at
`/opt/homebrew/llvm-23.1.0-rc1`; it must be built with
`LLVM_ENABLE_RTTI=ON`, or Tessera's pass and dialect typeinfo cannot link.
Before configuring or testing, set and validate this exact prefix:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
test -x "$TESSERA_LLVM23_PREFIX/bin/llvm-config"
test -d "$TESSERA_LLVM23_PREFIX/lib/cmake/mlir"
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

"$TESSERA_LLVM23_PREFIX/bin/llvm-config" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-opt" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-tblgen" --version
```

All three version commands must begin with `23.`. If either path check fails,
stop rather than falling back to `brew --prefix llvm` or AppleClang's system
libraries. To recreate the dedicated toolchain, install the Xcode Command Line
Tools first, then build it:

```bash
xcode-select --install                    # omit if already installed
brew update
brew install cmake ninja lit
git clone --depth 1 --branch release/23.x https://github.com/llvm/llvm-project.git /private/tmp/llvm-project-23
cmake -S /private/tmp/llvm-project-23/llvm -B /private/tmp/llvm-23-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/homebrew/llvm-23.1.0-rc1 \
  -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
  -DLLVM_TARGETS_TO_BUILD='AArch64;AMDGPU;NVPTX;X86' \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON
cmake --build /private/tmp/llvm-23-build --target install --parallel 8

export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
"$(brew --prefix lit)/bin/lit" --version
```

Do not use AppleClang's system LLVM libraries or mix the stable LLVM 22 keg
with this LLVM/MLIR 23 prefix. Record the upstream commit plus
`LLVM_ENABLE_RTTI=ON` in the build evidence.

For compiler artifacts, build the Apple backend and portable MLIR tools:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
cmake -S . -B build-apple -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang++" \
  -DLLVM_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$(brew --prefix lit)/bin/lit" \
  -DTESSERA_BUILD_APPLE_BACKEND=ON \
  -DTESSERA_BUILD_EXAMPLES=ON
cmake --build build-apple --target tessera-opt tessera-translate-mlir \
  TesseraAppleRuntime
export TESSERA_OPT="$PWD/build-apple/tools/tessera-opt/tessera-opt"
export PYTHONPATH="$PWD/python:$PWD"
```

Use the actual Ninja output path if the local LLVM/MLIR build lays out
`tessera-opt` differently. Build or load one fresh Apple runtime image for the
device lane; duplicate or stale dylibs invalidate symbol and placement proof.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and removes the obsolete Apple dialect property switch. The
portable Apple sources are assessed by the shared-source migration, but the
current WSL host cannot build or execute the Darwin/Metal runtime; Apple
LLVM/MLIR 23 build and exact-device parity are **follow-up required** on the
named Apple hosts.

## Ordered work

This is a live queue, not a historical checklist. `closed` means the item's
stated gate is met; `landing` means the principal implementation and evidence
landed but a deliberately narrower follow-up remains; `active` is the next
implementation/proof work; `blocked` names an external prerequisite.

| Order | ID | Status | Current state and next action |
|---:|---|---|---|
| 1 | APPLE-TEST-1 | **closed** | The centralized hardware boundary collects 976 of 15,374 unit nodes, the structural scan finds zero inline Apple capability gates, and portable marker/provenance ratchets reject classification drift. |
| 2 | APPLE-CI-2 | **closed** | The host-free compiler ownership gate is executable and green for the declared Apple capability set, and now validates the exact LLVM/MLIR runner-utils path for every CMake cache type. |
| 3 | APPLE-TEST-2 | **closed** | Fresh-runtime correctness (**850/850**), fallback-injection negatives, ordering/stress, and the serial measured lane are complete. |
| 4 | APPLE-REG-1 | **closed** | ABI/target-map/exact-device/Tile drift gates are registered and passing. |
| 5 | APPLE-TILE-1 | **closed** | The selected f16/bf16 simdgroup fragment and its two-run corpus meet the completion gate. MPS retaining every measured end-to-end row is the valid production decision. |
| 6 | APPLE-GEMM-1 | **closed** | The Apple7 paired ledger has stable decisions for every measured row: three promotions and nineteen incumbent retentions. New devices/routes require a new corpus. |
| 7 | APPLE-EPILOGUE-1 | **closed** | Supported f32/f16/bf16 fusions have native-oracle/resource proof and stable end-to-end selection; MPSGraph now has an explicitly labeled Metal 4 whole-graph envelope, pending a fresh two-run device-domain corpus. |
| 8 | APPLE-ATTN-FWD-1 | **closed** | Native forward variants, resident/cooperative candidates, full stated corpus, two-run route ledger, and timing-domain selection are complete. Do not reopen it for backward work. |
| 9 | APPLE-ATTN-BWD-1 | **closed** | Native f32/f16/bf16 MHA/GQA/MQA serial, atomic, and split-reduce routes share one oracle and explicit workspace/determinism policy. The stable two-run Apple7 ledger selects end-to-end routes per exact row and retains serial for every device-domain row. |
| 10 | APPLE-PAGED-KV-1 | **closed** | Direct resident page-table MLA attention and the staged peer share a non-identity oracle, causal/window boundary proof, transactional exhaustion/leak telemetry, and a paired two-domain Apple7 corpus. The legacy corpus records direct wins; strict production ingestion retains staged until those rows are re-recorded with the v2 context envelope. |
| 11 | APPLE-REPLAY-1 | **closed** | Resident inputs, ordered asynchronous ring submissions, native deterministic checkpoint folding plus same-command-buffer ring clearing, forced flush/rollback/partial-rejection ordering, backpressure/cleanup stress, and paired selector evidence are complete. Unstable device-domain evidence retains the fused-block incumbent. |
| 12 | APPLE-RETUNE-1 | **active** | One paired Apple7 corpus now covers grouped GEMM, MoE, reduction, resident KV, MLA, and Replay decode with honest timing scopes. Expand shapes/dtypes/transport bandwidth and obtain complete device intervals for composed/mapped routes. |
| 13 | APPLE-ROUTE-1 | **active** | Strict v2 ingestion replaces the literal table and rejects stale, reference, wrong-context, and wrong-domain rows with citations. Re-measure/migrate every legacy family ledger and separate package-subgraph evidence before closure. |
| 14 | APPLE-DTYPE-1 | **blocked — SDK** | FP8/FP4/MX native execution awaits the public macOS 27 Metal tensor path. Keep older-host int4/int8/f16/bf16 regression coverage. |
| 15 | APPLE-CI-1 | **closed** | The local Metal 4 release gate serializes the physical Mac without registering a GitHub runner, builds fresh LLVM/MLIR 23 compiler/JIT/runtime artifacts, records power/thermal/GPU-contention availability, rejects incomplete or skipped evidence, runs correctness twice, and seals paired device/end-to-end evidence. The retained `docs/audit/evidence/apple/metal4/20260718-b1ee875/` packet proves two clean 11-test Apple7 runs under Xcode 26.6, two 8-row route reports with four Metal 4 rows each, and an 8-decision two-domain ledger against commit `b1ee87591ec701dd06a156cad8449f6498ae0891`. Portable CI validates its hashes and contents. Metal 3 remains non-blocking compatibility coverage. |
| 16 | APPLE-E2E-1 | **queued** | After the shared `E2E-SPINE-1/-2` contracts land, make canonical compilation select the executable typed Apple GPU pipeline, package its native-library/function contract in the shared image and launch-descriptor schema, and give GA/EBM/linalg/PPO value-mode lanes an executable or explicit terminal state. Preserve Metal-owned schedules, placement policy, and selectors. |
| 17 | APPLE-CPU-E2E-1 | **queued** | Extend the canonical Apple CPU executable pipeline beyond rank-2 f32 matmul where stable Accelerate/LAPACK ABIs exist; classify every remaining lane explicitly rather than implying execution from artifact IR. |

## Canonical validation lanes

After APPLE-TEST-1 establishes complete marker coverage, the Apple host should
run these as independent commands:

```bash
# Host-free compiler, selector, validation, rejection, and fallback contracts.
python3 -m pytest tests/unit -q \
  -m "not hardware_apple_gpu and not performance"

# Apple compiler artifacts; this lane does not claim device execution.
python3 -m pytest tests/unit -q \
  -m "compiler_tool and not hardware_apple_gpu" --durations=50

# Native Metal correctness, twice from the same fresh build/runtime image.
python3 -m pytest tests/unit -q \
  -m "hardware_apple_gpu and not performance" --durations=100 \
  --junitxml=/tmp/apple-device-correctness.xml

# Measured lane: serial execution only.
python3 -m pytest tests/unit -q -n 0 \
  -m "hardware_apple_gpu and performance" --durations=0 \
  --junitxml=/tmp/apple-performance.xml

# Metal 4 promotion runs locally on the named Mac, never through a registered
# GitHub runner. Push the sealed packet into the coordinating PR; portable CI
# rejects zero selected tests, skips, hash drift, unknown GPU families,
# reference rows, missing device intervals, and incomplete two-domain ledgers.
bash scripts/run_apple_metal4_release_gate.sh \
  --publish-dir docs/audit/evidence/apple/metal4/<run-id>
```

The first focused parity and characterization loop is:

```bash
python3 -m pytest -q \
  tests/unit/test_apple_gemm_schedules.py \
  tests/unit/test_apple_sdpa_schedules.py \
  tests/unit/test_apple_gpu_metal4.py \
  tests/unit/test_apple_gpu_mpsgraph_lane.py \
  tests/unit/test_apple_gpu_resident_block_paged.py \
  tests/unit/test_ssm_apple_gpu_fused.py

python3 benchmarks/apple_gpu/benchmark_route_characterization.py \
  --matmul-shapes 64x64x64 128x256x64 257x129x65 256x256x256 \
  --softmax-shapes 64x64 128x257 256x256 \
  --reps 30 --output /tmp/apple-routes.json

python3 benchmarks/apple_gpu/benchmark_ssm_replay.py \
  --shapes 1x128x128 1x256x128 4x128x64 \
  --tokens 512 --capacity 16 --reps 20 \
  --output /tmp/apple-ssm-replay.json

python3 benchmarks/apple_gpu/record_hot_path_baseline.py --reps 20 --margin 2.0
```

Focused tests are edit-loop aids, not substitutes for the full marker lanes.
Files under `/tmp` are review artifacts only. Update a committed baseline or
route corpus only after two stable runs, explicit native-placement review, and
before/after resource inspection.

## Failure and benchmark evidence contract

For each failure or candidate record retain:

- test node, proof layer, Apple GPU family, macOS/SDK/compiler, dtype, shape,
  seed, selected route, and observed placement;
- fresh-runtime identity and whether the result reproduces alone, serially, and
  on the second clean run;
- named diagnostic or runtime error kind, compiler output, and Metal validation
  messages;
- maximum absolute/relative error, first failing index, non-finite policy, and
  the exact shared oracle;
- GPU/kernel time versus end-to-end time, warmup/repetition policy, cold compile
  or package-authoring cost, and command-buffer/dispatch count;
- residency and traffic bytes, threadgroup memory, occupancy/concurrency proxy,
  compiler statistics, and spill evidence available from the Metal toolchain;
- disposition: product defect, test-state defect, stale route/baseline, duplicate
  proof, unsupported capability, or exact external environment blocker.

Do not widen numerical tolerances or latency caps solely to turn the lane green.
Derive numerical policy from storage/accumulation semantics and performance
policy from stable repeated-median evidence.

## Next update

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16`: shared typed Tile IR now
permits logical `scale_a`/`scale_b` fragments only on NVFP4 MMA descriptors.
Apple has no enabled NVFP4 cooperative-matrix route, so this is follow-up
required at capability rejection only; no NVIDIA nibble, lane, scale-selector,
or OMMA mapping applies to Metal.

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16`: the shared `FusedRegion`
oracle now names bias/activation/residual order and emits registered
`E_FUSED_EPILOGUE_*` rejection diagnostics. Apple retains its architecture-owned
MSL/Metal 4 schedules. NVIDIA validates the complete 43-case supported
execution matrix; Apple independently validated its supported semantic order,
dtype matrix, residual guards, and diagnostics on the exact Metal host before
closing APPLE-EPILOGUE-1. The schedules and exact-device claims remain
architecture-specific.

Cross-backend sync `PR420-REVIEW-2026-07-17`: not applicable to Apple compiler
or runtime behavior. The scale-origin repair and canonical `fp16` alias are
confined to the SM120 NVIDIA fragment materializer/selector, and the bootstrap
ordering repair is confined to Ubuntu apt.llvm.org setup. No Apple IR, Metal
layout, dtype support, ABI, schedule, or exact-device claim changes.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18`: not applicable to Apple
runtime execution. The change adds a CUDA-owned packed NVFP4 ABI, SM120
HMMA/QMMA/OMMA kernels, CUDA-event evidence, and device-keyed NVIDIA selector
rows. It changes no portable dtype spelling, ScaleLayout, epilogue order, or
autotune schema. Apple remains SDK-gated for FP8/FP4 tensor execution and does
not inherit CUDA fragments, resource values, timings, or promotions.

Cross-backend sync `E2E-SPINE-2026-07-18`: Apple participates in the shared
native-image and launch-descriptor contract through **APPLE-E2E-1** and
**APPLE-CPU-E2E-1**. The shared work may select and package an existing typed
Apple pipeline, but it does not transfer CUDA/ROCm schedules, change Metal
placement, promote a route, or convert host-free compilation into exact-device
proof. Existing runtime and artifact routes remain available until their
canonical replacements meet all four proof layers on the named Apple host.
The behavior-neutral E2E-SPINE-0 foundation is complete: Apple CPU/GPU exact
targets now have total declared-pipeline ownership and truthful partial-B/
absent-C inventory rows; Apple runtime selection is unchanged. E2E-SPINE-1 is
also complete: Apple will consume the shared image/descriptor identity,
bindings, generic geometry, workspace, ordering, and diagnostics, while Metal
threadgroup schedules and placement remain Apple-owned. No Apple route or
exact-device status changes until APPLE-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no Metal hook and
does not change value-mode classification, MPSGraph/Metal placement, pipeline
cache policy, or selectors; APPLE-E2E-1 still owns native package production,
Apple registration/submission, comparison, cleanup, and Level-C proof.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
now consumes items **8, 9, 10, 11, 13, 14** as its Apple execution vehicle — it
adds candidates/state-types under existing items rather than opening new ones,
and **inherits this plan's evidence contract unchanged** (native `native_gpu`
placement, separate GPU/end-to-end timing-domain keys, two-run + ≥5% promotion,
forced binding-miss → `reference_cpu`). Concretely: channel-wise KDA/GDN decode →
**APPLE-REPLAY-1** (extend ReplaySSM / `SSMStateHandle` / `DeltaNetStateHandle`);
`sliding_window`/full mixer forward has closed its current **APPLE-ATTN-FWD-1**
scope; any new Sequence Mixer forward candidate requires a separately scoped
follow-up rather than silently reopening that item. `windowed_kv` +
uniform-block planner → **APPLE-PAGED-KV-1**;
chunkwise-scan inner GEMMs → **APPLE-RETUNE-1**; mixer arbiter → **APPLE-ROUTE-1**;
low precision → **APPLE-DTYPE-1** (stays SDK-gated — no NVFP4 cooperative-matrix on
Apple, so the executing FP4 proof is on NR2 Pro sm_120); mixer backward →
**APPLE-ATTN-BWD-1**. This is a direction pointer; it changes no Apple gate,
route, or exact-device claim here.

After the first Apple-host collection, replace the provisional marker count
with the migrated exact-device totals and append a failure table by execution
family and device generation. Set `plan_state: landing` once implementation or
test migration begins. Move this plan to the Apple archive only after every
completion gate is met.
