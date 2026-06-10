# Apple Backend Audit

This document consolidates Apple backend audit material across Metal runtime,
Metal 4, packaged kernels, command-buffer work, and Apple-specific performance.

## Finished

- **Apple CPU runtime:** Apple CPU artifacts execute through Accelerate.
- **Apple GPU runtime:** Apple GPU artifacts execute through MPS, MPSGraph, and
  custom MSL runtime paths.
- **Metal 4 surface:** Metal 4 probes and lanes exist for selected kernels.
- **Runtime ABI breadth:** generated ABI inventory reports a large Apple symbol
  surface and 84 Apple GPU families.
- **One-command-buffer substrate:** encode-session APIs and chain planning exist
  for resident decode-style chains.
- **Auto-batch substrate:** Apple GPU chain planner and `apple_gpu_ops` tracing
  pieces exist.
- **Conv2d encode-session lanes:** f32/f16/bf16 wrapper lanes exist.
- **Packaged-kernel lifecycle:** PK1-PK7 are proven against a real Apple sample
  package, including reflection, validation, argument layout, and dispatch.
- **Packaged-kernel *authoring* (PK8, 2026-06-02):** Tessera authors its own
  production `.mtlpackage` from the MPSGraph lane — build MPSGraph →
  compile to `MPSGraphExecutable` → `serializeToMPSGraphPackageAtURL:` → wrap
  the `manifest.json` MLLibrary layout. The authored package flows through the
  *existing* PK1-PK7 lifecycle, GPU output **bitwise-exact** vs numpy. No
  coremltools, no DXIL. MPSGraph packages expose positionally-indexed *unnamed*
  bindings, so PK8 adds `fill_input_at`/`read_output_at` index addressing
  (`tensorsByIndex`). **Generalized beyond matmul (2026-06-02):** a single
  `author_op` path covers the whole MPSGraph-lane op vocabulary by reusing the
  *runtime's own* graph builders (`mpsg_unary_node` / `mpsg_binary_node` /
  `mpsg_rowop_graph`) — so packaged kernels are numerically identical to the
  live path: **unary** (relu/sigmoid/tanh/softplus/silu/gelu/exp/log/sqrt/
  rsqrt/neg/abs), **rowop** (softmax/log_softmax), **norms** (rmsnorm +gamma,
  layer_norm +gamma+beta), **binary** (add/sub/mul/div/max/min/silu_mul). All
  match numpy at fp32 tol (~1e-7). C ABI: `tessera_apple_gpu_mlpkg_author_matmul`,
  `_author_op`, `_first_function_name`, `_fill_input_at`, `_read_output_at`.
  Python: `apple_mlpkg.author_matmul_package` / `author_op_package` (+ `AUTHOR_OPS`
  vocabulary) / `first_function_name`. Locked by
  `tests/unit/test_apple_mlpkg_pk8.py` (26 tests).
- **Fused multi-op packages (PK8b, 2026-06-02):** `author_chain_package`
  composes a whole chain into one serialized MPSGraph executable (one GPU
  dispatch): `matmul_softmax`, `matmul_softmax_matmul` (attention block),
  `rmsnorm_matmul`. C ABI `tessera_apple_gpu_mlpkg_author_chain`; all match
  numpy at fp32 tol. Tests in `test_apple_mlpkg_pk8.py`.
- **Graph IR → package hook (PK8a, 2026-06-02):**
  `compiler/apple_package_author.py` *recognizes* an MPSGraph-lane region in a
  real `GraphIRModule` (single op / matmul / fused chain) and authors the
  matching package. Two layers: `recognize()` (static-IR dims → `AuthorPlan`)
  and the shape-free `recognize_op()` (op/chain identity from op-names only —
  the live `@jit` IR carries no static dims) + `plan_from_shapes()`. Pure/
  device-free. Locked by `tests/unit/test_apple_package_author.py` (12 tests).
- **`@jit` → package emission wired into the compile path (PK8a wiring,
  2026-06-02):** `@jit(target="apple_gpu")` now runs the shape-free recognizer
  during compile and exposes `JitFn.recognized_package` (the `RecognizedOp`,
  or `None`). `JitFn.emit_package(out_path=None, example_args=...)` derives
  concrete fp32 shapes from the example tensors (the realistic AOT path,
  mirroring `aot.export(fn, *examples)`) and authors a real `.mtlpackage` that
  loads + dispatches through PK1-PK7 numpy-correct. Recognition is device-free
  (runs on any host); only `emit_package` touches the GPU, and only when
  called — no surprise authoring on decoration. Locked by
  `tests/unit/test_apple_jit_emit_package.py` (15 tests).
- **Compile-time auto-emit (PK8d, 2026-06-02):** `@jit(target="apple_gpu",
  emit_package=True | "<path>")` authors the package *at decoration* — no
  manual call — when the arg annotations are static integers (`Tensor[8, 6]` →
  `dim_names` all-numeric, read by `JitFn._static_input_shapes()`). Opt-in;
  symbolic-shape fns are a silent no-op (best-effort AOT, never a compile
  error); apple_gpu-only (raises otherwise). `emit_package()` (no examples)
  also uses this static-annotation path.
- **MSL-source dynamic-library AOT (Lane c, 2026-06-02):** the parallel AOT
  lane to MPSGraph packages — serialize Tessera's *MSL-source* custom kernels
  into a reloadable `.metallib` dynamic library. `apple_dylib.serialize_msl_dylib`
  / `load_dylib` over C ABI `tessera_apple_gpu_dylib_serialize` / `_load`,
  grounded in the SDK headers: `newLibraryWithSource:` (libraryType=Dynamic +
  installName) → `newDynamicLibrary:` → `serializeToURL:` →
  `newDynamicLibraryWithURL:` (reload). Round-trips a `[[visible]]`-function
  library to disk + back. Locked by `tests/unit/test_apple_dylib.py` (5 tests).
- **Package is a real execution path (PK8e, 2026-06-02):**
  `@jit(target="apple_gpu", dispatch_via_package=True)` now *executes through*
  the authored `.mtlpackage` — `__call__` routes to `_call_via_package`, which
  authors-on-first-use + caches an authored package path *and* a prepared
  PK1-PK7 pipeline per `(op, shape)`, fills inputs positionally, dispatches,
  and reshapes to `AuthorPlan.output_shape`. Repeated same-shape calls reuse
  both caches; a new shape adds one entry; non-fp32 / unrecognized calls fall
  back to the live MPS/MSL path (sentinel-guarded, authoring nothing). Verified
  numpy-equivalent for matmul + matmul→softmax across shapes. apple_gpu-only
  (raises otherwise). Locked by `tests/unit/test_apple_jit_emit_package.py`
  (21 tests). The package is no longer just an AOT side artifact — it is a
  selectable execution lane.
- **Package-lane perf benchmark (PK8f, 2026-06-02):**
  `benchmarks/apple_gpu/benchmark_package_lane.py` times the package lane vs the
  live MPS/MSL lane for the same jitted fn (steady-state, fp32, + a one-time
  `cold_author_ms`). Measured rule: **single matmul → live** (package +1.3–2.8×
  overhead); **fused `matmul→softmax` → package**, up to ~**14×** faster at
  256×256×256. Sample at `sample_package_lane_report.json`; findings table in
  the benchmarks README. This is the decision data behind the `dispatch_via_package`
  flag.
- **Production packaged-kernel coverage expanded (2026-06-02):** the
  Tessera-authored fixture set grew from 2 to **7 — one representative per
  authorable family**: matmul, silu (unary), softmax (rowop), rmsnorm
  (weighted norm), and the chains matmul_softmax / matmul_softmax_matmul /
  rmsnorm_matmul. Each is committed, drift-gated, and numerically dispatch-
  validated vs numpy (`test_apple_mlpkg_pk8c.py`). **Crucial placement fix:**
  the production fixtures live in `tests/fixtures/apple_gpu/production/`, NOT
  the top-level fixtures dir — the PK1-PK7 lifecycle tests `iterdir`-scan the
  top level and compile whatever `.mtlpackage` they find, and piling more
  packages there drove the per-process MTL4-ML-compile count past a Metal
  abort ceiling (`newMachineLearningPipelineState` → `MTLReportFailure` →
  `abort()`, a hard SIGABRT). Isolating them keeps the lifecycle tests on the
  Apple sample only.
- **Auto-route default — evaluated and deliberately NOT flipped (PK8h,
  2026-06-02):** making `dispatch_via_package="auto"` the unconditional
  default for apple_gpu was tested and **rejected** — under suite volume
  (hundreds of jitted chains each authoring an ML pipeline) it SIGABRTs the
  Metal runtime. Auto-routing stays opt-in, exposed two safe ways: per-fn
  (`@jit(..., dispatch_via_package="auto")`) and globally via
  `TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE=1`. `_resolve_dispatch_via_package`
  centralizes the policy; locked by `test_apple_jit_emit_package.py`.
- **Single runtime loader + auto-route (PK8g, 2026-06-02):** (1) **Duplicate-
  dylib fix** — `_apple_gpu_dispatch` is now the single Apple-GPU runtime
  loader (env-var → CMake build → from-source, with a newest-symbol staleness
  gate); `runtime._load_apple_gpu_runtime` delegates to it. Previously the
  ctypes/`bind_symbol` lane and the MPS lane each compiled + `dlopen`'d their
  own dylib (versioned vs unversioned) into the same cache dir, so a process
  using both defined every ObjC class twice (the `TesseraMlpkgPipeline`
  duplicate-class warning). Now one image is shared. (2) **Auto-route** —
  `dispatch_via_package="auto"` applies the benchmark verdict automatically:
  fused chains → package lane, single matmul/unary → live lane. Locked by
  `tests/unit/test_apple_jit_emit_package.py` (24 tests).
- **Committed production packages + manifest rows (PK8c, 2026-06-02):** two
  Tessera-authored `.mtlpackage` fixtures committed under
  `tests/fixtures/apple_gpu/tessera_authored_*` (matmul 8×8×8, fused
  matmul→softmax 4×6×5; 16K each), regenerable via
  `scripts/author_apple_packages.py`. `PACKAGED_PRODUCTION_KERNELS` now has 2
  rows (`apple_binding_spec=None` — MPSGraph packages are positionally bound).
  Committed artifacts load + dispatch + match numpy; load failure on an older
  macOS is treated as a portability skip. Locked by
  `tests/unit/test_apple_mlpkg_pk8c.py` + `test_apple_packaged_manifest.py`.
- **GA/EBM Apple specialization:** fused Apple kernels and benchmarks exist for
  important GA/EBM paths.
- **Descriptor-driven dispatch — single-source envelope (P1, 2026-06-09):**
  the Apple GPU envelope tables (21 lane sets + opcode dicts, 166 runtime ops)
  are single-sourced in `compiler/apple_gpu_envelope.py`; `driver.py` and
  `runtime.py` import them (their literal duplicate tables deleted), the
  runtime per-op dispatcher is a lane→handler table built from
  `APPLE_GPU_LANE_BY_OP` (the 200-line elif chain is gone), and
  `AppleKernelDescriptor` carries the new `lane` field. The C++ Tile→Apple
  recognizer's hand-maintained `kRuntimeOps` is now **generated** from the same
  registry (`scripts/generate_apple_runtime_ops_table.py` →
  `apple_runtime_ops.inc`), which also closed glass-jaw #10 (projection +
  reduction ops now tag `metal_runtime`); the drift gate covers the full
  envelope. Oracles: `test_apple_gpu_envelope_dispatch.py` (lane-table vs the
  legacy elif routing, hard-pinned per lane),
  `test_apple_runtime_ops_table_in_sync.py` (table),
  `test_apple_gpu_tile_pass_status_matches_envelope.py` (real tessera-opt over
  all 166 ops), phase8 lit 57/57 non-skipped, full apple_gpu behavior suite
  green.
- **Feature-table-driven selection (P2, 2026-06-09):** the three remaining
  hard-coded decisions now consult `apple_target`: bf16 native-vs-host-upcast
  (`apple_supports_native_bf16` — live `MTLGPUFamily` probe wins, static arch
  default off Metal; consumed by GEMM + conv2d dispatch), fused-chain /
  flash-attn head_dim ceilings (`apple_fused_chain_score_cap` derives 256 from
  the 1 KiB per-thread fp32 stack budget; all 12 runtime cap checks point at
  it), and threads-per-row (`apple_threadgroup_threads_per_row` =
  `simdgroup_size`, ready for the next .mm change). Locked by 7 new tests in
  `test_apple_feature_limit_lowering.py`.
- **Perf ratchets — manifest-attached benchmarks + CI gate (P2, 2026-06-09):**
  Apple GPU hot-path manifest rows (matmul / softmax / rmsnorm / flash_attn /
  bmm / conv2d) carry `benchmark_json` →
  `benchmarks/baselines/apple_gpu_hot_paths.json` (recorded live on this
  M-series machine via `benchmarks/apple_gpu/record_hot_path_baseline.py`;
  caps = median × 2.0); all 7 `PACKAGED_PRODUCTION_KERNELS` rows carry the
  PK8f package-lane report. `benchmarks/perf_gate.py` gained `--ratchet`
  (`evaluate_ratchet`: regression + coverage failure). Locked by
  `test_apple_gpu_perf_ratchet.py` (manifest linkage + evaluator + slow live
  re-time gate).
- **auto_batch polish — auto-detection + emission skip (P3, 2026-06-09):**
  `@jit(target="apple_gpu")` `auto_batch` now defaults to `None` (auto-detect).
  A recognized decode chain — a body of ≥2 encode-eligible ops and nothing else,
  detected by a conservative AST whitelist scan (`_recognized_decode_chain`:
  rejects arithmetic on op results, subscripts, control flow, non-op calls) —
  turns the one-command-buffer route on by default. When the route is on, the
  AST Graph IR that the tracer never reads is no longer emitted (an
  `_AutoBatchSkipEmission` sentinel installs the deferred state; emission still
  runs when `emit_package` needs the recognized region). Explicit
  `auto_batch=True`/`False` override detection. Locked by
  `test_apple_gpu_jit_auto_batch_autodetect.py` (19 tests: detection
  truth-table, emission-skip introspection, encode-name-vs-registry drift gate,
  misuse guards).

## Still Open

The 2026-06-09 sprint closed the four remaining themes — descriptor-driven
dispatch, feature-table-driven selection, perf ratchets, and the auto_batch
polish (all in Finished). **No Apple-compiler items remain open;** the standing
constraints/decisions below are by design, not tasks.

---

> **Corrected — packaged kernels are NOT open work.** The earlier
> "Production packaged kernels are empty / 0 `status=packaged` rows" bullet is
> **stale**: `PACKAGED_PRODUCTION_KERNELS` has **7 rows** backed by 7 committed,
> dispatch-validated fixtures under `tests/fixtures/apple_gpu/production/` (PK8c).
> The whole author → recognize → wire → auto-emit → execute → benchmark →
> auto-route arc closed 2026-06-02 (PK8–PK8h, see Finished). The ~60 lines of SDK
> lane-grounding (`metal-package-builder`, `serializeToMPSGraphPackageAtURL:`,
> Lanes 1/2/3) were the justification for that *now-completed* work and have been
> moved out of Still Open. Two standing items survive as **constraints/decisions,
> not open tasks**:
> - **Auto-route is opt-in by design.** Making the package lane the unconditional
>   apple_gpu default SIGABRTs the Metal runtime under suite volume (hundreds of
>   ML-pipeline compiles → `newMachineLearningPipelineState` abort ceiling).
>   Exposed safely per-fn (`dispatch_via_package="auto"`) and globally
>   (`TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE=1`).
> - **Metal Shader Converter / DXIL (Lane 2) is out of scope.** Tessera emits MSL;
>   the dylib AOT lane (`apple_dylib`, Lane 1/c) + MPSGraph packages (Lane 3)
>   cover the need. Revisit only if a DXIL-import requirement ever appears.

## Next Work

1. ~~Promote binding specs / descriptors to all Apple kernel families.~~
   **Landed 2026-06-02** — `AppleKernelDescriptor` unifies the dispatch
   contract across every family. ~~Follow-on: have runtime dispatch *consume*
   the descriptor instead of pattern-matching op names.~~ **Landed 2026-06-09**
   — envelope single-sourced in `apple_gpu_envelope.py`; runtime dispatch is
   lane-table-driven; C++ `kRuntimeOps` generated from the registry.
2. ~~Add Apple kernel descriptors for MPSGraph, MSL, Metal 4, packaged, and
   encode-session paths.~~ **Landed 2026-06-02** (same descriptor surface).
3. ~~Wire Apple feature limits into schedule/tile/kernel selection.~~
   **First wire-up landed 2026-06-02** (tiled matmul→softmax N cap derived
   from threadgroup-memory budget). ~~Follow-on: drive bf16 gating / head_dim
   ceilings / threadgroup sizing from the same feature table.~~ **Landed
   2026-06-09** (`apple_supports_native_bf16`, `apple_fused_chain_score_cap`,
   `apple_threadgroup_threads_per_row`).
4. ~~Finish `@jit(target="apple_gpu", auto_batch=True)` canonical
   one-command-buffer route.~~ **Landed 2026-06-02** — canonical
   `tessera.ops.*` decode through `@jit` runs on one cb, with
   `max_ops_per_cb` chunking threaded through the decorator. ~~Follow-on
   (optional): make auto_batch bypass unused Graph-IR emission; consider
   auto-detection so the route is on by default for recognized decode loops.~~
   **Landed 2026-06-09** — `auto_batch` defaults to `None` (auto-detect): a
   recognized decode chain (≥2 encode-eligible ops and nothing else, via the
   `_recognized_decode_chain` AST scan) turns the route on by default, and the
   unused AST Graph IR emission is skipped (an `_AutoBatchSkipEmission` sentinel
   lands the deferred state; `JIT_APPLE_GPU_AUTO_BATCH` diagnostic). Explicit
   `True`/`False` override detection. Locked by
   `test_apple_gpu_jit_auto_batch_autodetect.py` (detection truth-table +
   emission-skip + registry drift gate).
5. ~~Author production packaged kernels from the MPSGraph lane.~~ **Landed
   2026-06-02 (PK8)** — `author_matmul_package` builds → compiles →
   `serializeToMPSGraphPackageAtURL:` → wraps `manifest.json`, and the
   authored package dispatches through PK1-PK7 bitwise-exact vs numpy.
   See the "Finished" entry above; `tests/unit/test_apple_mlpkg_pk8.py`.
   **Follow-ons:** (a) ~~generalize beyond matmul~~ **done — `author_op`
   (unary/rowop/norm/binary) + `author_chain` (fused chains, PK8b)**;
   ~~graph→package compiler hook~~ **done — `apple_package_author.recognize` /
   `author_package_from_graph_ir` (PK8a)**. (b) ~~commit a real `.mtlpackage`
   + `status="packaged"` row~~ **done — 2 Tessera-authored fixtures +
   `PACKAGED_PRODUCTION_KERNELS` rows (PK8c)**. ~~wire the recognizer into the
   actual `@jit` compile path~~ **done — `JitFn.recognized_package` +
   `emit_package(example_args=...)` (PK8a wiring)**. ~~(c) MSL-source
   dynamic-library AOT chain~~ **done — `apple_dylib` serialize/reload (Lane c)**.
   ~~automatic emit during compile~~ **done — `@jit(emit_package=True)` +
   static-annotation shape specialization (PK8d)**. ~~dispatch an authored
   package as the live execution path~~ **done — `dispatch_via_package=True`
   routes `__call__` through a per-shape package + pipeline cache (PK8e)**.
   The full author → recognize → wire → auto-emit → execute arc is closed;
   the package is a selectable execution lane, not just an AOT artifact.
   ~~a benchmark comparing the package lane vs the live MPS/MSL lane~~
   **done — `benchmarks/apple_gpu/benchmark_package_lane.py` (PK8f)**. Verdict:
   **single matmul → live wins** (MPS optimal; package adds 1.3–2.8× ML-encoder
   overhead); **fused chains like `matmul→softmax` → package wins**, ≈**14×
   faster at 256×256×256** (MPSGraph fuses the whole graph; the live MSL
   softmax over a materialized score matrix scales poorly). Cold authoring
   ~11 ms/shape, amortized by the per-shape pipeline cache. ~~turn that rule
   into an automatic heuristic~~ **done — `dispatch_via_package="auto"` (PK8g)
   routes only fused chains (`kind=="chain"`) through the package and keeps
   single matmul / unary ops on the live lane, exactly the measured-optimal
   split.** The author → recognize → wire → auto-emit → execute → benchmark →
   auto-route arc is fully closed.
6. ~~Attach benchmark metadata for Apple hot paths such as matmul, matmul
   epilogues, conv2d, decode chain, and packaged kernels.~~ **Landed
   2026-06-09** — manifest rows carry `benchmark_json`; recorded ratchet
   baseline + `perf_gate --ratchet` + `test_apple_gpu_perf_ratchet.py`.

## Source Material Consolidated

- `archive/2026_06_01_apple_gpu_chain_audit.md`
- `archive/single_command_buffer_decode_plan.md`
- `archive/apple_ga_ebm_native_execution_gap.md`
- `../../compiler/archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`

