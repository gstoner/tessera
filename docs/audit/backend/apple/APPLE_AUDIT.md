# Apple Backend Audit

This document consolidates Apple backend audit material across Metal runtime,
Metal 4, packaged kernels, command-buffer work, and Apple-specific performance.

## Finished

- **Apple CPU runtime:** Apple CPU artifacts execute through Accelerate.
- **Apple GPU runtime:** Apple GPU artifacts execute through MPS, MPSGraph, and
  custom MSL runtime paths.
- **Metal 4 surface:** Metal 4 probes and lanes exist for selected kernels.
- **Runtime ABI breadth:** the drift-gated `docs/audit/generated/runtime_abi.md`
  is the count authority (currently a large Apple symbol surface across 109 Apple
  GPU kernel families); do not hard-code the number here — read the dashboard.
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
- **GPU dispatch error channel (2026-06-10):** the Apple GPU runtime exposes a
  thread-local last-error (`tessera_apple_gpu_last_error_kind` / `_message` /
  `_clear_last_error`) set at the shared command-buffer choke point
  (`commit_and_wait_with_timeout`, ~72 callers); the matmul / unary / binary /
  rowop / bmm Python lanes arm + consume it (`_apple_gpu_run_checked`) so a
  silent internal GPU failure funnels (`TESSERA_STRICT_DISPATCH` raises) +
  recomputes on host instead of returning a garbage buffer. See
  `../../compiler/CODE_AUDIT_2026_06_10.md`.
- **Numerical-proof discipline for Apple GPU fused rows (2026-06-10):** 21
  Apple GPU `fused` ops (GA/Clifford ×17, complex ×2, EBM ×2) that had genuine
  dedicated GPU execute-compare tests but no wired fixture now carry their
  `execute_compare_fixture` in the backend manifest (each verified to run the
  op's kernel and `assert_allclose` vs a numpy/GA reference, re-run green on
  this Metal host). Fixed a latent bug where `manifest_for`'s clifford/ebm/
  complex early-returns bypassed `_attach_numerical_fixtures`, so those domains
  could never have received a fixture. New manifest-level gate
  `test_apple_gpu_numerical_proof_discipline.py` freezes the remaining
  no-execute-compare allowlist (`ebm_self_verify` / `ebm_langevin_step` /
  `kv_cache_read`) and locks `hardware_verified ⟹ fixture`.

## Open Work

_No open Apple-compiler items._ The 2026-06-02 PK8–PK8h arc closed
packaged-kernel authoring end to end, and the 2026-06-09 sprint closed the four
remaining themes (descriptor-driven dispatch, feature-table-driven selection,
perf ratchets, auto_batch polish). Everything previously tracked under "Still
Open" / "Next Work" has landed and is summarized under **Finished** above. The
only remaining frontier is cross-backend **real-hardware** proof for NVIDIA /
ROCm / Metalium — tracked in the per-platform audits, not here.

## Hardware capability reference (grounded 2026-06-17)

Grounded against the Apple **Metal Feature Set Tables** (rev 2026-05-21, the authoritative
"when each feature is available" doc — Decision #27). GPU-family map: **M1 = Apple7**, M2 = Apple8,
M3/M4 = Apple9, M5 = Apple10 (A14 = Apple7; all M-series support Metal 3 & 4). A feature is available
when the Metal column (3 / 3&4 / 4) you target appears *and* the device family ≥ the Apple column.

**This dev Mac (M1 Max = Apple7) already has the full Metal-4 ML *compute* surface:** `bfloat`/bf16
(Apple6), **`simdgroup_matrix`** SIMD-scoped matrix multiply (Apple7), SIMD-scoped reductions (Apple7),
**`MTLTensor`** + **machine-learning encoding** (Metal 4, Apple7), float atomics (Apple7). What's gated
*past* Apple7 is overwhelmingly graphics/sparse/texture, **not** compute: full 64-bit atomics (Apple9),
lossy/universal texture compression (Apple8/10), SIMD shift-and-fill (Apple8), sampler min-max
reduction / depth-bounds / sampler-LOD-bias (Apple10), placement sparse (Apple8). So Apple-backend ML
work is **not** blocked waiting on newer silicon.

**The one ML gap on Apple7 is toolchain, not hardware:** the FP8/FP4/MX `MTLTensor` *dtypes*
(`MetalFloat8E4M3/E5M2`, `MetalFloat4E2M1`, `MetalFloat8UE8M0` block-scale) + multi-plane scale-tensor
machinery are **macOS-27.0 SDK-gated**, not hardware-gated — the same M1 Max gets them on a 27.0 SDK
(this machine is macOS 26.5.1). The hardware-free Tessera bridge for this is
`python/tessera/compiler/microscaling.py`.

**MLX** (`ml-explore`) is the production Apple-Silicon ML framework and Tessera's Apple-lane reference:
unified memory (no CPU↔GPU copies) + custom MSL kernels + `simdgroup_matrix` GEMM — the same path
Tessera takes. Reference/numeric oracle only, **never a runtime dependency** (Decision #23).

*Architecture finding (MLX source, 2026-06-17):* MLX's Metal backend mirrors Tessera's structure
(per-op host dispatch; a JIT-MSL + precompiled dual path; a fusion compiler `compiled.cpp`; custom
MSL kernels). Crucially, **MLX does NOT use MPS for GEMM** — it ships its own templated MSL library
**"steel"** (`kernels/steel/{gemm,attn,conv}`) built on `metal::simdgroup_matrix` +
`simdgroup_multiply_accumulate`, structured as a tile-fragment IR (`mma.h` fragments + `loader.h` tile
loads + `transforms.h` fused `axpby` epilogue). The production framework chose custom
`simdgroup_matrix` MSL over MPS for the fusion control a compiler needs — concrete support for
Tessera's "clear-MPS / native MTL4 matmul" direction, and a blueprint for a steel-like
simdgroup_matrix GEMM lane as the peer of NVIDIA `ct::mma`/Tile-IR and AMD `rocdl.wmma`.

*MLX mining pass (2026-06-17) — 5 areas to harvest as Tessera POLICY (not copied kernels):*
**(1) GEMM/attention autotune seeds** — MLX's `matmul.cpp` tile heuristic (`bm/bn/bk/wm/wn` keyed on
device-class × dtype × transpose × size + `align_M/N/K` / `do_axpby` / `swizzle_log` function
constants) is now encoded as Tessera **schedule candidates** in
`python/tessera/compiler/apple_gemm_schedules.py` (`MLX_SEED_TILES` sweep set + `select_seed_tile`
seed + `schedule_axes_for`), every tile feeding `emit_steel_gemm_msl`. **SDPA seeds landed** →
`apple_sdpa_schedules.py` (`scaled_dot_product_attention.cpp`: full NAX/Metal vs vector-decode vs
vector-2pass routing; `bq/bk/bd/wm/wn` block; `has_mask/do_causal/has_sinks`+GQA specialization).
**(2) Feature probes — NAX landed** → `apple_target.py` `nax_available()` (macOS 26.2+ AND
arch_gen≥(arch=='p'?18:17), MLX device.cpp:899) + the `AppleProbeKind` compile-required-vs-runtime
vocabulary. *(Grounded bug found, spun off task_fbb4d13b: `_APPLE_FEATURES[APPLE7]` wrongly marks
simdgroup_matrix/bfloat `not_supported` vs the Feature Set Tables + the in-repo simdgroup_matrix
emitters; M4/M5 family map also off.)* **(3) Allocator/residency** — MLX's memory/cache/wired-limit + heap
+ residency-set policy as a pool checklist. **(4) Microscaling oracle** — MLX's software-packed
`fp4.h`/`fp8.h` MSL as a bridge for the fp8/fp4/MX/NVFP4 contract before native MTLTensor (macOS-27.0
gated). **(5) Custom MSL hook** — `mx.fast.metal_kernel` as a debug/escape-hatch reference.

## Standing Decisions (by design, not tasks)

- **Auto-route is opt-in.** Making the package lane the unconditional apple_gpu
  default SIGABRTs the Metal runtime under suite volume (hundreds of ML-pipeline
  compiles → `newMachineLearningPipelineState` abort ceiling). Exposed safely
  per-fn (`dispatch_via_package="auto"`) and globally
  (`TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE=1`).
- **Metal Shader Converter / DXIL (Lane 2) is out of scope.** Tessera emits MSL;
  the dylib AOT lane (`apple_dylib`, Lane 1/c) + MPSGraph packages (Lane 3)
  cover the need. Revisit only if a DXIL-import requirement appears.

## Source Material Consolidated

- `archive/2026_06_01_apple_gpu_chain_audit.md`
- `archive/single_command_buffer_decode_plan.md`
- `archive/apple_ga_ebm_native_execution_gap.md`
- `../../compiler/archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`

