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

## Still Open

- **Binding specs / descriptors — unified surface landed (2026-06-02);
  runtime still pattern-driven.** `compiler/apple_kernel_descriptor.py`
  synthesizes a declarative `AppleKernelDescriptor` (family / status / dtypes /
  runtime_symbol / shape_envelope / encode-eligibility / packaged binding spec)
  for **every** Apple GPU kernel family — mps, msl, mpsgraph, conv, reduction,
  projection, linalg, encode_session, packaged — from the manifest + driver
  envelope + encode registry (no duplicated truth). Locked by
  `tests/unit/test_apple_kernel_descriptor.py`. Residual: Target IR / runtime
  dispatch don't *consume* the descriptor yet (they still pattern-match); the
  packaged reflection-level binding spec remains packaged-only by nature
  (other families expose no buffer-index reflection).
- **Feature limits — first wire-up landed (2026-06-02); broader use open.**
  The threadgroup-tiled matmul→softmax N ceiling is now derived from the
  per-arch threadgroup-memory budget
  (`apple_target.apple_threadgroup_tiled_softmax_n_cap`) instead of a magic
  `8192`, consulted by the runtime dispatcher and self-scaling on a
  higher-memory SKU. Locked by `tests/unit/test_apple_feature_limit_lowering.py`.
  Residual: other selection decisions (bf16 gating, head_dim ceilings,
  threadgroup sizing) still use ad-hoc constants rather than the feature table.
- **One-CB path — canonical route landed (2026-06-02); residual polish.**
  `@jit(target="apple_gpu", auto_batch=True)` now runs a decode body written
  with the canonical `tessera.ops.*` surface on one command buffer per encode
  segment (the global interception shim forwards to the encode session under
  an active trace), and `max_ops_per_cb` threads the chunking budget through
  the decorator. Locked by `tests/unit/test_apple_gpu_jit_auto_batch_canonical.py`.
  Residual: it is **opt-in** (`auto_batch=True`), not auto-detected; and an
  auto_batch body still pays Graph-IR-emission overhead it never uses, and
  must keep shape kwargs as literals/args (a general `@jit` AST-lowering
  constraint, not auto_batch-specific).
- **Production packaged kernels are empty — but the authoring path is
  on-host and NOT blocked (corrected 2026-06-02, third revision).** PK1-PK7
  prove the full load → reflect → validate → dispatch lifecycle against
  Apple's licensed **sample** `matrix-multiplication.mtlpackage`, but there
  are **0** live `status="packaged"` manifest rows. **Two earlier "blocked /
  no authoring path" claims were both wrong — reached from memory without
  consulting the SDK (exactly the Decision #27 anti-pattern, three
  recurrences now).** The verified reality:
  - **`metal-package-builder` ships on this host** at
    `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal-package-builder`
    — `metal-package-builder [-ml] -o out.mtlpackage <coremlpackage>` authors
    a `.mtlpackage` from a CoreML package.
  - **The sample `.mtlpackage` is just** `manifest.json`
    (`"pkgtype":"MLLibrary"`, `"content":{"mpspkgname":"library.mpsgraphpackage"}`)
    **wrapping a serialized MPSGraph** —
    `library.mpsgraphpackage/{model_0.mpsgraph, reflection.fb, manifest.plist}`.
  - **MPSGraph serializes to exactly that package format on this host:**
    `MPSGraphExecutable serializeToMPSGraphPackageAtURL:descriptor:`
    (`MPSGraphExecutable.h:205`, Swift `serialize(package:descriptor:)`,
    macOS 14+), with `initWithMPSGraphPackageAtURL:` /
    `initWithCoreMLPackageAtURL:` to reload.

  **This aligns with Tessera's *existing* MPSGraph lane.** The Tessera-native
  authoring chain needs no coremltools and no DXIL: (1) build the `MPSGraph`
  Tessera already builds in `apple_gpu_runtime.mm` for its MPSGraph-lane ops;
  (2) compile → `MPSGraphExecutable`; (3) `serializeToMPSGraphPackageAtURL:`;
  (4) wrap into a `.mtlpackage` — either via `metal-package-builder -ml`, or
  by writing the trivial 11-line `manifest.json` + dropping the
  `.mpsgraphpackage` dir beside it; (5) PK1-PK7 already load/reflect/dispatch
  it (and the package's `reflection.fb` is what PK6's reflection-validation
  gate checks). **So Lane 3 is achievable end-to-end on this Mac.** The
  remaining work is real engineering (wire the serialize call into the
  MPSGraph runtime, emit the package, add a numerical-compare fixture, declare
  the `status="packaged"` row) — not a toolchain block. Tracked as the
  next-work item below.

  **Parallel AOT lane (also verified, also not blocked) — runtime dynamic
  library for the MSL-source kernels:** (1) `MTLDevice
  newLibraryWithSource:options:error:` (`MTLDevice.h:786`); (2) `MTL4Compiler
  newDynamicLibrary:error:` (`MTL4Compiler.h:117`, Swift
  `makeDynamicLibrary(library:)`); (3) `MTLDynamicLibrary serializeToURL:error:`
  (`MTLDynamicLibrary.h:77`); (4) reload via `newDynamicLibraryWithURL:`
  (`MTL4Compiler.h:126`). (`MTLBinaryArchive.serialize(to:)` +
  `MTL4Compiler.pipelineDataSetSerializer` / `lookupArchives` give a parallel
  pipeline-state cache route.)

  **Three distinct compiled-artifact lanes (grounded 2026-06-02 in SDK
  headers + the Metal Shader Converter doc):**
  - **Lane 1 — runtime dynamic library (usable today):** the
    `newLibraryWithSource:` → `newDynamicLibrary:` → `serializeToURL:` →
    `newDynamicLibraryWithURL:` chain above. Tessera already emits MSL, so
    this is the directly-actionable AOT/precompiled-kernel path; no external
    toolchain.
  - **Lane 2 — Metal Shader Converter (OUT OF SCOPE — decided
    2026-06-02):** `libmetalirconverter` / `metal-shaderconverter` convert
    **DXIL** (DirectX LLVM-IR bytecode) → `.metallib` (`IRObjectCreateFromDXIL`
    → `IRCompilerAllocCompileAndLink` → `IRObjectGetMetalLibBinary`), and
    `metal-tt` finalizes → `.gpubin`. This is Apple's **DirectX-port** path
    (root signatures, top-level Argument Buffers, graphics/RT stages). **We do
    not need DXIL support** — Tessera emits MSL and Lane 1 covers AOT, so this
    lane is deliberately not pursued. (Toolchain is also not installed on this
    host: `/opt/metal-shaderconverter` absent. Download from
    developer.apple.com/metal only if this decision is ever reversed.)
  - **Lane 3 — `.mtlpackage` ML package (ACHIEVABLE on this host; rides the
    MPSGraph lane):** PK1–PK7 already consume/dispatch. Authoring is on-host:
    `MPSGraphExecutable serializeToMPSGraphPackageAtURL:` → wrap into
    `.mtlpackage` (trivial `manifest.json` + dir, or `metal-package-builder
    -ml`). Built from the same MPSGraph Tessera already constructs for its
    MPSGraph-lane ops. Remaining work is engineering, not a toolchain block —
    see next-work item 5.
- **Target IR does too much.** Apple source strings, fusion recognition, and
  runtime dispatch decisions need a descriptor-backed backend registry.
- **Performance gates are uneven.** Benchmarks exist, but manifest-attached
  benchmark metadata and ratchets are not systematic.

## Next Work

1. ~~Promote binding specs / descriptors to all Apple kernel families.~~
   **Landed 2026-06-02** — `AppleKernelDescriptor` unifies the dispatch
   contract across every family. Follow-on: have runtime dispatch *consume*
   the descriptor instead of pattern-matching op names.
2. ~~Add Apple kernel descriptors for MPSGraph, MSL, Metal 4, packaged, and
   encode-session paths.~~ **Landed 2026-06-02** (same descriptor surface).
3. ~~Wire Apple feature limits into schedule/tile/kernel selection.~~
   **First wire-up landed 2026-06-02** (tiled matmul→softmax N cap derived
   from threadgroup-memory budget). Follow-on: drive bf16 gating / head_dim
   ceilings / threadgroup sizing from the same feature table.
4. ~~Finish `@jit(target="apple_gpu", auto_batch=True)` canonical
   one-command-buffer route.~~ **Landed 2026-06-02** — canonical
   `tessera.ops.*` decode through `@jit` runs on one cb, with
   `max_ops_per_cb` chunking threaded through the decorator. Follow-on
   (optional): make auto_batch bypass unused Graph-IR emission; consider
   auto-detection so the route is on by default for recognized decode loops.
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
   ~11 ms/shape, amortized by the per-shape pipeline cache. The arc is fully
   closed — correctness *and* a measured lane-selection rule. **Next genuine
   step (open):** turn that measured rule into an *automatic* heuristic so
   `dispatch_via_package` becomes a smart default — route fused chains
   (`kind=="chain"`) through the package, keep single matmul on the live lane —
   instead of a blanket per-fn opt-in.
6. Attach benchmark metadata for Apple hot paths such as matmul, matmul
   epilogues, conv2d, decode chain, and packaged kernels.

## Source Material Consolidated

- `archive/2026_06_01_apple_gpu_chain_audit.md`
- `archive/single_command_buffer_decode_plan.md`
- `archive/apple_ga_ebm_native_execution_gap.md`
- `../../compiler/archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`

