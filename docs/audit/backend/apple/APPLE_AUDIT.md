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
5. **Author production packaged kernels from the MPSGraph lane (achievable
   on this host — see corrected analysis above).** Concrete steps: (1) in
   `apple_gpu_runtime.mm`, compile the MPSGraph Tessera already builds for an
   MPSGraph-lane op into an `MPSGraphExecutable`; (2) call
   `serializeToMPSGraphPackageAtURL:descriptor:` (`MPSGraphExecutable.h:205`);
   (3) wrap into a `.mtlpackage` (write the 11-line `manifest.json` +
   `.mpsgraphpackage` dir, or shell out to `metal-package-builder -ml`);
   (4) feed it through PK1-PK7 (load → reflect → validate → dispatch); (5) add
   a numerical-compare fixture vs the live MPSGraph path and declare the
   `status="packaged"` `BackendKernelEntry` row. **Parallel lane (b):** the
   MSL-source runtime dynamic-library AOT chain (`newLibraryWithSource:` →
   `newDynamicLibrary:` → `serializeToURL:` → `newDynamicLibraryWithURL:`,
   plus `MTLBinaryArchive`/`pipelineDataSetSerializer`/`lookupArchives`).
   Neither lane is toolchain-blocked.
6. Attach benchmark metadata for Apple hot paths such as matmul, matmul
   epilogues, conv2d, decode chain, and packaged kernels.

## Source Material Consolidated

- `archive/2026_06_01_apple_gpu_chain_audit.md`
- `archive/single_command_buffer_decode_plan.md`
- `archive/apple_ga_ebm_native_execution_gap.md`
- `../../compiler/archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`

