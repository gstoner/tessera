# Compiler and Apple Backend End-to-End Audit

**Date:** 2026-06-02

This audit reviews the current compiler path, IR handoffs, Apple backend
performance work, and remaining end-to-end proof gaps. It is based on the
current repo state across:

- `python/tessera/compiler/canonical_compile.py`
- `python/tessera/compiler/driver.py`
- `python/tessera/compiler/graph_ir.py`
- `python/tessera/compiler/schedule_ir.py`
- `python/tessera/compiler/tile_ir.py`
- `python/tessera/compiler/target_ir.py`
- `python/tessera/compiler/backend_manifest.py`
- `python/tessera/compiler/apple_packaged_manifest.py`
- `python/tessera/compiler/apple_target.py`
- `python/tessera/compiler/pipeline_gates.py`
- `python/tessera/compiler/execution_matrix.py`
- `python/tessera/runtime.py`
- `python/tessera/apple_gpu_batched.py`
- `python/tessera/apple_gpu_chain.py`
- `python/tessera/apple_mlpkg.py`
- `docs/audit/op_target_conformance.md`
- `docs/audit/single_command_buffer_decode_plan.md`

## Executive Verdict

The compiler stack is no longer a loose collection of claims. There is now a
recognizable proof spine:

1. Canonical compile entry point.
2. IR bundle with graph/schedule/tile/target text.
3. Capability gates with named failing axes.
4. Backend manifest statuses, including `hardware_verified` and `packaged`.
5. Runtime execution matrix as the launch source of truth.
6. Apple encode-session and packaged-kernel substrates.

The remaining glass jaw is not "no architecture." It is proof depth and
handoff discipline. Many paths are now represented, gated, and partially
executable, but optimization decisions and runtime proof still split across
Python metadata, Target IR string emission, runtime dispatch heuristics, and
backend manifests. The next work should reduce those split-brain boundaries.

Highest-leverage next steps:

1. Make canonical compilation component-aware, not first-op-only.
2. Promote Apple tensor/kernel binding specs from packaged-only scaffolding to
   the normal Apple GPU Target IR/runtime contract.
3. Wire `AppleGPUTargetProfile` and live Apple runtime limits into tile and
   kernel selection.
4. Replace Target IR embedded Apple source/fusion logic with a backend kernel
   descriptor registry.
5. Make conformance numerical proof fixture-driven for every claimed complete
   row.
6. Finish `tessera.ops`/`@jit` integration for the one-command-buffer Apple
   decode chain.
7. Populate production packaged kernels; the current packaged manifest only
   proves the lifecycle through an Apple sample fixture.

## Current End-to-End Shape

The implemented compiler flow is:

```text
frontend / Python trace
  -> Graph IR object model
  -> Schedule IR object model
  -> Tile IR object model
  -> Target IR object model / Apple GPU MSL metadata
  -> CompileArtifactBundle
  -> canonical CompileResult
  -> RuntimeArtifact metadata
  -> execution_matrix row
  -> runtime.launch executor
```

This is the right shape. The weakness is that several important decisions are
still made late and outside the IR contract:

- Apple fusion recognition is in `target_ir.py` and repeated in
  `runtime.py` metadata pattern checks.
- Apple runtime dispatch falls through a large Python envelope in
  `_execute_apple_gpu_mps_metadata`.
- Conformance can report lower IR columns as green even when runtime/numerical
  proof is not direct for that op/target.
- `canonical_compile._extract_primary_op()` still treats the first op as the
  primary operation, which is not enough for fused chains, decode blocks,
  pipelines, or real multi-op lowering.

## IR-Level Audit

### Graph IR

What is strong:

- Graph IR has a structured object model, verifier, types, mesh declarations,
  numeric policy records, constants, and KV-cache state concepts.
- Recent frontend fixes closed concrete lowering drops such as `AugAssign`
  subtraction/division.
- Dynamic control-flow support is better surfaced through diagnostics instead
  of silent eager fallback.

Glass jaws:

- Multi-op graph identity is weak at the compiler-driver boundary. The
  canonical compile path still extracts one "primary" op from the first
  function and first operation.
- Shape, dtype, layout, and aliasing contracts are not yet strong enough to
  drive backend selection without runtime envelope fallback.
- State-effect ops such as KV cache have Graph IR presence but incomplete
  end-to-end backend proof.

Needed improvements:

- Add a component-op vector to canonical compile metadata and pipeline gates.
- Treat graph outputs, effects, layout, and shape envelopes as first-class
  contracts, not only op metadata.
- Add source-span/error context consistently so unsupported dynamic constructs
  point to the Python construct and the failing compiler axis.

### Schedule IR

What is strong:

- Schedule IR has a verifier and knows key families such as matmul, conv2d,
  rope, flash attention, collectives, tiles, and pipeline regions.
- It preserves mesh/layout/placement metadata into downstream IR.

Glass jaws:

- Scheduling is still rule-based and canned. It does not yet behave like a
  planner that can choose among Apple MPS, MPSGraph, custom MSL, Metal 4, or
  packaged-kernel strategies based on shape, dtype, and target limits.
- Fusion intent is not consistently represented here, so downstream Target IR
  and runtime dispatch rediscover patterns.
- Apple feature limits exist in `apple_target.py`, but they are not yet the
  dominant input to schedule selection.

Needed improvements:

- Add a schedule planner that consumes `AppleGPUTargetProfile`, shape envelope,
  dtype policy, and static/live limit floors.
- Represent fusion groups and command-buffer grouping intent before Tile IR.
- Add cost-model hooks for "compose per op" vs "fused MSL" vs "MPSGraph" vs
  "Metal 4 tensor path" vs "packaged kernel."

### Tile IR

What is strong:

- Tile IR preserves metadata and can express pipeline regions, async copy,
  queues, barriers, collectives, and selected control-flow markers.
- The single-command-buffer architecture has a concrete encode-session
  substrate and a chain planner in Python.

Glass jaws:

- Tile IR does not yet own Apple command-buffer planning. The actual one-CB
  behavior is mostly a Python/runtime path through `apple_gpu_batched.py` and
  `apple_gpu_chain.py`.
- Tensor layout, buffer indices, command-buffer lifetime, resident resources,
  and hazard assumptions are not yet normal Tile IR contracts.
- Packed/decode chains are proven as runtime ergonomics, but not fully through
  canonical `tessera.ops` or `@jit(target="apple_gpu")`.

Needed improvements:

- Add Tile IR attributes for Apple resident tensors, buffer bindings, command
  segment IDs, and synchronization boundaries.
- Lower schedule fusion/chain intent into tile command segments.
- Complete `tessera.ops` interception or `@jit(..., auto_batch=True)` so
  encode-session planning is part of the canonical user path.

### Target IR

What is strong:

- Apple GPU Target IR is broad: MPS, MPSGraph, custom MSL, fused kernels, Metal
  4 lanes, and packaged-kernel adjacency are represented.
- It can emit fused Apple kernels for several important chains such as
  matmul-softmax, matmul-softmax-matmul, matmul-gelu, matmul-rmsnorm, bias/act
  epilogues, and SwiGLU-style paths.

Glass jaws:

- `target_ir.py` is too responsible. It contains source strings, fusion
  recognition, backend path decisions, MSL emission, diagnostics, and target
  metadata construction.
- Apple kernel ABI expectations are not uniformly represented as descriptor
  records. Packaged kernels have `AppleKernelBindingSpec` /
  `AppleTensorBindingSpec`, but normal Apple GPU kernels mostly rely on
  runtime wrapper conventions and symbol naming.
- Target IR emits artifacts that the runtime then interprets through another
  large pattern dispatcher, creating drift risk.

Needed improvements:

- Move Apple kernel definitions into a backend kernel descriptor registry:
  kernel ID, supported dtypes, layout, binding spec, feature requirements,
  runtime symbol, numerical fixture, benchmark fixture, and MSL/package source.
- Have Target IR emit kernel IDs and binding/layout specs rather than embedding
  large source strings and rediscovering fusion rules.
- Make packaged and non-packaged Apple kernels share the same binding-spec
  model.

## Apple Backend Performance Audit

### What Has Landed

- Apple CPU/Apple GPU are runtime-backed on capable Darwin hosts.
- MPS/MPSGraph/custom MSL paths exist for core tensor ops.
- Metal 4 lanes exist for scan, simdgroup matmul, matmul2d f16/bf16, epilogues,
  MLP sessions, archive telemetry, and conv2d routing.
- `AppleGPUTargetProfile` now models Apple7-Apple11 features, dtype sets,
  conservative static limits, and live runtime probing.
- Encode-session APIs support resident device tensors and one command buffer
  for meaningful decode-style chains.
- `apple_gpu_chain.py` has an encode-op registry for f32/f16/bf16 decode ops,
  including conv2d.
- Packaged-kernel PK1-PK7 lifecycle exists: load, reflection, validation,
  argument layout, dispatch, drift diagnostics, and real fixture proof.

### What Is Still Open

1. **Production packaged kernels are empty.**

   `apple_packaged_manifest.py` has a fixture-backed Apple sample package but no
   production `PACKAGED_PRODUCTION_KERNELS`. The runtime ABI is proven; the
   compiler has not yet gained real packaged backend coverage.

2. **Apple feature limits are present but underused.**

   The table and probe exist. Schedule/Tile/Target selection still needs to
   consume them to choose tile shapes, Metal 4 paths, packaged eligibility, and
   fallback paths.

3. **Single-command-buffer decode is not yet canonical JIT behavior.**

   The substrate is strong, and docs report real benchmark wins. The remaining
   open work is routing canonical `tessera.ops.*` and/or
   `@jit(target="apple_gpu", auto_batch=True)` through that path.

4. **Runtime dispatch is still envelope-based.**

   `_execute_apple_gpu_mps_metadata` is a large dispatcher that interprets
   metadata and checks fusion patterns. That should become a descriptor-driven
   executor keyed by compiled kernel/chain IDs.

5. **Conformance proof is not always direct enough.**

   `docs/audit/op_target_conformance.md` still shows rows where numerical is
   green despite missing op capability entries or runtime execution. The
   direction is right, but the numerical column should be driven by
   `BackendKernelEntry.execute_compare_fixture` for all completed cells.

6. **Conv2d and KV-cache remain high-priority end-to-end gaps.**

   Conv2d has encode-session and Metal 4 routing work, but conformance still
   reports Apple GPU runtime incomplete. KV-cache read has graph/lowering shape
   but no complete runtime/numerical proof row.

7. **Fusion coverage is useful but ad hoc.**

   Important chains are recognized, but there is no general fusion planner with
   a cost model and backend descriptor lookup. `matmul_relu` still composes
   instead of using a fused single kernel.

## Handoff Problems To Fix

### Compiler Driver to Gates

Current risk:

- `CompileResult.primary_op` is first-op-based.
- Pipeline gates can answer a single op more cleanly than a real program.

Fix:

- Add `component_ops`, `fusion_groups`, `effects`, `shape_envelope`, and
  `layout_contracts` to canonical metadata.
- Gate both the program and its component ops. A program should only be
  "complete" when the chosen execution strategy has runtime and numerical proof.

### Gates to Manifest

Current risk:

- Manifest rows are improving, but some rows still lag runtime reality.
- `hardware_verified` is correctly strict, but most executable-looking paths
  do not carry compare fixtures.

Fix:

- Require every `complete` conformance cell to resolve to one of:
  `hardware_verified`, `packaged` with validation and dispatch proof, or a
  manifest row with an explicit `execute_compare_fixture`.
- Keep `fused` as "compiled fused path exists" unless a fixture promotes it.

### Target IR to Runtime

Current risk:

- Target IR and runtime both know fusion patterns.
- Runtime dispatch depends on Python metadata conventions rather than a compact
  kernel ABI descriptor.

Fix:

- Introduce `AppleKernelDescriptor`.
- Emit a target artifact containing descriptors:
  `kernel_id`, `chain_id`, `runtime_symbol`, `binding_spec`, `layout`,
  `dtype_policy`, `feature_requirements`, `execution_mode`, and
  `compare_fixture`.
- Runtime dispatches descriptor IDs instead of recognizing op lists.

### Tile IR to Apple Runtime

Current risk:

- One-command-buffer optimization exists outside the IR handoff.

Fix:

- Add Tile IR command segments and resident tensor handles as explicit lowering
  products.
- Let `apple_gpu_chain.py` consume Tile/Target metadata instead of only Python
  trace records.

## Prioritized Next Work

### P0: Proof Honesty and Multi-Op Metadata

Deliverables:

- `CompileResult.component_ops`.
- `RuntimeArtifact.metadata["component_ops"]`.
- Conformance numerical cells require direct compare fixtures or packaged
  validation.
- Fix confusing rows where runtime is missing but numerical is green.
- Add manifest rows or explicit incomplete diagnostics for Apple GPU conv2d and
  KV-cache read.

Why first:

This prevents the dashboard from overclaiming and gives later optimization work
a reliable scorecard.

### P1: Apple Binding Contracts Everywhere

Deliverables:

- Promote `AppleTensorBindingSpec` / `AppleKernelBindingSpec` beyond packaged
  kernels.
- Add binding specs for MPSGraph, MSL, Metal 4, and encode-session kernels.
- Runtime validates descriptor/binding compatibility before dispatch.

Why:

This closes the biggest Target IR to runtime ABI gap and makes packed/non-packed
Apple kernels share one contract.

### P1: Apple Feature-Limit-Guided Tiling

Deliverables:

- Schedule/Tile planners consume `AppleGPUTargetProfile`.
- Tile choices include threadgroup memory, simdgroup matrix support, bf16,
  Metal 4 availability, argument-buffer budget, and live probe overrides.
- Add tests that force Apple7/Apple8/Apple10 choices and prove fallback reasons.

Why:

The feature table has landed. The value comes when it controls tile and backend
selection.

### P1: Canonical One-Command-Buffer Decode

Deliverables:

- Add `@jit(target="apple_gpu", auto_batch=True)` or equivalent.
- Route canonical `tessera.ops.*` through trace capture while auto-batch is
  active.
- Emit command-segment metadata in compile artifacts.
- Add a conformance/perf gate for "N ops, one command buffer."

Why:

The runtime substrate shows large small-batch decode wins. It needs to become a
compiler path, not only a separate Apple helper surface.

### P2: Apple Kernel Descriptor Registry

Deliverables:

- Move embedded Apple MSL/package descriptors out of `target_ir.py`.
- Registry rows include source/package path, ABI binding spec, feature
  requirements, supported shapes/dtypes, runtime symbol, fixtures, and benchmark
  metadata.
- Target IR emits descriptor references.

Why:

This reduces drift and makes adding kernels less invasive.

### P2: Production Packaged Kernels

Deliverables:

- Add first production packaged kernel entries, likely matmul or MLP epilogue
  variants where binding reflection and argument layout are already proven.
- Promote package rows only with real compare fixture plus drift validation.
- Decide package cache/archive strategy.

Why:

PK1-PK7 proved the surface. Production rows turn it into backend capability.

### P3: Optimization Loop and Perf Gates

Deliverables:

- Apple backend benchmark JSON attached to manifest rows.
- Stable perf ratchets for:
  - matmul f32/f16/bf16
  - matmul epilogue
  - conv2d f16/bf16
  - decode chain one-CB
  - packaged matmul/MLP once production rows exist
- CI separates "hardware unavailable" from "performance regression."

Why:

After proof and descriptors, performance optimization needs durable feedback.

## Recommended Sprint Order

1. **S1: Conformance proof hardening**
   - Component-op metadata.
   - Fixture-driven numerical column.
   - Conv2d/KV-cache manifest/runtime status cleanup.

2. **S2: Apple binding spec unification**
   - Descriptor/binding spec for non-packaged Apple GPU kernels.
   - Runtime validation before dispatch.

3. **S3: Feature-limit-guided tile planner**
   - Apple7-Apple11 selection tests.
   - Tile strategy metadata in compile artifacts.

4. **S4: Canonical one-CB JIT path**
   - `tessera.ops`/`@jit` integration.
   - Tile/Target command-segment metadata.
   - Perf gate for command-buffer count.

5. **S5: Target IR Apple registry refactor**
   - Move source strings and fusion descriptors out of `target_ir.py`.
   - Runtime dispatch by descriptor ID.

6. **S6: Production packaged kernels**
   - First packaged production rows.
   - Compare fixtures, reflection gates, cache policy.

## Bottom Line

Tessera's compiler now has enough structure to support serious end-to-end
backend work. The main architectural improvement is to make the runtime ABI and
performance plan explicit earlier in the IR chain. Apple backend performance
should not be a late Python runtime decision; it should be a scheduled,
tile-aware, descriptor-backed lowering result with direct numerical and
benchmark proof.

