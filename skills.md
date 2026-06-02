# Tessera — Claude Code Skill Reference

Maps common Tessera development tasks to the `/skill` commands available in Claude Code.
Invoke a skill with `/skill-name` or via the Skill tool.

---

## Audit Flow (start here for "what's done / what's open" work)

The `docs/audit/` folder is the canonical status surface (reorganized
2026-06-02: one root audit + theme audits + generated dashboards +
theme-local archives). See **CLAUDE.md Architecture Decision #26** for
the normative rule. When asked to audit, assess status, review open
efforts, or pick next work, follow this order — do **not** reconstruct
status by grepping scattered docs:

1. **`docs/audit/MASTER_AUDIT.md`** — all-up snapshot: finished work,
   still-open work per area, and the **P0/P1/P2 priority queue**.
   This is the single entry point.
2. **Theme audit** for focused status:
   - `compiler/COMPILER_AUDIT.md` — IR handoffs, lowering, spec gaps
   - `backend/BACKEND_AUDIT.md` — cross-target runtime/ABI/proof rules
   - `backend/apple/APPLE_AUDIT.md` — Apple CPU/GPU, Metal 4, packaged kernels
   - `backend/nvidia/NVIDIA_AUDIT.md`, `backend/rocm/ROCM_AUDIT.md` — hardware frontier
   - `coverage/COVERAGE_AUDIT.md` — primitive/op/KV-cache/examples coverage
   - `domain/DOMAIN_AUDIT.md` — GA/EBM, attention, CorrDiff, sharding, autodiff
   - `roadmap/ROADMAP_AUDIT.md` — execution roadmap, deferred items, sprint history
3. **`docs/audit/generated/`** = count/status truth (script/test-owned,
   drift-gated). **Never hand-edit.** Regenerate via each doc's CLI and
   the `scripts/check_generated_docs.sh` gate (pre-commit hook). Key
   dashboards: `runtime_abi.md`, `runtime_execution_matrix.md`,
   `e2e_op_coverage.md`, `s_series_status.md`, `support_table.md`,
   `test_coverage_classification.md`; plus root `op_target_conformance.md`
   and `standalone_primitive_coverage.md`.
4. **`*/archive/`** = provenance only (original narrative/acceptance
   criteria), not the current status surface.

**When finishing audit-relevant work:** update the **theme audit** (and
`MASTER_AUDIT.md` if the all-up picture or priority queue shifts); let
generated dashboards carry the numbers. For a broad multi-track review,
fan out one read per theme audit, cross-check against the live tree, then
synthesize — never trust the prose over the generated dashboards.

---

## Engineering Skills

| Task | Skill | When to use |
|------|-------|-------------|
| Design a new IR dialect or backend | `engineering:system-design` | Cerebras/Metalium backend architecture, Neighbors dialect design, new solver dialect |
| Create or evaluate an architecture decision | `engineering:architecture` | ADR for a pass pipeline change, dialect versioning policy, lowering strategy |
| Debug a failing pass or IR verification error | `engineering:debug` | Pass produces wrong IR, lit test fails, `tessera-opt` crashes, shape mismatch |
| Review a set of pass or frontend changes | `engineering:code-review` | Before committing any C++ pass, ODS change, or Python compiler module |
| Design the test plan for a new phase | `engineering:testing-strategy` | Planning Phase 7 test suite, extending lit tests, coverage gaps |
| Write pass reference docs, specs, ADRs | `engineering:documentation` | Pass reference markdown, IR specs in `docs/spec/`, API docs |
| Pre-ship checklist before tagging a phase | `engineering:deploy-checklist` | Before declaring a phase complete — tests green, lit tests pass, validate.sh clean |
| Audit stubs and incomplete pass bodies | `engineering:tech-debt` | Cross-check the audit flow first (`docs/audit/MASTER_AUDIT.md` → theme audit → generated dashboards) so "open" work is grounded in the canonical surface, not guessed |
| Structured incident response for CI breakage | `engineering:incident-response` | When validate.sh CI spine breaks across multiple tests |
| Daily standup summary from recent commits | `engineering:standup` | Summarize what changed in the compiler or test suite |

---

## Product / Planning Skills

| Task | Skill | When to use |
|------|-------|-------------|
| Scope and write a Phase 7+ spec | `product-management:write-spec` | Neighbors dialect spec, Cerebras backend feature spec |
| Sprint planning for a phase | `product-management:sprint-planning` | Breaking Phase 7 into concrete tasks with estimates |
| Brainstorm architecture or design options | `product-management:brainstorm` | Open-ended: "how should we structure the Metalium lowering?" |
| Competitive / prior art analysis | `product-management:competitive-brief` | Comparing Tessera's tile IR to Triton, XLA, or Halide approaches |
| Stakeholder update on phase progress | `product-management:stakeholder-update` | Status update summarizing what's done and what's blocked |

---

## Anthropic API / Claude Skills

| Task | Skill | When to use |
|------|-------|-------------|
| Build or tune a Claude-powered analysis tool | `claude-api` | Adding LLM-assisted IR analysis, auto-generating pass documentation |

---

## Skill Gaps (no matching skill — use general-purpose Claude)

| Domain | Notes |
|--------|-------|
| MLIR TableGen / ODS authoring | Core Tessera work — no dedicated skill. Use Claude directly with `src/compiler/ir/TesseraOps.td` as reference. |
| C++ CMake build system | `CMakeLists.txt` changes, new dialect registration — use Claude directly. |
| GPU kernel development (WGMMA, TMA, MFMA) | Hardware-specific IR lowering — use Claude directly with backend docs in `src/compiler/codegen/`. |
| Python scientific stack (numpy, Optuna) | Autotuner and benchmark work — use Claude directly. |
| Lit test authoring | MLIR FileCheck patterns — use `tests/tessera-ir/phase2/` as templates, Claude directly. |

---

## Quick Invocation Cheat Sheet

```
/engineering:system-design    — new backend or dialect design
/engineering:architecture     — ADR for a significant decision
/engineering:debug            — failing pass, lit test, or IR error
/engineering:code-review      — before committing C++ or Python compiler changes
/engineering:testing-strategy — test plan for a new phase or component
/engineering:documentation    — pass reference, spec, or API doc
/engineering:deploy-checklist — pre-phase-completion gate
/engineering:tech-debt        — stub and scaffold audit (read docs/audit/MASTER_AUDIT.md first)
/product-management:write-spec         — phase spec or feature design doc
/product-management:sprint-planning    — phase task breakdown
/product-management:brainstorm         — open design questions
```

---

*Skills are provided by the Claude Code harness. This file documents which ones apply to Tessera — it does not define them. Run `/help` for the full skill list.*

---

## Metal / Apple GPU — how to get authoritative API facts (READ FIRST)

When you touch the Apple GPU backend (`apple_gpu_runtime.mm`, Tile→Apple
lowering, MTL4 / MPSGraph / MSL kernels, packaged kernels), **ground every
API claim in a real source before concluding a thing is possible or
"blocked."** The 2026-06-02 `.mtlpackage` miss came from declaring an API
path nonexistent without checking — don't repeat it.

**Authoritative sources, in order of reliability on this machine:**

1. **On-machine SDK headers — the ground truth for the installed SDK.**
   `xcrun --show-sdk-path` → `…/System/Library/Frameworks/Metal.framework/Headers/`
   (also `MetalPerformanceShaders`, `MetalPerformanceShadersGraph`,
   `MetalPerformancePrimitives`). These are the exact interfaces the runtime
   links against. To answer "does API X exist / what's its signature":
   `grep -nE "<symbol>" "$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers/"*.h`.
   Example facts established this way: `MTL4Compiler` has
   `newDynamicLibraryWithURL:` and `pipelineDataSetSerializer` (pipeline-data
   serialization), `MTL4MachineLearningPipelineDescriptor` takes a
   `machineLearningFunctionDescriptor`, `MTLBinaryArchive`/`MTLDynamicLibrary`
   expose `serialize(to:)`.
2. **User-provided doc dumps** (e.g. a `…/Downloads/MTLLibrary docs`
   markdown). When the user hands you a doc export, Read it — it's a curated
   slice of the live docs.
3. **The `apple-metal-docs-urls` memory file** — canonical
   developer.apple.com URLs (Metal root + machine-learning-passes). Use these
   to know *what* to look up; see the WebFetch caveat below for *how*.

**WebFetch caveat — developer.apple.com is a JS-rendered SPA.** A plain
`WebFetch(https://developer.apple.com/documentation/metal/…)` returns only
the page *title*, not the API body (verified 2026-06-02). So WebFetch alone
is NOT a reliable Metal-doc source. Prefer the SDK headers (1) and user doc
dumps (2). If you need the rendered doc body, ask the user for a dump or use
a browser tool that executes JS — don't conclude "no such API" from an empty
WebFetch.

**Anti-pattern that caused a real miss:** declaring a Metal capability
"blocked / no path exists" from absence of evidence. Absence in *one* source
≠ absence in the SDK. Check the headers before writing a blocker into an
audit doc. Distinguish carefully: shader-library + pipeline-data
serialization (runtime APIs that **do** exist) vs. authoring a `.mtlpackage`
ML package on disk (an *offline* Apple toolchain step — coremlcompiler /
package tooling — with no runtime ABI).

---

## Lessons learned — external Apple Metal 4 ML sample (2026-05-31)

Source: Apple's `RunningAMachineLearningModelOnTheGPUTimeline` sample (915 LOC Objective-C, `MTL4MachineLearningCommandEncoder` end-to-end). Six patterns extracted, all confirmed against the actual code at `/Users/gregorystoner/downloads/RunningAMachineLearningModelOnTheGPUTimeline`.

### Patterns + Tessera-side mappings

| # | Pattern | Apple sample anchor | Where it lands in Tessera | Priority |
|---|---------|---------------------|----------------------------|----------|
| 1 | **Reflection-driven tensor setup** — `MTLFunctionReflection` + `MTL4ShaderReflectionBindingInfo`, filter `MTLBindingTypeTensor`, validate `rank`, reject sentinel `-1` for dynamic dims | `MLMatrixMultiplier+PipelineCompilation.m:25-62`, `+TensorSetup.m:20-64` | Future packaged-kernel path (MLA/MLP `.mtlpackage` per `docs/apple_gpu_tier2_tier3_plan.md`). NOT for the current MSL-source path. | Defer to packaged-kernel sprint |
| 2 | **MTL4ArgumentTable binding by reflected buffer index** — `[argumentTable setResource:tensor.gpuResourceID atBufferIndex:binding.index]` instead of hand-counted `setBuffer:index:` | `MLMatrixMultiplier.m:201-207` | Composes with (1). Migrate `apple_gpu_runtime.mm` dispatchers as MTL4 dispatch encoding lights up. M2/M3 fallback keeps `setBuffer`. | Defer; gated on MTL4 dispatch |
| 3 | **Intermediates heap sized from pipeline** — `pipelineState.intermediatesHeapSize` → `MTLHeap` → `dispatchNetworkWithIntermediatesHeap:` | `MLMatrixMultiplier.m:212, 232` | Only when adopting `MTL4MachineLearningPipelineState`. Replaces hand-tuned M8 resident-session reservations with pipeline-driven sizing. | Defer; gated on packaged kernels |
| 4 | **Shared-event sync with timeout** — `MTLSharedEvent` + `waitUntilSignaledValue:timeoutMS:N` distinguishes "Metal 4 path absent" from "kernel hung" | `MLMatrixMultiplier.m:87, 241-255` (constant `kMLPassTimeoutMilliseconds = 100`) | `apple_gpu_runtime.mm` per-dispatch synchronous waits. Replaces `waitUntilCompleted` (no timeout) and the wrong-layer 300s subprocess timeout in the `--verify-fixtures` CLI. | **Land now** — small, high CI/reliability value |
| 5 | **Reusable resource lifetime** — `init` creates device / queue / library / compiler / sharedEvent / commandBuffer / commandAllocator once; `configure` creates per-workload resources; `encode` reuses everything | `MLMatrixMultiplier.m:62-103, 137-220` | Validates Tessera's M8 `MetalDeviceContext` approach. **Missing**: `MTL4Compiler` + `MTL4CommandAllocator` are not yet cached on `MetalDeviceContext`. Add when MTL4 dispatch goes live. | Land now — small additive cache |
| 6 | **Stride-from-dimensions helper** — `tensorStridesForDimensions:` walks dims, emits row-major strides (innermost = 1) | `Matrix+TensorUtilities.m:61-70` | Tessera's `make_buffer_tensor_4d` in `apple_gpu_runtime.mm` does this ad-hoc — the conv2d spike had to debug the innermost-first contract. Lift as a named helper. | **Land now** — trivial; documents intent |

### Anti-patterns the sample also clarifies

* **Don't hand-count binding indices** in dispatchers. The sample shows that the reflection metadata IS the binding contract; the runtime trusts reflection rather than the compiler's parallel knowledge. (Tessera's current `setBuffer:0`/`:1`/`:2` pattern is fine for the MSL-source kernel inventory but fragile for packaged kernels.)
* **Don't `waitUntilCompleted` without a timeout** in tests that exercise GPU execution. A hung kernel becomes a hung test, which becomes a hung CI lane.
* **Don't compute strides inline** every time you build a `MTLTensorDescriptor`. The innermost-first contract is non-obvious; centralize it.

### Cross-reference

* `docs/apple_gpu_overview.md` — the architectural story.
* `docs/apple_gpu_tier2_tier3_plan.md` — packaged kernels roadmap (where patterns 1, 2, 3 land).
* `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm` — where patterns 4, 5 (follow-on), 6 land.

---

## Lessons learned — Apple Metal 4 doc-deep review (2026-05-31 round 2)

A wider review using Apple's developer documentation (Metal Feature Set
Tables, MTLTensor / MTLTensorDescriptor.strides, MTL4ArgumentTable,
MTL4MachineLearningPipelineState reflection / intermediatesHeapSize,
MTL4Compiler / MTL4Archive, dispatchNetwork(intermediatesHeap:)) surfaced
10 architectural patterns the sample alone didn't expose. Mapping each
to Tessera's current state:

| # | Pattern | Doc anchor | Status | Next action |
|---|---------|-----------|--------|-------------|
| 1 | **Capability-first lowering** — record presence of mtl4 / tensor / ml_encoding / argument_table / archive / command_allocator on the artifact, not just `target=apple_gpu` | Metal Feature Set Tables (family mapping + feature rows) | Probe exists (`mtl4_caps()` at L9640) but not surfaced in `compile_result` / `RuntimeArtifact` | Action 1 — small, do now |
| 2 | **Tensor-aware IR contracts** — record rank/dims/dtype/strides/offset/usage/resource-id per binding | MTLTensor + gpuResourceID docs | Not in tree | Action 2 — medium, soon |
| 3 | **Stride/layout rules first-class** — innermost-first, second-stride 64-byte-aligned for ML usage, 128-byte for sub-byte dtypes | MTLTensorDescriptor.strides | ✅ Landed (Pattern 6, this round) | — |
| 4 | **Reflection as ABI verification** — diff compiler-expected bindings against reflected pipeline bindings before executable=true | MTL4MachineLearningPipelineState.reflection.bindings | Not in tree | Defer — gated on packaged-kernel adoption |
| 5 | **Argument tables as compiler artifact** — emit ArgumentLayout (name/index/kind/dtype/rank/residency) beside backend IR | "Understanding the Metal 4 core API" | Runtime has `mtl4_argtable` but not as a compile-time artifact | Defer — gated on packaged-kernel adoption |
| 6 | **ML passes on the GPU timeline** — keep prefill/decode/attn/MLP/projection on one command buffer; avoid CPU turnarounds | MTL4MachineLearningCommandEncoder | Partially: M8 resident MLP keeps weights on-GPU; full decode chain not yet | Defer — separate workstream |
| 7 | **Pipeline intermediates from metadata** — read `intermediatesHeapSize` from compiled pipeline | dispatchNetwork(intermediatesHeap:) | Not in tree | Defer — gated on packaged-kernel adoption |
| 8 | **Compiler/AOT cache path** — surface cache key / archive lookup / hit-miss / fallback reason on artifact | MTL4Compiler, MTL4Archive, MTL4CompilerTaskOptions.lookupArchives | Partially: MTL4Archive plumbed (P4 in MetalDeviceContext), telemetry not exposed | Action 6 — small, do now |
| 9 | **Command allocator discipline** — session-cache MTL4CommandAllocator, especially for decode loops | "Understanding the Metal 4 core API" | ✅ Landed (Pattern 5 audit, this round) | — |
| 10 | **Feature-limit-guided tiling** — drive threadgroup size / shared memory / argument-table capacity / matmul gates from Apple's published limits, not CUDA-shaped assumptions | Metal Feature Set Tables (implementation limits) | Scattered: SVD / conv2d / buffer pool all reference limits ad-hoc; no centralized table | Action 7 — plan (its own sprint) |

### Recommended actions (filtered)

* **Now**: Action 1 (`AppleMetal4Capabilities` artifact metadata) + Action
  6 (archive/cache telemetry). Both flow into `CompileResult.to_dict()`
  and the conformance dashboard; both are small.
* **Soon**: Action 2 (`AppleTensorBindingSpec`) — its own focused PR.
  Even if only the MSL-source path populates it sparsely today, the
  data-model lands now so packaged kernels don't need a migration.
* **Defer to packaged-kernel sprint**: Actions 4 + 5 — both require
  `MTL4MachineLearningPipelineState` adoption.
* **Plan as its own sprint**: Action 7 — touches every Apple tile
  decision; needs a designed `AppleFeatureLimits` table that all
  call sites consult.

### Anti-patterns this round also clarifies

* **Don't conflate `target=apple_gpu` with capability presence.** A
  developer Mac with macOS 26 has MTL4 + tensors + argument tables; an
  older host has MPS only. The artifact must carry the per-feature
  answer, not the family label.
* **Don't compute strides ad-hoc** — Apple's documented rule (innermost
  = 1, second stride 64-byte-aligned for ML usage, 128-byte for
  sub-byte dtypes) is non-obvious. Centralize in one helper. (Pattern 3
  ✅; the helper currently honors the innermost-1 rule but does NOT
  yet enforce the 64-byte / 128-byte alignment — flag for the sub-byte
  / ML-usage work.)
* **Don't size scratch heaps by hand.** Future packaged-kernel paths
  must read `intermediatesHeapSize` from the compiled pipeline.

---

## Graphify scope in this repo — empirical (2026-05-31)

Graphify ships grammars for 19 languages (Python, JavaScript, TypeScript,
Ruby, PHP, Lua, PowerShell, Go, Rust, C, C++, Zig, Swift, Objective-C,
Java, Kotlin, Scala, C#, Elixir). The set is documented in graphify's
own README / install output.

**Critical distinction:** the grammar set is the *upper bound* on what
graphify CAN parse. The actual contents of ``graphify-out/graph.json``
are what graphify queries traverse. Always verify the latter before
assuming the former.

### What Tessera's graph actually indexes today

Empirically confirmed by ``graphify query`` probes against the current
index:

| Surface | In graph? | Notes |
|---------|-----------|-------|
| Python (`python/tessera/`) | ✅ Yes | Full coverage — primary use case |
| C++ (`*.cpp` in `src/compiler/`, `tools/`, etc.) | ✅ Yes | Indexed; symbols like `ErrorReporter.cpp::PyLoc` surface |
| C++ headers (`*.h` in `src/`) | ⚠️ Mixed | Some headers indexed; coverage spotty |
| Objective-C++ (`*.mm`) | ❌ **Not in graph** | `apple_gpu_runtime.mm` (13K LOC, the entire Apple GPU runtime) does NOT appear in any graphify query result. Use direct `rg` / Read. |
| MLIR TableGen (`*.td`) | ❌ Not in graph | Grammar not supported |
| MLIR / lit fixtures (`*.mlir`) | ❌ Not in graph | Grammar not supported |
| CMake (`CMakeLists.txt`) | ❌ Not in graph | Grammar not supported |
| Markdown (`*.md` in `docs/`) | ⚠️ Partial | Some headings/sections appear as nodes; not a code-symbol graph |

### Implications for tool choice

* **Python / `.cpp`** — graphify FIRST. `codegraph_context "how does X
  work"` returns a scoped subgraph in one call; vastly cheaper than
  ripgrep + Read across thousands of files.
* **`.mm` (Apple GPU runtime)** — direct `rg` / Read. The runtime is a
  single 13K-line file; grep is the right tool. Don't waste a graphify
  call expecting `MetalDeviceContext` to surface — it won't.
* **`.td` (ODS)** — direct Read with line-number anchors from
  `src/compiler/ir/TesseraOps.td`.
* **MLIR `.mlir` lit fixtures** — direct Read; the file count is small
  and the contents are short.
* **CMake** — direct Read.

### Verification recipe (copy-paste, ~5s)

Before assuming graphify covers a file, probe:

    graphify query "<symbol or topic likely in that file>"

If results all point to other files / no hit for the target file, the
file isn't in the graph. Switch to `rg` for that surface.

### Lesson learned (2026-05-31)

I previously assumed `.mm` was out of graphify's scope (correctly, as it
turns out — but for the wrong reason). When corrected that Objective-C
is on graphify's grammar list, I conceded the wrong concession.
**Empirical answer**: the graph in *this* repo doesn't include `.mm`
regardless of grammar availability. The lesson is to verify against
the actual graph, not the grammar list. ``graphify query`` is the
fastest check.
