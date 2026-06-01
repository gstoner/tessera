# Tessera — Claude Code Skill Reference

Maps common Tessera development tasks to the `/skill` commands available in Claude Code.
Invoke a skill with `/skill-name` or via the Skill tool.

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
| Audit stubs and incomplete pass bodies | `engineering:tech-debt` | Spectral/FFT passes, TPP solver wiring, Cerebras/Metalium scaffold gaps |
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
/engineering:tech-debt        — stub and scaffold audit
/product-management:write-spec         — phase spec or feature design doc
/product-management:sprint-planning    — phase task breakdown
/product-management:brainstorm         — open design questions
```

---

*Skills are provided by the Claude Code harness. This file documents which ones apply to Tessera — it does not define them. Run `/help` for the full skill list.*

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
