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
