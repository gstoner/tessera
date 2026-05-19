---
status: Normative
classification: Spec
authority: Distinguishes unit / lit / build / artifact-only / hardware-runtime checks across the Tessera validation spine
last_updated: 2026-05-18
---

# Tessera validation spine — M5 deliverable

Tessera ships multiple validation layers; this document is the
canonical map of what each layer proves and how to run it.  Until
M5 the layers were documented piecemeal in benchmark and audit
READMEs; M5 consolidates the picture so that **claim**
(`implemented` / `lit-testable` / `mock-runtime` / `hardware-runtime`)
maps unambiguously to **command**.

## Layer overview

| Layer | What it proves | Canonical command | Hardware required | CI default |
|-------|----------------|-------------------|-------------------|------------|
| **Python unit tests** | Python frontend, IR layers, audit, registries, error semantics | `pytest tests/unit -m "not slow"` | CPU only | yes |
| **MLIR lit tests** | C++ pass + dialect contracts on canonical input MLIR | `lit tests/tessera-ir -v` (the console script; the `python -m lit` form does not work — lit's package has no `__main__`) | CPU only (after building `tessera-opt`) | opt-in: `TESSERA_VALIDATE_LIT=1 scripts/validate.sh` runs lit when both `lit` and `tessera-opt` are on PATH (or in `build/tools/tessera-opt/`); otherwise the layer is skipped with a clear diagnostic. |
| **C++ build checks** | the C++ tree compiles against MLIR 21 / LLVM 21 | `cmake -B build && cmake --build build` | toolchain only | partial — gated on environment |
| **Artifact-only target lowering** | `tessera-lower-to-{rocm,metalium,apple_gpu}` produce textual artifacts; **no execution** | `tessera-opt ... | FileCheck` | none | yes |
| **Hardware-runtime smoke tests** | a fused symbol actually dispatches on the device | `pytest tests/unit -m hardware_apple_gpu` (or `-m hardware_nvidia`, etc.) | the named accelerator | no (skipped without hardware) |
| **Generated audit drift** | `op_catalog` / `primitive_coverage` / `backend_manifest` / `capabilities` haven't drifted from the generated support table | `python -m tessera.compiler.audit support_table --check` | CPU only | yes |
| **Canonical-program reports** | each shipped canonical program runs CPU-only and produces a deterministic `CompileReport` | `pytest tests/unit/test_compile_report.py tests/unit/test_canonical_program_registry.py` | CPU only | yes |
| **Benchmark schema validation** | every emitted benchmark row matches the M5 canonical schema | `pytest tests/unit/test_benchmark_row.py` | CPU only | yes |

## Mode → claim mapping

The status taxonomy in [`docs/README.md`](../README.md) maps onto
the spine layers as follows:

| Status label | Minimum layer that must pass | Notes |
|---|---|---|
| `implemented` | Python unit tests | The code exists and has coverage. |
| `lit-testable` | MLIR lit | Pass contract is locked but native execution isn't implied. |
| `mock-runtime` | Python unit tests + a deterministic CPU/mock fallback | Most non-CPU backends today. |
| `hardware-runtime` | Python unit tests + benchmark row with `proof_routes` for the named target | The `M5` no-silent-native rule: a hardware-runtime claim requires either a `JitBridgeRoute` row, a `compiled_artifact`, a `plan_hash`, or a `symbols` list. |
| `artifact_only` | MLIR lit (artifact path) — explicitly NOT hardware | Tracked in `backend_manifest.BackendKernelEntry.status="artifact_only"`. |
| `planned` | none — design direction only | Tracked in registry as `planned`. |

## Cross-references

- Status sources of truth: `python/tessera/compiler/op_catalog.py`,
  `primitive_coverage.py`, `backend_manifest.py`, `capabilities.py`.
- Generated support table (drift-gated):
  [`docs/audit/generated/support_table.md`](../audit/generated/support_table.md).
- Canonical end-to-end programs (M1 / M1.5):
  `python/tessera/compiler/canonical/`.
- CompileReport schema (M1):
  `python/tessera/compiler/compile_report.py`.
- Stable fallback taxonomy (M3):
  `python/tessera/compiler/fallback.py`.
- Canonical benchmark-row schema (M5):
  `python/tessera/compiler/benchmark_row.py`.

## How to add a new claim

1. **Decide the claim layer.**  Look the desired status label up in
   the table above; you need to clear at least that layer.
2. **Wire it through the audit sources** so the support table picks
   it up automatically.
3. **Regenerate the support table**:
   `python -m tessera.compiler.audit support_table`.
4. **Add tests** at the right layer (unit for `implemented`, lit
   for `lit-testable`, hardware-gated for `hardware-runtime`).  If
   you're emitting a benchmark row that claims native execution,
   include either a `JitBridgeRoute` proof, a `compiled_artifact`,
   a `plan_hash`, or a non-empty `symbols` list — the
   `validate_benchmark_row` gate rejects unproven native claims.
5. **Update the doc that names the feature** to link the
   regenerated support table rather than restating per-op status.

## Hardware gating in CI

Hardware-runtime checks live in `tests/unit/` but are tagged with
markers (`@pytest.mark.hardware_apple_gpu`, etc.) and are excluded
from the default sweep.  CI configurations that have the matching
hardware opt in by passing `-m hardware_apple_gpu` (or whichever
marker).  Tests without hardware emit a `skip` with the canonical
reason — never silently mark themselves green.

## What's intentionally out of scope this milestone

- A unified C++ build / lit / hardware-runtime CI job (deferred to
  whichever post-M5 sprint owns CI infrastructure).
- A cross-target benchmark sweep that runs once on each accelerator;
  today the only end-to-end hardware path is Apple GPU.
- Tooling that auto-promotes a primitive's claim layer when the
  required tests appear — for now the audit drift gate is enough.
