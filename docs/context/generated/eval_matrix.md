# Project-Level Eval Matrix

Generated from docs/context/ontology.yaml, docs/context/knowledge_map.yaml, and tests/COMPILER_TEST_PLAN.md. This is a derived navigation artifact; canonical specs remain authoritative.

This is an agent-friendly rendering of the project eval strategy. `tests/COMPILER_TEST_PLAN.md` remains the source of truth.

## Project-Level Eval Matrix

| Eval Family | Required Coverage | Gate |
| --- | --- | --- |
| Spec conformance | Canonical API symbols, named pipeline aliases, dialect symbols, and README phase/status claims match the implementation. | No stale public contracts or unsupported status claims. |
| End-to-end compiler evals | Representative Python/API samples lower through Graph IR, Schedule IR, Tile IR, and Target IR with semantic invariants preserved. | Required IR layers emitted and no illegal effect, shape, layout, or memory-space transition. |
| Numerical correctness | Supported operators compare against NumPy/PyTorch-style references across shape classes, seeds, and dtype-specific tolerances. | Results stay within the declared tolerance for each dtype and backend mode. |
| Shape/layout evals | Symbolic shapes, tile boundaries, layout transforms, sharding plans, halo inference, and neighborhood topology cases. | Valid programs infer stable metadata; invalid programs fail before lowering. |
| Diagnostics quality | Invalid programs for effects, distributions, target support, memory spaces, and shapes produce stable useful errors. | Diagnostics include source context, compiler stage, violated invariant, and actionable category. |
| Documentation evals | Documentation code blocks, links, referenced paths, public symbols, and pipeline diagrams stay current. | Executable examples run or are explicitly marked pseudo-code; all referenced project paths exist. |
| Sample/tutorial evals | Getting-started samples and tutorials import, use canonical APIs, and produce expected outputs or artifacts. | CPU-promised samples run without accelerators and finish within a local smoke-test budget. |
| Compatibility/project health | CPU-only build path, optional hardware gates, CLI help, import-time budget, package metadata, and script/README command agreement. | Deterministic health checks pass without requiring hidden local state. |

## Eval Tiering

| Tier | Eval Scope | Default |
| --- | --- | --- |
| Fast local | Spec conformance, documentation smoke checks, sample import checks, and CLI/package health. | Developer opt-in and cheap enough for pre-commit use. |
| CI deterministic | Project evals that require no accelerator and do not depend on machine-specific timings. | Always on once the corresponding eval harness exists. |
| Scheduled | Numerical sweeps, broader sample execution, documentation execution, and performance regression checks. | Nightly or weekly depending on runtime cost. |
| Hardware-marked | SM80/SM90/SM100, ROCm, TPU, and distributed backend evals. | Opt-in with explicit hardware environment flags. |
