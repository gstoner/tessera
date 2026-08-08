# Project-Level Eval Matrix

Generated from docs/context/ontology.yaml, docs/context/knowledge_map.yaml, and tests/COMPILER_TEST_PLAN.md. This is a derived navigation artifact; canonical specs remain authoritative.

This is an agent-friendly rendering of the project eval strategy. `tests/COMPILER_TEST_PLAN.md` remains the source of truth.

## Project-Level Eval Matrix

| Eval Family | Required Coverage | Gate |
| --- | --- | --- |
| Spec conformance | Canonical API symbols, named pipeline aliases, dialect symbols, and README phase/status claims match the implementation. | No stale public contracts or unsupported status claims. |
| End-to-end compiler evals | Representative Python/API samples produce lineage-linked Graph, Schedule, Tile, Target, backend, image, descriptor, and runtime evidence; package-only and compiler-boundary E2E remain distinct. | Every adjacent digest joins, the declared producer consumed the preceding artifact, semantic metadata survives or has a named drop, and fallback is disabled for execution claims. |
| Numerical correctness | Supported operators compare against NumPy/PyTorch-style references across shape classes, seeds, and dtype-specific tolerances. | Results stay within the declared tolerance for each dtype and backend mode. |
| Shape/layout evals | Symbolic shapes, tile boundaries, layout transforms, sharding plans, halo inference, and neighborhood topology cases. | Valid programs infer stable metadata; invalid programs fail before lowering. |
| Diagnostics quality | Invalid programs for effects, distributions, target support, memory spaces, and shapes produce stable useful errors. | Diagnostics include source context, compiler stage, violated invariant, and actionable category. |
| Documentation evals | Documentation code blocks, links, referenced paths, public symbols, and pipeline diagrams stay current. | Executable examples run or are explicitly marked pseudo-code; all referenced project paths exist. |
| Sample/tutorial evals | Getting-started samples and tutorials import, use canonical APIs, and produce expected outputs or artifacts. | CPU-promised samples run without accelerators and finish within a local smoke-test budget. |
| Agent context graph evals | Ontology, knowledge map, generated JSON/Markdown outputs, and agent workflows stay parseable, path-valid, deterministic, and marked non-authoritative. | `scripts/generate_context_outputs.py --check` passes and generated outputs match source YAML plus this test plan. |
| Compatibility/project health | CPU-only build path, optional hardware gates, CLI help, import-time budget, package metadata, and script/README command agreement. | Deterministic health checks pass without requiring hidden local state. |

## Eval Tiering

| Tier | Eval Scope | Default |
| --- | --- | --- |
| Fast local | Spec conformance, documentation smoke checks, context graph generation checks, sample import checks, and CLI/package health. | Developer opt-in and cheap enough for pre-commit use. |
| CI deterministic | Project evals that require no accelerator, including context graph output freshness, and do not depend on machine-specific timings. | Always on once the corresponding eval harness exists. |
| Scheduled | Numerical sweeps, broader sample execution, documentation execution, and performance regression checks. | Nightly or weekly depending on runtime cost. |
| Hardware-marked | Apple7, NVIDIA SM120, ROCm gfx1151, x86/AVX-512, access-gated AMX, and distributed backend evals. | Opt-in through the architecture-owned exact-device command; required before changing that architecture's execution or promotion claim. |
