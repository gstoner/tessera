---
status: Informative
classification: Informative
authority: Reader-facing visual-complex status card; generated dashboards own current target and execution claims.
scope: Public visual-complex API, evidence routing, health checks, and non-claims.
last_updated: 2026-07-13
---

# Visual-Complex Status

Visual-complex exposes analytic complex-valued programs through
`tessera.complex.*`, `@analytic`, and `@complex_jit`. This card describes where
to find the current support and execution evidence without turning a dated
milestone inventory into a live status page.

The public semantic surface includes `mobius`, `stereographic`, and Wirtinger
calculus; consult the API specification for their contracts.

## Current evidence

| Question | Authority |
|---|---|
| Is a particular visual-complex operation supported on a target? | [Generated support table](../audit/generated/support_table.md) |
| Is that operation backed by a proven native execution path? | [Runtime execution matrix](../audit/generated/runtime_execution_matrix.md) |
| What semantics apply to the public API? | [Python API specification](../spec/PYTHON_API_SPEC.md) and the source-level API contracts |
| What is the current Apple fused-program evidence? | The `visual_complex_fused` entries in the [generated support table](../audit/generated/support_table.md) |

Use the exact operation-and-target row when making a capability claim. A public
complex function, analytic differentiation contract, reference implementation,
and native target proof are separate facts.

## Health checks

```bash
scripts/check_generated_docs.sh
PYTHONPATH=python python -m tessera.compiler.audit support_table --check
PYTHONPATH=python python -m pytest tests/unit/test_complex_*.py -q
```

The tests cover public semantic behavior such as Möbius transforms,
stereographic projection, and Wirtinger differentiation. They do not by
themselves prove native execution on each target.

## Known non-claims

- Do not infer a native kernel from an operation appearing in the public API.
- Do not generalize a fused-program result to every visual-complex operation.
- Do not use the May 2026 planning snapshot as current status: [Visual-complex
  milestone snapshot](../audit/domain/archive/visual_complex_milestone_2026_05.md).
