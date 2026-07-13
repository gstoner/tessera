---
status: Informative
classification: Informative
authority: Reader-facing GA / EBM status card; generated dashboards own current target and execution claims.
scope: GA and EBM public semantics, evidence routing, health checks, and non-claims.
last_updated: 2026-07-13
---

# GA / EBM Status

GA and EBM are public Tessera capability families with their semantic contracts
defined in the API specification. This card routes readers to current evidence;
it does not duplicate target rows, kernel counts, or benchmark measurements.

## Current evidence

| Question | Authority |
|---|---|
| Is a particular GA or EBM operation supported on a target? | [Generated support table](../audit/generated/support_table.md) |
| Is that operation backed by a proven native execution path? | [Runtime execution matrix](../audit/generated/runtime_execution_matrix.md) |
| What semantics and public APIs apply? | [GA / EBM execution specification](../spec/GA_EBM_EXECUTION_STATUS.md), [Clifford API specification](../spec/CLIFFORD_SPEC.md), and [EBM API specification](../spec/EBM_SPEC.md) |
| How is the Apple health lane exercised? | [Apple GA / EBM benchmark](../../benchmarks/apple_gpu/benchmark_ga_ebm.py) and its CI configuration |

Use the exact operation-and-target row when making an execution claim. A
registered public API, compiler lowering, reference implementation, and native
target proof are distinct levels of support.

## Health checks

Refresh or validate generated evidence before relying on a changed status:

```bash
scripts/check_generated_docs.sh
PYTHONPATH=python python -m tessera.compiler.audit support_table --check
```

The Apple benchmark is a health signal for its declared lane, not proof that all
GA or EBM operations execute natively on every Apple configuration.

## Known non-claims

- Do not infer native execution from an API name, a manifest row, or a Python
  reference path.
- Do not generalize a target-specific proof to a different target, dtype, shape,
  or execution mode.
- Treat the May 2026 plan as historical context only: [GA / EBM Apple milestone
  snapshot](../audit/domain/archive/ga_ebm_apple_milestone_2026_05.md).
