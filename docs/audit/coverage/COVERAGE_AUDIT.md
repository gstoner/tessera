---
last_updated: 2026-09-04
audit_role: theme
---

# Coverage Audit

This page explains how to interpret Tessera's coverage evidence. It does not
own counts or per-operation status. Current values come from the generated
dashboards:

- [`compiler_progress.md`](../generated/compiler_progress.md) — phase and
  integration rollup;
- [`support_table.md`](../generated/support_table.md) — per-operation
  API→Graph→Schedule→Tile→Target→runtime state;
- [`s_series_status.md`](../generated/s_series_status.md) — primitive contract
  axes and exact-target backend proof;
- [revision-bound test coverage](#test-coverage-evidence) — direct, family,
  structural, and hardware-gated evidence;
- [`runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md) —
  paths that actually launch;
- [`single_gpu_closeout.md`](../generated/single_gpu_closeout.md) — ownership
  classification for the remaining rows.

## Closed foundation

The compiler-facing API, frontend-capture, Graph-registration, Schedule-IR,
Tile-IR, runtime-readiness, verifier, batching, transpose, and lowering axes are
closed in the current generated rollup. These are regression gates, not active
implementation programs.

`Graph IR registered` does not mean every registry entry is a device operation.
Host APIs, runtime-only surfaces, and explicit `not_applicable` dispositions are
tracked separately. Likewise, a registry-level `backend_kernel=partial` does
not erase exact-target proof already earned by x86, Apple, ROCm, or NVIDIA.

## Active queues

### Target and distributed closure

The four reference Target-IR rows are `all_gather`, `all_reduce`, `all_to_all`,
and `reduce_scatter`. Their Schedule and Tile contracts are complete; native
multi-rank transport and exact-device packets are the missing proof. They must
not be treated as single-GPU compiler gaps.

The live sharding queue is classified by the generated dashboard rather than
copied here. Its ownership groups are:

- one local compiler case, `factorized_matmul`, needs layout/shard metadata
  preservation proof;
- `moe_dispatch` and `moe_combine` require real multi-rank transport;
- the attention, EBM, solver, sparse, spectral, state-space, and
  state-update rows need domain-specific propagation rules and mock-mesh or
  native evidence.

### Target-specific backend proof

Backend promotion is architecture-owned. The registry's conservative
`backend_kernel` axis is a routing surface, not an all-up compiler veto. Only
exact-target `device_verified_jit` or `device_verified_abi` evidence closes a
hardware pathway. `fused` or `packaged` establishes implementation ownership;
`artifact_only` establishes compilation only.

### Autodiff and structured programs

Planned VJP/JVP rows are transformation work, not missing forward execution.
Prioritize model-facing spectral, solver, collective, and structured-region
products. Bounded `if`, counted `for`, canonical bounded `while`, and forward
`control_scan` already exist. General CFG recovery, multi-block regions,
Presburger shape proof, scan adjoints, and lowering selected checkpoint plans
into the generated region adjoint remain open.

### Evidence quality

Structural-only rows and missing benchmark inventory are proof queues, not
automatically implementation queues. Add direct numerical fixtures where the
operation has meaningful value semantics. Add performance packets only for
native/fused paths whose selection or promotion depends on timing. Host-only
metadata constructors, serialization, and structural transforms should not be
forced into artificial kernel benchmarks.

## VLM status

The June VLM connector audit is historical. Its useful result was the addition
of explicit preprocessing, modality-splice, position, resampling, and
cross-attention vocabulary. Those operations now have canonical Graph
dispositions and the project-wide `lowering_rule` axis is closed.

The remaining VLM work is physical and model-level:

- exact-target backend promotion for connector and attention families;
- native fused paths only where measured against the composed implementation;
- full checkpoint/processor/model execution with architecture-owned evidence;
- distributed placement for variable-length media and cache state.

Do not reopen a generic "VLM Graph lowering" project from the historical P0/P1
labels. Select concrete rows from the generated support and backend tables.

## Reading rules

1. Start with `compiler_progress.md` for the phase containing a gap.
2. Use `support_table.csv` to identify the operation and first incomplete layer.
3. Use `s_series_status.md` and the backend plan for exact-target ownership.
4. Use `test_coverage.csv`, benchmark packets, and runtime provenance to decide
   whether the missing item is implementation, correctness evidence, or
   performance evidence.
5. Never copy generated totals into authored plans.

## Source material consolidated

- `archive/advanced_examples_capability_gap.md`
- `archive/kv_cache_coverage_matrix.md`
- `archive/partial_ops_uplift_plan.md`
- `archive/primitive_coverage_state.md`

## Test coverage evidence

Decision #26 assigns `test_coverage.csv` and its Markdown companion to the
required Validate audit job's `coverage-evidence-<tested SHA>-<attempt>` artifact.
Find the [Validate run](https://github.com/gstoner/tessera/actions/workflows/validate.yml)
for the revision being assessed, then download its coverage artifact. Check
`manifest.json` for the source commit/tree digest and file hashes before citing
rows. PR runs describe the tested merge revision, not necessarily the PR head.
Artifacts expire after 90 days; unavailable evidence must not be cited as current.
For an older revision, check out that revision and regenerate, recording that
the evidence was regenerated rather than recovered from the original run.

Local command (from the repository root):

```sh
PYTHONPATH=python python scripts/coverage_evidence.py --output /tmp/coverage-evidence
```

For a convenient local dashboard, use the existing
`python -m tessera.compiler.generated_docs --write test_coverage` command. Its
outputs are ignored by Git. CI checks generator determinism and the semantic
row/heading contracts; static test references do not prove that tests passed
or that a kernel executed on a device. Other dashboards remain committed.
