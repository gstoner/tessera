---
last_updated: 2026-10-11
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

**Graph-less rows no longer claim `lowering_rule: complete`
(GRAPHLESS-LOWERING-1, 2026-10-11).** The category table
`_LOWERING_RULE_BY_CATEGORY` stamped `complete` on any family it said
"decomposes" to catalog ops. Measured on the Mac before the fix: 134 of 524
rows had `graph_name=None` and `lowering_rule=complete`, 132 of them from the
category table alone (e.g. `wasserstein_distance`, an `np.sort` numpy
reference no tracer sees; `ctc_loss`, `triplet_loss`, `vmap`, `switch`,
`max_pool`, the `rng_*` key/sampler helpers). `pipeline_gates` legality reads
this axis and admitted them. `_apply_category_overrides` now takes a required
`has_graph_lowering`, and only promotes `lowering_rule` to `complete` when the
row has a Graph op: its own `graph_name`, an `op_catalog` OpSpec (the GA4/EBM
`_planned` rows), a tracer emission (`_TRACED_GRAPH_LOWERINGS`:
`fori_loop`/`cond`/`while_loop`/`scan` → `tessera.control_*`), or a
domain-dialect op that a producer actually emits (`_DIALECT_GRAPH_LOWERINGS`:
`tessera_ebm.energy` and `tessera_ebm.langevin_step{manifold=sphere|bivector}`,
both emitted by `ebm/native_langevin.langevin_loop_module`).
`test_graphless_lowering_rule_gate.py` traces each control-flow entry, calls
each dialect producer and checks the op is in its output, and re-derives each
dialect op's tier from `ods_consumer_audit` (must be `compiler`). An ODS
declaration alone is not evidence (review of PR #941): `tessera_ebm.partition_z`
is `fixture_only`, `tessera_clifford.integral` is `unreferenced`, and
`tessera_ebm.decode_init` is read by a pass but emitted by nothing, so
`ebm_partition_{exact,monte_carlo,ais}`, `clifford_integral` and
`ebm_decode_init` stay open; a negative test pins them, including that
`pipeline_gates` legality refuses them. **105 rows reopened** (94 `partial`, 11
`planned`); each carries `metadata.lowering_rule_reason`, and its
`graph_ir_lowering` moves from `registered` to `stub_required` (the same claim
one field over). `memory_read` / `memory_write` / `memory_evict`, whose Graph
IR disposition is already `runtime_only`, close as the terminal `host_runtime`
instead. The S-series dashboard reports the count instead of "closed
project-wide". The two explicit per-name `complete` claims on Graph-less rows
(`edm_precondition` / `edm_loss_weight`) are withdrawn too. Note what the
`{x86,rocm,apple_gpu}_structured_compute_compiled` lanes are: a hand-built
artifact naming `tessera.<op>` that `runtime.py` (`_STRUCTURED_COMPUTE_OPS`)
dispatches straight back to the numpy reference (`losses.ctc_loss`,
`ops.image_resize`, `D.edm_precondition`, …) — execution-compare evidence,
not a lowering. Re-promote a reopened row by giving it a Graph op, not by
editing the category table.

Checked in the same pass, **not changed here** (recorded for follow-up), on the
105 reopened rows:

- `sharding_rule` is `complete` on 101 of them, from the category table. No
  per-primitive consumer reads this axis (`python/tessera/sharding.py` never
  consults the registry), so for a Graph-less row it states a rule nothing
  propagates. Same overstatement class.
- `transpose_rule` is `complete` on 66 of them, 27 of which have no `complete`
  `vjp`: those rows claim a linear transpose with no registered VJP to derive it from.
- `shape_rule` / `batching_rule` are `complete` on all 105, but both axes
  already carry `AXIS_CONSUMER_WAIVERS` in `test_governance_declarations.py`
  for every row (no consumer anywhere), so they are a known registry-wide
  debt, not specific to Graph-less rows.
- `math_semantics` / `dtype_layout_rule` / `tests` are justified by the Python
  reference and its tests.
- The seven supplemental public ops are stamped `graph_name=f"tessera.{name}"`
  whether or not that op exists, so they pass this gate by construction.
  Three have no `op_catalog` OpSpec (`depthwise_conv1d`, `online_softmax`,
  `online_softmax_state`) yet read `lowering_rule: complete`; the three Block
  AttnRes ops are already reopened by name, and `selective_ssm` has an OpSpec.

**`lowering_rule` overstates Python-reference losses (recorded open
2026-10-11 by #938; resolved 2026-10-11 by GRAPHLESS-LOWERING-1, PR #941).**
The original record: the `loss`-category override stamped `lowering_rule` (and
the shape / batching / transpose / sharding axes) `complete` on loss rows with
no Graph op (`graph_name=None`), e.g. `wasserstein_distance`, `ctc_loss`,
`triplet_loss`, so the axis meant "the category has a rule", not "this op
lowers". Resolution: the `lowering_rule` half is fixed by the entry above
(those rows now read `partial` with `lowering_rule_reason`); the
shape / batching / transpose / sharding half is still open, as listed above.
Nuance since that record: #938 then catalogued `brier_loss`,
`binary_brier_loss`, `ordinal_logistic_loss` and `soft_binned_ece` as
`op_catalog` Graph ops (`tessera.loss.*`), so those four are no longer
Graph-less and keep `lowering_rule: complete` on that evidence.

**Frontend gap (measured 2026-10-11).** A native-target `@jit` over a
`tessera.ops` name with no `op_catalog` spec emits an empty Graph IR function
(effect `top`) and runs eagerly with no `last_fallback_reason`, reproduced on
`main` with the pre-existing `jepa_l2_loss`.

**Calibrated decision heads (added 2026-10-11).** Proper scoring rules and
calibration measurement for bounded-choice heads: `losses.brier_loss`,
`binary_brier_loss`, `ordinal_logistic_loss` (CORAL-style) and
`ordinal_probs`; `tessera.calibration` for top-label ECE/MCE, reliability
bins, the differentiable `soft_binned_ece` penalty, and post-hoc
`fit_temperature`. Python reference with analytic VJP/JVP rules swept by the
autodiff law engine. `brier_loss`, `binary_brier_loss`, `ordinal_logistic_loss`
and `soft_binned_ece` are `op_catalog` Graph ops (`tessera.loss.*`);
`ordinal_probs` is not. No lowering or device result is claimed beyond what
the generated rows show; native lowering is a backend-proof queue item.

## VLM status

The June VLM connector audit is historical. Its useful result was the addition
of explicit preprocessing, modality-splice, position, resampling, and
cross-attention vocabulary. Those operations have Graph dispositions, but
`lowering_rule` is not closed for all of them: `image_resize` and
`interpolate` have no Graph op and were reopened by GRAPHLESS-LOWERING-1
(their structured-compute lane dispatches back to the numpy reference).

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
