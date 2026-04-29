---
status: Draft
classification: Normative
authority: Differentiable neural architecture search design guidance; defers operator semantics to docs/operations/Tessera_Standard_Operations.md and compiler layers to docs/spec/LANGUAGE_AND_IR_SPEC.md
last_updated: 2026-04-28
---

# Tessera Differentiable NAS Guide

Differentiable Neural Architecture Search, or DNAS, should be a Tessera compiler
feature. It fits the existing architecture because Tessera already treats Graph
IR, Schedule IR, numeric policy, effects, autotuning, and replay artifacts as
first-class objects. DNAS extends that model by making architecture choices and
some schedule choices differentiable during search, then freezing them into a
discrete Graph IR before final autotuning and lowering.

DNAS is not a replacement for the existing autotuner. It sits one level higher:

| Layer | DNAS role |
|-------|-----------|
| Graph IR | Represents searchable model structure: mixed ops, architecture parameters, gates, switches, and specialization. |
| Schedule IR | Represents searchable performance knobs: tile sizes, layouts, stages, prefetch plans, and movement choices. |
| Autotuner | Supplies measurements and schedule artifacts; trains or updates hardware-cost surrogates. |
| Runtime/distributed | Provides deterministic all-reduce for architecture logits and checkpoint/replay for search state. |

## 1. Search Space As Graph IR

Tessera should represent each differentiable choice as a `MixedOp`:

```python
from tessera import arch

attn = arch.MixedOp(
    [
        "flash_attention",
        "performer_attention",
        "multi_query_attention",
        "gmlp_block",
    ],
    relax="gumbel",
    temperature=4.0,
    name="block0.attn",
)
gate = attn.gates()
```

Planned Graph IR objects:

| Object | Semantics |
|--------|-----------|
| `arch.Parameter` | FP32 architecture logits, optimized separately from model weights. |
| `arch.MixedOp` | K candidate operators plus one architecture parameter vector. |
| `arch.GumbelSoftmax` | Differentiable categorical relaxation. |
| `arch.HardConcrete` | Sparse gate relaxation for edge pruning. |
| `arch.STEOneHot` | Straight-through hard selection. |
| `arch.weighted_sum` | Differentiable weighted merge of candidate outputs. |
| `arch.switch` | Soft or hard candidate selection. |

Graph IR verification:

- candidate result shapes must match unless the `MixedOp` declares a projection
- candidate dtypes and numeric policies must be compatible
- architecture parameters must be FP32
- gates must have length equal to candidate count
- random relaxations must be seedable under deterministic mode

## 2. Searchable Dimensions

DNAS should cover:

| Search axis | Examples |
|-------------|----------|
| Operator choice | FlashAttention vs. SDPA vs. Performer, MLP vs. gated MLP. |
| Dimensions | hidden size, FFN expansion, head count, groups, kernel sizes. |
| Topology | skip/residual choice, depth expansion, layer replication. |
| State policy | KV cache page size, rolling-window length, optimizer state layout. |
| Schedule knobs | tile sizes, stages, vector width, tensor layout, prefetch strategy. |

Shape-changing choices need either a common super-shape, explicit projections, or
specialization before lowering.

## 3. Bilevel Optimization

DNAS uses separate parameter classes:

- weights `W`, updated on training loss
- architecture logits `alpha`, updated on validation loss plus hardware cost
- optional cost-model parameters `phi`, updated from measurements

Canonical training pattern:

```python
opt_w = tg.Adam(model.weight_parameters(), lr=3e-4)
opt_a = tg.Adam(model.arch_parameters(), lr=5e-3)

for train_batch, val_batch in tg.zipcycle(train_loader, val_loader):
    loss_w = task_loss(model(train_batch.x), train_batch.y)
    opt_w.zero_grad()
    loss_w.backward(wrt="weights")
    opt_w.step()

    with tg.no_grad_for("weights"):
        task = task_loss(model(val_batch.x), val_batch.y)
        lat, energy, mem = arch.hw_cost(model)
        loss_a = task + 1e-3 * lat + 1e-4 * energy + 1e-4 * mem
        opt_a.zero_grad()
        loss_a.backward(wrt="arch")
        opt_a.step()
```

Supported variants should include unrolled inner optimization, implicit
gradients, and straight-through path binarization. Guardrails should include
gradient clipping for `alpha`, entropy regularization early, L0 sparsity late,
and deterministic temperature schedules.

## 4. Hardware-Aware Objective

Hardware cost must be smooth during search and measurable during validation.

Two-tier model:

1. Analytical estimator from Graph IR and Schedule IR features.
2. Learned surrogate trained from on-device measurements.

Feature schema:

```text
flops, bytes, params, tiles, seq_len, dtype, layout,
sm_arch, bandwidth, clock, movement_plan, schedule_hash
```

The current Python foundation exposes `arch.CostFeatures`,
`arch.HardwareCost`, `arch.AnalyticalCostModel`, `arch.hw_cost(...)`, and
`arch.measure(...)`. Future compiler passes should emit the same features from
Graph and Schedule IR.

Example:

```python
lat, energy, mem = arch.hw_cost({
    "flops": 2.0e12,
    "bytes_moved": 1.0e9,
    "params": 10.0e6,
})
```

## 5. Joint Architecture And Schedule Search

Schedule search can be relaxed with `ScheduleSpace`:

```python
sched = arch.ScheduleSpace({
    "tile_m": [64, 128],
    "tile_n": [128, 256],
    "stages": [2, 3, 4],
    "layout": ["row_major", "col_major", "tile(64)"],
})
current = sched.current()
```

Rules:

- Schedule logits are separate from model architecture logits.
- Schedule choices must produce legal Schedule IR before cost prediction.
- Autotuner measurements update the surrogate dataset.
- Final discrete schedules become normal schedule artifacts.

## 6. Freeze And Specialize

Search-time graphs are not deployment graphs. Deployment must freeze:

1. select each `MixedOp` by argmax or sampled low-temperature gate
2. delete unused branches and parameters
3. specialize shapes, layouts, and state policies
4. run normal Schedule IR autotuning
5. lower to Tile IR and Target IR
6. optionally finetune the discrete model

The specialization artifact must include choices, architecture logits, schedule
logits, cost-model hash, graph hash, schedule hash, RNG seeds, and target.

Python foundation:

```python
choices_op = arch.argmax({"attn": attn})
frozen_ops = arch.specialize({"attn": attn}, choices_op)
choices_sched = arch.schedule_argmax(sched)
```

## 7. Distributed And Deterministic Search

Architecture logits are tiny but semantically important. Distributed DNAS must:

- all-reduce `alpha` gradients across the DP mesh
- use ordered reductions under deterministic mode
- checkpoint `{W, alpha, phi, schedule_alpha, RNG, graph_hash, schedule_hash}`
- support submesh search variants, but log every population member and merge
  event for replay

## 8. Implementation Map

| Area | Current foundation | Needed next |
|------|--------------------|-------------|
| Python surface | `tessera.arch` parameters, relaxations, mixed ops, schedule spaces, bilevel plans | Integrate with full model modules and optimizer wrappers. |
| Graph IR | ODS anchors for `arch.parameter`, `arch.gumbel_softmax`, `arch.hard_concrete`, `arch.ste_one_hot`, `arch.weighted_sum`, `arch.switch`, `arch.mixed` | Lower Python search modules into these ops automatically. |
| Schedule IR | `schedule.knob`, schedule artifacts, autotuner cache | Feature extraction pass from Schedule IR. |
| Cost model | Bayesian autotuner measurements, analytical proxy, learned online surrogate | Replace linear surrogate with MLP once tensor autodiff is ready. |
| Runtime | Deterministic alpha all-reduce helper and replay docs | Distributed runtime binding for alpha gradients. |
| Tooling | QA/reliability docs | Alpha histograms, cost curves, predicted-vs-measured drift reports. |

## 9. Graph IR And Schedule IR Expansion

DNAS has concrete compiler anchors today, but only the foundation is wired.

Graph IR dialect anchors live in `src/compiler/ir/TesseraOps.td`:

```mlir
%tau   = tessera.graph.constant 4.0 : f32
%alpha = tessera.graph.arch.parameter {size = 4, name = "block0.attn"}
%gate  = tessera.graph.arch.gumbel_softmax %alpha {temperature = 4.0, seed = 17}

%y0 = tessera.graph.op.flash_attention(%x)
%y1 = tessera.graph.op.performer_attention(%x)
%y2 = tessera.graph.op.multi_query_attention(%x)
%y3 = tessera.graph.op.gmlp(%x)
%y  = tessera.graph.arch.weighted_sum %y0, %y1, %y2, %y3 by %gate
```

Schedule IR anchors live in
`src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td`:

```mlir
%tm = tessera.schedule.knob %matmul {name = "tile_m", choices = [64, 128]}
%tn = tessera.schedule.knob %matmul {name = "tile_n", choices = [128, 256]}
%st = tessera.schedule.knob %matmul {name = "stages", choices = [2, 3, 4]}
```

Compiler work still needed:

- Python lowering from `arch.MixedOp` into `arch.*` Graph IR ops.
- Verifiers for candidate shape/dtype/numeric-policy compatibility.
- Autodiff registration so `backward(wrt="arch")` reaches architecture logits.
- Schedule feature extraction from `schedule.knob` and movement plans.
- On-device measurements feeding the learned surrogate through the autotuner.

Examples:

- `examples/dnas_graphir_sketch.mlir`
- `examples/dnas_schedule_autotune.py`

## 10. Autodiff And Runtime Plumbing

DNAS uses explicit autodiff partitions:

```python
loss_w.backward(wrt="weights")
loss_a.backward(wrt="arch")
loss_phi.backward(wrt="cost")
```

The current Python foundation validates these names with
`arch.validate_backward_wrt(...)` and keeps architecture parameters as FP32
`arch.Parameter(kind="arch")`. Runtime implementations must preserve this
partitioning so AMP can apply to model weights without casting architecture
logits.

Architecture-gradient all-reduce must be deterministic:

```python
avg_alpha_grad = arch.deterministic_alpha_all_reduce(rank_grads, op="mean")
```

The reduction is ordered and uses compensated summation in the Python reference
helper. Production collectives should preserve the same rank order under
deterministic mode.

## 11. Phase Guidance

DNAS should be staged:

1. Phase 5 design surface: Python objects, docs, tests, and cost features.
2. Phase 5 compiler surface: Graph IR arch ops and Schedule IR knob ops.
3. Phase 6 runtime surface: deterministic distributed alpha updates and replay.
4. Production tooling: dashboards, measurement datasets, and export bundles.

This keeps the feature aligned with Tessera's architecture instead of building a
separate NAS framework beside the compiler.
