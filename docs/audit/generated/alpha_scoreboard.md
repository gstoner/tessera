# Functional-Complete Alpha Scoreboard

**Generated. Do not hand-edit.** Regenerate with
`python -m tessera.compiler.generated_docs --write alpha_scoreboard`.

Measures the release definition in
[MASTER_AUDIT §Functional-complete alpha](../MASTER_AUDIT.md#functional-complete-alpha-definition-and-guard-rails):
every alpha family, on every fleet lane, through every stage, produced by
MLIR passes. A cell is alpha-complete only when all five stages are
`native`; anything unproven counts against it (fail closed).

**Alpha-complete cells: 0 of 104.**

## Stages

| Stage | Source | Cells native |
|---|---|---|
| `frontend` | AST: does `graph_ir._OpExtractor` still exist (E2E-REAL-6) | 0 |
| `graph_opt` | not yet derivable per family — shown `unmeasured` | 0 |
| `schedule_tile` | [`bootstrap_prune_gap`](bootstrap_prune_gap.md) family routes | 14 |
| `native_lowering` | [spine](compilation_spine_inventory.md) Level C for the lane target | 0 |
| `execution` | [E2E fleet](e2e_fleet.md) release packet for the lane | 17 |

## Guard rail 1 — bypass surfaces (may only shrink)

| Surface | Count |
|---|---|
| Graph-input `package_*` constructors | 45 |
| Packagers that delegate to a runtime compiler / library | 3 |
| `emit/*` source emitters (`KernelEmitter` subclasses) | 5 |
| Target IR ops without a required contract | 104 |

`tests/unit/test_alpha_scoreboard.py` ratchets these counts and the
per-stage / per-lane native counts against
`tests/unit/alpha_ratchet_baseline.json`: a regression fails CI, and an
improvement fails until the baseline is tightened, so gains are locked.

## Lanes

| Lane | Host | Device | Native stage-cells |
|---|---|---|---|
| `mac_cpu` | Mac M1 Max | arm64 CPU | 1 |
| `mac_gpu` | Mac M1 Max | Apple7 GPU | 3 |
| `luna_cpu` | Princess-Luna | Zen 5 AVX-512 | 2 |
| `luna_gpu` | Princess-Luna | gfx1151 | 5 |
| `bear_cpu` | The-Super-Bear | Zen 2 AVX2 | 4 |
| `bear_gpu` | The-Super-Bear | sm_120 | 13 |
| `taj_cpu` | Tajasarus | Zen 5 AVX-512 | 2 |
| `taj_gpu` | Tajasarus | gfx1201 | 1 |

Known limits: `taj_gpu` shares the ROCm route module with gfx1151, so
its chip-specific state comes from the spine and fleet columns.

## Cells

Legend: ✅ native · 🔴 bypass · 🟡 partial / pending · ⬛ absent / unrouted · ❔ unmeasured

| Lane | Family | `frontend` | `graph_opt` | `schedule_tile` | `native_lowering` | `execution` | Alpha |
|---|---|---|---|---|---|---|---|
| `mac_cpu` | `matmul` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `mac_cpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `reduction` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `attention` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `linalg` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `mac_cpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_cpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `matmul` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `mac_gpu` | `softmax` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | ✅ native | — |
| `mac_gpu` | `reduction` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `attention` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `linalg` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `mac_gpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `mac_gpu` | `ppo` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `mac_gpu` | `ebm` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `mac_gpu` | `clifford` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `reduction` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | 🟡 packet_pending | — |
| `luna_cpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_cpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `luna_gpu` | `reduction` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `luna_gpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `luna_gpu` | `moe` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `luna_gpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `luna_gpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `bear_cpu` | `reduction` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | ✅ native | — |
| `bear_cpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_cpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `bear_gpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ✅ native | — |
| `bear_gpu` | `softmax` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ✅ native | — |
| `bear_gpu` | `reduction` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ✅ native | — |
| `bear_gpu` | `norm` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ⬛ absent | — |
| `bear_gpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ✅ native | — |
| `bear_gpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | ⬛ absent | ✅ native | — |
| `bear_gpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ✅ native | — |
| `bear_gpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `bear_gpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ✅ native | — |
| `bear_gpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ✅ native | — |
| `bear_gpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `bear_gpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `bear_gpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_cpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | 🟡 packet_pending | — |
| `taj_cpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `taj_cpu` | `reduction` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | 🟡 partial | 🟡 packet_pending | — |
| `taj_cpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | 🟡 partial | 🟡 packet_pending | — |
| `taj_cpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `moe` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | 🟡 packet_pending | — |
| `taj_cpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_cpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | 🟡 partial | ⬛ absent | — |
| `taj_gpu` | `matmul` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `softmax` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `reduction` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `norm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `attention` | 🔴 bypass | ❔ unmeasured | ✅ native | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `paged_kv` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `moe` | 🔴 bypass | ❔ unmeasured | 🔴 bypass | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `linalg` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `epilogue` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `replay_ssm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `ppo` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `ebm` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
| `taj_gpu` | `clifford` | 🔴 bypass | ❔ unmeasured | ⬛ unrouted | ⬛ absent | ⬛ absent | — |
