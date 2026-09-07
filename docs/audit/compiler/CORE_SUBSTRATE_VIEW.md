---
last_updated: 2026-09-07
audit_role: reference
---

# Core Substrate View — capability demands on one native compiler

Start at [README.md](README.md). This reference maps capability demands onto
existing owners; it creates no work IDs or independent build sequence.
[The integrated plan](INTEGRATED_COMPILER_PLAN.md#capability-plan-reconciliation--2026-09-07)
owns global order. The [August snapshot](archive/CORE_SUBSTRATE_VIEW_2026_08.md)
preserves the original demand analysis, mathematics and proposed phases.

## 0. Scope and evidence

The September reconciliation checks repository implementation and recorded
packets. It does not reproduce external papers or renew every historical
measurement. The 13 checks in
[verify_substrate_math.py](../../../research/core_substrate/verify_substrate_math.py)
are a bounded model/oracle harness, not a proof of all compiler or performance
claims. Graham's bound assumes identical machines; resource-overlap bounds do
not include every dependency, fill/drain or runtime cost.

Keep reference laws, serialized native contracts, exact-target execution and
measured candidate admission distinct. Generated dashboards are inventory
entry points; reconcile conflicts with source and revision-bound evidence.

## 1. The demand matrix

| Source | Shared demand | Current owner |
|---|---|---|
| CAKE / compiler enhancement | Derived legality, stated/derived schedules, measured pruning | W2.4a / W5.2 / F0–F3 |
| Game theory | Butterfly compatibility, solver certificates, batching | REF-TIER-PHYS-1 / LAYOUT-ALG-1 / AD / F4 |
| TileSight | Target calibration, resource models and rasterization evidence | W5.2 / F3 and target queues |
| TileRT | Dependence edges, composition, completion and overlap | W2.4a / W5.2 / F4 |
| SparDA | Data-dependent movement, cache invariants and layout | LAYOUT-ALG-1 / W2.4a / F2 / F4 |
| FORGE | Stateful epilogues and materialization legality | LAYOUT-ALG-1 / NUMPOL / F3 |
| PDE/stencil | Operator/domain/BC contracts and stability consumers | FA-7 / REF-TIER-PHYS-1 / AD |

## 2. The shared substrate — nine investments, many consumers

### S1. Derived facts across block-argument edges (the legality substrate)

Derived Tile provenance, loop-carried values and allocation/token lifetime
checks exist in `TileValueProvenance`, `TileDataflowLegalityPass` and
`TileBarrierReuseLegalityPass`. The earlier blanket block-argument fail-open
finding is historical. W2.4/W2.4a own extensions to general ownership,
outstanding generations and control flow. Each extension needs alias/SSA
forwarding negatives and a proved completion boundary; a typed token alone
is insufficient.

### S2. One schedule/action-DAG object that survives into IR, with two entry points

Content-addressed Schedule/Tile artifacts and native replay exist for bounded
families. F0/F2 still own the remaining Graph constructors and complete
ABI/layout/policy/provenance projections. W5.2 owns action-DAG composition and
automatic dependence edges. A typed Python wrapper is not proof that a whole
program body or its schedule is owned by serialized IR.

### S3. Calibration + certificate/resource-aware arbitration

W5.2 and F3 own calibration, candidate evaluation and admission. WSL host-wall
packets can provide regression or pruning evidence without becoming selector
evidence. PR #733's runtime-route promotion and declined package-level
promotion illustrate separate scopes: an admitted kernel does not establish a
complete-package speedup. Require paired measurements, target identity, timing
domain, resources and unchanged acceptance thresholds.

### S4. Semantic keys + emitted certificates as the safety architecture

F0/F2 require semantic keys and certificates to reach the actual consumer.
Missing status must not be interpreted as native execution: low-precision
Apple softmax now uses status-returning ABIs, while remaining void-ABI routes
need review. F3 candidate identity must bind the emitted program and resource
record to the original/transformed pair. Reference mathematics does not
substitute for a checked invocation or a physical packet.

### S5. `numeric_policy` carried below Graph IR (the Decision #32 carrier)

NUMPOL-CARRIER-1 and FA-1–FA-7 own policy propagation, error-budget composition
and concrete approximation/spectral/PDE consumers. Native carriers exist in
bounded families; neither "no carrier below Graph" nor universal policy closure
is accurate. Reassociation, mixed storage and accumulated error require explicit
legality at the selected consumer, including unsupported-policy rejection.

### S6. The general structural-op tranche

Existing primitives should be consumed rather than proposed again:

| Demand | Existing boundary / remaining work | Owner |
|---|---|---|
| Prefix/region scans and segmented sums | Prefix scans, `control.scan` and `segment_reduce(..., op="sum")` exist; verify ragged reach weights and native program transforms. | W4 / native AD |
| Tridiagonal solve | Shared Schedule/Tile and bounded x86/gfx1151 physical consumers exist; extend only named envelopes. | REF-TIER-PHYS-1 / FA-7 |
| Coalition butterfly | One shared coalition carrier and independent native consumers exist. FFT replacement, layout and bit-identity compatibility remain open. | REF-TIER-PHYS-1 / GAME G1b / LAYOUT-ALG-1 |
| Depth statistics and merge | Block AttnRes has typed contracts and bounded gfx1151 execution. General attention LSE ownership is a different contract. | BLOCK-ATTNRES-1 / F2 / AD |
| Sparse index/movement and spectral BC handling | Check the selected operation's current producer and boundary conditions; no blanket missing-op list. | F2 / F4 / FA-7 |

Do not create a second generic-ops queue from the historical S6 proposal.

### S7. Memory tiers + data-dependent movement as scheduled, legal IR

W2.4a and F4 own storage lifetime and stateful movement; DIST-NATIVE-1 owns
real cross-rank transport. Allocation-specific arenas, bounded paged-cache
contracts and device-ring experiments exist. These do not prove general
host/device prefetch feasibility, cache-state invariants or distributed overlap.
SparDA's workload must consume the same alias/effect/completion authority.

### S8. The transform substrate (batching, transpose, implicit-diff, schedule AD)

The [AD execution plan](AUTODIFF_EXECUTION_PLAN.md) owns native transpose,
batching, structured products, persistent tapes, higher-order composition and
solver integration. Linear transpose and bounded native products exist;
"every backward is written twice" and "implicit differentiation has no owner"
are obsolete. General while/mixed-state tapes, checkpoint-plan execution,
constrained-solver certificates and composition remain explicit residuals.
Game, ES, PDE and OT consumers reuse that authority.

### S9. Locality + residency: fusion legality as declared, provable metadata (from FORGE)

LAYOUT-ALG-1, Schedule/Tile allocation ownership and F3 stateful fusion own
FORGE's locality/materialization demands. The original FORGE proposal is
[archived behind its routing note](FORGE_ASSESSMENT.md); it is not an instruction
to add a second locality lattice or residency registry. Preserve numeric policy,
effects, aliasing, optimizer-state order and reduction semantics; prove absence
of forbidden materialization in IR, then measure the complete package on its
own architecture.

## 3. Ownership map

S1 → W2.4/W2.4a; S2 → F0/F2/W5.2; S3 → F3/W5.2; S4 → F0/F2/F3;
S5 → NUMPOL/FA-1–FA-7; S6 → the existing family owners above; S7 → W2.4a/F4/
DIST-NATIVE-1; S8 → the AD execution plan; S9 → LAYOUT-ALG-1/F3.
These replace the old "unowned" claims. A mapped owner is not a completion claim.

## 4. Build sequence

Follow the [integrated dependency order](INTEGRATED_COMPILER_PLAN.md#dependency-order-and-exit-tests):
serialized ownership and invocation checks, AD/numerical legality, native
rewrites and specialization, then target workload admission. The archived
P0–P5 proposal is historical; do not run it as a parallel queue.

## 5. Honest limits

Architecture proof never transfers. Assertions-disabled LLVM cannot close the
assertions-enabled validation gate. Mock transport, host-wall timing and reference
algebra retain their own evidence scopes. The current route/envelope inventory
must be consulted before retiring a producer or promising broader support.
