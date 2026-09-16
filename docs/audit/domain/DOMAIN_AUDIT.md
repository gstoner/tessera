---
last_updated: 2026-09-15
audit_role: theme
---

# Domain audit

This audit maps mathematical and model-facing domains onto the shared native
compiler. It replaces the chronological June closeout narrative; that narrative
is retained in [the archive](archive/DOMAIN_AUDIT_2026-06-11.md).
Global order belongs to the [integrated compiler plan](../compiler/INTEGRATED_COMPILER_PLAN.md).

Support must be reported separately for **reference math**, **compiler
transformation**, **native execution**, and **measured selection**. The
[primitive coverage](../standalone_primitive_coverage.md),
[op/target conformance](../op_target_conformance.md),
[AD ledger](../generated/autodiff_connection_ledger.md) and
[execution matrix](../generated/runtime_execution_matrix.md) provide the
registry/evidence projections. Reconcile stale prose against source and exact
proof scope; do not promote all variants of a domain from one kernel.

**Status is generated, not written here.** The
[domain proof ladder](../generated/domain_proof_ladder.md) derives, per domain,
the registry rows, the AD-ledger rows (adjoint / device-verified), the native
executable rows per target and the plan owners from those registries; it is
drift-gated, and `tests/unit/test_domain_audit_routing.py` fails this document
when it cites a plan ID the integrated plan has retired or re-routed. Read the
ladder before this prose; an empty native column there is a finding.

## Routing to the integrated compiler plan

The plan reorganized into the foundation program (cuts F0–F5) after this audit
was written; the W-items it used to cite are now `successor` rows. Current
owners, as the plan's routing index states them:

| Domain boundary | Plan owner(s) | Was cited as |
|---|---|---|
| Batched native GA products, rotor fusion, packed grades, PGA/CGA | [W6.4](../compiler/INTEGRATED_COMPILER_PLAN.md#w64) (owner: [GA/EBM review](GA_EBM_ARCHITECTURE_REVIEW.md)); higher-order / jets under [AD-HIGHER-1](../compiler/INTEGRATED_COMPILER_PLAN.md#ad-higher-1) | W3.6 → W6.4; W6.3 → AD-HIGHER-1 |
| Traceable energy bodies, effects, resident sampler loops | [W4-PRODUCT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#w4-product-1) (frontend CFG/effects), [AD-SOLVER-IFT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1) (implicit differentiation, OT primitives) | W3.5 → AD-SOLVER-IFT-1; W4 → W4-PRODUCT-1 |
| Saved-state / residual policy, device ownership | [AD-RESIDUAL-EVAL-1](../compiler/INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1), [W2.4a](../compiler/INTEGRATED_COMPILER_PLAN.md#w24a) | W5.1 → AD-RESIDUAL-EVAL-1 |
| Attention / persistent state selectors and tiled SSD | [W5.2](../compiler/INTEGRATED_COMPILER_PLAN.md#w52), [W5.2f](../compiler/INTEGRATED_COMPILER_PLAN.md#w52f) | unchanged |
| Field calculus, PDE, spectral policy | [MSW-9](../compiler/INTEGRATED_COMPILER_PLAN.md#msw-9), [TSOL-POLICY-PHYS-1](../compiler/INTEGRATED_COMPILER_PLAN.md#tsol-policy-phys-1), [TSOL-PHYS-TAIL-1](../compiler/INTEGRATED_COMPILER_PLAN.md#tsol-phys-tail-1); PDE contract under [PDE-STENCIL-FOUNDATION-1](../compiler/PDE_STENCIL_CAPABILITY_PLAN.md) | unchanged |
| Game theory / structured contractions | [TSOL-PHYS-TAIL-1](../compiler/INTEGRATED_COMPILER_PLAN.md#tsol-phys-tail-1) | unchanged |
| Sharding and distributed execution | [DIST-NATIVE-1](../compiler/INTEGRATED_COMPILER_PLAN.md#dist-native-1), [TSOL-SHARD-1](../compiler/INTEGRATED_COMPILER_PLAN.md#tsol-shard-1) | W5.4 → DIST-NATIVE-1 |
| Riemannian OT workload | [RIEMANNIAN-OT](../compiler/INTEGRATED_COMPILER_PLAN.md#riemannian-ot) | unchanged |

## Current boundaries

| Domain | Existing capability | Remaining architectural boundary |
|---|---|---|
| Geometric algebra / Clifford | Signature/product-table references, canonical `clifford_*` operations, differentiated tensor shims, specialized kernels and native grade-pruning passes; since 2026-09-16 the batched geometric product lowers natively (`ExpandProductTable` over any static rank) and executes through MLIR/LLVM on the CPU lane (`cpu` row in the ladder). | The linear/bilinear family (products, contractions, inner/norm, involutions, Hodge star, grade projection, rotor sandwich) lowers natively and executes on the CPU lane and, as native storage packages, on gfx1151/gfx1201/sm_120 (`GA-NATIVE-GPU-2026-09-16`). Still open: exp/log and the field ops, an Apple package route, and measuring the native route against the Python-emitted x86/ROCm/Apple kernels, which remain a second implementation until displaced. General signatures, packed grades and physical derivatives require their own proof. |
| Energy-based models | Reference energies/samplers/losses and specialized update/loss kernels. `geo_sampling.py` uses tape gradients for traceable energies and finite differences otherwise. Since 2026-09-16 the quadratic energy loop differentiates and executes through the MLIR/LLVM CPU lane (`cpu` row in the ladder): compiler-derived gradient, on-device Philox noise, one native call per K-step loop. | The generic `energy.py` Langevin route still uses numerical gradients when no `grad_fn` is supplied. Host gradient evaluation is not a fully resident sampler. Trace an energy body into the native shared AD/loop/ownership path. |
| Attention / persistent state | Canonical families, scheduled packages and bounded native AD; resident O/LSE and isolated CUDA Q/K JVP have explicit packets. | Composed AD, variant breadth, general state lifetimes and cross-target evidence remain separate. KV tiering or a prefetch annotation does not prove overlap. |
| Matrix/field calculus, PDE and spectral | Reference/domain contracts and shared transform/solver surfaces exist. | Coordinate/boundary-condition semantics must reach native operators; use the existing layout, numerical-policy, solver and AD owners. Follow [MSW](../compiler/MATH_SOURCE_WORKSTREAM.md) and the [PDE plan](../compiler/PDE_STENCIL_CAPABILITY_PLAN.md). |
| Game theory / structured contractions | Reference butterfly/coalition operations and derivative laws provide consumers for shared transforms. | Native shared butterfly lowering, batching, numerical-policy transport and measured execution remain workload-specific gates. |
| Domain sharding and distributed execution | Placement/reshard IR and mock/reference execution provide the semantic baseline. | Validate native transports, region placement and communication overlap independently per topology/backend. |

## Corrected historical conclusions

- “GA autodiff fully closed” described a particular Python/canonical-op surface,
  not compiler-generated derivatives for every signature and backend.
- “Apple CPU NumPy is optimal” was unsupported performance reasoning. Retain
  NumPy as the reference; decide native CPU promotion using batched workload
  measurements, dispatch/allocation cost and an accuracy contract.
- “Manifold strings are unvalidated” is obsolete: EBM has an admitted-value
  constraint and fail-closed canonicalization. Native consumption of the
  semantic key is still a separate obligation.
- “Input grades have no consumer” is obsolete: Python grade masks and native
  `GradeFusion`/`ExpandProductTable` consume them. Batched lowering landed
  2026-09-16 (`GA-NATIVE-BATCHED-2026-09-16`); the GPU package route is open.
- A single-dispatch sampler update can still evaluate its energy gradient on
  the host. Whole-loop residency and whole-training throughput need separate
  measurements; old “all gaps closed” language does not establish either.

## Improving domain support through the shared foundation

1. **W6.4 (formerly W3.6): batched native GA first.** Carry algebra, grades, shape and numerical
   policy from the frontend into native products. Prove that the rotor fusion
   reaches the selected package; retain general product-table math as oracle.
2. **W4-PRODUCT-1 with AD-SOLVER-IFT-1 (formerly W3.5/W4): traceable energy and gradient bodies.** Preserve a defined reference
   path for opaque callbacks. For supported traces, generate native derivatives,
   explicit RNG/state and resident loops; reject unsupported captures/effects.
3. **AD-RESIDUAL-EVAL-1 (formerly W5.1) with W2.4a: measured state/residual policy.** Use shared persistent tapes,
   alias/generation proofs and completion ownership. Domain-only checkpoint
   annotations are not executable policies.
4. **W6.4 with AD-HIGHER-1 (formerly W6.3): native finite-algebra lowering.** Reuse a typed multiplication
   table and layout machinery across Clifford and jets. New shared production
   lowering goes through MLIR/LLVM, not another `emit/` source generator.
5. **Promote one end-to-end workload per backend.** Start with batched rotor
   products and a traceable quadratic energy loop. Record forward/derivative
   error, transfers, allocations, retained bytes and complete-step timing.
   Broader energies, AIS, PGA/CGA and domain fusion follow only after the
   relevant semantics and physical path are proved.

The [GA/EBM architecture review](GA_EBM_ARCHITECTURE_REVIEW.md) supplies detailed
acceptance criteria. The [active AD plan](../compiler/AUTODIFF_EXECUTION_PLAN.md)
owns derivative/tape obligations. Backend queues own execution and performance;
this documentation update grants no new device-support status.

## Drift review — 2026-09-15

Re-read against source, the generated dashboards and the plan after the
foundation-program reorganization (413 commits since this audit's date):

- **Plan nomenclature had drifted.** Every W-item the improvement list cited
  had become a `successor` row (W3.5 → AD-SOLVER-IFT-1, W3.6 → W6.4,
  W5.1 → AD-RESIDUAL-EVAL-1, W6.3 → AD-HIGHER-1); the routing table above and
  the renamed items are the correction. The new gate keeps it from recurring.
- **Native GA/EBM execution exists on ROCm and is not described here.** The
  execution matrix carries `rocm_clifford_compiled` (Cl(3,0) table-driven
  bilinear products), `rocm_ebm_compute_compiled` and
  `rocm_ebm_langevin_compiled` as `native_gpu` rows on gfx1151, alongside the
  Apple GPU Clifford/EBM lanes the test tree exercises. The "Existing
  capability" column's "specialized kernels" undersold this; the ladder now
  reports it per target so this prose need not.
- **Code claims still hold** (as of 2026-09-15; the first was closed the next
  day). `ExpandProductTable.cpp` restricted v1 to rank-1 static tensors until
  `GA-NATIVE-BATCHED-2026-09-16` lowered every static rank;
  `ebm/energy.py::langevin_step` still takes finite differences when no
  `grad_fn` is supplied (the numerical-gradient default under W4-PRODUCT-1).
- **Higher-order AD moved.** Native HVP execution and attention JVP landed in
  the compiler (2026-09-13, `native_hvp.py`, `native_attention_jvp.py`) under
  AD-HIGHER-1; the GA/EBM review's W6.4 acceptance ("native jet coefficients
  require scaling, order-zero control, numeric policy") is now the next slice
  on that owner rather than a pre-requisite the tree lacks.
- **Domain suites are green on the Mac** (852 passed, 18 skipped across the
  Clifford, GA, EBM, jet and algebra files on 2026-09-15); that is host-free
  and Apple evidence only and transfers nothing to ROCm or CUDA.

The corresponding engineering loops: the generated
[proof ladder](../generated/domain_proof_ladder.md) (drift-gated) and
`tests/unit/test_domain_audit_routing.py` (routing gate). Both are named in
the plan's maintenance rules.

## Drive through the native backbone — 2026-09-16

Owner direction: close domain gaps by driving each domain through the
MLIR/LLVM compiler rather than through Python-emitted kernels. First slice,
sync `GA-NATIVE-BATCHED-2026-09-16` (W6.4): the batched geometric product is one native lowering
(`GradeFusion` → batched `ExpandProductTable` → scf/tensor/arith) that both
`ts-clifford-opt` and `libtessera_jit` run, and it executes on three CPU hosts
against the GA reference. The proof ladder's GA row gained its `cpu` column
from the execution matrix, not from this prose. Found on the way: the Clifford
and EBM dialects were configured OFF on every WSL build tree (the Mac was the
only host that could even parse them), so the domain dialects had no fleet
lit coverage — Princess-Luna and Super-Bear now configure
`TESSERA_BUILD_CLIFFORD_BACKEND=ON`. Second slice (`GA-NATIVE-FAMILY-2026-09-16`): the
rest of the linear/bilinear family — wedge, contractions, inner, norm, the
involutions, Hodge star, grade projection, rotor sandwich — executes behind
the JIT on the same three hosts. Third slice (`GA-NATIVE-GPU-2026-09-16`): the same lowering
reaches gfx1151, gfx1201 and sm_120 as native storage packages — the kernel
skeleton carries the Clifford op and the dialect expands it; the ladder's GA
row now carries `nvidia_sm120=1` and a second `rocm` row from the matrix.
Next in this stream: measure the native route against the Python-emitted
device kernels (dispatch/allocation/kernel time separately) before any lane
change, an Apple package route (the ladder's `rocm`/`apple_gpu` GA rows are still the
Python-emitted kernels), then the traceable quadratic energy loop (EBM).

## EBM bivector integrator and the overhead measurement — 2026-09-16

Seventh slice, sync `EBM-BIVECTOR-OVERHEAD-2026-09-16` (W4-PRODUCT-1 /
AD-SOLVER-IFT-1). The last manifold and the last measurement. Driving the
bivector integrator through the compiler found what a hand-written kernel
would not have: the Clifford expansion's per-multivector loop nest expressed a
*diagonal* blade map, which no consumer mapping the trailing axis to lanes
could use — it now lowers elementwise, as a select against keep/negate masks
rather than a multiply by a {0, ±1} mask, because a multiply would turn NaN
into NaN where the map drops a blade. The row-program emitter gained
per-feature constant tables to carry those masks into a kernel. The overhead
measurement is structural: one launch per loop against one per step, flat in K
against linear in K. It promotes nothing, and it showed that on gfx1201 and
sm_120 the cooperative kernel is the only compiled Langevin lane there is.

## EBM nonlinear energies and the sphere integrator — 2026-09-16

Sixth slice, sync `EBM-NONLINEAR-MANIFOLD-2026-09-16` (W4-PRODUCT-1 /
AD-SOLVER-IFT-1). Driving three energies and a curved manifold through the
same compiler found what a per-energy kernel would have hidden: `softplus`
had only a placeholder adjoint, so an energy written with it would have
round-tripped its gradient through the Python VJP registry every step; the
row-program emitter could fail with no diagnostic, indexed an empty slot
after a failed lookup, and copied a slot reference into the map it was
inserting into; and registering the domain dialects in `tessera-opt` had left
three `phase7` fixtures red on main, because a registered dialect withdraws
`--allow-unregistered-dialect` from its own ops. All are closed on the branch.
The sphere's two singularities are reported per row and never repaired
silently. Next in this stream: the bivector integrator (M2), which needs the
Clifford `grade` op inside the EBM lowering, and the overhead measurement.

## EBM Langevin loop as one cooperative GPU kernel — 2026-09-16

Fifth slice, sync `EBM-NATIVE-GPU-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). The
tensor-level gradient now lowers inside a device kernel through the compiler
alone: the new row-program emitter turns the lowered `[rows, features]` loop
into one cooperative `gpu.func` (block per row, lane per feature, K steps and
Philox in registers, ordered reductions), the native storage route packages
it, and the ladder's EBM row gains `rocm` and `nvidia_sm120` from the matrix
rows `rocm_ebm_langevin_native_compiled` / `nvidia_ebm_langevin_native_compiled`.
Bit-exact on gfx1151, gfx1201 and sm_120; one launch per loop. Driving it
through the compiler found what no EBM kernel would have: `tessera-opt` had
no way to run a domain dialect's passes in one invocation (now it does, with
EBM and Clifford registered when built), and the assertions-ON driver on
Tajasarus falsified two dialect promises the NDEBUG fleet ran green through.
The Python-emitted device kernels remain the lanes until measured against
this route. Next: the sphere and bivector integrators and the nonlinear
energies — all row programs the same emitter maps — per
[EBM_NATIVE_LOOP_ARCHITECTURE.md](EBM_NATIVE_LOOP_ARCHITECTURE.md).

## EBM quadratic energy through the backbone — 2026-09-16

Fourth slice, sync `EBM-NATIVE-QUADRATIC-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). Driving the
energy through the compiler rather than through `energy_jit`'s Python plan
found three shared-compiler gaps that no EBM-specific kernel would have
exposed: the Graph IR subtraction had no reverse-mode adjoint, the
sum-reduce adjoint's `unsqueeze`/`broadcast` had no linalg lowering, and the
JIT's DPS rewrite did not follow intra-module calls. All three are closed on
the branch and are what let the paired autodiff pass carry `0.5·Σ(x−y)²` to a
native gradient. The loop is a CPU-lane result on three hosts; the ladder's
EBM row gains `cpu=1` from the matrix. The Python-emitted
`x86_ebm_langevin_compiled` / `rocm_ebm_langevin_compiled` kernels are
unchanged and remain the device lanes. Next in this stream, planned in
[EBM_NATIVE_LOOP_ARCHITECTURE.md](EBM_NATIVE_LOOP_ARCHITECTURE.md): G1 serial
device residency through the native-tape route (two precise admission changes),
then the sphere (M1) and bivector (M2) integrators, nonlinear energies (N1),
the cooperative Tile contract (G2) and one compiler driver (T1).

