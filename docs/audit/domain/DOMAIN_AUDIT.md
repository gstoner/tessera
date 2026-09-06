---
last_updated: 2026-09-06
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

## Current boundaries

| Domain | Existing capability | Remaining architectural boundary |
|---|---|---|
| Geometric algebra / Clifford | Signature/product-table references, canonical `clifford_*` operations, differentiated tensor shims, specialized kernels and native grade-pruning passes. | Native `ExpandProductTable` remains rank-1; connect batched typed products and rotor fusion to executable native packages. General signatures, packed grades and physical derivatives require their own proof. |
| Energy-based models | Reference energies/samplers/losses and specialized update/loss kernels. `geo_sampling.py` uses tape gradients for traceable energies and finite differences otherwise. | The generic `energy.py` Langevin route still uses numerical gradients when no `grad_fn` is supplied. Host gradient evaluation is not a fully resident sampler. Trace an energy body into the native shared AD/loop/ownership path. |
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
  `GradeFusion`/`ExpandProductTable` consume them. Batched lowering remains open.
- A single-dispatch sampler update can still evaluate its energy gradient on
  the host. Whole-loop residency and whole-training throughput need separate
  measurements; old “all gaps closed” language does not establish either.

## Improving domain support through the shared foundation

1. **W3.6: batched native GA first.** Carry algebra, grades, shape and numerical
   policy from the frontend into native products. Prove that the rotor fusion
   reaches the selected package; retain general product-table math as oracle.
2. **W3.5/W4: traceable energy and gradient bodies.** Preserve a defined reference
   path for opaque callbacks. For supported traces, generate native derivatives,
   explicit RNG/state and resident loops; reject unsupported captures/effects.
3. **W5.1/W2.4a: measured state/residual policy.** Use shared persistent tapes,
   alias/generation proofs and completion ownership. Domain-only checkpoint
   annotations are not executable policies.
4. **W6.4 with W6.3: native finite-algebra lowering.** Reuse a typed multiplication
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
