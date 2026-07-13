---
status: Active sprint plan
classification: Plan
authority: Sequences Task 4 (Philox + energy lowering) and the four cross-cutting concerns flagged at the GA/EBM milestone close
last_updated: 2026-05-18
---

# Sprint plan — Task 4 + cross-cutting concerns

> Current status: [`docs/status/ga_ebm.md`](../../../status/ga_ebm.md).
> The milestone page is the canonical "where we are"; this page is
> "what we're doing next, why, and how the pieces connect."

## Headliner

**Task 4 — Philox-in-MSL + arbitrary `energy_fn` lowering.**  The
biggest single perf + capability step on the GA/EBM roadmap.  Two
sub-deliverables:

  - **Part A: Philox-4x32-10 in MSL.**  Replace the host-supplied
    noise buffers in `ebm_langevin_step`, `ebm_decode_init`, and
    `ebm_sphere_langevin_step` with deterministic on-device
    generation from a 4-element key + counter.  Unblocks T-step
    Langevin chains where re-uploading noise each step is the
    dominant cost.
  - **Part B: Restricted Python AST → MSL energy lowering.**  v1
    `ebm_energy` is a quadratic specialization; an AST visitor
    handling scalar arithmetic + polynomials + a small activation
    set lifts arbitrary user `energy_fn(y)` to MSL.  Enables real
    EBT refinement with per-step gradient recomputation natively,
    not just the fixed-grad snapshot used today.

## Cross-cutting concerns

Four items flagged at the milestone close.  All except the backend
gate need plan + code; the backend gate is acknowledged.

### #15a — Tensor attributes interaction

GA introduces `grade` and `algebra` as per-tensor concepts.  Q2 of
`docs/audit/domain/DOMAIN_AUDIT.md` decided **Multivector is a sibling
tensor kind, not a 7th attribute** — but the canonical reference
[`docs/reference/tessera_tensor_attributes.md`](../reference/tessera_tensor_attributes.md)
still describes only the six canonical attributes
(`shape`/`dtype`/`layout`/`device`/`distribution`/`numeric_policy`)
and doesn't mention the parallel Multivector kind.

**Action.**  Update the tensor attributes doc to:
  - Document Multivector as the parallel kind for Cl(p,q,r) ops.
  - List its required attributes (`algebra`, `grade`, plus the
    inherited dtype/layout/device).
  - Clarify that the six canonical attributes are the **tensor**
    kind's attributes; Multivector has its own kind-specific
    schema and shares dtype canonicalization rules.

### #25 — Partial ≠ ready, category hardening sweeps

The GA + EBM primitives currently live at `status="partial"` in the
standalone-compiler coverage registry.  The category-based
hardening sweeps in `python/tessera/compiler/primitive_coverage.py`
that already closed long-tail axes (`math_semantics`, `shape_rule`,
`dtype_layout_rule`, `lowering_rule`, etc.) do not yet have entries
for `category="geometric_algebra"` or `category="ebm"`.

**Action.**  Extend each `_<axis>_BY_CATEGORY` table to include
explicit settings for both categories, sweeping their primitives
forward axis-by-axis exactly the way the long-tail categories were
closed earlier in 2026-05-10.  Refresh
[`docs/audit/coverage/COVERAGE_AUDIT.md`](primitive_coverage_state.md)
with the post-sweep counts.

### Backend kernel gate (acknowledgement)

GA9 lights up `x86` + `apple_cpu` + `apple_gpu`.  NVIDIA, ROCm,
Cerebras, and Metalium backends are `status="planned"` for both GA
and EBM ops in the manifest, gated on Phase G / H / I respectively.
This matches Decision #1 (CPU-first, then GPU).  **No action
required this sprint** — keep the GA/EBM stack on the CPU-first +
Apple-GPU paths.

### GA6 — Multivector autodiff complexity

GA6 (Clifford autodiff) is the **highest-risk sprint** on the GA
roadmap.  Multivector reverse-mode requires correctly threading the
reverse anti-automorphism through every chain-rule application:

  - VJP of `geometric_product(a, b)` w.r.t. `a` is `out̄ · reverse(b)`
    (not `out̄ · b`).
  - VJP of `rotor_sandwich(R, x) = R x R†` is non-trivial; both
    `R` and `R†` contribute.
  - VJP of `hodge_star` is `±hodge_star` (sign per signature parity).

The headline estimate for GA6 understates the failure modes.

**Action this sprint (preparation, not implementation).**
  - Ship a `multivector_check_grad` helper that finite-differences a
    multivector-valued function and compares against a candidate VJP.
  - Front-load test coverage by running it against the simple closed-
    form ops we already have analytic VJPs for (`norm_squared`,
    `inner`, `geometric_product` operand-wise) so the harness is
    proven before GA6's main implementation lands.
  - Write a planning doc capturing the failure modes + the 2× budget
    rationale.

## Sequence

User redirected mid-sprint (2026-05-18): finish the buffer-pool
sweep + close the 8/9 → 9/9 EBM gap first, then convert
`@clifford_jit` from trace-capture into a real AST → Graph-IR
lowering path for the same `rotor_sandwich → norm` demo.  Task 4
(Philox-in-MSL + energy lowering scaffold) bumps to the next
sprint.

| # | Phase | Item | Risk | Status |
|---|---|---|---|---|
| 1 | Plan | This document | low | landed |
| 2 | #15a | Tensor attributes update | low | landed |
| 3 | #25 | Category hardening sweep | low | landed |
| 4 | GA6 prep | `multivector_check_grad` + planning doc + starter test | medium | landed |
| 5 | EBM closure | Native `ebm_partition_exact` MSL kernel (logsumexp) — closes 8/9 → **9/9** | medium | **landed (2026-05-17)** |
| 6 | Perf | Finish buffer-pool sweep — migrate all remaining dispatchers + harden early-return release safety | medium (mechanical) | **landed (2026-05-18) — RAII `TS_METAL_BUF_ACQUIRE` macros; every exit path release-safe by construction; locked by `test_apple_gpu_buffer_pool.py`** |
| 7 | Compiler | `@clifford_jit` AST → Graph-IR lowering — replace trace-capture with real AST visitor + IR executor | high | **landed (2026-05-17) — extended (2026-05-18) to accept inline int/float/bool literals** |
| 8 | Wrap-up | Milestone + sample JSON | low | landed |
| 9 | Next sprint | Task 4A (Philox-in-MSL) + Task 4B (energy lowering scaffold) | medium / high | **next up** |

## What landed this sprint (2026-05-18 wrap)

- **EBM closure.** [`benchmark_manifest.py::_EBM_APPLE_GPU_FUSED`](../../python/tessera/compiler/backend_manifest.py)
  now lists `ebm_partition_exact` with a stable-logsumexp MSL kernel
  + `tessera.ebm.partition_exact_from_energies` public API.  The
  Python-only EBM set is empty: **9 / 9** native.
- **Buffer-pool sweep + hardening.**  The runtime defines two RAII
  macros (`TS_METAL_BUF_ACQUIRE` / `TS_METAL_BUF_ACQUIRE_WITH_BYTES`)
  that declare a stack-scoped `MetalBufferGuard` whose destructor
  returns the buffer to the pool on **every** exit path (success,
  early `return false;`, exception).  Every dispatcher in
  [`apple_gpu_runtime.mm`](../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm)
  uses the macros — there are no raw `[ctx.device newBufferWith*]`
  calls outside the pool primitive, and no explicit
  `metal_buffer_release` calls outside the guard's destructor.
  Locked by 5 regression tests in
  [`tests/unit/test_apple_gpu_buffer_pool.py`](../../tests/unit/test_apple_gpu_buffer_pool.py).
- **`@clifford_jit` AST → IR lowering.**  Decoration walks the
  function's AST, emits a `CliffordIRProgram` (SSA `%tN` refs +
  per-op `CliffordIROpCall`), validates every op against
  `_CLIFFORD_APPLE_GPU_FUSED`, and freezes a
  `CliffordCompiledArtifact` whose `as_metadata()` embeds the IR.
  Runtime walks the IR and dispatches each op through `jit_bridge`.
  Trace-capture remains as a fallback for source-unreadable
  callables.  Operand vocabulary covers function-arg Names, SSA
  refs from earlier ops, and inline literal refs (`#int:N` /
  `#float:V` / `#bool:0|1`) — so `ga.grade_projection(a, 2)`,
  `ga.grade_projection(a, -1)`, etc. lower without lifting the
  scalar into a synthetic op.

## Out of scope this sprint

- NVIDIA / ROCm / Cerebras / Metalium GA + EBM kernels.  Phase
  G / H / I gate.
- Full energy-lowering AST coverage beyond the quadratic-class
  demo (deferred with Task 4B).
- On-device RNG (deferred with Task 4A).
- GA6 actual VJP implementations — preparation only.
- AST → IR control flow (if / for / while).  The current lowerer
  accepts straight-line assignments + return; control flow lifts
  to a follow-on sprint alongside the broader AST → MSL energy
  lowering work.
