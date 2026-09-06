---
last_updated: 2026-09-06
audit_role: plan
plan_state: landing
---

# Autodiff execution and remaining integration

Start at the [compiler audit map](README.md) for document ownership.

This is the active scoped AD plan. It consolidates the unification phases P0–P6,
architecture findings A1–A8/B1–B8 and capabilities D1–D7, and next-generation
AD-LAW/WEIL/JET/OPERATOR work. Existing IDs are retained; no parallel scheduler
or new AD engine is proposed. Global order belongs to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md), particularly W4,
W5.1 and W6 under **IR-NATIVE-FOUNDATION-1**.

The [AD specification](../../spec/AUTODIFF_SPEC.md) owns public semantics.
The [connection ledger](../generated/autodiff_connection_ledger.md) and
[law audit](../generated/autodiff_law_audit.md) distinguish reference rules,
valid native derivatives, runtime binding and exact-target proof. A ledger or
review disagreement must be reconciled against source and revision-bound
results; neither an old paragraph nor a positive forward row proves backward.

## Implemented foundation to preserve

| Foundation | Current evidence boundary |
|---|---|
| Differentiation requests, paired ABI and proof projection (P0–P4) | `autodiff_request.py`, `autodiff_ledger.py`, `AutodiffPairedPass.cpp`; public requests and native products exist. Family-wide/target-wide closure does not follow. |
| Forward mode and exact compiler HVP (D2 / AD-FWD-* / AD-HIGHER-1) | `TangentInterface.cpp`, `AutodiffForwardPass.cpp`, `JitFn.compiled_hvp_ir`. The eager `autodiff/grad.py::hvp` finite-difference helper remains a different route. |
| Structured AD (D3/D4 / AD-REGION-1) | Native bounded SCF/CFG products, recorded effects and residual contracts exist. See [control-flow contract](../../spec/CONTROL_FLOW_CONTRACT.md) and [effect plan](W4_ADMISSIBLE_EFFECTS_PLAN.md) for exact envelopes. |
| Persistent tensor products | `native_persistent_tape.py` consumes independently bufferized forward/backward products. Static f32 slots ≤1024 elements, logical temporaries ≤4096 bytes; serial CUDA/HIP proof, retained outer SAVE state and inner replay. This is not general persistent while/mixed-state support. |
| Resident attention O/LSE and Q/K JVP | `native_attention_program.py` and the explicit `JitFn.compile_native_attention_jvp` bind one isolated SM120 attention trace. Active tangent order is preserved; arbitrary surrounding JIT compositions remain unsupported. |
| Law/algebra substrate (AD-LAW-1/2, AD-WEIL-1) | `autodiff/laws.py`, `algebra.py`, `derivative_contract.py`; Dual/TruncatedJet and a finite multiplication table exist. Do not schedule their invention again. |
| Structured jets and first rule retirements | `jet.py` and `RETIRED_HAND_RULES` distinguish derived production rules from retained oracles. Online attention jets are reference evidence, not native higher-order attention. |
| Operator tangents / IFT | `operator.py`, `implicit.py`, `src/solvers/core/passes/NewtonAutodiff.cpp`; operator composition and root-conditioning certificates exist. A nonsingular-root check is not universal constrained-optimization/KKT proof. |
| Reference Jacobian reuse (D1/B1) | `transforms.py::jacrev` records the forward once and retains its tape. The old repeated-forward defect is fixed. General batching and compressed seeds remain separate. |

[Loop11 evidence](../../../benchmarks/NATIVE_STORAGE_FOLLOWUP.md) records the
latest bounded tape/JIT attention implementation. Earlier loop packets retain
their own envelopes. No performance promotion follows from correctness packets.

## Carried-forward tasks and acceptance gates

Within each row, extend the existing producer and verify the emitted product.
Rows are dependency groups; the integrated plan determines cross-domain order.

| Existing owner / source items | Remaining deliverable | Completion evidence |
|---|---|---|
| **W4-PRODUCT-1 / AD-REGION-1** — A1–A5, D3/D4, P5 | General persistent nested tapes: dynamic valid extents/capacity, scalar/predicate and mixed tensor slots, bounded while/CFG replay, alias/activity joins and operation-owned effects. Extend the split native ABI rather than wrapping the combined recompute consumer. | Native forward saves the executed path and live state; separately launched backward consumes it after input mutation. Test zero-trip, nested branches, overflow, allocation failure, repeated backwards and invalid-after-close on each owning backend. |
| **W2.4a / W4-EFFECTS-1 / SO-2** — A5, D3/D4 | Concurrent backward users, generation-sensitive aliases, cross-queue completion and safe release of persistent state. | Multiple live generations and adversarial completion orders; release only after all consumers finish. Race/failure tests plus actual CUDA/HIP/Metal evidence, not host ownership alone. |
| **AD-FWD-NATIVE-1 / W6.1** — A4, D2, P5/P6 | Compose automatic attention Q/K/V AD with surrounding tensor operations; extend masks/bias/dropout/cache policies only through explicit contracts. Broaden loss, optimizer, spectral and solver products. | Compiler-generated derivatives match analytic/adjoint/finite-difference oracles; O/LSE identity, inactive tangents and paired ABI survive packaging; no Graph redispatch or hidden Python adjoint. |
| **AD-RESIDUAL-EVAL-1 / AD-TREEVERSE-1 / W5.1** — A6/A8/B6, D5 | Consume selected SAVE/RECOMPUTE/HYBRID plans in native counted-loop programs; extend from bounded executed candidates to retained production policies. Keep EBM annotations outside the default path. | Compare actual forward/replay/backward work and unique retained bytes against the selected plan. Complete-backward exact-device timing with policy identity; analytical and WSL pruning evidence cannot silently select. |
| **AD-HIGHER-1 / W6.1** — A2/B4, D6 | Broader exact forward-over-reverse and nested derivative programs, including structured residuals and policies. | Validate the emitted second-order program; distinguish it in provenance from finite differences. Do not remove the eager identity-tape restriction until its replacement covers the public contract. |
| **AD-BATCH-1** — B2/B3, D1/D7 | Real batching/seed-axis propagation over supported programs; explicit fallback when a rule is missing. | Output equivalence across nonleading axes, nested transforms and pytrees; count primal execution/dispatches to prove a transform rather than a hidden Python loop. This is a dependency of physical jets. |
| **AD-SPARSE-1 / W6.2** — B5, D7 | Jacobian/Hessian structural sparsity propagation, coloring and compressed seeds. | Reconstruct independent dense derivatives on sparse fixtures; prove structural-zero safety and demonstrate work proportional to colors. No comparative claim about other frameworks is required. |
| **AD-FWD-DIST-3 / P6** — B7 | Broader subgroup/process transport and native collective derivative packages. | Adjoint identity on actual multi-rank NCCL/RCCL or other admitted transport, with exact devices, rank maps and timing. Mock collectives remain reference evidence. |
| **AD-SOLVER-IFT-1 / W3.5 / AD-OPERATOR-1** — B8 | Broader residual/predicate/solver envelopes, demand-driven operator consumers and clean performance evidence; constrained/KKT cases need their own hypotheses. | Residual/convergence and conditioning certificates, adjoint tests, compiled child identity and owning-device packets. Preserve the shared Riemannian-OT consumer instead of adding another solver stack. |
| **AD-JET-STRUCT-1 / AD-RETIRE-*** — D6 | More structured families and per-family law dashboard evidence; finish safe hand-rule/oracle retirement and geometric-tape absorption. | Laws anchored to the canonical forward, quotient consistency, ties/guards/dtypes/kwargs preserved, full survivor envelope and recorded backend soak. Attention rules cannot retire merely because the dense no-dropout core works. |
| **AD-JET-IR-1 / W6.3** | Lower the coefficient axis and finite-algebra evaluation through MLIR/LLVM, with native policy/layout/residual contracts. | Depends on structured product ABI, real batching, LAYOUT-ALG-1 and NUMPOL-CARRIER-1. CPU oracle plus independent GPU proof, unsupported-pair rejection and measured high-order scaling/conditioning/footprint. No new Python emitter as the production owner. |
| **AD-WEIL-1 registry integration / AD-CLOSEOUT-1** — P0–P6, A7 | Integrate derivative semantic fields into appropriate coverage views; audit active families for unresolved `custom_adjoint_call` and proof-totality gaps. | Every requested native family either has the complete verified path or rejects explicitly. Reference registration, runtime-bound and exact-device axes remain independent. |
| **AD-CERT-1 / future algebra instances** | TaylorModel, ChebJet, MixedPartial and enclosure/estimator extensions remain consumer-gated. | Name the consumer first; outward rounding or stochastic convergence evidence must match the declared claim. A reference protocol does not establish native certified execution. |

The original estimator obligations survive: explicit RNG keys, declared
pathwise/score-function/constant-noise semantics, random effects and reproducible
streams. Mathematical acceptance also retains coefficient scaling, primal-only
control (`control_at_order=0`), kink selection, cotangent/coefficient numeric
policy and a justified `pd_witness`; none may disappear during native lowering.

## FA-2 adjoint-contract follow-up

The [functional-analysis consolidation](INTEGRATED_COMPILER_PLAN.md#functional-analysis-contracts--consolidated-ownership)
retains FA-2 under AD-LAW / AD-CLOSEOUT-1. The adjoint harness and
canonical-forward checks already exist in `autodiff/laws.py` and
`compiler/law_audit.py`; do not recreate them. Remaining scope is a public debug
adapter over that engine, explicit norm-aware tolerances and coverage evidence
integration. Acceptance retains wrong-VJP and matched-wrong JVP/VJP negative
fixtures, verifies adapter behavior, and ties any coverage transition to the
actual law result. Amend the coverage contract before changing auto-flips;
reference laws cannot imply native derivative execution. This work does not
wait for the FA-1 numerical-budget consumer.

## Backend acceptance

| Backend | Current bound and follow-up |
|---|---|
| CUDA | Static split tapes and isolated resident Q/K programs proved on RTX 5070. General composition, asynchronous retirement and performance promotion remain open. |
| ROCm | Static split tapes proved on gfx1151 with AMDGPU-owned private storage. Native Q/K consumer and hardware-counter attribution remain independent work. |
| Apple | Requires MSL dynamic/threadgroup binding and completion-owned residual/O/LSE integration. Shared IR or CUDA pointer ownership is not Metal proof. |
| x86 | Supplies the host companion for GPU products; that does not prove CPU execution of those products. Existing CPU/AVX-512 AD families retain their own evidence. |

All four backend queues retain their existing synchronization keys and own
hardware promotion. This consolidation changes documents, not support rows.

## Archive disposition and migration map

| Historical document | What survives here |
|---|---|
| [Unification P0–P6](archive/AUTODIFF_UNIFICATION_PLAN.md) | Foundation table; native family expansion, public provenance, paired ABI, collective proof and AD-CLOSEOUT-1 gates. P0–P4 are not reopened wholesale. |
| [Architecture A/B/D findings](archive/AUTODIFF_ARCHITECTURE_REVIEW.md) | A1–A5 → persistent/activity/forward/higher rows; A6/A8/B6 → measured policies; A7 → closeout; B1/B2/B3 → corrected Jacobian baseline and batching; B4/B5/B7/B8 → higher/sparse/distributed/IFT. |
| [Next-generation design](archive/AUTODIFF_NEXTGEN_PLAN.md) | Laws, finite-algebra mathematics and rejection/retirement obligations remain references. Implemented LAW/WEIL/OPERATOR substrate is preserved; STRUCT breadth, native JET, registry integration, geometric absorption and CERT remain explicitly carried forward. |

Archive means **superseded queue**, not completed feature. Original filenames
remain small routing documents so diagnostics and source references still resolve.
The new plan stays `landing` until its active deliverables and evidence gates
are complete or explicitly rehomed again.
