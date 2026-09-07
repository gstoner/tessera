---
last_updated: 2026-09-07
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
| Persistent tensor products | `native_persistent_tape.py` consumes independently bufferized forward/backward products. Static f32/f64 slots ≤1024 elements, dtype-sized logical temporaries ≤4096 bytes; serial CUDA/HIP execution of bounded SAVE/HYBRID/recompute-all and proven counted-while normalization. Integer/predicate/dynamic slots and data-dependent while remain open. |
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

## Capability-review consumers of the AD execution plan

The [differentiable-programming review](DIFFERENTIABLE_PROGRAMMING_REVIEW.md)
retains book-derived labels and mathematical rationale; its old status tables
are historical. The [integrated capability mapping](INTEGRATED_COMPILER_PLAN.md#capability-plan-reconciliation--2026-09-07)
owns cross-plan sequencing. Reuse the active work above:

- C1/C2/C3 map to native linear/nonlinear products, kink policy and explicit
  effect/RNG semantics under AD-CLOSEOUT-1 and NUMPOL. Their Python or bounded
  compiler foundations are not unstarted work.
- C5/R1 map to executable checkpoint/residual plans and complete-backward
  measurement. A ranked treeverse candidate is not an executed schedule.
- T3 and game G2/OT solver consumers map to matrix-free native residual products
  and conditioning/constraint certificates. A nonsingular-root pilot does not
  establish strict complementarity, KKT validity or general equilibrium AD.
- C4/C6/R2 remain consumer-gated under structured AD, AD-HIGHER and AD-WEIL:
  define the recurrence or estimator, numeric/RNG contract and native acceptance
  workload before adding algebra, semiring or estimator interfaces.
- EGGROLL W2 reverse mode reuses linear transposition with fixed member/key
  identity. Block AttnRes native scheduling preserves block-state lifetime and
  typed product semantics. Neither should grow a separate AD implementation.

## Mixed storage and selected checkpoint execution — 2026-09-07

F3/FA-1 admission and AD-RESIDUAL-EVAL-1 execution now have the bounded consumers
described in the integrated plan's native ANN/checkpoint section. Native GPU
temporary capacity uses SSA-derived interval bounds for replay loops, including
checkpoint-selected starts, and dtype-sized private slots. Source `max_iters`
alone never permits an unproved loop. Pure zero-origin/unit-step counted whiles
can normalize before paired AD; general while termination and typed scalar
residual storage remain separate work. Five cases on each owning CUDA/HIP device
validate repeated backward execution, with saved values unchanged. See
[the recorder](../../../benchmarks/record_tape_checkpoint_execution.py) and
[reports](../../../benchmarks/baselines/tape_checkpoint_20260907/README.md).
This does not close automatic checkpoint policy selection, asynchronous frame
retirement, parallel tape schedules or Apple's MSL tape binding.


## Data-dependent exits and physical predicate storage — 2026-09-07

The integrated plan's native tape/ANN increment advances AD-RESIDUAL-EVAL-1.
A proven counter-bound conjunction permits pure data-dependent whiles to become
frozen-state counted loops before existing checkpoint/paired AD. Exported
index/predicate residuals use native i64/i8 tensor storage; the region pullback
separates discrete replay state from floating cotangents. This supersedes the
previous paragraph's scalar-storage gap within this bounded envelope.

CUDA/HIP persistent frames expose stream submissions, completion polling and
explicit derivative-generation release. Arbitrary downstream readers still
require a context completion barrier before allocation release. Shape-varying
residuals, fully asynchronous reclamation, noncanonical while/CFG control and
Apple MSL-owned tape execution remain open. Exact-device results and the
source-specific bounds are recorded in
[the current packet](../../../benchmarks/baselines/native_tape_ann_20260907/README.md).


### Shape-varying host tapes and GPU ANN arbitration — 2026-09-07

Owners: **F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1**.

Native Tessera-to-Linalg binary lowering now derives dynamic output extents
from the operand and checks unresolved operand/result equality before identity
indexing. Dynamic `zeros_like` adjoints use logical primal extents. The host JIT
accepts signature-checked i64/i8 residual buffers without widening its floating
high-level math API. Three shrinking-loop cases (input widths 4, 8, 16) execute
native x86 forward and repeated backward products with saved shape tapes and
unchanged persistent payloads. The host JIT now runs upstream ownership-based
deallocation after DPS result copies, including loop-carried temporary allocations.
Native extent violations currently lower through `cf.assert` to process abort,
not a recoverable Python exception. This is bounded shape-varying **host execution**;
CUDA/HIP persistent slot allocation still rejects dynamic extents.

Counter-bound recovery also recognizes the frontend's false-else short-circuit
`scf.if`, and equivalent `arith.select`. An arbitrary else value cannot prove
capacity. Arbitrary source CFG recovery, unbounded termination, dynamic device
slot allocation and generalized checkpoint selection remain open.

Terminal absolute value joins ReLU as a nonexpansive consumer of the affine
absolute-error bound. GPU candidates bind the logical domain and numerical
budget to both native artifact digests and the owning architecture. Scoped
registration retains the original under a zero rewrite budget and unregisters
its candidates only after successful binding close. Both GPUs execute all four
ReLU/absolute-value budget cases. The optional upstream elementwise-fusion
pipeline is serialized and replayed as part of artifact ancestry.

Nine fresh independent processes per GPU still refuse performance promotion:
CUDA median 0.99809x, lower bound 0.97773x; ROCm median 0.99728x, lower bound
0.98430x. Neither clears the 1.02x threshold. These are warm package wall times,
including transfers, for the small serial schedule; they neither measure kernel
clocks nor establish a tuned tensor-core/WMMA schedule. See
[raw reports and scope](../../../benchmarks/baselines/shape_tape_ann_20260907/README.md).

Fully asynchronous reclamation remains open: producer-event completion cannot
prove completion of readers of exported views. The next implementation needs
reader leases that prevent new acquisitions after retirement, stream-ordered
allocator/free support, and failure retention until all recorded readers finish.
The existing context barrier is retained. Further numerical consumers need
operator-specific induced-norm/error propagation; absolute value does not prove
reduction, spectral or approximation legality. Assertions-enabled MLIR validation
and Metal-owned tape storage remain required follow-ups.
