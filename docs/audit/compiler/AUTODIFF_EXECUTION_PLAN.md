---
last_updated: 2026-09-09
audit_role: plan
plan_state: landing
---

# Autodiff execution and remaining integration

Start at the [compiler audit map](README.md) for document ownership.

This is the active scoped AD plan. It consolidates the unification phases P0–P6,
architecture findings A1–A8/B1–B8 and capabilities D1–D7, and next-generation
AD-LAW/WEIL/JET/OPERATOR work. Existing IDs are retained; no parallel scheduler
or new AD engine is proposed. Global order belongs to
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md), particularly [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).

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

The [functional-analysis consolidation](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)
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
are historical. The [integrated live queue](INTEGRATED_COMPILER_PLAN.md#live-queue)
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


### 2026-09-07: bounded recovery with logical shape residuals

W2.4a now proves nonnegative constant starts, positive strides, and signed
inclusive/exclusive counter bounds for data-dependent while normalization.
Declared capacity remains checked against the derived trip count. Both CUDA
SM120 and ROCm gfx1151 execute the static-slot strided products with early exits
and repeated backward calls. Native x86 executes logical shape-varying while
tapes for widths 3/4/5/8/16 and zero through three trips. See
`tests/unit/test_native_loop_next.py` and the integrated plan's corresponding
increment for the exact envelope and raw owning-device evidence.

Still open: arbitrary multi-block recovery, dynamic GPU capacity materialization,
and fully asynchronous reclamation. Reader tracking and stream-ordered allocator
ownership are prerequisites; existing externally exported views retain the
context barrier. This increment does not change that lifetime contract.


### 2026-09-07: internal dynamic capacity and tracked generations

Native bounded CFG replay now handles typed multiway switch edges. GPU internal
allocations can preserve dynamic logical extents within SSA-proved capacity;
external shape-varying tape arguments remain open. See the integrated plan's
dynamic temporary capacity increment for the exact envelope and device packets.

Opt-in tracked backward generations use stream-ordered pool storage and scoped
reader events. Retirement closes acquisitions and queues frees after every
reader. Faulted records and partial frees retain owners for explicit wait/close.
Unrestricted frame exports still require the context barrier. The next consumer
integration must preserve the declared reader stream and scope, not extract and
retain an untracked pointer.


**Allocator-failure boundary:** a failed free quarantines the frame and retains
its owners for device teardown; normal close must not retry an ambiguously freed
pointer. Completion-event failures alone retain the explicit stream-wait recovery
path. Quarantine is intentionally not reclaimed by garbage collection.


Cross-block SSA values are also owned state: dominating operation results and
foreign block arguments used by a successor are mapped to distinct state slots,
not left pointing into the erased CFG region. Native forward/reverse regressions
cover direct shared definitions across all switch cases and the default. Dynamic
saved values still require explicit shape envelopes for their new state slots.

### Checked native CFG products and scoped composition — 2026-09-07

W2.4a / AD-RESIDUAL-EVAL-1 now admits bounded native function-body CFGs through
existing state-machine recovery. The opt-in CUDA/HIP `guard-v1` status ABI handles
top-level product assertions and prevents suffix execution after failure.
Synchronous capture/backward checks status before exposing results; asynchronous
checked products and nested assertions still refuse. Scoped derivative owners
can submit into native tensor bindings or another frame's backward call, closing
reader scopes even after an enqueue error. This is reverse-program composition,
not automatic higher-order differentiation.

Current sequencing belongs to the live integrated-plan tasks:
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1) owns the
returned logical-shape ABI, validated loaded extents and checked-result protocol;
[W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1) owns source CFG/effect
recovery; [W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) owns scoped readers and
generation retirement. The [historical increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--checked-persistent-products-and-composed-readers)
records bounded evidence only. Static capacity allocation alone does not close
exported shape-varying GPU tapes.


### Checked host tickets — 2026-09-08

[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1) now includes
asynchronous static-product submission with independent per-generation status.
Successful `wait()` or `poll()` must consume status before `outputs` is exposed.
Failure is retained, while release/close still drains outstanding device work.
`tracked=True` remains refused for checked products: event completion is not a
success predicate for device readers. Dynamic public shapes, nested guards and
fully asynchronous reclamation remain open under the live owners above.


## 2026-09-08 — Nested guards and checked device readers

The latest [AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1)
consumer accepts explicitly authored rank-one capacity-buffer programs with
compiler-projected shape sidecars. Device status and host extent validation
precede public logical views. This supersedes the earlier blanket nested-guard
refusal for supported single-block `scf.for`/`scf.if`, including scalar yields.
General tensor-returning AD still needs an automatic producer for this ABI.

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) now includes dependent checked backward
submission: an event orders producer completion, and an incoming device status
prevents the child body from reading failed derivative data. SM120 and gfx1151
prove the successful chain and injected failure. Allocation and release retain
context completion barriers; fully asynchronous reclamation remains open.
Apple has no corresponding MSL status/shape producer; x86 proof is independent.


## 2026-09-08 — Automatic result ABI and checked pool retirement

[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1) now has an
automatic native export-to-public-result adapter for a single rank-one AD
result with static external arguments. Capacity is a checked storage budget;
logical length is computed by the device. Dynamic backward inputs and saved
multi-result/multidimensional products still need their own carriers.

[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a) now retires checked derivative data
and status through the pool after every registered reader event. Gated child
backwards do not need a host success readback; generic scoped reads require
`wait_success()`. Healthy retirement/poll does not wait on the context. Whole
frame capture/close and exceptional teardown retain their completion barriers.


### Runtime-shaped products and scoped frames — 2026-09-08

AD-RESIDUAL-EVAL-1 now generates multiple rank-one through rank-four public
result sidecars and bounded dynamic input descriptors from exported native AD.
`public-input-capacity` binds flat physical storage and separate per-axis i64
shapes; native guards reject negative, overflowing, static-axis-mismatched and
incompatible elementwise dimensions before access. Output copies pack logical
row-major data and check total volume. `NativePublicResult.submit` queries device
completion before reading status/shapes and exposing a logical view.

W2.4a adds `capture(scoped=True, stream=...)`, scoped primal/residual readers and
`retire` / `poll_retired` for whole-frame storage. Generation readers precede
frame frees. Pool/event failures preserve ownership; module release happens only
after tracked completion. Scoped persistent frames still use static tensor shapes; runtime-shaped public
frames retain synchronous close. Capture, exceptional recovery and unrestricted-export
close remain synchronous. Module unload latency and general external-reader
adoption are not closed. No overlap or performance claim follows from this proof.

Source CFG remains owned by W4-PRODUCT-1: the tracer rejects Python data-dependent
truth conversion. Native bounded CFG normalization and nested guard execution do
not recover Python merge values, early return, break/continue or arbitrary effects.
See the live integrated plan for sequencing and
`benchmarks/baselines/runtime_shape_frames_20260908/` for exact-device scope.

The private temporary allocator still reserves the product of independently
proved dimension maxima under its 4096-byte ceiling. It does not yet reuse the
new input volume guard as a joint allocation bound. Consequently larger dynamic
multidimensional products can refuse even when their actual logical volume fits
the public capacity. AD-RESIDUAL-EVAL-1 owns carrying that joint bound through
allocation/view lowering; the current small-shape packets do not close it.


## Source recovery and scoped asynchronous ownership (2026-09-08)

Owners: W4-PRODUCT-1, AD-RESIDUAL-EVAL-1 and W2.4a; synchronization key
`SOURCE-ASYNC-FOUNDATION-2026-09-08`. Sequencing remains in the live integrated plan.

`trace(..., source_control_flow=True, max_steps=N)` recovers pure local Python
branches, nested branch early returns and one initialized tensor loop carry.
The source adapter calls the existing TraceBuilder; `to_native_source_ir` emits
SCF from its typed SSA edges. Native while exhaustion asserts rather than
silently truncating. The comparison boundary explicitly converts the native i1
result to the tracer's historical floating mask. Ordinary tensor truth tests,
object mutation, arbitrary calls, break/continue, multi-value loop carries and
loop early returns remain refused. This is opt-in source capture, not automatic
JIT migration or arbitrary Python CFG closure.

`NativePublicResult.submit(..., scoped=True)` exposes logical views only inside
`read(stream)` after successful completion/status/shape checks. `retire(stream)`
orders all capacity/sidecar/status frees after registered readers.
`PersistentTapePair.capture_async(stream, ...)` uses pool allocation, ordered
copies and forward submission without explicit context synchronization; generic
readers require `poll_capture`, while native backward uses the captured status.
Legacy one-status packages require a successful capture check before replacing
that dependency. The two-status continuation below preserves both checks. Persistent captures retain static
storage shapes; dynamic public frames are a separate ABI.

Asynchronous owners defer module unloading to at most eight worker slots.
`poll_retired` does not wait for driver unload. Stalled workers retain modules,
contexts and slots; failed unloads are quarantined without retries. This bounds
admission and keeps polling responsive, not the driver's completion latency.
The caller must retain its owning context until retirement completes. Exceptional
buffer cleanup and unrestricted exports retain conservative synchronous recovery.

Temporary storage can use an exact dominating `ule(product(extents), capacity)`
then-edge proof. Wrong products, reverse predicates and non-dominating/else-edge
bounds cannot tighten storage. Logical dimensions and distinct iteration slots
are preserved; unknown aliases and more general relational volume proofs remain
open. Evidence is recorded under
`benchmarks/baselines/source_async_foundation_20260908/`.


### Effect-aware CFG and two-status composition (2026-09-08)

Owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a). Sync key:
`CFG-STATUS-COMPOSITION-2026-09-08`.

Pure single-carry while retains native `scf.while`. Loops with multiple initialized
tensor variables and break/continue use bounded expansion (at most 16 iterations,
256 statement/region visits), merging each iteration's state before constructing
the next. Assertions use registered scalar MLIR and `cf.assert`; their effects
cannot be dropped as unused tensor results. Budget exhaustion is an assertion,
not silent truncation. Loop-return payloads, external aliases/mutation, exceptions
and general JIT plumbing remain open. Calls outside the admitted pure vocabulary
are rejected before tracing, including unreachable statements.

`materialize_persistent_tape(..., checked_status=True, gated_input=True,
status_inputs=2)` serializes `tessera.autodiff.input_status_count = 2 : i64`.
The native producer checks both statuses before body effects. The package projects
two distinct readonly status arguments; capture preserves distinct storage rather
than weakening the no-alias contract. A checked derivative's `backward_into`
retains its reader scope and orders its event before the consumer, while the
consumer's capture event orders its own status. Neither success can overwrite or
substitute for the other. Without an external dependency, both inputs reflect the
capture result. One-status packages remain compatible with their checked-host
replacement rule. Count/ABI tampering is rejected by native replay and projection.

Independent CUDA SM120 and ROCm gfx1151 recorders exercise all four combinations
of capture/upstream success and failure, expected derivative values, reader-aware
retirement and no host capture check. These are correctness packets, not overlap
or performance evidence. Arbitrary fan-in, heterogeneous dynamic capture and
exceptional asynchronous recovery remain open.

### Completion state and bounded status fan-in (2026-09-08)

Owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a); sync key
`COMPLETION-STATE-FANIN-2026-09-08`. This supersedes the corresponding open
boundaries in the preceding increment, without implying general AD closure.

Expanded loops now carry return flags and floating tensor/tuple payloads;
nested returns and finally overrides execute through typed native conditionals.
Explicit raises of unshadowed builtin ValueError, RuntimeError and AssertionError
resolve to statically matching handlers, including across loop exits. Exception
objects, dynamic classes, re-raise/causes and an uncaught exception result ABI
remain excluded. Bounds remain 16 expanded iterations and 256 continuation visits.

`compile_source_state(..., mutable=(...))` is an explicit native CPU adapter.
It serializes exact input alias groups, shape/dtype and state-result contracts
in `tessera.source_state`. Full-slice writes update shared SSA roots; execution
uses snapshots and separate outputs before copying declared state back. All
aliases in a mutable group must be writable. Partial overlaps, mutable return
aliases and changed alias topology refuse. The caller exclusively owns the
arrays during execution; concurrent mutation and GPU state binding remain open.
Graph conversion refuses these state traces until it has an effect consumer.

`status_inputs` now admits one through eight. A checked generation's
`backward_into(..., dependencies=(...))` adds status-only prerequisites, retains
all reader scopes through submission and orders every producer before the
consumer. Its own capture remains a separate status for counts above one.
Duplicate dependencies, insufficient slots and mismatched targets refuse.
The legacy single-status host-check rule remains intact.

Four incoming statuses passed all sixteen combinations on SM120 and gfx1151;
counts three/four/eight also have compiler contract tests. These correctness
packets do not establish eight-input device proof, overlap, performance,
heterogeneous dynamic capture or general exceptional reclamation. Evidence:
`benchmarks/baselines/completion_state_fanin_20260908/`.

### Source JIT and state generations (2026-09-08)

Owners: W4-PRODUCT-1 / W2.4a; sync key `SOURCE-JIT-STATE-2026-09-08`.
`jit(source_control_flow=True, source_mutable=(...), source_error_specs=(...),
source_max_steps=...)` selects an explicit native CPU consumer, without Graph
reconstruction or Python execution fallback. Use the returned owner as a context
manager or close it. At most four shape/alias specializations retain native modules.
Incompatible target, batching and AD options refuse; this is not general JIT closure.

CPU state accepts injective strided/reversed views; partial overlapping aliases,
broadcast/self-overlap and arbitrary objects refuse. Writes commit after native
completion, including writes before a transported explicit builtin exception.
`source_error_specs` supplies static floating return shapes so failure paths have
a typed payload. Codes transport AssertionError/RuntimeError/ValueError with a
generic message; original messages, implicit operation errors, dynamic exception
objects and source-bound exhaustion transport remain open.

The GPU adapter admits one declared state input until multi-input alias
projection reaches its binding. The native GPU producer accepts serialized source-state results and constructs
capacity/shape sidecars through the existing public-result path. Two state
steps execute on gfx1151 while original input storage remains unchanged.
Results retain their owning frames; this functional generation protocol is not
arbitrary in-place device mutation, GPU Python exception transport or AD support.
ROCm also proves all 256 eight-status cases. NVIDIA follow-up is blocked by SSH
access; no Apple proof transfers. Evidence: `benchmarks/baselines/source_jit_state_20260908/`.

### Source object and paired-product boundary (2026-09-09)

Owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
`NativeSourceJit.vjp` executes compiler-exported forward/backward products on CPU
for pure tensor inputs. It is explicit seeded VJP, not automatic differentiation
of effects. Mutable fields, exception transport and object arguments refuse AD.
Declared dict/SimpleNamespace fields and static exception payloads now have native
source execution; custom objects and full exception semantics remain open.
Read-only overlapping views snapshot safely; writable aliases need a common
backing-storage SSA model with ordered extract/insert operations before admission.
Owned GPU state has synchronous SM120/gfx1151 copyback proof and scoped-reader
exclusion. Asynchronous writes and externally owned mutation remain open.
Evidence: `benchmarks/baselines/source_object_ownership_20260909/`.

### Mixed alias and state derivative follow-through (2026-09-09)

Owner: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1), with
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1) and
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
Plain instance dictionaries can supply declared tensor fields; descriptors and
custom lookup hooks still refuse. Alias checks classify each pair by writable
state participation, so disjoint mutable state does not prohibit overlapping
read-only inputs. Writable overlap still needs shared backing-storage SSA.
State VJP accepts explicit seeds for public and next-state outputs, computes
native products and leaves caller inputs unchanged. Mutable input aliases,
object AD and exception AD remain excluded.

Dynamic exceptions need a typed completion payload in the serialized frame,
including exception tag, payload storage, cause/context relationships and
ownership across handlers/finally. A class-code table alone cannot implement
those semantics. Exception paths must preserve preceding state writes, and AD
must define replay and differentiation of each admitted effect before promotion.
This is the next architectural gate, not implemented runtime support.

Single-stream GPU submit/poll now validates the computation before enqueuing
copyback. Readers and additional writes refuse while pending. Copy/event errors
poison the owner and keep storage until synchronized close succeeds. This is
asynchronous update completion, not nonblocking failure teardown or concurrent
multi-writer mutation. Evidence: `benchmarks/baselines/mixed_alias_async_state_20260909/`.

### Shared view roots and typed exception values (2026-09-09)

Owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1) and
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1).
The native CPU state path now admits contiguous rank-one overlapping writes
when an input contains every participating view. One SSA root owns the state;
standard tensor slices project reads and insert writes in program order. The
serialized view map binds argument indices, offsets and lengths, is checked at
capture and invocation, and participates in specialization identity. No hidden
NumPy allocation is assumed to be a usable compiler input. Noncontiguous views,
absent containing inputs and alias-aware adjoints remain open.

Explicit object-state VJP flattens declared fields, returns gradients in field
order, and seeds public outputs followed by next-state outputs. It does not
mutate caller fields. Effectful accessors and exception AD still refuse.

Dynamic exception transport now has one typed f32 tensor payload output of
shape (1,), alongside the completion tag. Explicit exception-value expressions
are snapshots at the raise; handler/finally re-raise preserves the appropriate
payload through nested completion. Raising a mutable alias directly refuses
because a value snapshot would not preserve alias changes made by finally.
This is not full Python exception identity, dynamic strings, causes, traceback
or implicit-error transport. Those remain explicit ABI/ownership tasks.

Evidence: `benchmarks/baselines/source_views_exception_20260909/`. CUDA/HIP
packets cover regression of the existing single-state asynchronous consumer;
they do not promote multi-input writable views or exception execution on GPUs.

### Positive-stride roots and exact-alias adjoints (2026-09-09)

Owners: [W4-PRODUCT-1](INTEGRATED_COMPILER_PLAN.md#w4-product-1),
[AD-RESIDUAL-EVAL-1](INTEGRATED_COMPILER_PLAN.md#ad-residual-eval-1),
[W2.4a](INTEGRATED_COMPILER_PLAN.md#w24a).
View records retain compatibility with old rank-one triples/quadruples. Bounded
injective negative and same-rank multidimensional maps retain explicit root
coordinates (up to 256 mapped elements). Local static slices compose those maps;
standard tensor singleton slices avoid negative-stride MLIR slice assumptions.
This is a correctness fallback, not scalable model-state code generation.

Native slice adjoints now serve source JIT: gather/scatter accumulates at the
canonical root and overwritten destination gradients are masked. Duplicate
alias argument gradients remain zero. Finite differences cover sequential
partially overlapping writes. Users must not re-sum the root derivative for
each alias. SM120 and gfx1151 independently execute a mapped backward product.

Static caught exception bindings support identity tests and named/bare re-raise.
The host decoder retains cause/context identity and suppression. Native raise
sites appear as notes, never synthetic Python traceback frames. Dynamic f32
payloads remain typed completion results; dynamic context payload retention,
new loop context slots, full traceback semantics and exception AD remain open.

GPU source/AD bufferization copies before writes. Checked public frames decode
exception sidecars after successful completion/shape validation and before
exposing any result. Repeated asynchronous polls preserve the decoded error
object. Error cleanup can synchronize; owned in-place mutation still rejects
exception-bearing contracts. Exact-device proof covers one state input, not
arbitrary external GPU alias sets or collective-safe exception propagation.
Evidence: `benchmarks/baselines/source_exception_gpu_20260909/`.

[Block AttnRes integration](BLOCK_ATTNRES_ROCM_PLAN.md#iii5a-how-source-views-adjoints-and-completion-help-both-gpu-lanes)
uses these contracts for state/lifetime oracles; cooperative workload kernels
and target-specific timing remain independent gates.

### Runtime maps and exception-bearing products (2026-09-09)

Owners remain W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Positive rectangular views now use compact offset/size/stride records and native
slice operations at model-sized CPU shapes. Native CPU product allocation has
an independent 16M-element slot cap; it no longer imports the GPU 1024-element
cap. General negative/permuted maps retain the 256-element fallback bound.

The runtime-even-columns fixture uses descriptor dimensions, not Python shape
buckets. Both GPUs execute forward/backward for several shapes. Dynamic slice
pullbacks assert seed extents before scatter; removed/inverted guards refuse.
Allocation intervals admit unsigned division only with nonnegative numerators
and positive divisors. Dynamic copies require identical logical dimensions or
an exact dominating equality guard, never equal capacities alone. Automatic
runtime-sized Python slicing remains a frontend integration task.

Per-dynamic-raise-site SSA slots preserve distinct f32 cause/context payloads.
They do not provide arbitrary generation-indexed exception objects across loop
iterations. CPU exception-aware VJP executes the native forward, decodes its
completion, and launches backward only on success; status/payload cotangents
are zero and state is not committed. GPU exception AD explicitly refuses until
its exported product ABI can bind the corresponding checked forward. Ordinary
GPU exception transport retains its synchronous/asynchronous completion proof.

**Full Python traceback boundary:** LLVM execution does not create CPython
frames. Current errors have genuine host-bridge tracebacks and native source
location notes. Full Python-compatible traceback objects would require an
explicit interpreter/frame ABI, locals/closure state and reference ownership,
exception chaining across native/host calls, GIL interaction and frame lifetime
through asynchronous retirement. Do not synthesize dummy frames or relabel
location notes as this implementation. Keep that compatibility work independent
of numeric kernel promotion and measure its success-path overhead before any
default integration.

Evidence: `benchmarks/baselines/runtime_source_maps_20260909/`.


### Source indices, generation slots and checked GPU VJP (2026-09-09)

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a. Sync key:
`SOURCE-GENERATION-AD-2026-09-09`. This increment supersedes the corresponding
remaining boundaries in the preceding historical entries.

Source capture admits rank-preserving slices with runtime single-element int64
tensor bounds and positive steps on statically ranked roots. Native signed
normalization clips negative/out-of-range bounds; zero/negative runtime steps
fail through the status guard. Empty results are valid. Bounds are not read from
trace samples. A tracer int64-to-f32 misclassification is fixed. Read-only views
retain the defining SSA root after Python local rebinding. The native CPU DPS
caller currently supplies the dynamic output shape; automatic dynamic-result
allocation in `NativeSourceJit`, Python integer/index protocols, negative runtime
strides and nested dynamic views are still open.

Expanded loops now carry one completion code, with an exception table that may
grow while handlers are traced. Each syntactic dynamic raise and bounded loop
iteration has its own SSA payload slot (at most 32). Cause/context chains can
escape a loop with the correct payload and shared exception identity. This is
bounded generation storage, not a heap of arbitrary exception objects carried
between iterations; the existing expansion/depth/edge caps still apply.

`materialize_source_vjp` binds an isolated single-input generated pair; projected aliases and object fields refuse. The synchronous owner
validates and snapshots device inputs, executes the forward product, checks its
primal completion prefix (before saved residuals), then launches backward with
those same snapshots/residuals and zero metadata seeds. Failed forward execution
never launches backward. Scalar i8/i64 products retain logical rank zero with
one physical storage element and a checked shape sidecar. Standalone source
exception backward packaging still refuses. Dynamic/async VJP staging, arbitrary
exception values and derivative-through-exception-object semantics are absent.

**CPython traceback architecture decision:** keep the real host traceback and
native location notes distinct. CPython's
[traceback API](https://docs.python.org/3.12/c-api/exceptions.html#tracebacks)
prepends an actual frame; it does not recover a native instruction's locals or
call history. Full compatibility needs a separate compiler-owned debug-frame
contract: code identity/instruction mapping, live local and closure ownership at
the throw point, native-to-Python caller links, and lazy materialization under the
owning interpreter/GIL. First validate one frame with locals and cause/context;
then nested calls and asynchronous frame retention. Do not replay the function
or fabricate execution frames from filename/line notes. Repeated GPU failure
polls now restart the same exception's host traceback, preventing indefinite
retention of earlier callers and their locals. This fixes ownership, not native
frame reconstruction. No full-CPython support or performance promotion is claimed.

Evidence: `benchmarks/baselines/source_generation_ad_20260909/`.


## Signed runtime views and asynchronous source products (2026-09-09)

Owners remain W4-PRODUCT-1 and AD-RESIDUAL-EVAL-1 in the live integrated plan.
Signed/nested runtime source views now execute through compact tensor.generate
maps. Their reverse gather/scatter rule remains unsupported and must refuse;
static slice adjoints do not prove this path. CPU source JIT automatically
allocates dynamic outputs from the compiler's capacity/shape projection, with a
1024-element capacity bound. Bounded exception references survive iterations
without losing the original payload. Arbitrary custom exception objects do not.

CUDA SM120 and ROCm gfx1151 independently execute checked same-stream source
VJP: asynchronous snapshots retain inputs, forward completion gates backward,
and failed forward execution exposes neither primal nor derivative. Explicit
close may synchronize. Fully asynchronous reclamation and cross-queue writer
coordination remain open. Evidence and exact envelope:
`benchmarks/baselines/source_nested_async_20260909/`.

Full CPython frame reconstruction is a separate ABI requirement: preserve the
original code and instruction identity, throw-site live locals and closures,
caller links, and interpreter/GIL ownership through asynchronous completion.
Current native source notes and host bridge frames do not satisfy that contract;
replaying the source or constructing dummy frames would misrepresent execution.
