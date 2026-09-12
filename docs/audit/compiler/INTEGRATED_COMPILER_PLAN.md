---
last_updated: 2026-09-11
audit_role: plan
plan_state: open
---

# Integrated Compiler Plan

Start at [`README.md`](README.md). This document alone owns cross-domain compiler
sequencing. [MASTER_AUDIT](../MASTER_AUDIT.md#proof-vocabulary) and generated
inventories supply evidence; scoped plans own detailed acceptance contracts;
backend queues own target-specific proof. Reconcile contradictory evidence with
source and tests rather than inventing a second status registry here.

`last_updated` means last substantive review of current-priority content, not a
formatting change. The [engineering log](INTEGRATED_COMPILER_LOG.md) preserves
revision-bound outcomes. The [archived queues and waves](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md)
are provenance, including superseded priority statements and estimates.

## Foundation program

One MLIR-owned semantic program must descend through verified native compiler
passes. Python remains the public API, capture/orchestration layer and oracle.
After canonical IR exists, code generators must not reconstruct semantics from
GraphIRModule, operation names or Python descriptor/fusion objects. Shapes,
layouts, effects, derivative products, numeric policy and schedule choices must
survive in verified IR. Deployment facts and measurements remain identity-bound
inputs; runtime loads images and executes their ABI.

| Cut | Architectural purpose | Sequencing rule |
|---|---|---|
| F0 | Route census, proof integrity and usable compiler tooling | Cross-cutting; no hardware prerequisite for inventory or contract fixes. |
| F1 | First canonical-artifact package migration | Historical NVIDIA scheduled-matmul slice; remaining families belong to F2. |
| F2 | Migrate surviving Graph-owned packaging and callers | Select a family from the census; retain its ABI, dtype and policy envelope. |
| F3 | Native recipe instantiation, fusion, ANN and measured scheduling | Requires the selected family's IR boundary, not completion of every F2 route. |
| F4 | Structured AD/effects/memory, numerical and distributed semantics | Extend the same compiler foundation through verified carriers. |
| F5 | Retire redundant generators and broaden mathematical workloads | Delete only after differential and owning-target proof covers surviving callers. |

The cuts are an architecture map, not a waterfall. Priority is local to each
cut below. Dependencies identify the specific prerequisite portion; they do not
require the entire referenced program to close. Tuned/library candidates remain
legitimate when their selection and ABI are explicit in IR.

Native destinations: x86 uses MLIR/linalg/vector/SCF to LLVM/object or ORC;
NVIDIA uses Tile/Target to NVVM/PTX/image; ROCm uses GPU/ROCDL/LLVM to HSACO;
Apple GPU uses compiler-owned MSL and the supported Metal compiler/metallib path.
Do not imply a general LLVM-IR-to-Metal backend or transfer schedules/proof
between targets. The [foundation survey](MLIR_NATIVE_FOUNDATION_SURVEY.md) owns
migration seams; the [backend map](../backend/README.md) owns device queues.

## Start here

This is a derived navigation view, not another queue. The first `host-free`
action in each cut is the entry point below; each linked record names its gate
and prerequisite portion. Host-free means its next engineering step can start
without device access, not that its eventual promotion needs no device proof.
`device` actions require an owning-host availability check at execution time.
Dependencies and required evidence must be checked before starting dependent work.

<!-- ready-view:start -->
- F0: [E2E-REAL-6F](#e2e-real-6f).
- F2: [E2E-REAL-6](#e2e-real-6).
- F3: [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1).
- F4: [W4-PRODUCT-1](#w4-product-1).
- F5: [W5.2f](#w52f).
<!-- ready-view:end -->

## Live queue

Records below are remaining obligations, not support/promotion statuses.
`Gate` is the next missing deliverable and acceptance boundary. `Latest` points
to historical context, which must be read at its recorded scope. `Start` names
the next action's host requirement; it is not a live fleet-availability claim.

## F0

### E2E-REAL-6F

**Route census and exact-device certificates**

- Owner: [MLIR_NATIVE_FOUNDATION_SURVEY.md](MLIR_NATIVE_FOUNDATION_SURVEY.md)
- Gate: The expanded census includes x86 breadth packaging: 46 Graph inputs, 14 scheduled inputs and 14 raw/unclassified entries. Import-resolved caller candidates and local helper-to-emitter paths expose reconstruction paths; scope-aware import candidates now reject parameter/rebinding shadowing and sibling-scope leakage; indirect dispatch and per-envelope certificate joins still require review. Counts do not authorize constructor deletion.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--expanded-route-callers-and-f64-ownership)

### COMPILER-DEVEX-1

**Assertions-enabled validation and usable tools**

- Owner: [COMPILER_REFACTOR_PLAN.md](COMPILER_REFACTOR_PLAN.md)
- Gate: Assertions-enabled LLVM/MLIR 23.1.1 and a hardware-free all-target Tessera compiler now pass all 475 active lit fixtures on Super-Bear. Backend-owned fixtures declare their feature requirements, the data-only x86 execution input is outside lit discovery, and the union gate requires every active fixture to pass in at least one lane. The opt-in CI lit lane requests the same full portable target matrix. Installed drivers now pass relocated-prefix smoke on Super-Bear with loader overrides removed; the CI lane runs this check after installation. Preserve these regression gates; owning-device correctness and performance remain separate backend gates.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--installed-drivers-and-owning-device-measurements)

### EVIDENCE-PACKET-1

**Evidence packet consumers**

- Owner: [EVALUATOR_PLAN.md](EVALUATOR_PLAN.md)
- Gate: Unify missing packet consumers around artifact/image identity, clock validity and promotion eligibility; malformed or incomplete evidence must refuse. The [benchmark alignment review](../../../benchmarks/COMPILER_ALIGNMENT.md) maps six suites to actual callers: math remains metadata-driven, GA/EBM composition lacks route receipts, and direct-IR AD probes need paired public-frontend coverage. New diagnostic math/AD runs no longer inherit performance eligibility from target names or historical packets. The extended suite review retires synthetic SuperBench timing, labels DLOP dispatch counts as estimates and lattice timings as host-wall, and adds per-submission native guards for Apple policy-loss timing. Bounded CUDA/HIP ANN adapters now record driver-launch receipts; public SSD VJP has independent numerical comparisons. Dedicated CUDA SSD kernel/API attribution and serial/cooperative SSD adapters are validated; sequential synchronous mixed-artifact CUDA attribution and CUDA GEMM/attention adapters now pass. ROCm GPUFuncOpLowering asserts on duplicate attributes for both matrix families; resolve and revalidate, then extend asynchronous attribution and clean performance admission.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--matrix-adapters-and-mixed-attribution)

### TPROF-NATIVE-1

**Native clocks and counter attribution**

- Owner: [INTRA_KERNEL_FEEDBACK_PLAN.md](INTRA_KERNEL_FEEDBACK_PLAN.md)
- Gate: The CUDA adapter now calibrates seven launch-inclusive event windows against exact Nsight product-kernel spans, checks captured device identity and rejects excessive profiler overhead. After removing SSD's unrelated 1024-element tape-parser cap, a 512x2x32x8 cooperative workload agrees within 0.51% with 0.24% profiler overhead. The source remains dirty and the host WSL. A nine-pair collector now binds fresh process nonces and Nsight PIDs and refuses unqualified hosts before production collection. Validate per-process clean-image calibration and counter attribution on each owning host; WSL regression evidence is not clean timing.
- Depends on: —
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--program-retirement-and-declared-slot-discovery)

### DISPATCH-BREAKER

**Safe waits and checked-output ownership**

- Owner: [../backend/README.md](../backend/README.md)
- Gate: Audit remaining bridge waits and checked-output readers; bounded failure must retain live resources and poison unsafe ownership before wider adoption.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Next: migrate legacy callers, health-check replacement workers and validate actual driver-failure behavior. Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- Latest: [recorded exploration](INTEGRATED_COMPILER_LOG.md#2026-09-11--gated-metadata-owner-and-isolated-recovery)

## F2

### E2E-REAL-6

**Remaining Graph-owned packaging and frontend retirement**

- Owner: [MLIR_NATIVE_FOUNDATION_SURVEY.md](MLIR_NATIVE_FOUNDATION_SURVEY.md)
- Gate: F32, BF16-to-f32, FP64 and mixed uint8/int8-to-int32 x86 matmul now enter Graph-to-Schedule-to-Tile and project native ABI from replay-verified artifacts. The mixed recipe carries ui8/i8 signedness, rejects wider-than-i32 runtime extents, and has owning-CPU ragged/overflow differential proof. Cohort, elementwise and breadth Graph constructors remain. Require per-target differential proof and preserve policy before deleting each route.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--mixed-integer-ownership-and-dtype-arithmetic)

### W1.1

**NVIDIA typed-fragment closure**

- Owner: [../backend/nvidia/todo.md](../backend/nvidia/todo.md)
- Gate: Reconcile remaining NVIDIA fragment producers and Target proof against current typed routes; retain uncovered dtype/architecture obligations and exact-device gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)

### W3.3

**Tile dialect ownership review**

- Owner: [IR_STACK_INTEGRATION_REVIEW.md](IR_STACK_INTEGRATION_REVIEW.md)
- Gate: Reassess surviving Tile primitive/kernel/domain/solver ownership before any dialect split; move only consumed semantics with parse/lower/execute regression gates.
- Depends on: [E2E-REAL-6F](#e2e-real-6f): census and proof requirements for the selected route, not all certificates.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-05--deleted-functionality-reassessment)

## F3

### FRONTEND-IR-MEDIUM-1

**Native recipes and broader raising**

- Owner: [FRONT_END_LOWERING_ASSESSMENT.md](FRONT_END_LOWERING_ASSESSMENT.md)
- Gate: Exact f32 attention loops, including a positive exactly representable f32 literal post-dot scale, explicit grouped-head indexing and end-aligned causal masking, now raise to a symbolic recipe and instantiate two native buckets through Schedule/Tile. The opt-in NVIDIA binding now projects native Schedule fields, replays the recipe and executes both buckets without Graph reconstruction. Apple and x86 now project target-specific parents; the x86 package executes on Zen 5. CUDA now proves ragged Q/K causal GQA buckets, with native rejection of nondivisible head counts. Bounded asymmetric window masks now execute on CUDA; mandatory serialized Presburger constraints reject fully masked rows, including when callers supply additional constraints. Finite full-shape additive bias now crosses native instantiation and Schedule projection and passes CUDA differential execution. NVIDIA raised binding now accepts irregular full-shape additive negative-infinity masks and rejects empty rows after causal/window composition; CUDA differential execution passes, including irregular masks combined with ragged GQA, causal and left-window masking for Q>K and K>Q. Extend Boolean/padding/broadcast masks, fully masked-row semantics and recognition, Apple device proof, ROCm-compatible storage and measured candidate admission; preserve complete witnesses and IR/image-bound admission.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--resident-dag-accumulation-and-snapshot-readers)

### MSW-9

**Broader ANN admission and tuned candidates**

- Owner: [ANN_CALCULUS_DESIGN_SPIKE.md](ANN_CALCULUS_DESIGN_SPIKE.md)
- Gate: Terminal square now has explicit analytic error amplification and intermediate-overflow refusal alongside ReLU/absolute-value consumers. Proved row-private mutable temporaries now admit the 64x8 square workload within the unchanged 4096-byte limit; constants and nested generations stay fully allocated. Extend measured nonlinear families and physical schedules. Admit only exact-artifact scoped candidates passing the measured lower-bound gate; retain incumbents when evidence is insufficient.
- Depends on: [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1): recipe/native identity for the candidate; broader raising is independent.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--dynamic-reader-fanout-and-row-private-ann)

### W5.2

**Measured schedule selection**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Connect remaining producers and target calibration to measured schedule selection; preserve inferred dependencies and select only with eligible exact-device evidence.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--bounded-while-recovery-and-row-parallel-ann)

### W5.5

**Consumer-driven canonicalization**

- Owner: [OPTIMIZING_COMPILER_PLAN.md](OPTIMIZING_COMPILER_PLAN.md)
- Gate: Checked permutation composition, matrix-transpose flag folding, attribute-bearing cast retention and fusion guards now share legality across generic/custom canonicalization and direct transpose lowering. Extend live consumer coverage and measure candidate effects before promotion; equality saturation still requires a demonstrated ordering problem.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-09--four-canonicalization-legality-improvements)

## F4

### W4-PRODUCT-1

**Source CFG and effect-aware recovery**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: A bounded native C++ host allocator/collector now exposes generation-checked handles, roots, cause/context cycles and reusable payload holes through a copy-out C ABI. Serialized static source exception tables now compile through MLIR/LLVM to native allocation/root/edge calls and can be enabled on the native CPU source-state exception path. Fresh-heap construction cleans up on failure. Runtime numeric tensor payloads now enter native allocation through pointer/size operands, with a one-MiB bound, owned copies, runtime shape changes and cleanup after exhaustion. The host allocation is at synchronous completion. A bounded GPU producer now transactionally copies runtime f32 site payloads into preallocated global frames with caller-supplied generation/offset/length records and exhaustion status; CUDA/HIP execution and copy-back decoding are proven. A separate bounded nonmoving GPU slot pool now generates allocation and stop-the-world mark/sweep kernels, reuses payload slots, follows up to 32 generation-checked edges per live node and collects unrooted cycles on CUDA/HIP. Opaque byte records and a resident owner now order root/edge updates, allocation and collection after closed reader leases across streams. Completed event records are reaped; failed event recording retains retryable ownership until explicit completion. Private snapshot marking now runs independently of active-graph mutations; final seeded remark/sweep remains exclusive. Bounded sweep batches now logically retire the entire unreachable cohort before range reclamation, preventing cross-batch dead-cycle edges from becoming invalid. Allocation reuses only reclaimed slots; new roots cannot resurrect retired generations. CUDA/HIP validate mutation between batches. Owned immutable pool snapshots now permit snapshot reader scopes during live-pool sweeping without racing current-storage reads; copies are writer-ordered and remain owned through reader completion. This is snapshot isolation, not unrestricted same-storage concurrent sweeping. Quiescent exact builtin containers and plain instance dictionaries are discovered without user hooks, preserving cycles and aliases. General-key dictionaries now retain key/value references; exact sets, frozensets and bytearrays are supported. Builtin subclasses with native payload/descriptors require an explicit extractor instead of silently dropping their hidden state. Opt-in fully declared slot hierarchies now use native member descriptors without attribute/property hooks; mixed instance dictionaries and undeclared opaque extension storage refuse. Explicit exact-type ExtensionLayout extractors now copy opaque payload bytes and strong referents under a caller-enforced quiescence contract, with schema/type identity and existing budgets; no global registry or automatic extension traversal is implied. Before concurrent sweeping, implement and verify a mutation barrier, generation-aware retirement epoch and reader-protected reclamation; snapshot marking alone does not authorize freeing against live mutation. Automatic throw-site fusion, concurrent sweeping, variable-size payload storage and arbitrary extension/object semantics remain open. The existing arena also supports explicit live-node rooting and cause/context edge updates. Completion publication bypasses custom exception field hooks; source records remain diagnostic, not CPython frames. Exception arena ABI leases now exclude allocation/collection until readers complete; partial collections reuse payload holes without moving live roots. Opt-in source recovery records native single-carry while and bounded multi-variable break/continue expansion, merging each iteration before the next. Assertions remain ordered native effects; unmodelled calls refuse even on dead paths. Bounded nested/tuple loop returns and statically handled builtin exceptions now execute natively; explicit CPU state slots preserve exact input aliases and non-overlapping strided views through serialized SSA state and post-completion copyback. An opt-in native CPU JIT owns four shape/alias specializations; explicit result specs transport builtin exception classes with preceding writes. Declared plain-instance/dict/SimpleNamespace tensor fields and read-only overlapping snapshots now execute, including disjoint mutable state in the same invocation; static exception args, inherited/tuple handlers and bare re-raise are transported. Source JIT exposes explicit native paired VJP of functional results and declared next-state outputs, including projected object fields, without copyback; exact aliases accumulate their adjoint at the canonical root; native slice adjoints now accumulate mapped and overlapping reads at the containing root and mask overwritten destination gradients; exception objects remain nondifferentiable; checked source VJP differentiates successful numeric paths. Rank-one positive strides and bounded injective negative/multidimensional views share an explicit contiguous containing-input SSA root; local rank-preserving slices are captured, with general mapped code generation bounded to 256 elements; positive rectangular maps use compact native slices. Native runtime-shaped slice products reuse artifacts across shapes with guarded cotangent dimensions. Source slicing captures signed int64 bounds/steps on static ranked roots, including negative and nested runtime views, clipping and empty results. Python integer arguments reuse the same CPU JIT specialization; compiler-owned capacity/shape sidecars allocate multiple multidimensional outputs up to the 1024-element host capacity. Index-only runtime gather adjoints now use shape-guarded accumulating scatters; Python index protocols and runtime-shaped original roots remain open. Single-element f32 exception-value payloads cross loop/finally completion as typed outputs; per-site/generation SSA slots retain distinct dynamic cause/context values in bounded expanded loops, and CPU/GPU VJP gates backward on successful forward completion; bounded loop-carried caught references retain their original generation payload; static caught identities, named re-raise, explicit causes and implicit contexts are reconstructed at the host boundary with native source-location notes. CUDA SM120 and ROCm gfx1151 execute mapped forward/backward and checked synchronous/asynchronous exception completion; failed frames expose no result. Extend custom object access, ownerless/noncontiguous writable views, dynamic strings, unbounded or object-carried loop exception identity and real CPython frame/traceback semantics, changing loop state and automatic effectful AD; arbitrary CFG is not closed.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Next: migrate legacy callers, health-check replacement workers and validate actual driver-failure behavior. Epoch ordering remains; no measured overlap or promotion.
- Depends on: —
- Start: host-free
- Latest: [recorded exploration](INTEGRATED_COMPILER_LOG.md#2026-09-11--gated-metadata-owner-and-isolated-recovery)

### AD-RESIDUAL-EVAL-1

**Logical-shape ABI and persistent checkpoint execution**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Paired source VJP retirement resumes only children that have not submitted frees after a dependency failure, on the original stream. Native AD products now bind bounded dynamic GPU inputs and scalar through rank-four floating/i8/i64 results using checked capacity/shape sidecars. A synchronous checked source VJP retains device snapshots and the matching forward residuals; exception completion is checked before backward and completion metadata receives zero seeds. Same-stream asynchronous snapshots and forward-gated backward submission now expose derivatives only after successful completion. Scoped source VJP now supports reader-aware retire/poll without a context wait on the healthy path. Explicit close can still synchronize; failed-free quarantine, deferred module unload, cross-queue writer ownership and broader product bindings remain open. Exact dominating SSA product guards now tighten joint temporary capacities. Extend relational/aliased volume proofs, saved heterogeneous products, general layout envelopes and automatic frontend wiring; exported shapes do not establish arbitrary Python CFG capture.
- Depends on: [W4-PRODUCT-1](#w4-product-1): existing bounded product carrier; arbitrary source CFG closure is not a prerequisite.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-10--ssd-recurrence-and-retryable-completion)

### W2.4a

**Generation ownership and scoped readers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Late worker death can now reconcile a failed recovery ticket without repeating termination, releasing its admission slot and owner exactly once; no device-health inference follows. Isolated ANN admission now executes independent numerical probes in the fresh worker before readiness; replacement requires confirmed predecessor death. Multi-stream external readers now include checked dynamic public frames and paired source-VJP products, and record all declared completion edges, and eventless dependencies refuse implicit host synchronization. Actual wedged-driver recovery remains unproven. Bounded asynchronous isolation teardown now retains failed owners and slots, and ANN/module owners finalize only after confirmed process death. CUDA/HIP stopped-worker replacement is independently remeasured; actual driver-hang recovery and device-wide health admission remain open; bounded workload probes are implemented. Opt-in native ANN workers now own their CUDA/HIP context and return checked private host outputs; uncertain requests quarantine the worker until confirmed process death. Stopped-worker teardown/replacement is measured separately from actual driver-hang recovery. Scoped runtime-shaped public frames and asynchronous static capture now order frees after declared readers. Module unload can run off-thread with bounded admission and non-waiting polls; a stalled/failed driver retains its owner. One through eight serialized incoming statuses now support bounded fan-in with scoped readers for each prerequisite; all 256 eight-status combinations have independent SM120/gfx1151 truth-table proof. Source state can produce immutable next-state GPU generations; exclusive synchronous owned-state copyback now reuses a private allocation with independent SM120/gfx1151 proof and blocks active scoped readers. Single-stream submit/poll now gates asynchronous copyback and excludes readers until completion; failure poisons the owner and retains pending storage. External borrowed-pointer mutation and concurrent multi-writer updates remain open. Extend heterogeneous dynamic persistent capture, unbounded/heterogeneous effect joins and external-reader adoption. Driver unload itself is not cancellable or latency-bounded, and unrestricted views retain synchronous close.
- Next architecture gate: [gated owner and isolated recovery](HEAP_BARRIER_ARCHITECTURE_REVIEW.md#gated-metadata-owner-and-isolated-recovery-2026-09-11): every admitted metadata operation in the opt-in gated owner has replay/device proof; live metadata and legacy import/snapshot paths refuse. Process-owned heap recovery requires confirmed death. Next: migrate legacy callers, health-check replacement workers and validate actual driver-failure behavior. Epoch ordering remains; no measured overlap or promotion.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [recorded exploration](INTEGRATED_COMPILER_LOG.md#2026-09-11--gated-metadata-owner-and-isolated-recovery)

### NUMPOL-CARRIER-1

**Numerical policy and analytic budget consumers**

- Owner: [FUNCTIONAL_ANALYSIS_TSOL_PLAN.md](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
- Gate: Extend policy preservation and analytic budget consumers across missing boundaries, with induced norms, intermediate-overflow checks and explicit approximation/spectral contracts. Use `scripts/record_dtype_codegen_inventory.py` alongside the operator dtype-flow report to expose every canonical/planned dtype, scalar/vector handling and matrix accumulators. The 2026-09-11 packet verifies 24 scalar/vector basic-arithmetic rows independently on CUDA and HIP, including signedness-sensitive division; the four previously failing FP8 rows per device now have byte-conversion legalization and exhaustive input-pair proof. Bool logic and bounded complex component probes pass on both devices; ten NVIDIA packed/scaled layout probes pass. Apple native f32 multiply/divide flush three expected subnormal results, so strict gradual-underflow admission remains open. Extend remaining packed/scaled, Apple storage, general complex, numerical-policy and matrix consumers before claiming closure; TF32 remains a math mode. See [dtype follow-through](../../../benchmarks/baselines/dtype_followthrough_20260911/README.md).
- Depends on: —
- Start: host-free
- Latest: [conversion and numerical boundary increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--fp8-conversions-and-numerical-boundaries)

### LAYOUT-ALG-1

**Remaining physical layout envelopes**

- Owner: [CORE_SUBSTRATE_VIEW.md](CORE_SUBSTRATE_VIEW.md)
- Gate: Preserve proved static/dynamic layout consumers; extend only unresolved nonseparable tuple/layout envelopes with capacity, alias and lifetime proof. Matrix acceleration additionally needs operand packing, signedness, accumulator type, fragment shape, scale layout and target instruction witnesses; Zen 5 VNNI/vector GEMM does not establish AMX support.
- Depends on: —
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--descriptor-projection-and-seven-program-continuation)

### AD-SOLVER-IFT-1

**Native implicit-solver consumers**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend residual/predicate/solver envelopes and Apple/NVIDIA consumers through compiler-owned children; require convergence/conditioning certificates and owning-device proof.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

### AD-HIGHER-1

**Broader AD, batching, sparsity and jets**

- Owner: [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md)
- Gate: Extend composed attention and broader native AD families, higher-order products, real batching, structural sparse derivatives and native jets through the existing AD owners; retain oracle, structural-zero and conditioning gates.
- Depends on: [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1): the selected product's residual and ownership ABI; independent static slices may proceed.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--domain-and-autodiff-documentation-consolidation)

### DIST-NATIVE-1

**Real multi-rank transport**

- Owner: [SCHEDULE_OBJECT_DESIGN.md](SCHEDULE_OBJECT_DESIGN.md)
- Gate: Extend bounded MPI proof beyond two ranks and proper-subgroup participants; add native NCCL/RCCL and other transports with process ownership and real multi-rank packets.
- Depends on: [EVIDENCE-PACKET-1](#evidence-packet-1)
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-09--architecture-sweep-and-failure-boundary-reconciliation)

## F5

### W5.2f

**Shared tiled-SSD compiler family**

- Owner: [SEQUENCE_MIXER_ENGINEERING_PLAN.md](SEQUENCE_MIXER_ENGINEERING_PLAN.md)
- Gate: The registered internal `schedule.ssd` now owns a static f32 scalar-decay recurrence, immutable initial/final carry and chunk-end checkpoints. Schedule-to-Tile lowers it to structured tensor loops, numerically executed through the native CPU JIT including partial chunks. Replay-bound serial CUDA SM120 and ROCm gfx1151 packages now execute three chunk sizes with output/carry/checkpoint and immutable-input checks. Opt-in cooperative CUDA/HIP kernels now assign each head/value column to a block, keep one state element per active lane, and reduce in state-index order through shared memory and barriers. Both owning devices pass correctness. Nine independent paired process runs per backend now retain exact artifact identities and resident event-window confidence bounds; native clock calibration still prevents promotion; explicit measured binding is now wired. Checkpoint VJP differentiates all five inputs and all three result cotangents, recomputes within chunks, and now executes on CUDA/HIP using checkpoints from the cooperative forward. A scoped native CPU owner integrates Y with automatic host-tape grad and private checkpoints. Exact-artifact measured binding now recomputes policy and retains incumbents without native clock calibration. ResidentSSDProgram now automatically pairs GPU forward/VJP packages, snapshots inputs device-to-device, owns checkpoints and all five gradients, and exposes a synchronous first-order value_and_grad API with scoped lifetime. Asynchronous SSD gradients now compose through projected scoped generations on distinct CUDA/HIP streams and retire after external readers. Public vjp now dispatches explicitly owned resident SSD programs. Capture, VJP and whole-frame retirement use stream events and async allocation/free; scoped forward and derivative readers delay reclamation. Explicit close remains synchronous; program retire_async/poll_close now orders all frame/reader retirements before bounded-admission off-thread module unload. Partial queueing is retryable and new capture stays closed. CUDA/HIP both validate two public-VJP frames without context synchronization; driver unload itself is not cancellable or latency-bounded. SSD packages now use a separate checked 64 MiB per-buffer bound; both GPUs validate a larger cooperative workload. ResidentSSDTrace now traces connected acyclic SSD compositions before submitting work, automatically captures forward/checkpoint frames and runs reverse calls with scoped readers. Fan-out and shared public inputs accumulate cotangents through replay-validated native f32 addition kernels. A three-call DAG and all five shared public input gradients pass an independent float64 finite-difference oracle on CUDA/HIP; addition allocations and modules participate in frame/program retirement. This is host orchestration of replay-bound packages, not a fused or canonical whole-program IR. Serializing composition into the compiler foundation, unused-input zeros, same-call input alias admission, effectful/data-dependent traces and additional operation families remain open. Next: arbitrary traced public tape integration, broader owner adoption and uncertain-unload isolation recovery, broader tiling/reduction tuning, public mixer/frontend integration, broader mutation/alias lineage, ReplaySSM comparison and selector-grade promotion.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-11--resident-dag-accumulation-and-snapshot-readers)

### TSOL-POLICY-PHYS-1

**Spectral policy breadth**

- Owner: [TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)
- Gate: Reconcile current spectral policy envelopes, then close remaining target-specific strides/full-spectrum/window, broadcasting and streaming execution gaps with independent adjoint proof.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### TSOL-SCALE-1

**ND and large-transform execution**

- Owner: [TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)
- Gate: Add missing ND/large/prime transforms with explicit algorithm, plan/twiddle/workspace identity and per-target retain/promote/reject comparisons.
- Depends on: [TSOL-POLICY-PHYS-1](#tsol-policy-phys-1)
- Start: device
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### TSOL-SHARD-1

**TSOL placement and sharding**

- Owner: [SCHEDULE_OBJECT_DESIGN.md](SCHEDULE_OBJECT_DESIGN.md)
- Gate: Reconcile spectral/solver/sparse placement rules and reshard consumers; separate mock-mesh correctness from actual multi-rank execution.
- Depends on: [DIST-NATIVE-1](#dist-native-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### TSOL-PHYS-TAIL-1

**High-value physical math families**

- Owner: [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md)
- Gate: Advance high-use solver, sparse/segment and layout families from target-specific gaps; preserve PDE discretization/halo and coalition-lattice obligations.
- Depends on: [E2E-REAL-6](#e2e-real-6): canonical artifact boundary for this workload, not every family migration.
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### W6.4

**Native batched finite algebra**

- Owner: [../domain/GA_EBM_ARCHITECTURE_REVIEW.md](../domain/GA_EBM_ARCHITECTURE_REVIEW.md)
- Gate: Carry W3.6 batching and grade-aware algebra through native consumers; packed grades/PGA/CGA/exp/log require independent semantic, AD and device proof.
- Depends on: [AD-HIGHER-1](#ad-higher-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-07--math-audit--foundation-reconciliation)

### RIEMANNIAN-OT

**Geometric primitives and OT validation**

- Owner: [RIEMANNIAN_OT_PLAN.md](RIEMANNIAN_OT_PLAN.md)
- Gate: Retain missing geometric primitives and constrained/KKT hypotheses under shared solver/AD owners; validate the OT workload without creating another solver stack.
- Depends on: [AD-SOLVER-IFT-1](#ad-solver-ift-1)
- Start: host-free
- Latest: [recorded increment](INTEGRATED_COMPILER_LOG.md#2026-09-06--functional-analysis-contracts--consolidated-ownership)

## Reconciliation before archival

The move preserves every old queue/wave paragraph and historical “Next” list.
These are the disposition rules applied to the review inventory; an uncertain
remainder stays with its owner and is not silently declared complete.

| Earlier obligation | Current destination and reconciliation |
|---|---|
| W1.1; W2.4; W2.4a NVIDIA producers/Target/barriers | W1.1 and W2.4a retain architecture closure and wrapper/ownership review. Shared legality and recent bounded reader proofs are not all-route closure. |
| W3.1, W3.2, W3.4, W3.7 | E2E-REAL-6 owns remaining frontend, JitFn and forward/direct constructor work. W3.2's old estimate is retired; estimate each census-selected route. |
| W3.3 | Retains a consumed-semantics inventory gate before a dialect split. |
| W3.5; OT geometric primitives | AD-SOLVER-IFT-1 and RIEMANNIAN-OT retain broader consumers/hypotheses; shared solver slices are not universal device support. |
| W3.6; W6.4 | W6.4 retains batched grade-aware algebra and actual fold-marker consumers. |
| W4.1–W4.3 | W4-PRODUCT-1 and AD-RESIDUAL-EVAL-1 retain frontend CFG/effects, exported logical shapes and checked asynchronous products. Recent native function CFG/status work is bounded. |
| W5.1 | AD-RESIDUAL-EVAL-1 retains selected production checkpoint policies and complete exact-device comparisons; bounded native SAVE/HYBRID/recompute execution already exists. |
| W5.2; W5.2e; W5.2f; W5.5 | W5.2 retains producer/calibration/selector work; W5.2f owns tiled SSD; W5.5 owns consumer-justified canonicalization. Existing inferred DAGs and prune-only search are not promotion proof. |
| W5.4 | DIST-NATIVE-1 retains transport beyond the bounded MPI slice; TSOL-SHARD-1 retains spectral/solver placement. |
| W6.1–W6.3 | AD-HIGHER-1 routes to the active AD plan's higher-order, batching, sparse and native-jet obligations; do not rebuild landed reference algebra. |
| Ordered queue 8–11 | TSOL-POLICY-PHYS-1, TSOL-SCALE-1, TSOL-SHARD-1 and TSOL-PHYS-TAIL-1 preserve policy, scale, placement and physical-tail gates. |
| Ordered queue 12–14 | TPROF-NATIVE-1, EVIDENCE-PACKET-1 and COMPILER-DEVEX-1 retain clocks, packet consumers and installed/assertions-enabled tooling. |
| Six non-wave scoped notes | PDE, math-source, AttnRes, reference-tier, layout and frontend retain their scoped owners in the routing index. They are workload contracts, not new waves. |
| Later numerical and ANN “Next” lists | NUMPOL-CARRIER-1 and MSW-9 retain broader analytic consumers and tuned native admission; recent bounded admission/row schedules are not performance promotion. |

## Maintenance and proof rules

- A substantive delivery PR updates its active task record and appends a log
  entry naming that ID. Completion removes the record only after the routing
  index points to a scoped owner, successor or archived disposition.
- CI checks structured IDs, destinations, dependencies, latest-log links and
  the derived start view. A merge-base check requires an owner-record change
  when a new log entry is added; it does not infer ownership from arbitrary code.
- Keep five proof questions independent: parse/verify; lowering consumes the
  incoming IR; runtime binds that artifact; owning device executes correctly;
  performance evidence is eligible. Use existing proof vocabulary and F0-derived
  inventories, not hand-maintained five-column status copies.
- Preserve ABI/layout/policy/provenance through serialized IR and replay;
  migrated routes need differential tests and exact-target execution before
  retiring their old constructor. Presence of bytes or hashes is insufficient.
- Shared changes assess all four [backend queues](../backend/README.md). Runtime
  timeouts must retain in-flight resources and reject unsafe reuse. Reference
  execution and WSL regression timing never silently become native promotion.
- Follow [AGENTS.md](../../../AGENTS.md), [governance and fleet facts](../../../CLAUDE.md),
  and the [compiler test plan](../../../tests/COMPILER_TEST_PLAN.md). Do not copy
  OS versions, command pins, budgets or governance decisions into this queue.
- Scoped acceptance remains in its owner document; the global queue chooses
  sequencing, not duplicate domain designs. The log has no current next-order.

## Routing index

Fixed columns are a navigation contract. `owner`, `successor` and `archive`
describe routing, not readiness. Historical mentions need not be active tasks.

| ID | Canonical destination | Relationship |
|---|---|---|
| AD-HIGHER-1 | [AD-HIGHER-1](#ad-higher-1) | owner |
| AD-RESIDUAL-EVAL-1 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | owner |
| AD-SOLVER-IFT-1 | [AD-SOLVER-IFT-1](#ad-solver-ift-1) | owner |
| COMPILER-DEVEX-1 | [COMPILER-DEVEX-1](#compiler-devex-1) | owner |
| DISPATCH-BREAKER | [DISPATCH-BREAKER](#dispatch-breaker) | owner |
| DIST-NATIVE-1 | [DIST-NATIVE-1](#dist-native-1) | owner |
| E2E-REAL-6 | [E2E-REAL-6](#e2e-real-6) | owner |
| E2E-REAL-6F | [E2E-REAL-6F](#e2e-real-6f) | owner |
| EVIDENCE-PACKET-1 | [EVIDENCE-PACKET-1](#evidence-packet-1) | owner |
| FRONTEND-IR-MEDIUM-1 | [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1) | owner |
| LAYOUT-ALG-1 | [LAYOUT-ALG-1](#layout-alg-1) | owner |
| MSW-9 | [MSW-9](#msw-9) | owner |
| NUMPOL-CARRIER-1 | [NUMPOL-CARRIER-1](#numpol-carrier-1) | owner |
| RIEMANNIAN-OT | [RIEMANNIAN-OT](#riemannian-ot) | owner |
| TPROF-NATIVE-1 | [TPROF-NATIVE-1](#tprof-native-1) | owner |
| TSOL-PHYS-TAIL-1 | [TSOL-PHYS-TAIL-1](#tsol-phys-tail-1) | owner |
| TSOL-POLICY-PHYS-1 | [TSOL-POLICY-PHYS-1](#tsol-policy-phys-1) | owner |
| TSOL-SCALE-1 | [TSOL-SCALE-1](#tsol-scale-1) | owner |
| TSOL-SHARD-1 | [TSOL-SHARD-1](#tsol-shard-1) | owner |
| W0.1 | [W0.1](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w01) | archive |
| W0.10 | [W0.10](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w010) | archive |
| W0.2 | [W0.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w02) | archive |
| W0.3 | [W0.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w03) | archive |
| W0.4 | [W0.4](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w04) | archive |
| W0.5 | [W0.5](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w05) | archive |
| W0.6 | [W0.6](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w06) | archive |
| W0.7 | [W0.7](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w07) | archive |
| W0.8 | [W0.8](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w08) | archive |
| W0.9 | [W0.9](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w09) | archive |
| W1.1 | [W1.1](#w11) | owner |
| W1.1b | [W1.1b](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w11b) | archive |
| W1.2 | [W1.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w12) | archive |
| W1.3 | [W1.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w13) | archive |
| W1.4 | [W1.4](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w14) | archive |
| W2.1 | [W2.1](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w21) | archive |
| W2.2 | [W2.2](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w22) | archive |
| W2.3 | [W2.3](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w23) | archive |
| W2.4 | [W2.4a](#w24a) | successor |
| W2.4a | [W2.4a](#w24a) | owner |
| W3.1 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.2 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.3 | [W3.3](#w33) | owner |
| W3.4 | [E2E-REAL-6](#e2e-real-6) | successor |
| W3.5 | [AD-SOLVER-IFT-1](#ad-solver-ift-1) | successor |
| W3.6 | [W6.4](#w64) | successor |
| W3.7 | [E2E-REAL-6](#e2e-real-6) | successor |
| W4-PRODUCT-1 | [W4-PRODUCT-1](#w4-product-1) | owner |
| W4.1 | [W4-PRODUCT-1](#w4-product-1) | successor |
| W4.2 | [FRONTEND-IR-MEDIUM-1](#frontend-ir-medium-1) | successor |
| W4.3 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | successor |
| W5.1 | [AD-RESIDUAL-EVAL-1](#ad-residual-eval-1) | successor |
| W5.2 | [W5.2](#w52) | owner |
| W5.2a | [W5.2a](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52a) | archive |
| W5.2b | [W5.2b](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52b) | archive |
| W5.2c | [W5.2c](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52c) | archive |
| W5.2d | [W5.2d](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52d) | archive |
| W5.2e | [W5.2](#w52) | successor |
| W5.2e-PRODUCER-1 | [W5.2](#w52) | successor |
| W5.2f | [W5.2f](#w52f) | owner |
| W5.2g | [W5.2g](archive/INTEGRATED_COMPILER_PLAN_2026-08-02_WAVES.md#w52g) | archive |
| W5.3 | [W5.2](#w52) | successor |
| W5.4 | [DIST-NATIVE-1](#dist-native-1) | successor |
| W5.5 | [W5.5](#w55) | owner |
| W6.1 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.2 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.3 | [AD-HIGHER-1](#ad-higher-1) | successor |
| W6.4 | [W6.4](#w64) | owner |
| PDE-STENCIL-FOUNDATION-1 | [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md) | owner |
| MATH-SOURCE-WORKSTREAM-1 | [MATH_SOURCE_WORKSTREAM.md](MATH_SOURCE_WORKSTREAM.md) | owner |
| BLOCK-ATTNRES-1 | [BLOCK_ATTNRES_ROCM_PLAN.md](BLOCK_ATTNRES_ROCM_PLAN.md) | owner |
| REF-TIER-PHYS-1 | [PDE_STENCIL_CAPABILITY_PLAN.md](PDE_STENCIL_CAPABILITY_PLAN.md) | owner |
| FA-1 | [FUNCTIONAL_ANALYSIS_TSOL_PLAN.md](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-2 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-BATCH-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-SPARSE-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| AD-JET-IR-1 | [AUTODIFF_EXECUTION_PLAN.md](AUTODIFF_EXECUTION_PLAN.md) | owner |
| IKF-1 | [INTRA_KERNEL_FEEDBACK_PLAN.md](INTRA_KERNEL_FEEDBACK_PLAN.md) | owner |
| F0 | [F0](#f0) | owner |
| F2 | [F2](#f2) | owner |
| F3 | [F3](#f3) | owner |
| F4 | [F4](#f4) | owner |
| F5 | [F5](#f5) | owner |
| F1 | [first migration](INTEGRATED_COMPILER_LOG.md#2026-09-04--foundation-f1-implementation) | archive |
| E2E-REAL-0 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-1 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-2 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-3 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-4 | [E2E-REAL-6](#e2e-real-6) | successor |
| E2E-REAL-5 | [E2E-REAL-6](#e2e-real-6) | successor |
| MSW-1 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-2 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-3 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-4 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-5 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-6 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-7 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| MSW-8 | [math-source owner](MATH_SOURCE_WORKSTREAM.md) | owner |
| FA-3 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-4 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-5 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-6 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| FA-7 | [functional-analysis owner](FUNCTIONAL_ANALYSIS_TSOL_PLAN.md) | owner |
| AD-FWD-NATIVE-1 | [active AD owner](AUTODIFF_EXECUTION_PLAN.md) | owner |
