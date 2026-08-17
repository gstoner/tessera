---
last_updated: 2026-08-16
audit_role: plan
plan_state: open
supersedes_queues_in:
  - COMPILER_ARCHITECTURE_SWEEP.md §4
  - FRONTEND_GRAPH_SCHEDULE_REVIEW.md §5
  - IR_STACK_INTEGRATION_REVIEW.md §5
  - AUTODIFF_ARCHITECTURE_REVIEW.md §5
  - TARGET_IR_REVIEW.md §5
  - AUTODIFF_UNIFICATION_PLAN.md §7
  - COMPILER_REFACTOR_PLAN.md §9
  - EVALUATOR_PLAN.md §10
  - OPTIMIZING_COMPILER_PLAN.md §5
  - SEQUENCE_MIXER_ENGINEERING_PLAN.md §7
  - ../domain/GA_EBM_ARCHITECTURE_REVIEW.md §4
  - RIEMANNIAN_OT_PLAN.md §4
---

# Integrated Compiler Plan

Start at [`README.md`](README.md) for the folder authority map. This is the sole
cross-domain compiler sequencing authority.

One plan across the compiler reviews and scoped plans. Each document's ranked queue stays as *evidence
and rationale*; **this document owns sequencing and de-duplication.** Where a
review's queue and this document disagree on ordering or cost, this document
wins — the reviews were written independently and double-counted overlapping
work.

Status truth remains `MASTER_AUDIT.md` and `docs/audit/generated/`
(Decision #26). Nothing here reclassifies a row. Effort figures are engineering
estimates for a single track with no hardware gates, not commitments.

## How to execute this plan now

Earlier E2E-REAL slices established content-addressed Graph→Schedule→Tile
lineage and bounded x86/gfx1151 physical consumers. The next route is no longer
E2E-REAL-0:

1. **Completed — AD-CORE-LINEAR-1:** `LinearTransposeInterface` is the Graph-IR
   authority for transpose/reshape, broadcast/expand, the structural view
   family, and operand-wise multilinear matmul. Both autodiff passes consume
   it; paired CPU execution covers inverse shape transport, every matmul
   transpose-flag combination, and fail-closed unsupported operations. The
   Python linear registry remains an oracle, not the production authority.
2. **Compiler slice complete — AD-TSOL-SPECTRAL-1:** Graph IR now owns explicit
   normalization, logical length, spectrum layout/Hermitian identity, and DCT
   type. FFT/IFFT/RFFT/IRFFT/DCT transposes have paired CPU numerical proof;
   STFT/ISTFT/filter/convolution emit a content-addressed multi-output
   Schedule→Tile carrier. Real-input full FFT autodiff fails closed and directs
   callers to the explicit packed-real `rfft` contract. The first native
   compound-backward slice now consumes complex-f32 spectral filter and
   unbroadcast full-f32 spectral convolution on AVX-512 and gfx1151 without
   returning to Graph IR. STFT/ISTFT backward, broader axes/dtypes/broadcasting,
   and performance promotion remain architecture-owned and fail closed.
3. **Completed foundation audit — GRAPH-VERIFY-SIGNED-1:** registered Graph and
   canonical-attention integer legality now reads the signed `IntegerAttr`
   spelling rather than MLIR 23's unsigned native value accessor. Negative
   extents, block counts, seeds, cache windows, control bounds, and indices fail
   closed, with direct negative IR proof. Registration and `hasVerifier`
   coverage are not treated as semantic-verifier proof.
4. **Completed — AD-CORE-EFFECT-CONTROL-1:** canonical `stop_gradient` is a
   Graph operation and compiler barrier; both autodiff passes compute backward
   SSA activity, carry registered Graph effects, reject active stochastic work,
   permit inactive regions, and fail closed for active regions or stopped
   residuals that cannot be replayed safely. Direct lit and paired CPU proof
   cover the legal and negative paths.
5. **AD-SOLVER-IFT-1 — bounded physical pilot landed:** the registered solver
   dialect owns the general IFT semantics. The first content-addressed physical
   contract carries `R(theta,x)=x²-theta` through Schedule and Tile into native
   AVX-512 and compiler-generated gfx1151 packages. Both packages execute the
   residual, transposed matrix-free solve, and residual adjoint and have
   compiled numerical packets. A general physical parent now binds immutable
   residual and solution/parameter JVP/VJP children and executes restarted
   GMRES on AVX-512 and gfx1151. The compiler now generates the five physical
   children from verified Graphs. The typed child-program envelope now includes
   pointwise operations, sum/mean with broadcast-correct cotangent reduction,
   rank-2 matmul/transpose, distinct parameter and solution spaces, bounded
   dynamic dimensions, explicit f16/bf16-to-f32 package-boundary widening, and
   statically bounded `control_for` expansion. The nonlinear
   `R=x*x+sin(x)-theta` packets remain the architecture-owned proof baseline.
   Pure scalar data-dependent `if` and bounded `while` now lower to explicit
   compare/select SSA with digest-bound recomputation in every child. Expanded
   30-sample AVX-512 and gfx1151 WSL packets cover reduction, reduced-storage
   matmul, bounded-dynamic mixed storage, both predicate regions, and ISTFT
   window products. Clean selector-grade timing remains open.
6. **AD-RESIDUAL-EVAL-1 — bounded execution bridge landed:** complete backward
   samples and unique retained residual bytes stamp Graph rematerialization
   only from eligible evidence. Counted-region treeverse now executes exact
   checkpoint capture, primal replay, and backward callbacks, converting an
   estimated candidate into an eligible row only after complete execution.
   General MLIR region adjoints and broader exact-family packets remain open.
7. **E2E-REAL-6:** remove duplicate lowering authorities only after migrated
   families satisfy lineage, correctness, and architecture-owned evidence.
8. **COLLECTIVE-NATIVE-FOUNDATION-1 — landing:** the C++ NCCL/RCCL adapters
   issue real NCCL-compatible calls, query initialized communicator properties,
   and own symmetric registrations through move-only RAII windows. Target
   artifacts bind initiation, registration, ordering, capture policy,
   backend/source identity, and the exact communicator-capability digest;
   device initiation rejects a mismatched runtime topology. The two-gfx1151
   LSA harness is compiled and registered as an opt-in hardware CTest. Current
   WSL communicator discovery passes but reports device API false, zero LSA
   teams, no GIN, host RMA true, and one visible GPU; symmetric registration
   and peer correctness therefore remain blocked. Zero-CU Copy Engine,
   GIN/RMA, and gfx1250 DDA now have independent typed artifact lanes and
   legality gates. Copy Engine binds zero-CTA communicator initialization;
   GIN/RMA binds strict registered windows and public one-sided operations;
   gfx1250 DDA binds architecture plus selector-evidence identity without
   importing an RCCL-internal selector. Exact-device correctness, route proof,
   and performance packets remain separate open slices under sync
   `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09`.
   GIN now has registered Target operations for window lifetime and ordered
   put/signal/wait, a content-addressed lifetime verifier, and rank-local
   runtime dispatch. Its native executable binds explicit Tessera, OpenMPI,
   PMI, or Slurm ranks to `ncclCommInitRank`, shared rendezvous, symmetric
   windows, exact ring readback, and independent HIP-event/host-wall clocks.
   Exact multi-node gfx1151 proof remains open; ordinary collective records
   cannot opt into the one-sided lane.
9. **AMD-ISA-DTYPE-1 — cross-generation foundation landing:** gfx1151,
   gfx1200, and CDNA5 gfx1250/gfx1251 now have a dtype-total architecture
   contract separating scalar/vector storage, dense WMMA, sparse SWMMAC,
   accumulator, scale-operand, compiler-state, and exact mnemonic identity.
   CDNA5 is modeled as wave32 XDL-WMMA rather than inherited wave64 MFMA;
   MI455X maps to gfx1250 and MI430X to gfx1251 with distinct cost identities.
   The C++ fragment boundary recognizes a distinct CDNA5 ABI and lowers the
   already-proven f16/bf16 K32 forms. FP8/BF8, IU8, F8F6F4/FP4/MX scaling,
   sparse materialization, and gfx1251 FP64 WMMA remain fail-closed until typed
   Target operations, LLVM serialization, and exact-product packets land.

Scoped plans own the design and acceptance details for these items. Backend
plans own exact-device promotion. Neither creates a competing global order.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` normalizes the
codegen dashboard to one `BackendKernelEntry` op×target grain. Runtime paths
are now independent evidence rather than a second additive denominator;
exact-device verification is split from `fused`/`packaged` implementation
presence; explicit no-kernel terminals close Tile/Target phases; and closed
rows no longer appear as open work. This changes reporting and drift gates,
not physical packages, selectors, or device evidence.

The standalone primitive dashboard now generates its live registry totals,
compiler-layer summary, exact-target manifest rollup, and open queues. Its
aggregate compiler row is explicitly best-available evidence, not a universal
backend claim. The 2026-08-08 reconciliation removed the stale Adafactor
single-GPU terminal override and registered existing physical TSOL/Adafactor
benchmark harnesses; it changed audit truth, not implementation or selector
state. The four collective Tile contracts are now first-class ODS operations
with exact Schedule→Tile lowering and shared shape/reduction verification.
They lower into a registered asynchronous `tessera_collective` Target dialect
and a content-addressed runtime-adapter package; deterministic two-rank tests
execute all four operations without returning to Graph IR. The functional
portable Target/runtime axis is complete. As of
`COLLECTIVE-ASYNC-UNIFY-2026-08-09`, the older forward/autodiff transform
producers also emit this registered future/await contract and rewire SSA users;
no active compiler producer emits an unregistered collective pseudo-dialect.
Native architecture transport,
exact multi-rank execution, and performance proof stay open and
architecture-owned. `DIST-SHARD-ALIAS-1` also removes the misleading "nine
backend contracts" framing: `named_sharding` and `partition_spec` are
compile-time metadata, `shard_map` is a compile-time region contract,
`psum`/`pmean`/`pmax`/`pmin` map to registered all-reduce records, and
`broadcast_to_axis` maps to all-gather. Those five aliases now execute through
the portable two-rank runtime. `collective_permute` is now a distinct typed
Tile/Target operation carrying an immutable source/destination peer map. The
portable runtime and the one-process/multi-device NCCL/RCCL launcher execute
it with grouped send/receive calls, zero-fill unmatched destinations, and
reject duplicate or out-of-range peers. The remaining foundation queues are
frontend capture of these aliases, MPI/OFI/SHMEM process launch, subgroup
communicator construction, exact NCCL/RCCL multi-rank evidence, broader
compiler autodiff/sharding,
target-specific benchmark packets, and duplicate-authority deletion in
E2E-REAL-6.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` also closes
the actionable direct-test bucket: the five differentiable relaxations now
have public-forward/oracle coverage and the two fused training rows have exact
x86/gfx1151 fused-versus-unfused proof. The raw thin-reference count remains a
scanner metric; its 87 structural rows are explicitly non-numerical contracts,
not hidden primitive test debt.

The x86 production tool/image boundary also follows the common fail-closed
build selection as of 2026-08-08: `TESSERA_BUILD_DIR` selects both
`tessera-opt` and `libtessera_x86_elementwise.so`. This prevents a current
Python contract from pairing with a stale compiler or image in another CMake
tree. The exact-host FFT suite passes against the selected clean LLVM/MLIR 23
tree, and the standalone comparison now measures normalized forward and
inverse C2C execution independently.

**Source reviews:**
[GA/EBM](../domain/GA_EBM_ARCHITECTURE_REVIEW.md) ·
[Autodiff](AUTODIFF_ARCHITECTURE_REVIEW.md) ·
[Sweep](COMPILER_ARCHITECTURE_SWEEP.md) ·
[Frontend→Graph→Schedule](FRONTEND_GRAPH_SCHEDULE_REVIEW.md) ·
[IR Stack Integration](IR_STACK_INTEGRATION_REVIEW.md) ·
[Target IR](TARGET_IR_REVIEW.md) ·
[Riemannian OT](RIEMANNIAN_OT_PLAN.md)

---

## 0. End-to-end reset — one artifact must cross every boundary

The original waves are useful root-cause groupings, but they are not by
themselves an executable compiler program. A source review on 2026-08-04 found
that Tessera can report all of Graph, Schedule, Tile, Target, native-image, and
launch-descriptor stages without proving that stage N+1 consumed stage N. This
section is therefore the **delivery order** for the plan. The thematic W0-W6
queues remain the owning backlog; when their ordering conflicts with this
vertical program, this section wins.

### 0.1 Rechecked current truth

The repository has substantially more real machinery than the seven source
reviews originally credited:

- `canonical_compile()` is the common orchestration entry and
  `CompileArtifactBundle` carries typed stage artifacts, native images, launch
  descriptors, content hashes, and runtime state.
- The Python `lower_graph_to_schedule_ir` → `lower_schedule_to_tile_ir` →
  `lower_tile_to_target_ir` ladder is a real object-model lowering and is useful
  as an oracle.
- NVIDIA SM120, ROCm gfx1151, x86, Apple GPU, and Apple CPU all have bounded
  native package producers and descriptor launch paths. Existing E2E-SPINE work
  made the image/descriptor/cache/runtime contracts real.
- Schedule runtime steps are consumed. Pipeline-step execution, supported
  collective overlap, and optimizer-shard state transitions are no longer a
  missing-runtime item.
- W1.1's current ROCm work proves a typed
  `view → fragment_pack → mma → fragment_unpack → store` chain on gfx1151,
  including ragged shapes. That is a valid backend-lowering proof.

Those facts do **not** yet make one compiler spine:

| Boundary | What exists | Why it is not yet canonical end to end |
|---|---|---|
| Python source → Graph | The tracer now promotes directly into verified `GraphIRModule` and is the production authority for the native-forward-product cohort; `@jit` AST extraction, textual parsing, and constrained adapters remain for unmigrated families | More than one producer remains globally, but migrated JVP packages bind tracer Graph identity and no longer consult decoration-time AST Graph IR. |
| Graph → Schedule | Python lowering is substantive; C++ `GraphToSchedulePass` only stamps `schedule.artifact_hash = "__pending__"` | The C++ pass rewrites no operation. The Schedule ODS is tablegen input but has no built dialect library/registration in production `tessera-opt`. |
| Schedule → Tile | Python lowering is substantive; C++ `ScheduleToTilePass` only stamps `tile.staged` | The Python schedule model drops value types in its textual form, while C++ `schedule.tile` is metadata-only and has no SSA inputs/results. Neither is a production value-carrying boundary. |
| Shared Tile → backend package | Typed value and launch-level Tile contracts exist | Native packagers accept the original `GraphIRModule`, classify it again, and synthesize backend-owned Tile text. They do not consume the shared ladder's Tile artifact. |
| Target → image → runtime | Image/descriptor validation and launch are real | This validates the package-internal Target digest, not continuity from the earlier shared Schedule/Tile artifacts. |

The concrete discontinuity is in `driver.compile_graph_module`: it builds a
`cpu_plan`, then a native branch calls `*_native.package_native(module, ...)`
with the **original Graph module** and replaces `bundle.tile` and
`bundle.target_ir` with the package's artifacts. `spine_stages()` records that
all stages exist, but carries no parent digest and cannot detect the fork. The
existing E2E-SPINE claims are therefore correctly interpreted as
**package/runtime E2E**, not yet **compiler-boundary E2E**.

Backend recheck:

| Target | Proven today | Remaining lineage break |
|---|---|---|
| x86 / AVX-512 | Broad typed native packages, runtime descriptors, exact Zen 5 execution, and one canonical scheduled f32 matmul consumer | Matmul now preserves adjacent Graph→Schedule→Tile→Target/image lineage. Other admitted families still re-derive a family-specific `tile.*_kernel` from Graph IR. |
| ROCm / gfx1151 | Exact-device package families, the typed WMMA GEMM differential path, and one canonical scheduled f16/f32 matmul consumer | Matmul now preserves adjacent Graph→Schedule→Tile→Target/HSACO lineage. Other admitted families still synthesize their own Graph/Tile envelopes. |
| NVIDIA / SM120 | Typed PTX/native packages for bounded families | `nvidia_native` classifies Graph IR and emits NVIDIA-owned launch Tile programs; shared Schedule/Tile is bypassed. |
| Apple GPU/CPU | Descriptor launch and exact-host packets for bounded families | Packaging largely binds a prebuilt runtime image to a synthesized Target call. It does not compile the preceding shared Tile artifact, and Apple execution must be re-proven on the Mac. |

### 0.2 Definition of a real E2E compiler path

A lane is **compiler-boundary E2E** only when one compile request satisfies all
of the following:

1. Each artifact records `producer`, `input_digest`, `output_digest`, target,
   and contract version. For every adjacent pair,
   `next.input_digest == previous.output_digest`.
2. Graph semantics are never reconstructed from op names after Graph lowering.
   A backend may select among candidates, but the selected compiler candidate
   consumes the canonical Tile or Target artifact rather than the original
   `GraphIRModule`.
3. Schedule IR preserves the SSA computation it schedules. The initial design
   is mixed-level: Graph ops remain the semantic payload while `schedule.*`
   records decisions; Schedule→Tile atomically replaces the payload with typed
   Tile values/launch envelopes. A metadata-only `schedule.tile` is not a
   substitute for the computation.
4. Every boundary parses and verifies in production `tessera-opt`; Python
   models may be differential oracles but cannot be the only executable
   lowering behind a C++ capability claim.
5. Selected tile sizes, warps, stages, raster policy, numeric policy, layouts,
   and dynamic-memory expressions survive into the physical package or carry a
   named, verified drop reason.
6. The native image is derived from that Target artifact, the descriptor is
   validated against that image, and the runtime launches the descriptor with
   fallback disabled.
7. A hardware lane compares against the numerical oracle and records the
   correct timing domain. Promotion remains separate: a correct compiler lane
   does not replace a faster production candidate until its architecture-owned
   performance ratchet passes.

### 0.3 Delivery program — vertical slices, not horizontal completion

The first capability is deliberately narrow: one static rank-2 matmul semantic
fixture with two typed instances — x86 f32 and ROCm f16 storage with f32
accumulation/output. This is the smallest workload with existing Graph
semantics, schedule knobs, logical and launch Tile contracts, two locally
available physical backends, numerical oracles, and performance baselines. A
future shared bf16 instance can compare both architectures under one storage
contract; the initial proof does not pretend target-specific Graph digests are
identical.

| Order | Work item | Deliverable | Stop-the-line gate |
|---|---|---|---|
| 0 | **E2E-REAL-0 — lineage truth** — **implemented 2026-08-05.** | `LoweringArtifact`/`CompileArtifactBundle` carry parent/output digests, producer identity, representation, contract version, and `lineage_complete`. Existing Graph-owned native packages now report their Tile fork instead of inheriting boundary-E2E from stage presence; no package was deleted or demoted. | `test_compiler_artifact_lineage.py` proves digest adjacency, substituted-Tile detection, stage-presence independence, tamper rejection, and JSON/runtime-metadata round trips. |
| 1 | **E2E-REAL-1 — make Schedule IR real** — **bounded matmul slice implemented 2026-08-05.** | The generated Schedule ODS library is compiled, linked into, and registered by both full and lean production `tessera-opt`. `schedule.matmul` now preserves the Graph result as an SSA subject and carries a lowercase SHA-256 decision contract; the matching `schedule.artifact` and Graph attribute use the same digest. The prior `__pending__` annotation skeleton is gone. | Production builds list `schedule`; the intermediate fixture proves Graph semantics remain live behind the scheduled SSA edge, all three digest copies agree, and malformed/tampered decisions fail closed. |
| 2 | **E2E-REAL-2 — Graph/Schedule → launch Tile matmul** — **implemented 2026-08-05.** | The C++ Schedule→Tile pass atomically consumes `schedule.matmul`, its matching durable artifact, and the retained Graph producer for bounded static rank-2 x86-f32 and ROCm-f16/f32. It materializes A/B/D/M/N/K and emits one verified `tile.matmul_kernel` with numeric, layout, tile, pipeline, raster, and schedule-digest attributes. | `e2e_matmul_graph_schedule_tile.mlir` proves no Graph/Schedule op survives and exactly one six-operand launch op is produced for both targets. Dynamic shapes, ROCm f32 storage, forged hashes, altered knobs, and missing/duplicate artifacts fail closed. |
| 3 | **E2E-REAL-3 — two physical consumers on this host** — **implemented for the bounded matmul slice 2026-08-05.** | `ScheduledMatmulArtifact` is the typed package boundary. The canonical x86 and ROCm matmul routes consume its exact launch Tile text; x86 runs `TileToX86Pass`, while ROCm runs `GenerateWMMAGemmKernel` before `TileToROCM`/ROCDL/HSACO. Their retained Graph-owned matmul emitters remain separately callable candidates. | Per-request lineage is adjacent through Graph/Schedule/Tile/Target/backend; lit proves both physical consumers remove Graph/Schedule/launch Tile; descriptor launches agree numerically without fallback on the established three-shape Zen 5 corpus and aligned plus ragged gfx1151 cases. Performance and selector promotion remain E2E-REAL-4. |
| 4 | **E2E-REAL-4 — performance and promotion decision** — **completed for Zen 5; gfx1151 retained pending selector-grade timing, 2026-08-05.** | The shared contract now distinguishes the 16x16 instruction tile from an architecture-selected macro tile. Zen 5 preserves 16x16 and promotes: aligned/ragged outputs are bit-identical and scheduled/production medians are 1.031x/0.988x. gfx1151 preserves its committed 32x64 (2x4 WMMA) choice and recovers direct performance: 18.29 versus 18.34 TFLOP/s, 2.281x the 8.02 floor, with zero route difference on aligned/ragged cases. | The committed reports name device, toolchain, artifact digests, warm/cold state, resources, numerical error, and timing domain. x86 promotes its already-canonical selection without another selector mutation. ROCm retains because WSL host-wall timing is not selector-grade device-event evidence; the initial 16x16 result (4.14 TFLOP/s) was rejected and triggered the macro-tile contract. gfx1200/gfx1250 fail closed until their own profiles and device packets exist. |
| 5 | **E2E-REAL-5 — migrate breadth one family at a time** — **bounded f32 softmax/reduction slice implemented for x86 and gfx1151, 2026-08-05; broader families remain landing.** | `schedule.softmax` and `schedule.reduce` now bind Graph semantics, architecture-owned numeric/launch policy, and one SHA-256 identity before Schedule→Tile emits the exact launch artifact. `ScheduledKernelArtifact` carries that artifact into x86 and ROCm packaging without Graph re-entry. The first truthful envelope is static f32 last-axis softmax plus rank-reducing sum/mean/max; x86 reduction is last-axis, while gfx1151 also consumes arbitrary static axes. ROCm f16/bf16→f32 and `keepdims=true` remain on the explicit Graph-owned route because canonical `tessera.reduce` currently requires same-element-type, rank-reduced output. Attention forward/backward is next, followed by stateful/training families. | Structural and tamper lit gates pass; exact Zen 5 softmax/reduction and exact gfx1151 softmax/reduction descriptor launches agree with NumPy. Per-request Graph→Schedule→Tile→Target→backend lineage is adjacent and both packages prove exact Tile-text consumption. No selector or performance promotion is inferred. NVIDIA follows on SM120; Apple follows on the Mac; no evidence transfers between targets. |
| 5A | **E2E-REAL-5A — canonical attention forward** — **bounded x86/Zen 5 and ROCm/gfx1151 slice implemented 2026-08-05.** | `schedule.attention` content-addresses the static rank-4 recurrence, modifiers, launch dimensions, physical dtype, and architecture-owned backward-LSE policy. `ScheduledAttentionArtifact` retains the shared batch/query-head/KV online-softmax recurrence as semantic proof and carries exactly one launch-level `tile.attention_kernel` into TileToX86 or TileToROCm/ROCDL without Graph re-entry. x86 preserves `save_lse/saved`; gfx1151 preserves `gfx1151_auto_128` and its per-shape saved/recompute selection. | Production `tessera-opt` proves Graph→Schedule→Tile adjacency and rejects stale policy. Exact Zen 5 execution agrees with NumPy; exact WSL-visible Radeon 8060S/gfx1151 execution agrees with the shared streaming-attention oracle for GQA, causal windowing, ragged `Sq=17/Sk=19`, and f16→f32. gfx1200/gfx1250 fail closed until architecture-owned profiles and device packets exist. Attention backward remains the next family boundary; NVIDIA/Apple require owning-host follow-ups. |
| 5B | **E2E-REAL-5B — canonical attention backward** — **bounded x86/Zen 5 and ROCm/gfx1151 slice implemented 2026-08-05.** | `schedule.attention_backward` is one content-addressed three-result dQ/dK/dV edge. Its digest binds the tensor-valued dQ loop, two-way dK/dV split, ascending reduction, launch-owned workspace, modifiers, and architecture-owned LSE checkpoint identity. `ScheduledAttentionBackwardArtifact` carries the exact program Tile text into the x86 saved-LSE ABI or the gfx1151 five-entry package without Graph re-entry. | Production Graph→Schedule→Tile and both physical lowerings pass. Exact Zen 5 and WSL-visible gfx1151 tests cover MHA/GQA/MQA, aligned and ragged shapes, causal/window/bias/softcap, and all three gradients while preserving each architecture's established modifier semantics. gfx1200/gfx1250 fail closed; NVIDIA and Apple require architecture-owned Schedule/LSE instances and owning-host evidence. |
| 5C | **E2E-REAL-5C — stateful/training families** — **bounded Lion VJP, factored/full Adafactor VJP, and sequence-mixer backward slices implemented for x86/Zen 5 and ROCm/gfx1151, 2026-08-05.** | `tessera.state_buffer_lineage.v1` content-addresses logical buffer name, role, static shape, dtype, version, access, parent identities, and mutation policy independently of host object or device address. Typed `schedule.lion_vjp`, `schedule.adafactor_vjp`, and `schedule.sequence_mixer_backward` operations each lower to exactly one `tile.training_kernel`; the runtime consumes that exact artifact and does not retain or reconstruct Graph-op metadata. Adafactor binds factored row/column and full-state topologies. Sequence-mixer identity includes checkpoint, chunk-summary/prefix/fill, reverse phases, workspace, and fresh dQ/dK/dV/dgate/dbeta/ddecay outputs. | Schedule/Tile tamper and structural tests pass; exact Zen 5 and WSL-visible gfx1151 Lion, Adafactor, and gated/modified DeltaNet backward cohorts pass without fallback. gfx1200/gfx1250 remain fail-closed pending profiles and exact-device evidence. NVIDIA and Apple require architecture-owned consumers and owning-host validation. Broader stateful/training families remain under this row; these three bounded migrations are no longer open work. |
| 5D | **E2E-REAL-FFT — canonical spectral FFT** — **typed artifact boundary, persistent ROCm package, and the second x86/gfx1151 performance slice implemented; hardware follow-ups remain landing, 2026-08-05.** | `schedule.fft` content-addresses mode, shape/axis, direction, normalization, storage/accumulation, algorithm, radix policy/sequence through radix 17, Bluestein size, workspace policy, residency, twiddle policy, kernel family, and launch size. x86 caches Bluestein plans and owns native AVX-512 mixed-radix Stockham codelets. gfx1151 loads a prebuilt versioned shared image whose bounded persistent plan is keyed by the exact Tile digest; Bluestein owns four M buffers including an immutable transformed chirp. Rader remains candidate-only, Bailey is rejected, and gfx1151 fused LDS remains a separate candidate. | Production `tessera-opt`, native x86 images, and `libtessera_spectral_rocm.so` rebuild. ROCm persistent plans are 1.24x--1.45x faster than legacy per-call allocation at N=257/509/1009 in synchronized WSL host-wall timing. x86 cached Bluestein is 1.57x--1.76x faster and mixed radix wins 12/13 shapes. HIP events still return zero and rocprofv3 emits no WSL timestamps, so fused LDS remains experimental pending bare-metal evidence. gfx1200/gfx1250 fail closed. |
| 5E | **TSOL-ROCM-E2E-1 — compound spectral programs** — **typed Schedule→Tile carriers plus expanded x86/Zen 5 and ROCm/gfx1151 consumers implemented 2026-08-06.** | `schedule.spectral_program` content-addresses child FFT digests, bounded specialization template and exact shape, arbitrary axis, storage, normalization, layout, pad/crop, window/hop/frames, workspace, accumulation, native entry, and mutation lineage for all five compound spectral ops. Both runtimes consume one exact Tile artifact without Graph re-entry. Native package ABI v4 owns forward/ortho scaling, f16/bf16 conversion around f32 accumulation, and host-side arbitrary-axis pack/unpack. The HIP image exports one compiled architecture and stale cross-architecture images fail closed. | 36 combined Zen 5 contract/package/evidence tests and 15 exact gfx1151 package tests pass. Each architecture owns a 30-row full-family packet covering all five operations, seven digest-changing bounded specializations, every physical policy, and combined dynamic-axis-reduced-storage-ortho execution. x86 timing is selector-eligible; gfx1151 timing is synchronized WSL host wall and remains selector-ineligible. Separately stamped gfx1200/gfx1250 ABI-v4 packages cross-build, but their profiles remain `build_only`/fail-closed pending architecture-owned schedules and exact-device evidence. Bare-metal gfx1151 device events and Apple/NVIDIA physical consumers remain follow-ups. |
| 5F | **ROCM-MATH-EVIDENCE + MATH-PHYSICAL-2 — stable statistics, boundary semantics, and physical math efficiency** — **gfx1151 and Zen 5 bounded slices implemented 2026-08-06.** | ROCm var/std use centered parallel Welford; unary/binary codegen preserves difficult IEEE/NumPy domains; generated HIP math modules are process-cached by family/chip/op/dtype; and x86 arithmetic scans use an evidence-selected AVX-512 Hillis--Steele prefix while extrema retain their faster scalar recurrence. Binary physical packages reject mixed input storage. | Exact gfx1151 math passes 579 tests across fp32/fp16/bf16 storage; exact Zen 5 math passes 167 tests. The gfx1151 module cache improves seven f32 host-wall medians by 1.46x--3.58x. Paired Zen 5 `cumsum`/`cumprod` improve 1.48x/1.47x. ROCm timing remains selector-ineligible under WSL; sibling GPU backends require owning-device validation. |
| 5G | **TSOL-CONTRACT-GENERALIZE + X86-WELFORD-PARITY** — **shared contract, x86 Welford parity, and x86/gfx1151 TSOL policy expansion implemented 2026-08-06; packed fusion and DCT-I/III/IV landed 2026-08-08.** | `tessera.scheduled_spectral.v5` separates bounded template identity from exact physical specialization and carries dynamic bounds, arbitrary axes, fp32/fp16/bf16 storage, backward/forward/ortho normalization, explicit DCT-I/II/III/IV identity, and hashed fusion topology through verified Schedule→Tile lowering. Even-length compound paths bind packed N/2 children; gfx1151 v6 folds Hermitian work into fused LDS. The causal streaming-STFT policy content-addresses overlap state and fails closed for centred streaming without lookahead lineage. | Native x86 and gfx1151 correctness suites pass. Historical v5 packets remain historical; v6 promotion requires fresh clean Zen 5 and bare-metal gfx1151 evidence. Physical adoption of centred/n-FFT/full-spectrum/output-length STFT policies remains open. gfx1200/gfx1250 execution stays fail-closed; CUDA/SM120 and Apple/Metal physical consumers remain architecture-owned follow-ups. |
| 6 | **E2E-REAL-6 — delete duplicate authorities — active family-migration cohort (updated 2026-08-14).** | Ordinary pure straight-line tensor calls cache tracer-owned canonical Graph IR; native forward and reverse products consume that module rather than decoration-time AST Graph. A mandatory, cached content-addressed differential certificate compares SSA-independent topology and concrete outputs before native JVP and pure native VJP execution. It refuses effectful/stochastic double execution. Native-product plugins own family selection and package construction. Reduction uses two actual `schedule.reduce → tile.reduce_kernel → native descriptor` children. Normalization forward products use a digest-validated composite Schedule action program; normalization reverse products now declare their Graph, Schedule, Tile, and x86/gfx1151/SM120 Target consumers in the native-VJP registry. `JitFn` only binds inputs and records that plugin result for this family; its former `_native_norm_backward` package constructor is deleted. FFT/DCT/compound spectral retain their canonical carriers. The former private permissive `schedule` dialect in distribution lowering is deleted; `TesseraScheduleIR` is the sole ODS dialect authority. **Remaining debt:** `_OpExtractor` remains only as the named decoration-time candidate and CPU/oracle substrate; effectful/unmigrated execution needs non-reexecuting proof; the other backward-family dispatch branches and `_native_*` helper bodies still need extraction into plugins. | Fresh LLVM/MLIR 23 x86 and combined ROCm+x86 compiler builds pass the direct dialect-load regression, positive/negative x86 Target-IR fixtures, all 329 enabled tests in the 381-test shared lit suite (52 configuration-gated, zero failures), the 61/61 ROCm dialect suite, and the explicitly gated backend cohort. Exact WSL-visible gfx1151 tests pass for reduction, normalization, FFT/DCT, affine normalization, compound spectral, and ISTFT products. The v2 packet binds source, paired-JVP, Schedule-program, Tile-program, and parent digests. Clean selector evidence remains independent. Global Decision #31 remains open for effectful families and remaining backward-helper extraction. |

**Historical delivery note.** E2E-REAL-0 through the bounded E2E-REAL-5 family
slices landed using small, attributable PRs. Preserve that vertical-slice
discipline for the current autodiff route above; do not restart the already
completed Schedule-dialect bootstrap.

### 0.4 Relationship to W0-W6

- W1.1 supplies the typed Tile contract E2E-REAL-2/3 consume; finish its two
  NVIDIA producers and permissive-verifier removal without treating that as a
  full spine.
- W2 analyses should attach to the real mixed-level Graph/Schedule program after
  E2E-REAL-1, not to the Python shadow alone.
- W3.1 and W3.2 are delivered by E2E-REAL-0 through E2E-REAL-6. Their old
  three- and four-week estimates are retired; they omitted Schedule dialect
  construction, SSA preservation, bufferization, backend API migration, and
  exact-device gates.
- W4 control flow starts only after the matmul slice proves adjacency. Its
  acceptance test remains valuable, but it must traverse the same lineage gate.
- W5 schedule and residual decisions are admitted only when the chosen values
  appear in the physical artifact and descriptor provenance. Stamping an
  attribute into an unused Python artifact does not close W5.

---

## 1. The thesis

Across seven independent reviews, roughly forty findings reduce to **two root
causes and one consequence.**

### Root cause A — Declared but not consumed

The compiler computes, validates, and attaches the information a pass needs, and
then no pass reads it.

| Declaration | Consumer that ignores it | Review |
|---|---|---|
| `manifold` attribute on `ebm.langevin_step` | every backend codegen (6 grep hits, all comments) | GA/EBM §1.1 |
| `MultivectorSpec.grades`, `IsRotor`, `Even`/`Odd` | `geometric_product` — iterates all `dim²` pairs | GA/EBM §2.1 |
| `batching_rule` axis, closed across 487 primitives | `vmap` — a Python `for` loop | Autodiff §B3 |
| `shape_rule` axis, reported closed | `_infer_result_type` — a five-case if-chain | Frontend §G2 |
| `!tile.fragment`, `!tile.buffer`, `!tile.tmem`, … (9 types) | partially consumed; core `tile.mma`/`tile.async_copy` and compatibility envelopes remain open | IR Stack §T1 |
| `numeric_policy` (Decision #15a) | no carrier below Graph IR at all | IR Stack §T6 |
| `TilingInterface` on Matmul/Conv/FlashAttn | `fusion_core.py` — 7 hand-enumerated regions | Sweep §F3 |

### Root cause B — Told, not derived

Passes are *given* facts syntactically instead of *computing* them, so they are
wrong at the edges and fail open.

| Missing analysis | What happens instead | Review |
|---|---|---|
| Effect/purity on the IR | `ast.NodeVisitor` name-matching; aliased RNG ⇒ inferred pure ⇒ `deterministic=True` passes | Sweep §F1 |
| Differentiation activity | `AutodiffPass` builds adjoints for everything | Autodiff §A5 |
| Gradient demand / trajectory liveness | `CheckpointInnerLoop` marks every EBM step in a containing loop, but no downstream pass consumes those marks | GA/EBM §1.5 |
| Symbolic shape constraints | `dims_compatible` is `str(lhs) == str(rhs)` | Sweep §F2 |
| Fusion legality | region shapes enumerated by hand | Sweep §F3 |
| Sharding propagation | every layer annotated by hand; `validate()` contains `pass` | Sweep §F4 |
| Tile-level invariants | six `*LegalityPass` re-deriving what types would give free | IR Stack §T2 |

### The consequence — C: duplication

**A + B ⇒ C.** When the declared contract doesn't carry the information, and no
analysis derives it, the only way to ship is to write a second implementation
that does. Every duplication in the tree traces to this:

| Duplication | Why it exists |
|---|---|
| Two frontends (AST `_OpExtractor` / tracer), opposite failure policies | the AST path can't produce SSA through control flow |
| Two Graph→Schedule, two Schedule→Tile lowerings; Python canonical — **but see the correction below** | the MLIR types don't carry what codegen needs |
| Python AD tape + `AutodiffPass` | the tape can't be a transform (global monkey-patching) |
| GA Python fast paths + `RotorSandwichFold` marker | `ExpandProductTable` rejects batched operands |
| Two `Queue.td`, two `Attn.td` defining the same dialect | accretion, uncaught |
| Two remat passes with opposite rigor | no shared liveness analysis |

**This is why the plan is ordered A → B → C.** Collapsing a duplication before
the surviving path can carry the information just deletes a working system. Every
attempt to start at C fails.

> **Correction (2026-08-02, from W0.6 execution) — the Graph→Schedule and
> Schedule→Tile row above overstates the C++ side, and this makes W3.2 *larger*,
> not smaller.** There are not two competing lowerings at those boundaries. On
> the C++ side there is one **annotation-only skeleton**: `GraphToSchedulePass`
> stamps `schedule.artifact_hash = "__pending__"` on three op-name prefixes and
> `ScheduleToTilePass` stamps `tile.staged` on `schedule.async_copy`. Neither
> matches, replaces, or rewrites any op — the original source comment says so
> outright ("a real pass would pattern-match and replace ops"). Worse, the
> library holding them (`TesseraPM`) is linked **only into the test binary**;
> `tessera-opt --help` in the production driver does not list
> `-tessera-graph-to-schedule` at all.
>
> So W3.2 is not "converge two implementations onto MLIR" — the MLIR
> implementation does not exist yet, and the Python spine is not a duplicate of
> it but the only implementation. Re-scope W3.2 accordingly before funding it.
> The passes now carry `[annotation-only skeleton]` in their registered
> descriptions and a maturity contract in `PMPasses.h`, so nothing can cite them
> as evidence of a working boundary again.

---

## 2. Governance — the rules that stop regrowth

A cleanup without rules regrows. Six standing decisions, each derived from a
specific finding, each drift-gateable:

| # | Rule | Prevents | From |
|---|---|---|---|
| **#21a** | **Semantic keys never default.** An attribute selecting *semantics* fails closed on absence; one selecting *performance* may fall back with a diagnostic. Semantic: `manifold`, `algebra`, `math_mode`, `rounding_mode`, `distribution`, `dtype`. Performance: tile sizes, stage depth, `auto_batch`, checkpoint budget. | silent Euclidean fallback; `operand_types[0]`; unvalidated `StrAttr` | OT §H1 |
| **#10a** | **An eligibility-marking pass ships a negative fixture.** Any pass annotating work as rematerializable / fusable / pipelineable gates on demand analysis and ships ≥1 fixture where the correct output is *no annotation*. | 2500 dead steps marked rematerializable | OT §H2 |
| **#29** | **A declaration must have a consumer.** If the compiler declares metadata — an ODS type, a coverage axis, an attribute — a named pass must consume it, or the declaration is deleted. Drift-gated: a test asserting every `primitive_coverage` axis names its consumer. | root cause A, all seven instances | this doc |
| **#30** | **Derive, don't ask.** A pass needing a program fact queries the analysis layer. New bespoke walkers are rejected in review. | root cause B; the eighth hand-rolled analysis | Sweep §5 |
| **#31** | **One implementation per boundary.** Each level boundary has exactly one production lowering. A second implementation is either a declared oracle with a differential test, or deleted. | root cause C | IR Stack §T3 |
| **#32** | **Information loss across a level boundary must be declared.** A lowering carries each Decision #15a attribute forward or records a named reason it dropped it; a boundary verifier fails on silent loss. | `numeric_policy` vanishing above the MMA | IR Stack §U5 |

Adopt these **before** Wave 1, not after. #29 and #31 in particular change what
gets accepted in review, and they are what make Waves 1–3 stick.

---

## 3. De-duplication ledger

The seven reviews costed overlapping work independently. Corrections applied here:

| Double-counted work | Costed as | Merged into | Saved |
|---|---|---|---|
| Symbolic shapes (Sweep #10, 3w) + control-flow adjoints (AD D4, 6w) + frontend regions (E7) | 3 separate items | **W4** — one program, one gate | ~4w and 2 items that would each have "landed" with zero capability |
| Differential harness: trace-vs-AST (E2) + Python-spine-vs-MLIR (I5) | 2 harnesses | **W3.1** — one harness, ~~two uses~~ **one use today** (see below) | ~2w *(saving no longer holds as stated)* |
| Schedule-decision work (Frontend U3/E6) + (IR Stack U6/I6) | 2 items | **W5.2** | ~5w |
| Effects re-homing (Sweep #9) vs derive-from-traced-IR (Frontend E3) | 2 approaches | **W2.2** — E3 strictly supersedes | ~2w |
| Implicit diff: OT R2 `custom_root` + AD "finish NewtonAutodiff" | 2 items | **W3.5** — same pass | ~2w |
| Legality collapse (IR Stack I2) as independent work | standalone | **W2.4** — client of the dataflow layer | ~1w |
| Remat unification: ~~delete `EBMCheckpointInnerLoop`~~ **(done in W0.2)** + AD D5 | 2 items | **W5.1** — now AD D5 only | ~1w *(already banked)* |

**Two rows corrected after W0 execution (2026-08-02).** The ledger is accounting,
not a work queue — every row's work already lives in a wave item — but two rows
no longer describe reality:

- **Remat unification.** Deleting `EBMCheckpointInnerLoop` is **done**: W0.2
  removed it from the default EBM pipeline (its three attributes had zero
  consumers tree-wide), kept it as an explicitly experimental standalone pass,
  and shipped the Decision #10a `CHECK-NOT` fixture proving the default pipeline
  emits no checkpoint annotations. W5.1 therefore owns **only** AD D5 — the
  demand-aware residual policy as an arbiter axis. Do not re-scope W5.1 as if
  the deletion were still ahead of it.

- **Differential harness — this merge's saving does not hold.** It assumed two
  implementations to compare at the Graph→Schedule / Schedule→Tile boundaries.
  W0.6 established there are not: the C++ side is an **annotation-only skeleton
  in a test-only library** (`GraphToSchedulePass` stamps
  `schedule.artifact_hash = "__pending__"` and returns; `TesseraPM` is linked
  only into the test binary, so production `tessera-opt` never exposed
  `-tessera-graph-to-schedule`). The trace-vs-AST use is real and unaffected;
  the Python-spine-vs-MLIR use has no MLIR side to differ against until W3.2
  builds one. **Net effect: W3.1 keeps one use, and W3.2 grows** — see the
  correction under §1.

**~17 weeks of double-counting identified**, minus the corrections above.
Queue estimates are directional:
the source documents used different scopes, and the Target-IR corrections below
replace a blanket ROCm migration with an ownership-and-evidence gate.

---

## 4. The waves

Each wave has **one observable exit criterion**. A wave is not done when its
items are merged; it is done when the criterion holds.

### W0 — Stop the bleeding *(4 weeks · no dependencies · start immediately)*

Live defects, fail-open paths, inert machinery, and false documentation. Every item is
independent; run them in parallel.

> **W1.1 step 3 status (landing):** `GenerateWMMAGemmKernel{via-tile=true}`
> emits the complete
> view/pack/mma/unpack/store chain at production `mt=2, nt=4`; the gfx1151
> aligned and ragged differential gates are zero-difference. Flash and linear
> attention now attach typed fragments at their register-owned computed-value
> boundary and lower identically to their direct ROCDL lanes. Dynamic N/K are SSA leading dimensions
> (`#tile.memory_layout<leading_dim = 0>`), not a 64x64 specialization. The
> default direct lane remains unchanged. Two NVIDIA-owned tensor producers in
> `TileIRLoweringPass` remain; Python has no direct `tile.mma` producer.

| # | Item | Source | Effort |
|---|---|---|---|
| W0.1 | **Landed 2026-08-02.** `manifold` is now `EBM_ManifoldAttr` (a `StringBasedAttr` pinning `euclidean`/`sphere`/`bivector`), and `Canonicalize`'s Euclidean fallback is replaced with `emitError`+interrupt+`signalPassFailure`. Verified: unknown value and missing value are both rejected before any pass runs; negative fixture `canonicalize_rejects_bad_manifold.mlir`. The typed-`EnumAttr` upgrade stays with W1.1b. Two side findings, both fixed: the `.td` comment claiming ODS "doesn't support" a constrained string alias was false (`StringBasedAttr` is already used by the ROCm dialect in this tree), and `ts-ebm-opt` never registered `arith`, so **6 of its 12 lit fixtures could not parse** — invisible because `TESSERA_BUILD_EBM_BACKEND` is OFF by default. EBM lit is now 12/12. | GA/EBM §1.1 | 3d |
| W0.2 | **Landed 2026-08-02.** `checkpoint-inner-loop` is out of `tessera-ebm-pipeline`; verified that `tessera.ebm.{checkpoint_loop,checkpoint_budget,recompute_step}` have **zero consumers tree-wide**, which is why an unconsumed declaration must not ship in a default path (#29) and why an eligibility pass must gate on demand analysis (#10a). The pass remains registered and explicitly labelled experimental so its own fixtures keep running. The Decision #10a negative fixture lives in `full_pipeline_chain.mlir` (`CHECK-NOT` on all three attributes). Demand-aware loop rematerialization stays W5.1 — and per the de-duplication ledger correction, W5.1 now owns *only* that, not the deletion. | GA/EBM §1.5 | 1d |
| W0.3 | **Landed 2026-08-02.** A traceable EBM energy is now defined as one whose scalar result flows through `tessera.ops.*` on the state it was handed; `_tape_grad`/`_tape_grad_mv` return a reverse-mode gradient when a cotangent path is actually recorded and `None` otherwise, so raw-NumPy callbacks keep the central-difference path. Both `bivector_langevin_step` and `sphere_langevin_step` try the tape first. Measured on Cl(3,0) (D = 2³ = 8): **1 energy evaluation instead of 16**, and exact instead of first-order. **Root cause was tape identity, not the samplers** — `Multivector.coefficients` returns a fresh read-only *whole-array view* on every access, so the tape's `id()`-keyed identity could never match the state buffer and manifold energies were untraceable in principle. **Recorded negative result:** fixing this inside `Tape._describe` (resolving a whole view to its base) is the obvious move and is **wrong** — `Tape.record` keys an op's *output* on `id(output)`, so rewriting only the *input* side severs producer→consumer links and silently drops gradients. Measured: 12 Clifford/MoE autodiff failures, all numerically silent. Identity is therefore recovered *after* `backward` in `_cotangent_for_buffer`, local to the EBM helper, leaving global tape semantics untouched; a comment in `_describe` records why the tempting fix is rejected. Six regression tests cover both paths plus their agreement on a full Langevin step. | GA/EBM §2.6 | 1w |
| W0.4 | **Landed 2026-08-02.** `jacrev` records the forward pass once and re-runs `backward` with `retain_graph=True` per output element — measured 1 evaluation instead of 4 on a 4-element output, and the `retain_graph` machinery whose own docstring named `jacrev` as its motivating caller is finally wired to it. **`jacfwd` was evaluated and needs no fix:** one `jvp` per *input* dim is the definition of forward mode, not a forward-pass-per-element defect, and its docstring already says exactly that. Reviewed again in PR #490 — the single-tape rewrite removed an accidental shield (the old code wrapped `fn` in `sum(out*cotangent)` through `ops.*`, making the target tape-produced regardless), so `jacrev` of an identity or constant now resolves structurally. | Autodiff §B1–B2 | 3d |
| W0.5 | **Completed 2026-08-02:** Decision #5 in `CLAUDE.md` now states that the effect lattice walks the AST, not the IR | Sweep §F1 | done |
| W0.6 | **Landed 2026-08-02.** Deleted the dead `dialects/tessera_{queue,attn}/*.td` (no CMake referenced them, yet **six docs cited them as the authoritative source** — all repointed). The new Decision #31 drift gate then found a **third duplicate the reviews missed**: `src/compiler/programming_model/ir/tile/TileMemoryOps.td` declared the same `tile` dialect name as the production `Tessera/Dialect/Tile/TileOps.td`, with *contradictory* mnemonics (`mma.tcgen05` vs the live `tcgen05.mma`); it was tablegen'd but never `#include`d by any source and never registered. Deleted, and `CLAUDE.md`'s GPU-only tier corrected to the real mnemonic. PM passes moved to `lib/PMPasses.cpp` + `include/tessera/ProgrammingModel/PMPasses.h`. | IR Stack §T5, §T3 | 3d |
| W0.7 | **Landed 2026-08-02.** All three GA8 pass summaries now read `[annotation-only]` with an explicit "rewrites no IR" rather than the ambiguous `[GA8 stub]`, which conflated "does nothing" with "does something partial". The false claim that GA8 passes "gate on `canonical` and refuse to proceed on out-of-allow-list signatures" is removed from **both** `CliffordPasses.td` and `AnnotateAlgebra.cpp` — verified the GA8 passes reference `canonical` nowhere, making it a live #29 declaration-without-consumer, now recorded as such. (The one remaining "GA8 stubs" mention is a descriptive file-header line about where the passes live, not a capability claim.) | GA/EBM §1.4 | 1d |
| W0.8 | **Landed 2026-08-02.** All six decisions are in `CLAUDE.md`'s do-not-revisit list with their originating defect. #29 and #31 are drift-gated by `tests/unit/test_governance_declarations.py`: every `primitive_coverage` axis must name an existing consumer file, and no two ODS files may declare the same dialect name. The two genuinely-unconsumed axes (`batching_rule`, `shape_rule`) are explicit ratchet waivers naming their owning wave item, so they read as open rather than closed. The #31 half found a duplicate dialect on its first run (see W0.6). | §2 | 1d |
| W0.9 | **Landed 2026-08-02, and it found more than expected.** Substring assertions retained as smoke; a real parse + dialect-load + verifier harness now runs each emitter's text through `tessera-opt`. **Result: every Python-emitted "Target IR" fails a real MLIR parse.** Two stacked defects, both invisible to `in`-assertions: (1) module attributes are not dialect-prefixed (`arch`, `target`, `target_features`), which `builtin.module` rejects outright; (2) underneath that, the ops violate their own ODS — `tessera_rocm.mfma` is emitted as `() -> ()` carrying its result as a **string attribute** (`result = "v0"`) while the dialect requires one SSA result. So the Python lane emits text that *resembles* the dialect without being it, and Decision #19's contract was validated by a test that could never have caught this. NVIDIA targets **skip** rather than fail — `tessera_nvidia` is not compiled into the default build, and failing them would measure the build config rather than the emitter.

**The ratchet is now EMPTY — all of it was fixed, not just recorded.** Four distinct defects, each invisible to `in`-assertions: (1) module attributes are dialect-prefixed at MLIR-render time (`_mlir_module_attrs`), keeping the short Python-facing keys callers index; (2) the function container is `func.func`, replacing a hardcoded map of `tessera_apple.cpu.func` / `tessera_rocm.func` / `tessera_nvidia.func` / `tessera_x86.func` — **none of which any dialect defined**; (3) `mfma` / `async_copy` / `wait` emit their real ODS signatures with the async-copy token threaded into the wait; (4) five emitted-but-undeclared ops (`tessera_rocm.{elementwise,kv_cache_read,msa_block_sparse}`, `tessera_apple.cpu.{kv_cache_read,moe_solver}`) were added to their dialects. A second, duplicate emitter family in `matmul_pipeline.py` had the same defects and was fixed with it. The gate now also parses **every committed golden**, which is what caught defect (4): the single-matmul test passed while the multi-op `matmul_softmax` goldens did not.

**The `cpu` reference lane is now closed too (2026-08-02) — no exclusions remain.** It emitted `tessera.cpu.<source-op>`, one op name per Graph IR op, so its vocabulary grew with the op set and could never be enumerated in ODS. That name was pure redundancy: the CPU verifier already *requires* a `source` attribute naming the originating op. It now emits the single declared `tessera.cpu.reference` node (plus `cpu.profiler_probe` and `cpu.msa_block_sparse`, kept separate because they carry distinct contracts), and parses and verifies like every other lane. Every target the build compiles a dialect for — `cpu`, `x86`, `rocm`, `apple_cpu`, `apple_gpu` — now passes a real parse + dialect-load + verifier run; only NVIDIA skips, and only because its dialect is off in the default build. | Target §X4 | 1w |
| W0.10 | **Decided 2026-08-02: build `tessera_x86`.** No carve-out — Decision #19 stays universal. Evidence that settled it: `TileToX86Pass` lowers Tile IR to **21 `func::CallOp`s** into a hand-written C shim plus arith/memref glue, using neither a `tessera_x86` dialect nor MLIR's upstream `amx`/`x86vector` dialects — structurally the same `func.call`-to-a-C-symbol shape `CLAUDE.md` already flags for Apple GPU. The build is cheaper than the other backends' equivalents because the abstract ops largely exist upstream: the hardware-free layer (`tessera_x86.amx_tile_load`, `.amx_dpbf16ps`, `.avx512_gemm_microkernel`, pack/unpack) can lower into `amx.*`/`x86vector.*` rather than terminating in `func.call`. **Built 2026-08-02.** `tessera_x86` is defined, tablegen'd, linked into `tessera-opt`, and registered (`--show-dialects` lists it). It separates **value-carrying** ops — `amx_tile_load` / `amx_tile_zero` / `amx_dpbf16ps` / `amx_dpbusd` / `amx_tile_store` over a real `!tessera_x86.tile` type — from **directives** (`avx512_gemm_microkernel`, `pack_b_panel`, `elementwise`, plus the emitter's `kernel` / `kv_cache_read` / `unsupported`). `abi_call` models the C-shim boundary rather than hiding it, so Decision #28's arbiter can distinguish compiler-generated from delegated work. Positive **and negative** lit fixtures ship (`x86_target_ir{,_invalid}.mlir`); the negative one proves the typed layer rejects an AMX dot-product whose operands never came from a tile — exactly the property a substring test cannot check. The Python x86 emitter's output now parses, loads the dialect, and verifies. **Remaining, and re-scoped 2026-08-02:** lowering into upstream `x86vector.*` (AVX-512) instead of terminating in `func.call` is the live follow-on — it changes generated code and needs AVX-512 execute-and-compare on this box. **The AMX half is deprioritized to optional:** per the project owner, AMX is expected to be superseded by the ACE matrix instructions jointly agreed by Intel and AMD for future CPUs, so an AMX → `amx.*` lowering is not worth building now. (Recorded as owner direction; ACE specifics are not independently verified in this plan.) The AMX ops stay in the ODS as the IR-level contract — they cost nothing, they pin the tile/accumulator shape, and they give the eventual ACE ops a structure to follow. This also removes the fleet's only hardware blocker here: AVX-512 execution is available on this box, whereas no machine in the fleet has AMX. | Target §X1 | 1h to decide (done); dialect + fixtures done |

**Exit:** the open-string manifold key is verified, the EBM default pipeline
emits no unconsumed checkpoint policy, `CLAUDE.md` Decision #5 is accurate, and
every dialect has exactly one ODS.

### W1 — Make declarations binding *(8 weeks · depends on W0.8)*

Root cause A. Most items enforce existing declarations; fragment/buffer
parameterization and target matrix contracts require bounded, variant-aware type
design before migration.

> **W1 status — 2026-08-04. NOT closed.** Verified against `main`, not recalled.
>
> | item | state |
> |---|---|
> | W1.2 shape-rule registry | ✅ complete |
> | W1.3 metadata boundary verifier | ✅ complete |
> | W1.4 GA grade threading | ✅ complete |
> | W1.1b semantic `$kind` | 🟡 partial — the 3 fail-OPEN ops closed; 14 already fail closed, not yet hoisted to ODS |
> | W1.1 Tile IR typing | 🔴 **open** — **2 of 6 numbered steps** landed (1, 2) |
>
> **Landed but NOT numbered steps** — real work, and deliberately not counted
> as step completions: the 2b **guard** (NVWGMMA fails closed on an accumulator
> it would drop, #506) and **3a** (ragged masking in `materializeFragmentPack`
> plus the shared bounded-`tile.view` arity contract, #510). 3a is a
> prerequisite invented for option (a), not an entry in the design doc's step
> table. Counting either as a numbered step overstates progress — an earlier
> version of this block said "4 of 6" by doing exactly that.
>
> **Stack context — read before sizing any of this.**
> [`ROCM_LANE_MAP.md`](../backend/rocm/ROCM_LANE_MAP.md) measures the lane
> W1.1 is supposed to improve. The executing ROCm GEMM lane begins at a
> **Target-IR directive** built as a string in Python: no Graph IR, no
> Schedule IR, no Tile IR. `lower-tile-to-rocm` runs but is a verified
> no-op there. A Graph-IR lane exists and is compiled, but its only caller
> in the tree is a benchmark. Consequences: W1.1 changes no executing
> kernel until step 3. Two costs, kept apart: closing the Tile fragment
> contract (steps 3-5) spans **5 C++ `tile.mma` creation sites + the Python
> emitters** — step 5 is NOT behind the expander population; making the ROCm
> backend traverse Tile IR is a separate, unpriced **58**-expander question.
> The 5w estimate covers building the contract, not adopting it.
>
> **What is really open, in dependency order:**
>
> 0. ✅ **LANDED 2026-08-04 — the typed lowering COMPOSES (§4.6.1).**
>    `convertTypedFragments()` in `TileToROCM.cpp`: `TileFragmentTypeConverter`
>    + four conversion patterns + `applyPartialConversion`, running ahead of the
>    legacy walk, which still owns the bare `!tile.fragment` spelling. The
>    K-loop, an mma feeding an mma, and a non-`fragment_zero` accumulator all
>    lower — `rocm_typed_fragment_composition.mlir`, verified to fail when the
>    synthesized-zero defect is re-injected. `scf.for` was one library call, as
>    predicted. **Not yet on an executing lane: no producer emits typed
>    fragments until step 3, so this is proven by fixture only.** Two defects it
>    exposed, both with green positive tests, are in §4.6.1 and the paired
>    `_invalid` fixture. Original scoping below.
>
>    <details><summary>Original scoping (2026-08-04)</summary>
>
>    `TileToROCM`'s typed path is
>    a single-shot whole-chain pattern match (`view → pack → zero → mma → unpack
>    → store`, then erase), so an accumulator that is not a `fragment_zero`, an
>    mma feeding another mma, and a chain crossing a loop boundary are all
>    inexpressible *by construction*. Replace it with a `TypeConverter`
>    (`!tile.fragment` → `vector<N × T>`) + conversion patterns. `scf.for`
>    iter_args come free from
>    `populateSCFStructuralTypeConversionsAndLegality`, which this LLVM 23 ships
>    — the hand-rolled region conversion §4.2 sized as the largest step is a
>    library call. Cost is that **no pass in this tree uses a `TypeConverter`
>    yet**; this is the first.
>
>    </details>
>
> 1. **step 3 — restructure producers onto `tile.view` + `fragment_pack`.**
>    `fragment_pack` requires a `!tile.tile`; zero producers supply one
>    (`TileIRLoweringPass` passes tensors, the three `GenerateWMMA*Kernel` passes
>    pass lane-level vectors). Option (a) chosen 2026-08-04; 3a landed. **Made
>    materially smaller by (0)** — the producer then emits well-typed ops rather
>    than a pattern one matcher must recognise whole.
> 2. ✅ **step 2b — CLOSED by (0) as a capability.** A non-zero accumulator is a
>    converted operand, and "synthesise a zero" is now the lowering of
>    `fragment_zero`. Closed on the **ROCm** side only, and closed as a
>    capability rather than as shipped codegen — nothing emits it until step 3.
>    The NVIDIA fail-closed guard (#506) stays: `NVWGMMALoweringPass` has had no
>    equivalent conversion built, and step 6's NVIDIA half remains unverifiable
>    on this box (needs `-DTESSERA_ENABLE_CUDA=ON`).
> 3. **step 4** — the five Python text emitters.
> 4. **step 5** — delete `MMAOp::verify`'s permissive branch. Unreachable until
>    (1) and (3) complete; deleting it earlier breaks every producer.
> 5. **step 6 — Target IR dialects.** Remove unexplained `AnyType` from
>    `tessera_nvidia` (3/3) and `tessera_apple` (12/12), with `tessera_x86`
>    (0/0) as the reference shape. Independent of the producer chain, and it
>    was omitted from an earlier version of this list — which is how still-
>    required Target IR work disappears from the owning queue.
> 6. **W1.1b** — close/hoist the semantic `$kind` sets.
>    **Re-measured 2026-08-05: 11 `StrAttr:$kind` sites across FIVE
>    dialects** — Graph IR `TesseraOps.td` (6), Apple (2), NVIDIA (1),
>    ROCm (1), Neighbors (1) — not one backend's ODS. That makes it a
>    shared-contract change under AGENTS.md (same PR assesses every backend).
>
>    **The "already fail closed" label does not hold, and was inherited rather
>    than measured** (found by #521 review). Verified both directions:
>    `ROCM_Int4PackKernelOp` DOES fail closed — its generator validates against
>    an explicit set and errors naming
>    `kind=pack|unpack|relu|sparse_gather|cache_append`. But
>    **`tessera.neighbors.topology.create` fails OPEN**, and it is a
>    **Decision #21a violation**, not a layering nit: `CreateTopologyOp::verify()`
>    checks only that `kind` EXISTS, and `DynamicTopologyPass::isMutableKind`
>    dispatches by **substring** —
>    `contains("dynamic")||contains("adaptive")||contains("fault")||contains("custom_graph")`.
>    So a typo (`2d_mseh`) silently becomes a STATIC topology, and the substring
>    test is wrong in the other direction too — `not_dynamic` classifies as
>    MUTABLE. `kind` there drives `topology.dynamic`/`topology.replan`/
>    `topology.replan_hook`, so this is the same unnamed-semantic-default class
>    #505 closed for `predicate`/`optimizer`/`clifford`.
>
>    So the item is **per-site triage first** (fail-open ⇒ correctness fix with a
>    negative fixture; fail-closed ⇒ ODS hoist), and its remaining size is not
>    yet known. Each site's legal set must be derived from its actual consumer
>    dispatch: deriving one from a partial read is how
>    #499 shipped an optimizer enum missing `adafactor` and broke six
>    tests including one that executes on gfx1151, so budget per-site
>    derivation plus a run of the existing tests, not a bulk edit.
>    Independent of everything above; no longer purely a layering
>    improvement, since at least one site is a live fail-open defect.
>
> Items 1–4 are one chain. Items 5 and 6 can proceed in parallel with it.



| # | Item | Source | Effort |
|---|---|---|---|
| W1.1 | **ROCm steps 1–4 and typed performance closure landed; shared steps 5–6 remain open.** Step 2b is closed on ROCm by step 0. Design + inventory: [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md), [`W1_1_TYPING_INVENTORY.md`](W1_1_TYPING_INVENTORY.md). **Landed:** (1) `!tile.fragment` is parameterized on `m/n/k, elem, acc, role, layout, family`. (2) verification and ROCm dialect conversion consume the contract and thread loop/chained/non-zero accumulators. (3) `GenerateWMMAGemmKernel{via-tile=true}` emits the complete typed 2x4 view/pack/mma/unpack/store chain and remains bit-identical for aligned/ragged gfx1151 cases. Per-fragment address, scheduler, IGroupLP, K-pair, whole-pack, and occupancy experiments established that approximate reconstruction could not close the measured 0.711x gap. The retained solution gives the direct generator and typed consumer one target-owned complete physical emitter. The typed producer stamps a SHA-256 over its ordered semantic Tile body; the gfx1151 consumer verifies that digest and exact 24/24/32/16/16 topology before materializing the shared physical function, without Graph re-entry. Direct and typed Target IR and HSACO are byte-identical (same 256 VGPR/107 SGPR/spills). Exact gfx1151 packets measure 1.001x at 1024³, 0.993x at 2048³, and 0.998x at 4096³, closing the performance gap; the canonical scheduled package now selects the typed route. (4) Python emitters no longer construct physical `tile.mma`. **Open:** NVIDIA's two tensor producers, deletion of the permissive branch, and NVIDIA Target-IR typing on an SM120 host. gfx1200/gfx1250 remain fail-closed pending their own profiles and device evidence. | IR Stack §U1 + Target §X2 | 5w |
| W1.1b | **Partially landed; the row's premise did not survive measurement.** It said "62 × `$name`, 4 × `$kind`, 1 × `$mode`". Measured: **17** ops carry `$kind`, **3 of them are `I64Attr`** rather than strings, and **14 of 17 already fail closed** in their generators. `$name` is the emitted kernel SYMBOL (`flash`, `fc1`, `bwd`, …), an open set chosen by the caller — enumerating it would reject valid programs, so it is deliberately left a free string and gated as such. **Landed:** `$dtype` split into three per-op-family constraints (#499, after review showed one shared union let `softmax` accept `int8`); `reduction` / `mode` closed sets (#499); and the **three `$kind` ops that failed OPEN** — `predicate`, `optimizer`, `clifford` — closed (#505). Those three each had a trailing `else` doubling as an unnamed semantic default, so a typo silently computed `isfinite`, trained with Adam, or evaluated the **geometric product** instead of the requested Clifford operation. **Open:** hoisting the other 14 already-fail-closed `$kind` sets from their generators into ODS — a layering improvement (reject at verification, not in the generator), not a correctness fix. | Target §X3 | 1w |
| W1.2 | **Landed 2026-08-03.** Both halves now hold. (a) Unknown op ⇒ diagnostic: `_infer_result_type` raises when a catalog-declared rule has no implementation, instead of the old five-case if-chain ending in `return operand_types[0]` — correct for the ~60 elementwise ops and silently wrong for everything else. (b) **Auto-flip wired** — `primitive_coverage.shape_rule` is derived from `op_catalog` via `_catalog_shape_rule_status`, the mechanism `op_catalog`'s own source predicted and nobody had connected. It found a live defect: the dashboard promoted `shape_rule` off the LOWERING KIND and never consulted the catalog, so **all six ops whose rule the catalog had explicitly withdrawn reported `complete`** — the same bug `shape_rule_for` had already fixed one layer down. 456 complete → 450 + 6 partial, with no other entry moving, which is the proof the derivation agrees with the rest. 16 now-inert override lines deleted (Decision #29); the surviving 39 are gated by `test_shape_rule_autoflip.py` so a contradicting override fails the build rather than quietly winning. Ops the catalog does not own (~169 Python-reference/host-API) are deliberately untouched. | Frontend §U2 | 2w |
| W1.3 | **Landed 2026-08-03.** `--tessera-record-metadata` + `--tessera-verify-metadata-obligation`: the snapshot rides in the IR as a module attribute, so record → lower → verify is ONE `tessera-opt` invocation and is lit-testable (a `PassInstrumentation`, the more obvious idiom, is registered in the driver and could not be fixtured — an unfixturable verifier is what Decision #29 rejects). Comparison is per function and normalized to the attribute's last dot-component, so `tessera.layout` → `tile.layout` is not a drop; `shape`/`dtype` are untracked because they live in types. **Found a live bug on its first real program:** `TileIRLoweringPass` has two `tile.mma` producers and only the fused K-step forwarded `numeric_policy`, so the main matmul path stated the accumulator contract at Graph IR and lost it one level down — fixed. Five fail-closed refusals incl. STALE_DECLARATION (a declared drop that did not happen) and NO_SNAPSHOT (an unrun check must not look like a passed one); `not_yet_carried:<item>` keeps declared debt attributable. | IR Stack §U5 | 2w |
| W1.4 | **Landed (PB stack, 2026-08-03) — row was stale.** `python/tessera/ga/ops.py` derives grades before backend dispatch via `_blade_mask_for_grades` + `_product_grade_contract`, closing the Decision #29 violation the plan itself cites (`MultivectorSpec.grades` reaching no consumer). `GradeFusion.cpp` gained `InputGradeFusionPattern`, attaching `tessera.clifford.input_grades_lhs/rhs` — the MLIR half: `output_grades` prunes the Cayley table by which RESULTS are wanted, `input_grades` by which INPUTS can be non-zero. Fixtures `input_grade_fusion{,_prunes}.mlir`, including the Decision #10a negative case where the correct output is NO annotation. | GA/EBM §2.1 | 1w |

**Exit:** no true Tile primitive has an unexplained `AnyType`; a mismatched
`tile.mma` fails verification; every compatibility exception names its owning
level-migration item; no op reaches the `operand_types[0]` fallback; no Decision
#15a attribute drops across a boundary without a recorded reason.

> **W1.1 is a high-leverage contract project, not a mechanical ODS edit.** It is
> the precondition for W2.4, W3.2, W3.3, and every backend having a real contract
> to lower against, but it must land incrementally with producer/consumer and
> per-architecture variant coverage.

### W2 — Build the analysis layer *(8 weeks · depends on W0)*

Root cause B. One framework, then each analysis is a transfer function rather
than a subsystem.

| # | Item | Source | Effort |
|---|---|---|---|
| W2.1 | **Closed 2026-08-11 — shared Graph IR dataflow framework.** `GraphDataflowAnalysis` runs shape and alias product lattices plus liveness on MLIR `DataFlowSolver`, derives value-scoped memory dependence from registered effects/resources, and exposes reverse activity. Unknown producers, unranked shape, unknown effects, aliased memref arguments, nested regions, stale snapshots, and absent analysis state resolve conservatively to ⊤/unsafe. The analysis owns explicit `invalidate`/`recompute`; await sinking recomputes after every mutation. Reverse AD and await sinking are production C++ clients. `--tessera-graph-dataflow` exposes schema-v1 facts for inspection, while `python/tessera/compiler/graph_dataflow.py` provides a structural-digest-invalidated Python query mirror. Direct tests cover shape, alias lineage, dead values, activity/`stop_gradient`, unknown producers, stale mutation, ordered collectives, and the x86 no-async path. Wiring automatically inferred dependence edges into new Tile action DAGs remains a scheduling-client task, not missing analysis substrate. | Sweep §3 | closed |
| W2.2 | **Closed 2026-08-10 — effects from registered traced IR.** The canonical op catalog now emits explicit `tessera.effect_kind` for pure and effectful Graph operations plus alias and stochastic-identity contracts. Python `EffectLattice` consumes Graph records and `_EffectVisitor` is deleted. The C++ semantic analysis consumes those contracts, falls back to MLIR memory/cast/view interfaces, treats unknown behavior as `top`, rejects an attribute that erases a registered side effect, and reaches a fixed point across internal calls. Await sinking uses the same query. Direct tests cover indirect RNG, aliases, mutation, ordered collectives, regions, unknown operations, and x86/no-async. | Frontend §U7 | closed |
| W2.3 | **Region-aware client landed.** Reverse AD consumes W2.1 activity and `stop_gradient` semantics instead of its private walker. Active structured operations propagate to explicit operands and implicit captures; inactive regions remain pruned. `RegionAdjointInterface` owns internal block activity for admitted W4 forms. Whole-program memory activity remains a separate optimization. | Autodiff D3 | closed |
| W2.4 | Collapse six `*LegalityPass` → ODS constraints (post-W1.1) + one `TileDataflowLegalityPass` | IR Stack §U2 | 1w |
| W2.4a | **CAKE typed Tile sync/memory + SO-2 role surface.** Phase 1 §5.1–§5.4 is closed: typed waits/tokens, loop-carry provenance, registered sync vocabulary, hatch deletion, and production NVIDIA legality wiring. On 2026-08-16 the Phase 2 role carrier added loop-carry-safe `!tile.role`, role-bearing pipeline/mbarrier ownership, a ROCm role-producing and role-consuming wave/LDS path, and explicit x86 `no_async_noop`. The plan-named gfx1151 §5.5 cohort passed **8/8** on the changed tree (global→LDS, staged LDS/WMMA, via-Tile; no skips), explicitly reconciling rather than inheriting the older 1,569-test sweep. **Open:** NVIDIA barrier-at-birth emission, retirement of WarpSpecLegality's legacy ancestor marker, and row 8 after W1.1 step 5. | CAKE §5–§6 | gfx1151 gate closed; NVIDIA producer + row 8 open |

**Exit:** an aliased/indirect RNG call is detected by effect inference; an
inactive branch's adjoint is provably **not emitted** (`CHECK-NOT` fixture per
#10a); six legality passes are one.

> W2.2 is closed against the canonical Graph emitter and concrete trace
> certificate. W3.1 still owns making one tracer the only frontend; unresolved
> source calls fail closed until that promotion can trace them.

### W3 — Collapse the duplications *(10 weeks · depends on W1, W2)*

Root cause C. Now possible, because the surviving path can carry what the deleted
path was carrying.

| # | Item | Source | Effort |
|---|---|---|---|
| W3.1 | **One differential harness**, then promote the tracer to the only frontend; delete `_OpExtractor`. Delivery is E2E-REAL-0/6: the harness must compare typed artifacts and observable execution, not only op-name lists. | Frontend §U1 + IR Stack §U3 | re-estimate after E2E-REAL-3 |
| W3.2 | **Superseded in delivery shape by E2E-REAL-0 through E2E-REAL-5.** Build and register a real Schedule dialect, preserve Graph SSA under schedule decisions, lower one scheduled matmul to the launch-level Tile ABI, make x86/ROCm packages consume that artifact, then migrate families. The Python spine is the differential oracle. This is three boundaries plus bufferization/package API work, not a 3-week convergence edit. | IR Stack §U3 + Target §X5 | re-estimate after the matmul vertical slice |
| W3.3 | Split the Tile dialect by level: primitives stay `tile.*`; whole-kernel ops → Graph IR / `tessera.kernel.*`; domain ops → `tessera_ebm`; `svd`/`qr`/`cholesky`/`lu` → linalg solver | IR Stack §U4 | 2w |
| W3.4 | Decompose `JitFn` (11 `_native_*_backward` → `emit/candidate.py` candidates behind `@f__bwd`); split `__init__.py`'s 315 nested defs into `tessera/ops/` | Frontend §U5–U6 | 3w |
| W3.5 | **General shared IFT execution plus typed physical child-composite landed.** `NewtonAutodiff` accepts arbitrary typed residual bodies and emits value-producing IFT VJP/JVP functions with explicit matrix-free GMRES/CG convergence policy. The execution oracle and physical parent use restarted GMRES with true-residual checks; the parent hashes residual, solution JVP/VJP, and parameter JVP/VJP children and executes them on AVX-512/gfx1151 without materializing a Jacobian. The compiler generates all five children for pointwise, sum/mean, rank-2 reduced-storage matmul/transpose, distinct parameter/solution spaces, bounded-dynamic dimensions, explicit mixed-storage widening, statically bounded `control_for`, and pure scalar `if`/bounded-`while` predicate replay. Expanded 30-sample WSL correctness packets are committed for AVX-512 and gfx1151. Remaining: Apple/NVIDIA consumers, broader Krylov packets, non-pure/vector predicates, and clean selector-grade timing. | Autodiff §B8 + OT R2 | landing |
| W3.6 | Batched operands in `ExpandProductTable`; connect `RotorSandwichFold`'s marker to a consumer | GA/EBM §1.3 | 2w |
| W3.7 | **Define ROCm producer ownership per package family** across registered C++ generators, compatibility Target-IR text, and `emit/rocm_hip.py` candidates. Add differential gates and retire only producers proven duplicate; preserve C++ MLIR→ROCDL/HSACO as the canonical native spine | Target §X6 | 2w initial inventory/gate |

**Exit:** one frontend, one lowering per boundary, no target string in `jit.py`,
every `tile.*` op is a tile primitive.

### W4 — The control-flow program *(10 weeks · depends on W1.2, W2.1, W3.1)*

**This is the plan's most important structural correction.** Dynamic control flow
was blocked at three independent layers — structured tracing, symbolic shape
proof, and region differentiation. The first executable single-block SCF slice
now spans all three; the remaining work is the production execution/evidence
gate rather than three disconnected substrates.

| # | Item | Effort |
|---|---|---|
| W4.1 | **Tracer-owned multi-block carrier landed 2026-08-14.** `for`/`if`, bounded `while_loop`, and `control_scan` retain nested bodies and now recover a content-addressed basic-block graph with explicit branch/yield/backedge values, merge arguments, entry, and exit. Recovery consumes the canonical trace and never re-executes Python or consults `_OpExtractor`. General raw-Python source CFG interpretation, arbitrary unstructured CFG, and native MLIR multi-block region materialization remain open. | landing |
| W4.2 | **Typed Presburger boundary, including region propagation, landed 2026-08-14.** Python emits a versioned, content-addressed coefficient-vector carrier (`eq`/`ge`/`mod`) before compilation. `SymbolicDimEqualityPass` consumes it with MLIR `IntegerPolyhedron`; modular rows introduce existential quotient locals. Attaching a system now re-digests the structured CFG and stamps the same typed-system identity on every nested block, so a branch cannot silently drop the constraint environment. Remaining: native C++ consumption on general multi-block regions and nonlinear products, which remain explicit compatibility guards. | landing |
| W4.3 | **Single-block SCF products plus executable residual ABI carrier landed.** `RegionAdjointInterface` differentiates `scf.if`, positive-step counted `scf.for`, canonical bounded `scf.while`, and lowered `control_scan`. The execution-derived treeverse path now materializes a content-addressed `tessera.region_residual_abi.v1` binding policy, structured-CFG digest, steps, state dtype/shape, checkpoint identities, and retained bytes; SAVE and HYBRID captures execute and reject mismatched candidate/ABI identity. The C++ region adjoint still consumes only `recompute_all`. Remaining: emit those residual operands/results in native MLIR, general source CFG, native region-product execution, and target packets. | landing |

**Exit (single gate):** *a `@jit` function containing a data-dependent loop
compiles, differentiates, and executes, with numerical agreement against the
Python oracle.*

### PDE-STENCIL-FOUNDATION-1 — truthful discretization contracts

The first correctness slice landed 2026-08-12. Neighbors stencil definitions
carry one finite floating-point coefficient per tap, and loop materialization
performs the explicit multiply. TPP stencil operations require a scheme,
positive even order, and positive per-axis spacing; none is manufactured by
legalization. The CPU gradient ABI consumes the selected-axis spacing. Target
lowering now distinguishes `executable` from `artifact_only` and emits a
callable symbol only for a linkable implementation.

The two stencil stacks are declared differential oracles. Their shared
periodic central-gradient subset is tested against the compiled TPP CPU kernel
using the typed physical-coefficient contract, including non-unit spacing and
`h^-derivative_order` scaling. Temporal halos, general kernels, boundary modes,
and physical target consumers remain open. The first compiler-owned typed
constant-coefficient `pde.operator` semantic carrier, exact-rational
principal-symbol classifier, and fail-closed diagonal-diffusion FTCS bound are
now landed. Next ordered work is Graph ODS/pass materialization and a general
root-condition certificate, followed by physical consumers. ROCm owns the first
gfx1151 stencil+halo packet; x86 owns the CPU/AVX-512 packet. Apple and NVIDIA
remain independent follow-ups. Acceptance criteria live in
[`PDE_STENCIL_CAPABILITY_PLAN.md`](PDE_STENCIL_CAPABILITY_PLAN.md).

### BLOCK-ATTNRES-1 — ROCm-first depth-attention residuals

Synchronization key `BLOCK-ATTNRES-ROCM-2026-08-12` owns this project. The
delivery order is intentionally split by evidence domain:

1. **Host-free semantic phases 0–4.** Phase 0 established the balanced block
   partition, epsilon-qualified key normalization, forward/VJP equations, and
   merge lemma. Phase 1 now exposes fp32 `attn_with_stats`, `softmax_merge`,
   and `softmax_finalize` reference operations with associative/commutative,
   partition-invariance, dtype, shape, and fail-closed tests. These references
   remain `artifact_only` at the lowering registry. Phase 2 is also landed: the
   faithful stdlib recurrence and query-hoisted two-phase algorithm preserve
   zero-init, balanced nonempty blocks, absent-partial identity, fp32 statistics,
   storage dtype, and attention-based final aggregation. Phase 3 landed
   2026-08-13: typed statistics/finalize/depth-attention Graph operations carry
   required epsilon and numeric-policy attributes; `depth_attn` implements the
   compiler Adjoint and Tangent interfaces through typed product operations;
   direct numerical tests compare both analytic products with the canonical
   decomposition. Phase 4 carries the exact static all-f32 contract through one
   content-addressed `schedule.depth_attention` operation and one launch-level
   `tile.depth_attention_kernel`; its digest binds source count, rows, width,
   epsilon, numeric policy, source tile, workgroup, statistics recurrence,
   associative merge recurrence, and target architecture. Schedule→Tile
   rejects changed policy and gfx1200/gfx1250 remain fail-closed. This shared
   closure grants no Target execution claim.
2. **Exact-device phase 5 (landed 2026-08-13).** ROCm/gfx1151 now owns the
   typed Target record, fused statistics-attention plus associative
   merge/finalize kernel, exact content-addressed HSACO package, and runtime
   descriptor consumer. Three exact-device shapes pass the independent fp32
   oracle (maximum absolute error `1.96e-5`). The committed WSL packet records
   operation-total host-wall samples and is explicitly selector-ineligible;
   bare-metal HIP-event/ROCprofiler calibration remains open. gfx1200/gfx1250
   remain fail-closed.
3. **Architecture phase 6.** x86/AVX-512 owns an independent vectorized
   package and clean Zen 5 packet after the shared contract is stable. Apple
   and NVIDIA own separate packages and evidence; no schedule or result
   transfers between targets.

The scoped mathematical and acceptance authority is
[`BLOCK_ATTNRES_ROCM_PLAN.md`](BLOCK_ATTNRES_ROCM_PLAN.md). Global ordering and
promotion authority remain here.

### REF-TIER-PHYS-1 — solver-first physical lanes and shared lattice butterfly

Cross-backend sync `REF-TIER-PHYS-2026-08-16`. The PR #568 reference tier now
has its first compiler-owned physical boundary:

1. `tridiagonal_solve` lowers through one content-addressed Schedule→Tile
   contract. x86 owns a batch-blocked Thomas recurrence vectorized across
   independent systems; gfx1151 owns cooperative LDS parallel cyclic reduction.
   Both preserve the reference's fp64 accumulation and fp32 storage contract.
2. The four subset/superset zeta/Mobius operations lower through one
   `schedule.coalition_butterfly`/`tile.coalition_butterfly_kernel` carrier.
   `(half, sign, ascending_bit_yates_v1)` selects the recurrence, avoiding four
   target emitters. x86 and gfx1151 have independent physical consumers.
3. Host-free C++/MLIR and direct x86 numerical proof are required in the owning
   change. Exact-device gfx1151 and clean Zen 5 performance packets remain
   architecture gates; gfx1200/gfx1250 fail closed.
4. The other five coalition operations remain compositions over butterfly,
   reduction, softmax, bit-permutation, and segmented primitives until fusion
   evidence justifies a family-level package. This slice does not call the
   broader FFT/coalition G1b consolidation complete.

### LAYOUT-ALG-1 — one layout algebra under five asking consumers

Bound 2026-08-16 from [`CUTE_IR_ASSESSMENT.md`](CUTE_IR_ASSESSMENT.md), which
owns the mathematical verification and the acceptance criteria. Global ordering
and promotion authority remain here.

The finding that justifies an ID rather than a reference: **five independent
in-tree items already ask for a piece of layout reasoning, and each proposes a
different, weaker mechanism.** FORGE W1's read-locality lattice is a six-point
chain approximating a question that layout composition plus divisibility decides
exactly; FORGE W2's residency contract is `cosize`; SparDA §III.3's GQA-fold is
`group_modes`/`logical_divide` and is **not expressible today** — `rearrange`
fails closed on precisely that spec; TileSight §3.2's rasterization is
`composition(grid, raster_layout)`; and the G1b butterfly/FFT consolidation
needs a shared representation in order to be shared at all. This is root cause B
("told, not derived") about to recur five times, and Decision #30's "no eighth
bespoke walker" rule applied before the walkers exist. Against
[`CORE_SUBSTRATE_VIEW.md`](CORE_SUBSTRATE_VIEW.md), layout algebra is not a tenth
substrate investment — it is the missing implementation under **S9**, whose `⊑`
operator that document already flags unowned.

Ordered delivery, host-free throughout except where noted:

1. **L1 — the algebra and its binding.** Eleven operations over nested
   `(shape, stride)` with static-or-dynamic leaves; four types. **One C++
   implementation** in the support library with a ctypes binding, per the
   2026-08-16 home decision — not a Python/C++ pair, so Decision #31 is
   satisfied by construction rather than by declared-oracle exemption. The
   binding fails closed with a named diagnostic and ships **no fallback path**,
   because a fallback is a second implementation in disguise. Exhaustive proof
   over all layouts to size 64 runs on **both sides of the ABI**. Every operation
   is accepted on **two independent axes — function and result structure**;
   `test_layout_algebra_contracts.py` already enforces the pair, and an
   implementation that returns the right elements under the wrong mode grouping
   must fail, since grouping is what downstream slicing and partitioning read.
2. **L2 — first consumer, `rearrange`/GQA-fold.** Smallest real consumer;
   converts a documented `ValueError` into a working op without regressing its
   Decision #21a fail-closed behaviour. **L1 does not land without L2 committed**
   — otherwise it is a Decision #29 violation by construction.

**2026-08-16 checkpoint:** the versioned C++ dylib, no-fallback ctypes binding,
compact coordinate algebra, exhaustive size-64 ABI corpus, and executable
forward/inverse GQA-fold consumer have landed in the working slice. General
nested composition/complement/coalesce/inverse and product/divide/slice remain
inside L1; LAYOUT-ALG-1 is therefore landing, not closed. SO-1 is complete and
re-bases R3 action DAGs onto `ScheduleObject`; SO-2 has its first typed
role-to-barrier carrier and target-neutral ping-pong/Hopper verifier fixtures.
The gfx1151 numerical gate is closed: the earlier 1,569-test device sweep gives
broad parity evidence and the narrower plan-named SO-2 cohort passed 8/8 on the
role-carrying path. NVIDIA barrier-at-birth remains the missing physical role
producer; it does not reopen AMD numerics. L3/L4 and all raster promotion
decisions remain unchanged and evidence-gated.
3. **L3 — the `⊑` decision procedure.** FORGE W1/W2's legality query and
   residency check implemented as layout factorization and `cosize`. The lattice
   stays the declared interface; the algebra makes `block` precise instead of a
   promise. Closes the S9 implementation gap.
4. **L4 — emitter index math.** Consolidates the four hand-written block-index
   emitters and the hardcoded `A[row*K+k]`/`B[k*N+n]` templates onto the shared
   algebra. Acceptance is **bit-identical output** for every currently reachable
   `(raster_order, raster_group)` — a pure-refactor proof. Choosing a measured
   non-default raster is explicitly **not** in this item: per ROCM-CALIB-1 that
   is architecture-owned and blocked on a correlation/retain verdict, and it
   does not transfer between targets.
5. **L5 — MLIR carrier.** `#tile.layout` gains nesting and dynamic leaves;
   `MaybeStaticTypeInterface` plus a fold-static pass, delegating to L1 rather
   than reimplementing. **Sequenced after W1.1 step 4**, never before — the same
   Decision #31 ordering caveat [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md)
   applies to `#tile.mma_desc`.

Independent of the above and not blocking it: a negative-scoped
`tessera-target-opt` driver that registers the Target IR dialects without
`tessera`/`tile`, making Tile IR leakage a parse error. `test_target_ir_contract.py`
parses and verifies every emitter and golden, but registers everything, so it
cannot falsify a *leakage* claim — the same argument as Decision #19's standing
lesson that a host with the ISA cannot falsify a host-portability claim.

### W5 — Decisions become measured *(9 weeks · depends on W1, W3)*

Root cause: L5, "a constant where a measured decision belongs." Every item routes
an existing hardcoded choice through the arbiter Decision #28 already built.

| # | Item | Source | Effort |
|---|---|---|---|
| W5.1 | **Executable policy cohort landed; architecture packets remain open.** Complete-backward samples and unique retained residual allocations produce exact evidence; only exact-device rows stamp compiler selection attributes. Counted regions execute candidate plans and report the policy actually implemented (`SAVE`, `RECOMPUTE`, or `HYBRID`) plus forward, replay, and backward work. Model-only candidates remain ineligible. Remaining: lower chosen plans into the C++ region adjoint and collect exact-family x86/gfx1151 packets. | Autodiff D5 + GA/EBM §1.5 | landing |
| W5.2 | Scheduling decisions at Schedule IR — tile sizes, stage counts, raster order, warp roles chosen from `fusion_core` cost models via the measured arbiter, not from `--tile-q=64` | Frontend §U3 + IR Stack §U6 | 5w |
| W5.2a | **COMP-SCHED-OVERLAP-1/R1 — closed 2026-08-10.** Python Schedule→Tile async carriers now produce explicit token results and token-consuming waits; internal `tessera.queue.*` markers are rejected instead of filtered. The registered `--tessera-await-sinking` pass moves collective awaits to the first true SSA use through the W2.2 registered semantic query. Mutation, RNG/sample identity, aliases/casts/views, regions, unknown operations, and ordered collectives are fail-closed barriers. x86/no-async is a recorded no-op; the existing gfx1151 global→LDS/LDS-WMMA cohort remains 16/16. | TileRT R1 + async reconciliation | closed |
| W5.2b | **COMP-SCHED-OVERLAP-1/R2 — closed 2026-08-10.** Successful measured autotune rows now carry a validated `tessera.measured_resource_vector.v1` in canonical `hot_path_metadata`: compute time, dtype-correct bytes moved, communication bytes, queue/resource identity, timing provenance, and a content digest for the measured schedule candidate. Timing provenance survives SQLite warm-starts. Analytical rows cannot claim this vector, and the contract stamps `usage = composition_analysis_only` plus `selector_authority = latency_ms`; scalar measured latency remains the sole selector score. R3 is the first permitted composition consumer. | TileRT R2 + Evaluator | closed |
| W5.2c | **COMP-SCHED-OVERLAP-1/R3 — v2 scalable search landed 2026-08-14.** The compiler-owned `composition_cost` boundary consumes R2 vectors as explicit Tile actions, validates DAGs with deterministic Kahn traversal, and models compute/memory/communication lanes plus queue serialization. Production search is critical-path/list scheduling. The admissible bound is `max(dependency critical path, per-resource-lane work, queue work)`; an inexact candidate is pruned only when that lower bound already loses to another candidate's feasible upper bound. Results remain promotion-ineligible with scalar `latency_ms` as selector authority. W2.1/W2.2 supply registered effect, alias, liveness, and memory-dependence facts. | TileRT R3 + TileSight T3 | closed |
| W5.2d | **COMP-SCHED-OVERLAP-1/R4 — bounded functional consumer closed 2026-08-10.** MegaMoE now executes a content-addressed chunk/action plan rather than reconstructing an implicit loop policy. The plan owns contiguous chunk slices, per-chunk expert capacity, a two-live-frame workspace bound, dispatch→compute and compute→combine true-use dependencies, ordered collective dependencies, and deterministic chunk-combine order. R3 requires total measured evidence for every action and may only prune plans; exact-device scalar latency selects among retained plans before the chosen digest is passed to the runtime. Runtime telemetry records the digest, issued action order, true-use waits, combine order, and maximum live workspace. Mock multi-rank numerical and repeated bit-identity proofs pass. Native NCCL/RCCL/MPI/OFI/SHMEM transport plus architecture-owned correctness/performance packets remain open and do not transfer across targets. | TileRT R4 + TileSight T4 | closed |
| W5.2e | **Automatic dependence-edge generation — conservative parity gate landed; selector proof remains open.** `infer_action_dag` consumes a fresh W2.1 Graph snapshot and total R2 resource vectors. It emits explicit edges for SSA producers, overlapping or unknown alias sets, value-scoped memory dependence, mutation/state/I/O, registered stochastic identity, ordered collectives, unknown effects, and region boundaries. Unknown alias facts now carry an explicit reason rather than relying on a side effect of `may_alias`. `compare_inferred_action_dag` requires generated edges to cover every edge in an existing hand-authored R3 fixture; additional conservative edges are reported separately. Pure SSA matches the current fixture exactly, and opaque dataflow is serialized. `CompositionCandidate.from_graph` remains the physical-family entry point. **Open:** adopt it in remaining non-JVP physical producers and collect clean calibrated Zen 5/bare-metal gfx1151 packets before scalar measured latency may select a retained schedule. WSL and analytical vectors remain prune/rank-only. | W2.1 + TileRT R3/R4 | landing |
| W5.2f | **Tiled SSD family.** Define one content-addressed Schedule→Tile program for chunked SSD GEMM, reduction, recurrent carry, checkpoint/residency, and mutation lineage. Existing ReplaySSM and backend-specific sequence kernels are candidates/oracles, not semantic authorities. Physical lowering and promotion remain architecture-owned. | Roadmap tiled-SSD design + E2E-REAL-6 | open |
| W5.2g | **Scalable action-DAG search closed 2026-08-14.** Wide DAGs use deterministic critical-path/list scheduling and never enter factorial enumeration. DAGs through eight actions retain exhaustive enumeration as the declared oracle; tests compare feasibility, modeled makespan bounds, and deterministic identity. A lower bound combines dependency critical path, per-resource-lane work, and queue work. Proven lower-bound losers may be pruned; every inexact survivor remains rank/prune-only and scalar exact-device latency retains selection authority. Remaining producer wiring stays W5.2e, not a second scheduler. | PDE plan §IV.3/IV.4 routed into TileRT R3 | closed |
| W5.3 | Generic fusion region discovery over a legality oracle (a W2.1 client); keep the measured cost models | Sweep §F3 | *(folded into W5.2)* |
| W5.4 | **Typed placement lattice plus bounded reshard SSA landed; native execution remains open.** The compiler represents replicated, tiled, partial-reduction, and unknown placements and propagates registered pure rules to a content-digested fixed point. Consumer mismatches now produce digest-bound actions with exact consumer/operand identity; `all_gather`, `all_reduce`, `reduce_scatter`, and `all_to_all` are inserted as real Graph SSA, retain their def-use chain through `schedule.collective` and Tile IR, and carry plan digest, subgroup, and region path. Region values may flow only to the same or a nested region; sibling/escaping movement fails closed. The conflict-free collective scheduler gives all-to-all subgroups a deterministic cyclic 1-factorization whose rounds have unique senders and receivers and cover every directed peer pair exactly once. Replicated-to-tiled local slicing deliberately remains unmaterialized until mesh extent can determine a truthful result type. Remaining: encode and consume the 45 generated domain-specific/axis-changing contracts, typed local-shard shape materialization, general nested-region placement joins, mock-mesh family proofs, and native NCCL/RCCL/MPI/OFI/SHMEM packets. | Sweep §F4 + PDE plan §IV.1 | landing |
| W5.5 | Rule-table-driven canonicalization (PDL/PDLL). **Defer equality saturation** until the rule table is large enough that ordering demonstrably costs something | Sweep §F5 | 3w |

**Exit:** no tile size, residual policy, or fusion boundary is chosen by a
constant; sharding a model requires O(few) annotations, not O(layers).

### W6 — Exceed the state of the art *(14 weeks · depends on W2, W4)*

| # | Item | Source | Effort |
|---|---|---|---|
| W6.1 | **Bounded native products plus exact compiler HVP composition landed (`AD-FWD-CORE-1` / `AD-FWD-PRODUCT-2` / `AD-FWD-NATIVE-1` / `AD-FWD-DIST-3` / `AD-HIGHER-1`).** `TangentInterface`, paired functions, public request/provenance, stable `wrt_indices`, and direct tangent families are live. `--tessera-autodiff-hvp-pipeline` now emits the paired reverse program, marks only original differentiable primals tangent-active, and applies the exact forward transform to produce `@f__bwd__jvp`; the emitted product is numerically executed against an independent quadratic oracle. `JitFn.compiled_hvp_ir` exposes that fail-closed compiler product without finite-difference substitution. The eager `tessera.autodiff.hvp` compatibility helper still uses finite differences and is not this compiler proof. Remaining: broaden second-order TangentInterface coverage, broader ISTFT layouts/dtypes, MPI/OFI/SHMEM and subgroup transport, native multi-rank packets, clean performance packets, and Apple/NVIDIA packages. | Autodiff D2 | landing |
| W6.2 | Sparse AD — sparsity detection + coloring (client of W2.1/W4.2). PyTorch, TF, and JAX all lack this | Autodiff D7 | 5w |
| W6.3 | Taylor/jet mode over Weil algebras on a **new generic finite-multiplication-table substrate** potentially shared with GA. The current `ga/signature.py` is Clifford-specific (blade XOR, metric signs, anti-commutation) and cannot represent arbitrary commutative nilpotent Weil algebras | Autodiff D6 | research estimate required |
| W6.4 | Table-driven GA kernel synthesis via `emit/`; then PGA `Cl(3,0,1)` | GA/EBM §2.3–2.4 | 5w |

**Exit:** a defensible "exceeds SOTA" claim with a benchmark behind it — sparse
Jacobian scaling `O(colors)` not `O(rows)`; order-`k` derivatives sharing the
tuned GA kernels.

> W6.4 can supply useful table-lowering machinery, but W6.3 still requires a
> generic algebra representation and AD semantics. Treat reuse as a design
> hypothesis to prove, not as a sequencing-based cost reduction.

### Riemannian OT — re-scoped as validation, not a track

The [OT plan](RIEMANNIAN_OT_PLAN.md) proposed R0–R5 at ~14 weeks. Integrated, most
of it is **already funded by W1–W5**:

| OT need | Provided by |
|---|---|
| manifold as a hard dispatch key (H1) | W0.1 |
| remove inert EBM checkpoint policy from the default pipeline (H2) | W0.2; demand-aware loop rematerialization remains W5.1 |
| geometric primitive layer (R1) | new — 1.5w, first consumers are `ebm/geo_sampling.py` and `hyperbolic.py` |
| `stop_gradient` + implicit diff (R2) | W3.5 + W2.3 |
| `c_transform` fused loop (R3) | W4 (control flow) + W5.1 (residual policy) |
| backend lanes (R4) | W5.2 (schedule decisions) + existing EBM Langevin kernel precedent |
| oracles (R5) | W2.1 (fail-closed analyses) + the existing Evaluator |

**Net new OT scope: ~4 weeks** (the geometric primitive layer plus the RNOT
composite ops), down from 14. And RNOT becomes the plan's **acceptance test**: a
2500-step manifold inner loop with a stop-gradient boundary exercises W1 (typed
tiles), W2 (activity + liveness), W4 (control flow), and W5 (residual policy) in
one workload. If RNOT runs at wall-clock parity with RCPM, the plan worked.

---

## 5. Dependency graph

```
W0 ─────────────────────────────────────────────────────────────►  (start now)
 │
 ├─► W1 (declarations binding) ─────────┬──────────────────────────┐
 │      W1.1 type Tile IR ──────────────┤                          │
 │      W1.2 shape-rule registry ───────┼──► W4 (control flow) ────┤
 │                                      │      W4.2 needs W1.2     │
 └─► W2 (analysis layer) ───────────────┤                          │
        W2.1 dataflow framework ────────┼──► W4.3 needs W2.1       │
        W2.2 effects ───────────────────┤                          ├─► W6
        W2.3 activity ──────────────────┘                          │   (SOTA)
                │                                                  │
                └─► W3 (collapse duplication) ──► W5 (measured) ───┘
                       W3.1 one frontend ──► W4.1
```

Delivery-critical chain: **E2E-REAL-0 → E2E-REAL-1 → E2E-REAL-2 →
E2E-REAL-3 → E2E-REAL-4**. W1.1 is an input to E2E-REAL-2/3; W2.1 can proceed
once E2E-REAL-1 supplies a real program to analyze. W4 and W5 then extend a
proven spine instead of building capability beside it. The earlier
40-week/63-week totals are not retained as commitments: the old W3.2 estimate
omitted an entire registered Schedule dialect, SSA-preserving boundary design,
bufferization, package API migration, and hardware gates; W6.3 still needs
research scoping.

---

## 6. Three budget levels

**Minimum — W0 only.** Closes verified fail-open and invalid-value paths,
preserves the correct numerical fallback for untraceable EBM callbacks, removes
inert checkpoint policy from the default pipeline, corrects architecture
documentation, removes a duplicate-dialect trap, and upgrades the generic
Target-IR contract test. Adopts the six governance rules so nothing regrows.

**Recommended scope — finish the live W1.1 contract, then E2E-REAL-0 through
E2E-REAL-4, with W2.1 starting after E2E-REAL-1.** This produces one measured,
lineage-complete matmul on the two locally available architectures rather than
funding several horizontal subsystems before any one request traverses them.
Re-estimate frontend deletion and family breadth from that measured slice.

**Full — W0…W6, re-estimate after W1.1 and W6.3 design spikes.** ROCm ownership/inventory work is host-free;
subsequent kernel-producing migrations are hardware-routed individually (§6a).
Adds the control-flow capability, measured decisions, and two defensible
exceeds-SOTA claims.

## 6a. Fleet routing — what must run on which box

**Core compiler work is driven on the Strix Halo box (decided 2026-08-02).**
`AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, Ubuntu 24.04 under WSL2, 32 threads,
62 GB RAM, LLVM/MLIR 23 at `/usr/lib/llvm-23`, `gfx1151` visible to `rocminfo`.
It is both faster and larger-memory than the Mac M1 Max for `tessera-opt`
rebuilds, and it is the only box in the fleet with an executing GPU lane — so
compile-time contract work and its hardware gate live on the same machine. The
Mac is retained for Apple-backend work, which cannot move.

Most of this plan is compile-time contract work and runs anywhere. Two items are
hardware-bound, and one of them is the highest-risk item in the plan.

| Work | Box | Why |
|---|---|---|
| W0, W1 (typing, enums, shape rules), W2 (analyses), W3.1–W3.4 | **Strix Halo (primary)** | ODS, `tessera-opt`, lit, unit tests. No device needed; tightening a type is a compile-time change. Runs on the Mac too, but the primary is where the gates are expected to be green. |
| Any item touching the Apple backend or Apple lit fixtures | **Mac M1 Max** | Metal/Accelerate toolchain is Mac-only; this is the one thing that cannot be retargeted. |
| W0.10 build branch — `tessera_x86` dialect | **Strix Halo** for ODS/lit; **AVX-512 execution proof also here**; AMX execution proof has **no box in the fleet** | Zen 5 has AVX-512 (`avx512f` confirmed on this host) but **AMX is Intel-only** — the earlier routing to a "Zen5 box for AMX" was wrong. The NR2 Pro's Core Ultra 7 has neither. Anything AMX-gated is currently unexercised (see the note below). |
| **W3.7 — ROCm producer ownership + differential gate** | **Strix Halo** — inventory/IR equivalence is host-free, and gfx1151 for each later producer change is on the same box | The initial slice does not change kernels. Any retirement or migration that changes generated code requires execute-and-compare on the owning device. |
| W4 (control flow) end-to-end gate | **Strix Halo / gfx1151** | It has the broadest compiler-generated + hardware-verified op coverage. |
| ROCm arch breadth (gfx950 / gfx1201 / gfx1250) | deferred — no silicon | MASTER_AUDIT P2, unchanged. |

**AMX has no hardware behind it.** `tests/device/x86/test_amx_int8_gemm.py` and
`scripts/run_x86_amx_release_gate.sh` (merged in #489) gate on AMX capability,
which no current fleet box reports. The gating is honest, but the lane is
unexercised; do not let W0.10 lean on an AMX execution proof it cannot obtain.
Native x86 proof on this box means AVX-512.

**Two sequencing consequences.** First, W3.7 reuses the differential-harness
design from W3.1/W3.2 and records one owner per package family before anything
is deleted. Second, W1.1's ROCm typing may expose invalid assumptions in
registered generators; fix those compile-time contracts host-free, and require
gfx1151 evidence only when a generated kernel or selected producer changes —
both halves now on the same machine.

**What to cut first if squeezed:** W6.4 (GA synthesis / PGA) and W5.5
(canonicalization rule tables) are the most deferrable — genuine value, no
downstream dependents. **What never to cut:** W0.5 and W0.8 (an hour and a day)
— a wrong architecture decision and absent governance are what let all of this
accumulate.

---

## 7. Risks

| Risk | Wave | Mitigation |
|---|---|---|
| **W3.1 is a broad behavior change** — the AST frontend is the default on every non-Apple target | W3 | The differential harness ships *before* the switch, not after; promote per-target with the harness green |
| **W4 will exceed its estimate.** Region adjoints are the hardest item here and structured reverse mode is genuinely difficult | W4 | Land W4.1+W4.2 with a forward-only gate first, so partial progress is observable before W4.3 |
| **W1.1 touches every Tile-consuming backend** | W1 | Parameterized types and verifier contracts can invalidate producers and consumers; land per primitive/variant with parser, verifier, lowering, and backend fixtures |
| **Stage presence is mistaken for stage lineage.** A bundle can contain Graph/Schedule/Tile/Target/image/descriptor while its backend package was re-synthesized from Graph | E2E | E2E-REAL-0 records parent digests and a separate `lineage_complete` answer; no capability or dashboard may infer it from non-null stages |
| **Schedule IR cannot currently carry the computation it claims to schedule.** Its C++ `schedule.tile` is metadata-only and the Python textual model emits `() -> ()` | E2E | E2E-REAL-1 chooses and gates the mixed-level SSA contract before implementing a broad lowering |
| **Waves 1–3 produce no user-visible feature.** Fifteen weeks of "the compiler now enforces what it already said" is hard to fund | all | RNOT (§4) is the visible acceptance workload; state the intermediate gates as capability claims, not cleanup |
| **The governance rules are ignored under delivery pressure** | all | #29 and #31 are drift-gateable; make them tests, not conventions |
| **Someone starts at W3** (deleting duplications first, because they are the most visible waste) | — | It fails: the surviving path cannot yet carry what the deleted one carried. Ordering A→B→C is the plan's core claim |

---

## 8. Delegated detail and scope boundaries

The integrated queue owns sequence, not every implementation detail. The
following subjects remain delegated and must return here only when they change
global order or a shared contract:

- ~~Target IR dialects~~ — **reviewed 2026-08-02**
  ([TARGET_IR_REVIEW.md](TARGET_IR_REVIEW.md)). The `AnyType` finding **does**
  repeat, in `ROCM_{MFMA,WMMA}` and the NVIDIA mma ops; W1.1 grew 3w → 5w
  accordingly, and four new items landed (W0.9, W0.10, W1.1b, W3.7).
- The bodies of the 67 `GenerateROCM*Kernel` passes — these need review on the
  host for ownership classification, with gfx1151 required before any
  kernel-producing migration is accepted.
- `emit/nvidia_cuda.py` (4722 lines) internals. `emit/rocm_hip.py` was inspected
  far enough to establish that it is an arbiter candidate/runner surface, not a
  drop-in replacement for the canonical MLIR→ROCDL package spine.
- Spectral implementation details are owned by E2E-REAL-5D/5E/5G and the TSOL
  physical packages; spectral autodiff is now AD-TSOL-SPECTRAL-1. TPP solver
  families, collectives/neighbors, and RubinCPX still require separately bound
  work items before they enter this sequence.
- Quantization numerics; the KV-cache and memory model.
- The Evaluator implementation is owned by
  [`EVALUATOR_PLAN.md`](EVALUATOR_PLAN.md). It supplies proof and promotion
  decisions to every slice here; it does not independently set their order.
- `WarpSpecializationPass` and `AsyncCopyLoweringPass` bodies.

Absence from this plan is not a clean bill of health or authorization to start
an unbound queue. Add an owning integrated ID when delegated work becomes a
cross-compiler priority.
