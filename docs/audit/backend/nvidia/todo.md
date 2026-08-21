---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-08-21
---

# NVIDIA compiler test-suite evaluation and rearchitecture

Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; NVIDIA outcome: not applicable.**
`tessera_jit` is the host-CPU JIT lane; the NVIDIA backend's device
paths (NVRTC / tessera-opt pipelines) do not consume it, and NR2 Pro's
host-CPU lane is the same x86 lane the x86 entry validates. No
NVIDIA-owned surface compiles through the changed pipeline.


Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; NVIDIA outcome: follow-up required (sm_120).**
Same contract change as the rocm entry. Expected parity-neutral for the
same reason as the previous two keys (both sides of every parity
comparison read the same updated reference; dtype preserved; lgamma/
digamma primals mirror the canonical forwards bit-for-bit). Run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record here — one
NR2 Pro session can now close all three open keys (this one,
AD-RETIRE-1-POINTWISE-2026-08-20, AD-RETIRE-2-2026-08-20).


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; NVIDIA outcome: follow-up required (sm_120).** Same contract
change as the rocm entry. Expected parity-neutral for the same reason as
`AD-RETIRE-1-POINTWISE-2026-08-20` (both sides of every parity comparison
read the same updated reference; dtype preserved); run the CUDA-marked
autodiff/loss parity tests on NR2 Pro and record here. Note this key AND the
still-open AD-RETIRE-1 key can be closed by one NR2 Pro session.


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; NVIDIA outcome: follow-up required (sm_120).** PR #600
retires the ODE-family pointwise hand rules behind the `DerivativeContract`
datum (dtype-preserving reference rules; unified log/sqrt boundary guard —
see the rocm entry for the full contract statement). Expected impact on
sm_120: parity-neutral — the CUDA loss/optimizer backward lanes compare
against the same updated reference on both sides — but per the
no-evidence-transfer rule that expectation is not a claim: run the
CUDA-marked autodiff/loss parity tests on NR2 Pro and record the outcome
under this sync key. Boundary inputs below 1e-12 are outside every sampled
parity envelope; the dtype change moves the reference TOWARD the fp32 native
lanes, not away.


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; NVIDIA outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
`_apple_gpu_dispatch._prebuilt_candidate` decided whether a prebuilt dylib was
current by `ctypes.CDLL`-ing it and probing sentinel symbols. Loading is not a
read-only probe: it registers the runtime's Objective-C classes process-wide,
and skipping the candidate afterwards does not unregister them (the ObjC
runtime pins an image that has defined classes). A stale candidate therefore
stayed resident and the from-source dylib compiled next registered the same
classes again -- two images of one runtime, each with its own copy of every
file-static, including the thread_local last-error channel that
`_apple_gpu_run_checked` reads to decide whether a kernel failed. Staleness is
now read from the file's symbol table via `nm`, so a stale candidate is never
loaded; an undecidable probe (no `nm`) keeps the previous load-and-probe
behaviour rather than rejecting a library it cannot fault.
NVIDIA impact: none. The loader is Apple-specific (`_apple_gpu_dispatch.py`,
`libTesseraAppleRuntime` / `libtessera_apple_gpu_runtime`) and no NVIDIA path
reaches it. No sm_120 retest required and no device evidence is produced or
claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; NVIDIA outcome: not applicable.** The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
GPU elementwise-binary lane. `apple_gpu_runtime_stub.cpp` — compiled on every
NON-Darwin host so the C symbol exists — implemented opcodes 0-8 and its
`default:` arm assigned `out[i] = x`, so `mod`(9), `floor_div`(10), the six
comparisons(11-16), and the logical/bitwise ops(17-22) returned the LEFT operand
instead of computing. Because the symbol exists,
`_apple_gpu_dispatch_mpsgraph_binary` takes the kernel branch rather than its
numpy fallback, so those values came back as if computed, with no diagnostic
(Decision #21). Fixed by implementing opcodes 9-22 to match
`mpsg_binary_node` and the declared host reference
`runtime._apple_gpu_binary_numpy`, and by rejecting an unknown opcode through
the stub's last-error channel (new kind 3) so it routes to the host fallback
instead of returning a plausible buffer.
NVIDIA impact: none. `tessera_apple_gpu_mpsgraph_binary_f32` is referenced only
by Apple files (`runtime.py`, `_apple_gpu_backend.py`, `_apple_gpu_dispatch.py`,
`apple_exact_device_proofs.py`, the two Apple runtime TUs, and
`SiluMulToAppleGPU.cpp`); no NVIDIA lowering or runtime path reaches it. No
sm_120 retest required and no device evidence is produced or claimed.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; NVIDIA outcome: not applicable today; parity by construction
for future migrations.** The zero-function-candidate slice (PR #590) changes `JitFn`'s call ABI recovery
and adds one diagnostic code. A `@jit` function whose AST lowering produced no
function raised a bare `IndexError` from `_establish_tracer_authority`, and the
same absence left `_call_arg_names`/`_constraint_ir_args` empty — silently
mis-binding keyword calls and skipping call-time constraint re-checking. The ABI
is now derived from the Python signature via the shared
`graph_ir.ir_args_from_signature` (Decisions #30/#31), and the apple_gpu tracer
lane lifts foreign interpreter exceptions into the new registered
`JIT_APPLE_GPU_TRACE_FAILED` code (Decision #21). `TesseraTraceError` passes
through unwrapped.
NVIDIA impact: none today. The diagnostic is apple_gpu-scoped, and the
trace-defer route that reaches it is gated on `target == "apple_gpu"`. The ABI
recovery is target-independent and strictly widens what was previously an empty
tuple, so no NVIDIA behaviour changes. No sm_120 retest required and no device
evidence is produced or claimed. Future sm_120 frontends inherit the corrected
keyword binding and constraint re-check.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; NVIDIA outcome: not applicable today.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
NVIDIA impact: none. No NVIDIA lowering or runtime path consumes the `scalar`
kwarg — an exhaustive sweep for `get("scalar"`/`["scalar"]`/`get("other"` across
`python/tessera/` finds consumers only in `runtime._apple_gpu_dispatch_mpsgraph_binary`,
`runtime._execute_runtime_cpu_op`, and `matmul_pipeline._execute_op`, none of them
NVIDIA-specific. No sm_120 retest required and no device evidence is produced or
claimed. If an NVIDIA elementwise lane later accepts the lifted-scalar form, it
inherits this contract and must honor `scalar_side` for its non-commutative
opcodes.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; Nvidia outcome: not applicable today; parity by construction for future migrations.** The AD-LAW series (PR #588)
closes the swallowed-kwarg class registry-wide and adds the AD-WEIL-1 algebra
substrate plus all six executable laws. Reference-rule changes in this slice:
`stft`/`spectral_conv` JVPs **deleted** in favour of derivation from the
forward (both are bilinear, so every configuration key is honored by
construction); `jvp_istft` rewritten to honor axis/center/length/onesided/norm
while preserving its window quotient; `dequantize_nvfp4` fixed to accept the
per-block scale array its canonical forward takes (both modes previously
crashed); `jvp_lgamma`/`jvp_digamma` replaced (a dead zero-returning stub and
an identity placeholder); `jvp_cast` fixed for canonical dtype strings; the
shared polygamma helpers given reflection formulas (previously an O(n) loop
that hung on valid negative input — a live defect in the REVERSE path too).
NVIDIA impact: no native binding consumes these reference rules yet, so nothing retests. Future sm_120 backward family migrations inherit the law-checked oracle lane and the E2E-REAL-6 Law-3 gate. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; NVIDIA outcome: not applicable today, parity by
construction for future migrations; no sm_120 evidence changed.** AD-LAW-1
(PR #584) adds law oracles (adjoint + canonical-forward chain) over the
shared numpy reference JVP/VJP registries and the byte-gated
`autodiff_law_audit` dashboard. NVIDIA has no native binding to the two
fixed reference rules (`jvp_rmsnorm` eps default, `jvp_clamp` kwarg names),
so nothing retests; future sm_120 backward family migrations inherit the
law-checked oracle lane and the E2E-REAL-6 Law-3 gate (#583). Open shared
follow-up: the pinned swallowed-kwarg findings in `test_autodiff_laws.py`.
*Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for
`clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft
`norm` handling; five entries benign-classified; the open set is now the
stft/istft/spectral_conv and quantize families only. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan.

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; CUDA follow-up required.** Dynamic saved-slot data/shape
tapes, exact polynomial witness guards, and variadic branch CFG state are
target-neutral contracts. Effectful replay remains fail closed except for
compiler-owned extent assertions. NVIDIA has no native region-product consumer
or SM120 evidence and inherits no x86/gfx1151 proof.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared legality and
frontend authority landed; CUDA evidence unchanged.** Production pipeline,
WarpSpec, barrier-reuse, and derived Tile dataflow relations now run through
one staged `TileDataflowLegalityPass`; legacy CLI names are wrappers. Pure
static annotations abstract-trace before AST compatibility capture. The
existing SM120 Lion and causal DeltaNet launchers are now selected by explicit
family plugins instead of `JitFn` target dispatch. This is an ownership move,
not new device evidence; CUDA state-lineage package unification and an exact
SM120 packet remain open.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared contracts
landed; CUDA producer proof remains open.** CUDA-math kinds and TCGen05 operands
are ODS-typed, bare Tile fragments are rejected, and WarpSpec no longer trusts
legacy ancestor markers. Existing SM120 loss packages are reached through
explicit VJP plugins. The final tensor-valued MMA producers, barrier-at-birth
emission, SM120 Target proof, and Lion/DeltaNet shared-lineage proof remain
NVIDIA-owned; host-free verification transfers no device evidence.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
contract parity validated; CUDA physical follow-up required.** SAVE/HYBRID
selection now has a digest-bound Graph→SCF carrier. Shared paired AD consumes
dynamic branch-local residual extents, bounded SAVE and sparse HYBRID `while`
state tapes, and bounded-dynamic counted-loop tapes. The target-neutral
compiler now classifies source-CFG SCCs and structurizes bounded pure reducible
or irreducible native CFGs as a typed program-counter state machine while
preserving CFG/Presburger identity. Nested canonical structured bodies and
mixed control/tensor state are admitted. Saved dynamic slots require total
data/shape-tape envelopes; unbounded, unsupported-region, and unrecorded
effectful forms remain fail closed. SM120
still needs its own region-product correctness/performance packet and inherits
no sibling proof.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **shared
optimizer lineage validated; NVIDIA not applicable for the bounded physical
set.** The new `schedule.optimizer_vjp` → `tile.training_kernel` authority
does not declare CUDA consumers because no matching native reverse package
was proved in this slice. SM120 remains fail closed and inherits no AVX-512 or
gfx1151 evidence.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **shared
Adafactor/sequence-mixer authority validated; NVIDIA target follow-up
required.** Factored/full Adafactor and causal DeltaNet backward now have a
non-reexecuting Graph→Schedule→Tile package for x86/gfx1151. The existing
SM120 DeltaNet path is now plugin-owned but remains a compatibility package;
NVIDIA has no
Adafactor Target owner. Neither inherits sibling numerical evidence; each must
adopt the shared lineage carrier and produce CUDA exact-device proof.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **shared flat Lion
Graph and non-reexecuting proof parity validated; CUDA plugin follow-up
required.** The existing SM120 PTX Lion VJP remains numerically valid, but it
is now selected by the family plugin but does not yet consume the shared
`schedule.lion_vjp` state-lineage artifact. A CUDA follow-up must extend that
typed package to SM120 and rerun its owning-device packet. No x86/gfx1151 evidence
transfers.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **shared
rank-4 authority and bottom-right ragged-causal semantics validated; CUDA
plugin follow-up required.** The shared family registry now owns
`flash_attn`/GQA/MQA reverse through tracer Graph →
`schedule.attention_backward` → `tile.attention_backward_kernel`. NVIDIA’s
existing SM120 package is unchanged and inherits no x86/gfx1151 evidence. A
CUDA plugin consumer and independent SM120 numerical packet are still required.
The rank-3 `multi_head_attention` wrapper remains outside this bounded rank-4
migration, and active dropout needs keyed, non-reexecuting replay proof.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **shared tracer
and package-contract parity validated; CUDA physical follow-up required.**
Concrete AST specialization now resolves the same shape-derived spectral
identity as tracing, and compound spectral reverse products have one declared
Graph/Schedule/Tile/Target family-plugin boundary. NVIDIA is deliberately not
a Target consumer in this slice: no PTX package, CUDA runtime path, or SM120
evidence was added. A future CUDA consumer must own its package and independent
device proof.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no CUDA evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. NVIDIA code,
SM120 selectors, and CUDA packets are unchanged. A future SM120 corpus still
requires its own CUDA-event/CUPTI-correlated evidence.

Cross-backend sync `REF-TIER-PHYS-2026-08-16` — **shared Schedule contract
received; CUDA physical follow-up required.** Batched tridiagonal solve and
the four coalition zeta/Mobius transforms now have content-addressed
Schedule→Tile carriers. The coalition transforms share one parameterized
Yates butterfly rather than four emitters. NVIDIA gains no PTX consumer or
SM120 claim in this slice; a future lane must select an architecture-owned
parallel solver/butterfly schedule and provide independent device evidence.
The shared Schedule Object now snapshots nested resource metadata before
digesting, and dynamic rearrange/GQA-fold inference preserves ranked `?`
dimensions; parity is validated without transferring a physical schedule.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared carrier
parity validated; physical producer follow-up required.** The native layout ABI
and GQA fold transfer no CUDA layout or raster decision. SO-1 now owns the
content-addressed action/edge/role/residency value; SO-2 registers symbolic
producer/consumer roles on Tile mbarriers and proves the Hopper split with the
same rule as AMD ping-pong. Shared loop-carried role provenance and role-bearing
pipeline state are implemented, but NVIDIA still owns barrier-at-birth emission
and deletion of the legacy WarpSpec ancestor-role marker. No PTX, raster output,
selector, or SM120 evidence changed.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **no NVIDIA result transfers.**
ROCm's canonical split backward program was re-audited and x86 gained a
deterministic parallel implementation. The SM120 architecture-owned backward
package and performance evidence remain independent.

Cross-backend sync `X86-PASS-DIALECT-DEPENDENCY-2026-08-16` — **parity
validated; no NVIDIA physical follow-up.** The shared pass library now models
the optional hardware-free x86 Target dialect as a declared MLIR pass
dependency and fails closed when it is absent. No NVIDIA dialect, pipeline,
CUDA ABI, schedule, selector, or SM evidence changed.
The same closeout removes a private permissive `schedule` dialect from the
shared transform library; all transforms now consume the canonical ODS
`TesseraScheduleIR` authority. NVIDIA receives the build-parity fix only.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared semantic parity
validated; CUDA physical follow-up required.** Exact-rational PDE
classification and the first fail-closed diffusion stability certificate are
backend-neutral. No PTX stencil/boundary/halo package or evidence is claimed.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared compiler parity;
native NVIDIA follow-up required.** Reshard plans now materialize bulk
collectives as Graph→Schedule→Tile SSA with subgroup/region identity and
deterministic all-to-all matching rounds. Exact forward-over-reverse HVP now
exists as a compiler Graph product. NVIDIA receives those shared contracts but
adds no NCCL launch binding or SM120 HVP package in this slice; native
multi-rank and product evidence remain independent gates.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **normalization parity
validated; no new SM120 evidence claimed.** The existing SM120 normalization
backward launch is now owned by the shared native-VJP family registry with
explicit Graph/Schedule/Tile/Target declarations instead of package
construction inside `JitFn`. Runtime semantics and the existing SM120 evidence
row are unchanged. Other NVIDIA backward families remain compatibility paths
until migrated independently.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no NVIDIA
physical change required.** The new selector is AMD-specific and changes no
Graph dtype spelling, public operation, Schedule contract, NVVM/PTX ABI, or
SM120 capability. AMD FP8/BF8/FP4/MX and sparse-WMMA evidence cannot be used as
CUDA evidence. The shared `OP-DTYPE-FLOW-1` generator now audits SM120 by
operator and storage dtype; target-wide derived legality remains `legal_only`
unless an NVIDIA manifest owns the physical kernel.

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **not applicable to
NVIDIA, by existing architecture.** The `Validate / lit` lane was dead from
2026-08-11 to 2026-08-12 (pytest collection aborted on a missing `ml_dtypes`,
fixed in #554); the first green-collection run failed 27 of 367 fixtures on
unregistered `tessera_x86` / `tessera_rocm` dialects. No NVIDIA fixture is in
the failure set and this PR changes nothing for NVIDIA: the NVIDIA dialect is
off by default and its fixtures run under the separate `tessera-nvidia-opt`
driver (`%tnv`), not the `tessera-opt` binary this lane builds — the same
separation `test_target_ir_contract.py` already records as NVIDIA's documented
skip under Decision #19.

Architecture-specific reason this stays not-applicable rather than
follow-up-required: adding `TESSERA_BUILD_NVIDIA_BACKEND=ON` to the shared lane
would be actively harmful. With `ENABLE_CUDA` off — the only option on a
CUDA-less runner — `tools/tessera-opt/CMakeLists.txt:95-96` forces
`TESSERA_OPT_LEAN_ARTIFACT_DRIVER`, dropping core `TesseraIR`/`TesseraPasses`
and the x86 Target IR. That is the same trap documented for ROCm under this
sync key; NVIDIA's separate driver is the existing, correct answer. Revisit
only if a CUDA-capable runner joins CI, which would also be the point at which
sm_120 exact-device evidence becomes schedulable.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no CUDA package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **shared semantic
parity validated; CUDA physical follow-up required.** Explicit coefficients,
scheme/order, and per-axis spacing are compiler requirements. The absent
NVIDIA stencil/halo/boundary symbols are artifact-only rather than callable
Target records. No SM120 package or CUDA packet was added.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes plan establishes portable balanced-partition,
epsilon-qualified numeric, VJP, and softmax-merge oracle contracts. This PR
adds Phase-1 stats/merge/finalize references, the Phase-2 stdlib recurrence,
Phase-3 typed Graph/VJP/JVP contracts, and the Phase-4 content-addressed
Schedule→Tile artifact, but no SM120 package or CUDA evidence. NVIDIA must
later provide its own
stats-attention/merge Target consumer and exact-device packet; the small depth
shapes do not by themselves justify tensor-core lowering, and ROCm proof does
not transfer.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **shared MiniMax MSA
package lineage landed; SM120 consumption remains follow-up required.** x86
and gfx1151 now consume exact digest-bound MSA artifacts without Graph
redispatch. No CUDA package or SM120 evidence was added, and those schedules do
not transfer. NVIDIA must bind the same parent-digest contract to its canonical
NVVM/PTX package; DeepSeek MLA/DSA remain independently open.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; SM120 FP8/INT4 consumption remains follow-up required.** The
carrier preserves genuine checkpoint bytes and separate fp32 scales under a
content digest and forbids full-weight materialization. No CUDA package or
SM120 packet was added, and gfx1151 INT4 evidence does not transfer. NVIDIA
must bind this ABI to its architecture-owned FP8/INT4 package and exact-device
performance proof.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared analysis
contracts landed; SM120 consumption remains follow-up required.** Graph IR now
carries typed integer-affine plus exact modular/divisibility constraints into
the C++ Presburger consumer. The shared sharding layer has a fail-closed
replicated/tiled/partial-reduction fixed point and explicit reshard planner;
lowered `control_scan` owns shared recompute-all JVP/VJP products. This adds no
CUDA region product, reshard lowering, NCCL packet, or SM120 evidence; other
architecture results do not transfer.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; CUDA follow-up required.** The
tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No SM120 region product, physical producer
wiring, calibrated packet, or selection claim was added; architecture evidence
does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared native-product v3 and
automatic dependency consumption landed; SM120 remains follow-up required.**
Reduction and normalization now have truthful Schedule/Tile product carriers,
native JVP/VJP paths require cached tracer/AST differential proof for pure
programs, and Graph-derived R3 candidates consume compiler-generated edges.
No CUDA product child, physical dependency consumer, or SM120 packet was
added; x86/gfx1151 evidence does not transfer.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; SM120 remains follow-up
required.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare their Graph/Schedule/Tile/Target disposition and own
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. This adds no CUDA family package,
edge-consuming physical pipeline, selector promotion, or SM120 evidence.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **shared product
contracts landed; SM120 consumption remains open.** Graph IR now represents
the exact ISTFT spectrum/window product, and the general solver parent binds
residual plus solution/parameter JVP/VJP children under restarted GMRES with a
true-residual gate. The shared compiler now derives those five children for
typed pointwise, sum/mean, rank-2 matmul, bounded-dynamic/mixed-storage,
distinct parameter-space, and statically counted-region residual Graphs.
Pure scalar `if`/bounded-`while` predicates now have explicit compare/select
replay in the shared child contract. The AVX-512 and gfx1151 packages and
packets do not transfer to CUDA. NVIDIA still needs PTX children and independent SM120 correctness and
performance evidence.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **shared frontend and
family-plugin boundary landed; SM120 remains follow-up required.** Native
forward-product specialization now originates in tracer-produced canonical
Graph IR, and explicit family plugins own reduction, normalization, FFT, and
compound-spectral planning outside `JitFn`. General solver contracts bind exact
residual/JVP/VJP identities and can execute matrix-free reference products
without finite differences. This adds no CUDA family plugin, native package,
or SM120 evidence; the AST lane remains for unmigrated families.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP,
structured-region products, and typed point-to-point transport landed; SM120
evidence remains open.** Public JVP/jacfwd no longer substitutes finite
differences. Compiler forward mode carries primal/tangent state through bounded
SCF. `collective_permute` reaches the existing one-process/multi-device NCCL
launcher as grouped send/receive with an explicit peer map. SM120 still needs
architecture-owned JVP packages, subgroup communicators, and exact multi-GPU
correctness/performance packets.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; CUDA consumption is
follow-up required.** Portable tracing now emits bounded SCF, and the paired
compiler differentiates effect-safe single-block `if`, counted `for`, and
canonical bounded `while` with implicit captures. General residual execution
uses restarted GMRES/CG policy in shared IR. This adds no SM120 region executor,
solver package, checkpoint packet, or performance claim.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; CUDA behavior and evidence are unchanged.** Graph
IR now has one fail-closed, invalidatable shape/alias/liveness/memory-
dependence/activity analysis with C++ and Python query surfaces. Reverse AD and
await sinking consume it. This is target-independent legality infrastructure;
it transfers no SM120 schedule or performance evidence. Region-aware clients
and native CUDA overlap proof remain separately owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **shared affine,
compound-spectral, solver-product, and native-collective contracts landed;
SM120 consumption remains open.** Compound spectral Graph operations now own
direct tangent interfaces, including an exact ISTFT window-product carrier, and solver
artifacts distinguish JVP/non-transposed from VJP/transposed solves. The
multi-rank product accepts only a live NCCL hardware adapter, but no SM120
correctness/performance packet is claimed by this shared-contract slice.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **shared native-product
lineage landed; SM120 consumption is follow-up required.** The parent artifact
schema binds paired-JVP IR to immutable ordered child packages and detects
child substitution. Only x86/AVX-512 and ROCm/gfx1151 are executable in this
slice; no schedule or evidence transfers to CUDA. SM120 needs family-owned
product packages and exact-device numerical/performance packets before a
native JVP matrix row may be added.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; no CUDA evidence claim.** Canonical Graph records now carry
effect, alias, mutation, and stochastic identity; Python and C++ consume the
same fail-closed facts and internal calls reach a fixed point. Await sinking
uses that shared query. This changes scheduling legality only; SM120 still owns
native overlap execution and exact-device correctness/performance packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; NCCL/SM120 evidence remain open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured plan records; scalar CUDA-event
latency selects. Mock multi-rank execution transfers no performance claim.
SM120 still needs native NCCL stream/event binding and exact-device packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no CUDA selection claim.** R3 validates explicit
dependencies and calibration identity, uses deterministic critical-path/list
scheduling, and composes compute/memory/communication lanes with queue
serialization. Exact small DAGs and proven lower-bound losers may be pruned;
every estimate is promotion-ineligible and scalar
measured latency remains authoritative. SM120 needs its own calibrated vectors
and exact-device packet before using this analysis; R4 production consumption
does not transfer from another backend.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; CUDA evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. No gfx1151 or x86 timing,
queue, or resource identity transfers to SM120; CUDA event/activity providers
must populate their own provenance before R3 composition analysis can use it.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage and fail-closed await sinking landed; CUDA consumption remains
architecture-owned.** Python Schedule→Tile no longer emits internal
`tessera.queue.*` compatibility markers: async copies produce named tokens and
waits consume them. Collective awaits move only across operations proven
memory-effect-free; mutation, RNG, aliases/casts, regions, and ordered
collectives are barriers. Existing typed CUDA token lowering is unchanged;
SM120 needs independent executable overlap and exact-device proof.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **shared stochastic JVP
contract available; CUDA follow-up required.** Explicit key/counter Graph ops,
estimator provenance, dropout replay, fixed-key EGGROLL JVP, and derivative
proof obligations are target-independent. x86 and gfx1151 distribution kernels
do not transfer to CUDA; SM120 needs its own Philox distribution consumer,
compiler-JVP package, and exact-device proof.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
CUDA execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No PTX
package, selector, or SM120 evidence transfers; native JVP remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; NVIDIA physical consumption remains architecture-owned.**
The Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no PTX package or SM120 evidence; NVIDIA
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **shared schema
parity assessed; no CUDA physical change.** x86 now validates a closed Tile
family and registered Target marker before selecting its prebuilt AVX-512
image. NVIDIA may reuse the schema and fail-closed family discipline, but no
x86 ABI call, schedule, package, or Zen 5 evidence transfers to SM90/SM120.
The x86 backward-family allowance is narrowly one explicit forward-recompute
companion, not a general multi-carrier escape hatch. The canonical NVVM/PTX
image boundary remains NVIDIA-owned.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; NVIDIA physical consumption is follow-up required.** gfx1151 owns the
first GPU exact-device rank-1 proof and Zen 5 now owns an independent AVX-512
fp32 package; neither is a portable SM schedule. NVIDIA remains a lead
performance target: an SM-owned SGMV / `mma.sync` implementation and numerical
packet are open. The `s32` lane maps to native int8→s32 tensor cores and remains
a separate EGG expansion. W4 scalar-gather/member reconstruction passes
mock-mesh proof; native NCCL multi-rank execution and a target packet remain
open. Contract:
`docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
artifact discrimination adopted; AMD transports not applicable.** Advanced
collective artifacts now distinguish Copy Engine, GIN/RMA, and gfx1250 DDA and
bind target architecture plus selector evidence where required. These RCCL
lanes do not transfer to SM120. NVIDIA device-initiated communication remains
an architecture-owned NCCL follow-up with its own public API, legality, and
exact-device packet; it must not reuse an AMD transport or selector claim.
The shared Target dialect now registers explicit window-lifecycle and
put/signal/wait operations, but those records are RCCL GIN-gated and do not
imply NCCL device-initiated support on SM120.
The native multi-process harness binds only the RCCL GIN ABI; its launcher
metadata and evidence schema may be reused, but no AMD operation or result
transfers to NVIDIA's separately gated NCCL device-initiated lane.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **host adapter
and artifact contract landing; SM120 evidence open.** The C++ NCCL adapter now
executes all-reduce, reduce-scatter, all-gather, and grouped send/receive
all-to-all from an explicit communicator and CUDA stream instead of compiling
to successful no-ops. The shared Target artifact binds initiation,
registration, ordering, capture compatibility, backend/source identity, and a
capability digest. Shared communicator-property discovery, move-only symmetric
window ownership, and runtime-digest rejection are available, but still need a
CUDA-enabled build and SM120 packet. No AMD LSA, Copy Engine, GIN, or DDA claim
transfers.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; SM120 NCCL evidence open.** The legacy unregistered
`tessera.collective.*` markers are gone from active producers and fixtures.
Forward and adjoint passes emit registered futures, await their payloads, and
rewire SSA uses. Runtime topology validation now fails closed for unknown or
unsupported subgroup mesh axes, and native v1 forbids implicit non-fp32
conversion. Exact multi-GPU NCCL correctness/performance remains required; no
PTX selector or device claim changes.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **shared alias mapping
available; SM120 evidence open.** Five public reduction/broadcast aliases now
resolve to the registered all-reduce/all-gather transport; three sharding
entries remain compile-time placement/region contracts. `collective_permute`
correctly remains a distinct point-to-point gap rather than being mislabeled
as all-to-all. CUDA frontend capture and exact multi-GPU NCCL proof remain
architecture-owned; no SM120 claim transfers from portable execution.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded x86/ROCm
pilot landed; CUDA follow-up required.** The shared IFT chain now has a
content-addressed Schedule→Tile physical contract, and counted-region treeverse
can execute checkpoint replay before a row becomes eligible. SM120 has no
consumer for the diagonal-sqrt pilot, no general iterative solver, and no
complete-backward packet. This PR changes no PTX package, selector, policy, or
device claim.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
Graph/Tile/portable-Target contracts available; CUDA follow-up required.** Compiler activity,
effects, `stop_gradient`, stochastic rejection, and fail-closed region
adjoints are target-independent. The four collectives now lower into one
content-addressed asynchronous Target queue and execute through the shared
runtime-adapter ABI. SM120 still needs exact multi-GPU NCCL execution and a
device packet. No PTX selector, native performance, or device evidence is
claimed or transferred.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; no CUDA physical claim.** Graph and canonical-attention
integer bounds now consume signed `IntegerAttr` values, preventing MLIR 23
unsigned accessors from accepting negative schedules, seeds, cache windows, or
control bounds. Direct negative IR cases cover both dialects. No PTX ABI,
SM120 schedule, selector, package, or exact-device evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-1-2026-08-08` — **shared Graph contract
available; CUDA follow-up required.** Compiler spectral identity and the
FFT/RFFT/DCT transpose rules are target-independent and CPU-oracle proven.
SM120 still needs its own Schedule→Tile/native compound-backward package and
exact-device evidence; no CUDA support or performance claim follows from the
x86/gfx1151 carrier.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **SM120 follow-up
still required.** The bounded spectral-filter/convolution consumers and proof
land only for AVX-512 and gfx1151. No PTX image, schedule, correctness, or
performance evidence transfers to NVIDIA; its compound-backward path remains
fail-closed until an architecture-owned package lands.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR follow-up
available; no CUDA physical claim.** Compiler-owned linear transposition now
covers structural views, broadcast, and operand-wise matmul in both autodiff
passes with CPU numerical proof. SM120 backward packaging remains
architecture-owned; no CUDA image, selector, schedule, or device evidence is
transferred by this shared interface.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **SM90 and
SM120 proof separated; no CUDA physical change.** SM90 compile/artifact rows
are no longer added to SM120 runtime counts, and only exact-device statuses
close a hardware op×target grain. No package, selector, or device evidence is
changed.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **shared
fail-closed selection assessed; no CUDA package changes.** The x86 runtime and
native packager now honor `TESSERA_BUILD_DIR` and reject a missing selected
tree rather than loading a stale default image. NVIDIA keeps its own CUDA tool
and image discovery, and inherits no AVX-512 implementation or timing result.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **registry truth
adopted; no CUDA execution claim changes.** The standalone dashboard now
generates its counts, compiler-layer rollup, exact-target manifest summary,
and open queues from the live registries. It explicitly separates aggregate
best-available evidence from per-target support. SM120 still owns every CUDA
physical and benchmark follow-up in its manifest; x86 and gfx1151 TSOL or
Adafactor evidence does not transfer.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **shared artifact
follow-up required; no CUDA schedule transfers.** The target-neutral FFT
contract now binds logical/physical length, Hermitian layout, and packed-real
versus full-complex policy. Only x86 and gfx1151 own physical N/2 consumers and
evidence. SM120 must implement and measure its own real-transform package
before selecting the packed policy; it inherits no AVX-512 or RDNA schedule.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no CUDA physical change.** Shared compiler-test discovery now
accepts fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; SM120 packages, schedules, evidence, and selectors are
unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; CUDA physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no CUDA lowering
or SM120 evidence. They remain explicitly reference-only until a CUDA-owned
physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **shared dtype contract
assessed; CUDA follow-up required.** Physical binary math packages now require
matching input storage dtypes. The Zen 5 scan selector and gfx1151 HIP module
cache are architecture-owned and transfer no PTX schedule, dtype, or
performance claim. NVIDIA must run the reduced-storage and difficult-domain
corpus on its canonical CUDA math packages before claiming parity.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **shared semantic
contract adopted; physical consumer remains follow-up.** Bounded dynamic
dimensions, arbitrary axes, storage policy, and normalization are now explicit
before an exact TSOL specialization is emitted. Zen 5 and gfx1151 now consume
that wider contract, but their ABI and evidence do not transfer. NVIDIA still lacks the
prerequisite promoted FFT package, so Schedule→Tile lowering rejects CUDA
physical consumption and records no dtype, numerical, or performance claim.
The architecture-owned sequence remains: close canonical CUDA FFT, define an
SM120 workspace/residency ABI, implement the compound package, then gather
exact-device evidence.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared evidence schema is
executable on ROCm/x86; CUDA adoption remains follow-up.** The ROCm plan now records each
clock independently and forbids fallback relabeling. CUDA should adopt the same
provenance, validity, calibration, instrumentation, and verdict fields for host
wall, CUDA events, an architecture-qualified device clock, and CUPTI activity.
HIP `wall_clock64()`, `rtg_hsa_dispatch`, ROCprofiler, and gfx1151 evidence do
not transfer to SM120. CUDA clock choice and CUPTI/SM120 promotion remain
architecture-owned exact-device work.
The native-evidence extension adds content-digested provider captures,
clean-versus-instrumented image/resource comparisons, and exact-machine event
maps. These are shared evidence concepts only: ROCprofiler, RTG, Linux perf,
IBS, gfx1151, and Zen 5 records transfer no CUDA result. NVIDIA must populate
the corresponding fields from CUPTI activity/metrics/PC sampling on SM120.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared ODS vocabulary
adopted; CUDA physical execution remains follow-up.** The target-neutral
`schedule.spectral_program` and `tile.spectral_program_kernel` contract is
registered in production. NVIDIA still lacks a promoted canonical FFT package,
so it cannot consume the compound artifact or inherit ROCm/x86 evidence. A
future CUDA implementation must first close its FFT package gap, then bind its
own workspace/residency policy and SM120 device evidence.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **not applicable to
CUDA execution.** The content-addressed FFT vocabulary now carries gfx1151's
batched fused-LDS residency explicitly, but the HIP image dependency, AMD LDS
kernel, and WSL timing evidence establish no CUDA package or SM120 selector.
NVIDIA's architecture-owned FFT/TSOL follow-up is unchanged.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **shared DCT and
streaming policy adopted; CUDA physical follow-up unchanged.** DCT-I/II/III/IV
now carry distinct API, autodiff, Graph, Schedule, and Tile identities. The
target-neutral causal chunked-STFT state binds its policy digest and overlap
lineage, while centred streaming fails closed pending explicit lookahead.
NVIDIA still lacks the prerequisite CUDA FFT/compound package and inherits no
x86/gfx1151 physical or performance evidence. The length-one convolution and
one-sample STFT/ISTFT physical boundary repairs transfer no CUDA implementation
claim.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **not applicable to
NVIDIA codegen.** Centered Welford and the scalar boundary fixes alter ROCm C++
generators plus shared host-side atan2 quadrant semantics; no PTX, CUDA ABI,
math mode, or NVIDIA capability changed. NVIDIA must evaluate the same domains
on its own canonical math/reduction lanes.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; NVIDIA
still has no promoted canonical FFT package.** The ROCm opaque plan ABI and HIP
allocation policy are not transferred. A future CUDA package must define and
measure its own artifact-bound plan/workspace contract on NVIDIA hardware.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **not applicable to the unproven
CUDA lane; follow-up remains required.** The new cached Bluestein, Rader,
mixed-radix AVX-512 codelets, and rejected Bailey candidate are x86-owned.
They do not establish SM120 code generation or evidence. NVIDIA's existing
mixed-radix/Bluestein and exact-CUDA gaps remain unchanged.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **follow-up required;
radix-17 source changed without CUDA evidence.** The shared planner now admits
radix 17 and the CUDA generic-stage private array was widened accordingly, but
this Ubuntu/gfx1151 host cannot compile or execute the SM120 lane. NVIDIA must
compile and compare radix-17 direct execution against its prior rejection path
on CUDA before claiming parity. The expanded Schedule→Tile FFT identity remains
outside NVIDIA until its Bluestein gap closes.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **follow-up required; no support claim.**
ROCm corrected its public FFT authority to the proven Stockham/Bluestein
package and identified `schedule.fft`→launch Tile as the remaining shared
boundary. That shared content-addressed contract is now implemented and
consumed by x86/gfx1151, while NVIDIA remains deliberately outside its target
set. NVIDIA must join it only after its existing mixed-radix
hook gains Bluestein and passes exact CUDA/SM120 evidence; ROCm evidence does
not promote this lane.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **follow-up required — mixed-radix only, no Bluestein, unverified.**
Tessera's own FFT (Stockham, `TargetHooks/`) extends from powers of two to
every length: a generic radix-r stage for the odd small primes and Bluestein
for the rest. Shared contracts changed, so all four backends are affected:

* **Planning is now one implementation** (`TargetHooks/Common/FFTPlan.h`).
  CPU, AMD and NVIDIA each carried their own `while (n%4) ... while (n%2)`
  driver loop, and all three silently returned a HALF-FINISHED transform for
  any other N while reporting success. `LegalizeSpectral::pickRadixSequence`
  was a fourth copy, factoring over radices 7/5/3/4/2 and pushing a residual
  prime as a "stage" of that radix -- a stage nothing could execute.
* **Compiler routing was wrong independently of the kernels.**
  `LowerToTargetIR::stageSymbolFor` mapped every radix other than 4 to
  `ts_stockham_r2_*`, so a static N = 12 = 4x3 emitted a radix-2 call for a
  radix-3 stage. The runtime driver was correct; the compiler path was not, and
  direct driver tests could not see the difference.
* **New C ABI surface:** `ts_stockham_rn_<backend>(in, out, N, L, r, sign)`
  (note the extra radix argument, which r4/r2 do not take), plus
  `tessera.target_ir.stage_radices` carrying it, and a
  `tessera.target_ir.bluestein` marker routing those lengths to the driver.

NVIDIA gets the shared plan and the generic radix-r stage, so every
mixed-radix length is routed and emitted correctly. It does NOT get Bluestein.

There is no CUDA toolchain on the development box, so ~60 lines of device code
written for it could not be compiled, let alone checked against a reference.
Shipping unverifiable device code is the same unproven-claim pattern the silent
truncation was an instance of, so the driver DECLINES instead:
`ts_fft_supported_nvidia(N)` answers the question and the driver returns
without writing `d_out` rather than truncating.

**Nothing in this change has been compiled for NVIDIA.** The generic radix
kernel is a mechanical mirror of the gfx1151-verified AMD one, which lowers but
does not remove the risk. First task on the CUDA box: compile the `.cu`, run
the mixed-radix sizes against numpy, then implement and verify Bluestein.
Registering the lane as a `spectral_fft` arbiter candidate is a separate,
still-open item.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **follow-up required - scale operands changed, and NVIDIA has no FFT lane.**
PR #493 closed the Graph IR shape-rule registry: **303 declared / 6 deliberately
undeclared / 0 unexamined**, with the `MAX_UNCLASSIFIED` ratchet dropped 106 -> 0.
Shared contracts changed; all four backends are affected equally at the
reference level:

* **Result contracts.** Multi-result ops now emit every SSA result
  (`kv_cache.read -> (K, V)`, `top_k`, `qr`/`svd`/`lu`/`nonzero`), and tuple
  destructuring (`v, i = ...`) lowers. The emitter previously called the
  single-result `_infer_result_type`, so a declared multi-result contract
  stopped at Graph IR.
* **Stateful handles.** `!tessera.kv_cache` is now reachable from Python; the
  emitter had been printing `tensor<*x?>` for a type the ODS has always
  declared.
* **dtype policy.** An integer input to a float-producing op promotes to the
  declared `COMPUTE_FLOAT_DTYPE` (fp32) instead of NumPy's width-derived float
  (`cos(int8) -> f16`, `cos(int32) -> f64`); index/count results use a declared
  `INDEX_DTYPE`; complex is a LOGICAL dtype carried in an interleaved real pair,
  not a storage format.
* **Diagnostics.** The whole `GRAPH_IR_*` family (17 codes) is registered - the
  drift gate's scanner did not know the prefix, so it reported green while the
  family accumulated unregistered.

**This is the Python reference lane, not generated device code.** The NVFP4/MX matmul lane carried `scale_a`/`scale_b` as ATTRIBUTES holding SSA
names; they are now real operands 2 and 3. This corrects Graph IR toward what
NVIDIA already declared everywhere else: the ABI is
`tessera.nvidia.nvfp4.a_b_scale_a_scale_b_d_m_n_k.v1` and the kernel is
`tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d`, so Tile IR modelled the
scales as operands and only Graph IR demoted them. `nvidia_native.py`'s
packagers and its `requests_`/`supports_` predicates were updated; `bias` was
never affected (x86, ROCm and the unscaled NVIDIA lanes all append it as an
operand). Parity validated at the Python packaging level; **device evidence is
missing** - no exact-device run confirms the packaged buffer order end-to-end on
sm_120.

Recorded plainly: **complex FFT is REJECTED on nvidia_sm120**, because no NVIDIA
target declares an `fft` capability entry and zero NVIDIA source files mention
FFT at all. An earlier cut synthesised capability entries for absent ops and made
the sm_90 dashboard assert `fp8_e4m3` and `int8` FFT kernels - nine
`artifact_only` rows for a backend with no FFT. That was backed out. A target
with no `fft` entry is stating it has no `fft`.


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required; NVIDIA is the target where this matters most.**
The quantize family is now correctly declared as MULTI-RESULT `(codes, scale)`,
and `quantize_nvfp4` has its own rule because its scale is per-BLOCK (one per 16
elements along the last axis) rather than per-tensor — the micro-scaled form
Blackwell implements. A shared per-tensor rule would have misstated the format
for exactly the architecture that motivates it.
**The open gap is a backend-path one, not a shape-rule one.** The reference
returns codes as **f32** — fake-quant. `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1` and
`nvfp4` are canonical dtypes and the Graph IR type system can carry them, so
nothing in the compiler prevents real sub-byte storage; no lowering produces it.
NVIDIA owns this first: consumer/datacenter Blackwell has native FP8 and NVFP4,
so it is the one target where "the backend upcasts anyway" is NOT the answer.
The Target IR must carry fp8/fp4 as real storage into the mma path.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required, reference-level only.**
The shared reduced-precision policy changed: ops whose declared rule preserves
storage dtype now upcast reduced-precision inputs to f32, compute, and store
back. This repaired six ops whose INTERNAL arithmetic left fp16 range while
their answers fit easily — including `flash_attn` and `mla_decode`, both hot
SM120 paths, which previously returned float64 for f32 AND bf16 inputs.
**This is the Python reference lane, not generated CUDA.** The same hazard
class applies to NVIDIA kernels — a QK^T contraction overflowing fp16 before the
softmax rescales — and nothing here proves the generated kernels handle it.
NVIDIA owns verifying the accumulate-in-f32 contract on device; the reference
now states what the kernels must match.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **parity validated, and the prior NOT-VALIDATED status is now CLOSED.**
`MMAOp::verify()` now counts DATA operands, so the typed `tile.mma` fragment
form and the warp-spec `!tile.async_token` edge can coexist. NVIDIA needed the
same correction and had not received it: `NVIDIALowering.cpp` compared raw
`op->getNumOperands()` against 3/5 and then indexed raw operands, so the exact
typed-plus-token form this change unblocks would have hit `emitError` +
`signalPassFailure` during SM120 lowering — and operand 3 would have been the
token rather than an NVFP4 scale. Both the count and the indexing now use
`tessera::tile::dataOperands`.
**Verified by building it**, not by inference: ROCm and NVIDIA cannot both
register in one `tessera-opt`, so a second tree (`build-nvidia`,
`-DTESSERA_BUILD_NVIDIA_BACKEND=ON`) was configured and built. With
`tessera_nvidia` actually registered, the W0.9 parse/verify gate **passes for
sm90 / sm100 / sm120, plain and probe-annotated** — it had been SKIPPING every
run since PR #490. The contract test now discovers either build, so NVIDIA is
no longer silently unmeasured.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **follow-up required, NOT validated on NVIDIA.**
W0.9 added a real parse + dialect-load + verifier gate over every Target-IR
emitter, and it found that no Python-emitted Target IR was valid MLIR
(undialect-prefixed module attributes, an invented `<dialect>.func` container,
ops emitted with signatures their ODS rejects, and several undeclared op names).
Those defects were fixed and verified for `cpu`, `x86`, `rocm`, and `apple`.
**NVIDIA was skipped, deliberately and visibly**: `tessera_nvidia` is not
compiled into the default build, and failing it would have measured the build
config rather than the emitter. The shared fixes (module-attribute prefixing,
`func.func` container) apply to the NVIDIA lane too, but its dialect-specific
surface is unchecked — `TesseraNVIDIADialect.td` still carries 3 `AnyType` and
3 `Variadic<AnyType>` slots, and it may have the same undeclared-op class
(`tessera_nvidia.profiler_probe` in particular was not declared).
**Owning follow-up:** run `tests/unit/test_target_ir_contract.py` on a build
with `-DTESSERA_BUILD_NVIDIA_BACKEND=ON`; the gate skips rather than passes
today, so a green run there is not yet evidence.

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **follow-up
required, no NVIDIA contract change.** X86 adopted the shared rank-4 forward
and tensor backward loops and closed its optimizer adjoints. No Zen 5 ABI,
schedule, LSE policy, or timing transfers to SM120. NVIDIA retains direct
shared-loop forward/backward consumption, architecture-owned LSE selection,
and its remaining backward materializers.

## NVIDIA-SPINE-1: make the completed SM120 package the canonical default

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **landing.** The SM120
Graph/Schedule/Tile → NVVM/PTX image and launch-descriptor path was already
complete under NVIDIA-E2E-1/-2, but `canonical_compile()` did not select it by
default. NVIDIA now owns one `native_package_kind` / `package_native` entry
point; the shared driver no longer duplicates the vendor family-dispatch table,
and eligible static modules auto-promote when the complete SM120 toolchain is
available. Explicit `package_native = false` remains a stable opt-out.

Host-free focused validation covers canonical selection, typed package
production, runtime projection, and the native artifact contract. Existing
RTX 5070 Ti evidence remains the exact-device proof for the unchanged lowering,
PTX, ABI, and schedules; this slice changes selection authority, not emitted
code. The ROCm, Apple, and x86 selector reconciliations subsequently landed
under the same synchronization key. Apple keeps its explicit Value Target-IR
compatibility/probe route outside descriptor promotion. X86 has since separated canonical MLIR/native target
`x86` from its `x86_c` source candidate. NVIDIA already has that separation;
no PTX, ABI, schedule, or exact-device evidence changes.

APPLE-RASTER-1 subsequently consumed the shared map in emitted MSL and retained
row-major after mixed Apple7 timing. This is Apple-specific evidence, not an
NVIDIA selector or measurement result.

## NVIDIA-CALIB-1: supply the sm_120 corpus to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **superseded by the terminal
ROCm home-architecture rejection.** No NVIDIA arbiter-score adoption work remains.

The independent Zen 5 hierarchical T1 packet now also rejects T1 for latency
ranking (median rho -0.4062, 0/3 winner matches). Its x86 cache hierarchy,
bandwidths, candidates, and verdict do not transfer to SM120; NVIDIA's own
descriptor-complete correlation packet remains the only valid local decision.

**Correction that created this item.** `APPLE_AUDIT.md` originally scoped this
calibration to Apple alone, on the stated grounds that ROCm and NVIDIA kernels
"cannot be measured". That was false for NVIDIA: this backend already has a
committed, **consumed**, device-keyed `nvidia:sm_120` autotune corpus covering
64/256/512/1024/2048 square buckets plus fused GEMM and causal attention,
generated by `benchmarks/nvidia/record_autotune_corpus.py`. Excluding it would
have discarded the deepest per-shape latency evidence in the fleet.

**Historical calibration result and current subject.** The original two
static, device-free scores from
[`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8 — a step-distance locality histogram and an N-way bank-conflict
analyzer — were not retained after the step-distance line failed on its ROCm
home architecture. Do not tune or transplant that score. The current subject is
the shared T1 GEMM model: symbolic tile identities, capacity-bounded LRU reuse,
cache-derived DRAM traffic, and explicit target compute/bandwidth inputs
([`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §3).

**NVIDIA's role: shape depth.** The committed corpus already varies the shape
axis within GEMM and attention at fixed op kind, which is exactly the axis Apple
cannot supply and the one a cache/reuse score most needs to be tested against —
locality changes with shape at constant op. Apple supplies op breadth
(`APPLE-CALIB-1`); ROCm supplies a second, independent architecture
(`ROCM-CALIB-1`). Fitting on any one of the three reproduces the single-arch
overfit the assessment records for NeuSight.

**Note on translating the retired bank metric.** It was derived for AMD LDS with a known
bank count and a wave64 4-phase access pattern (survey §5.1). CUDA shared memory
is 32-bank and warp-synchronous; the *method* transfers but every constant must
be re-derived for sm_120 before a conflict number here means anything. Do not
port AMD constants.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 tested the metric where it
originated and reproduced 0/6 committed gfx1151 winners (median rho -0.1381, 0%
positive). The agreed home-architecture failure rule ends this latency-ranking
line without coefficient or target retuning. NVIDIA therefore owes no sm_120
promotion analysis for this score; CUDA-specific bank diagnostics remain a
separate future model and must not inherit AMD constants.

**Live-input update (2026-07-30).** The owning RTX 5070 Ti now reports through
CUDA 13.3 `cudaDeviceGetAttribute`: 70 SMs, 2.497 GHz core, 14.001 GHz memory,
a 256-bit bus, and `cudaDevAttrL2CacheSize = 50,331,648` bytes (48 MiB). The
registry consumes the measured L2 capacity; the observed memory clock and bus
corroborate the existing 896 GB/s peak-bandwidth derivation.

**Counter rerun (2026-07-30).** Nsight Compute 2026.2.1 is now installed on
the owning WSL host. A resource-only capture of the existing 512³ f16
production-route launcher retained `/tmp/nvidia-calib-sm120-2026-07-30.ncu-rep`
(`sha256:e0d0a1e2650dbfc1da921f72ebf0f36c27a1ec1165b278c51905336d02d6c79d`).
For the two Tile implementations, `tessera_tile_matmul_direct_f16` reported
97.65% L2-sector hit rate and 1,273,856 DRAM bytes, while
`tessera_tile_matmul_shared_f16` reported 91.24% and 1,101,312 bytes. This
confirms that CUDA counter collection works and that the two schedules have
distinct cache/traffic behavior. The profiler's replay duration is explicitly
not timing evidence, and these are different schedule shapes rather than a
controlled candidate-rank packet; neither value changes selection or supplies
a T1 correlation verdict.

**Correlation verdict: not identifiable from the committed corpus.** The
corpus retains latency keyed by named implementation but does not serialize the
candidate Tile M/N/K/raster descriptor needed to replay T1 for each competitor.
It therefore cannot supply an honest within-shape rank correlation, even with
the now-measured cache input. Retain T1 solely as a legal pruning estimator;
measured latency remains selection authority. A future CUDA corpus revision
must persist candidate schedule descriptors and profiler counter availability
before this item can produce a correlation verdict. Do not fabricate a rank or
revive the rejected AMD step-distance metric.

## NVIDIA-RASTER-1: consume the shared block-rasterization contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **follow-up required, owning
host NR2 Pro (RTX 5070 Ti, sm_120).**

**Shared contract changed.** Schedule IR gained two attrs and two knobs —
`raster_order` (`row_major` | `column_major` | `grouped_m` | `grouped_n`) and
`raster_group` — carried on `schedule.tile` and `schedule.knob`, mirrored by
`TuningConfig.raster_order`/`raster_group` and persisted in the SQLite tuning
cache. The order is a *permutation of block ids onto the tile grid*, defined in
the arch-neutral `compiler/tile_rasterization.py` with a `remap()` reference, an
`emit_c()` snippet valid identically under CUDA and HIP, and `is_bijection()` as
a total hardware-free oracle. Rationale and the 35%→72% L2 figure that motivated
it: [`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**Implementation landed (2026-07-30); selection remains open.** The SM120
`mma.sync` fused-GEMM and gated-matmul emitters now consume `raster_order` and
`raster_group`. `row_major` retains their established 2-D launch and direct
`blockIdx.x` / `blockIdx.y` coordinate arithmetic. Non-default orders flatten
the same block count to one dimension and inject the shared `emit_c()` mapping,
including ragged final panels. The compiled-artifact cache key includes both
knobs, so a swizzled binary can never alias a row-major one. Focused host-free
tests cover source selection and the shared permutation oracle; exact RTX 5070
Ti execute-and-compare covered grouped-M fused GEMM and grouped-N gated matmul
on ragged dimensions. No selector change is implied.

**Why it did not land in the contract PR.** Changing a hardware-verified
`mma.sync` kernel without sm_120 silicon to measure the result would be an
unverified edit to a proven path for no demonstrable gain. `row_major` is the
default and reproduces the existing index arithmetic exactly, so today's emitted
code is byte-identical.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property over ragged grids and **compiles the emitted C
with host clang, running it against the Python reference for every block id**
under `-Wall -Wshadow -Werror`. That covers the arithmetic and the emission's
scoping, on any host.

**Remaining exact-device evidence.** Whether a swizzle moves sm_120 latency, and
at which `raster_group`, for the GEMM shape buckets in the perf ratchet. This
needs a committed repeated-median CUDA-event matrix plus `ncu` L2 hit-rate
deltas on the NR2 Pro. Until then the axis is **carried, not selected**. T1 can
score the order symbolically, but it has not earned an sm_120 raster retain
verdict and cannot promote a choice.

## NVIDIA-AOT-1: decide whether NVRTC needs a precompiled peer — complete

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **follow-up required**.
Apple added `apple_gpu_air`, a precompiled-artifact lane behind the shared
`register_compiler(target, compile_fn)` seam, measured against its compile-on-
launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x; host-wall
timing on Apple M1 Max, not device-event evidence). NVIDIA is the backend
closest to Apple's position, not a distant one: its device code is NVRTC-
compiled at load (`nvrtc_jit.cpp`; runtime.py describes the mma.sync lane as
NVRTC-compiled for the device arch) and `runtime.py` has no cubin/fatbin
precompiled lane. So the AOT-vs-JIT question is genuinely open here. An earlier
version of this note said CUDA had 'nothing to catch up on' because
`emit/nvidia_cuda.py` contains no nvrtc reference — that was inferred from
absence of evidence in one file and is withdrawn. Follow-up: decide whether
SM120 wants a precompiled artifact lane, and if it is measured, reuse
benchmarks/apple_gpu/benchmark_aot_vs_jit.py *with its cache control* (a never-
before-compiled kernel per sample) — the driver's own cache is what made the
first Apple number 13x too good. No shared IR, ABI, dtype/op registration, or
numerical contract changed.

**Decision (2026-07-30): a precompiled peer is warranted.** The exact SM120
probe [`benchmark_aot_vs_jit.cu`](../../../../benchmarks/nvidia/benchmark_aot_vs_jit.cu)
uses a unique CUDA source and entry symbol for every sample, so neither NVRTC
nor the driver can serve a prior module. Both lanes load, launch, and verify the
same device result. On the RTX 5070 Ti (CUDA 13.3, driver 610.62), seven-sample
medians were **18.266 ms** for NVRTC compile + module load + launch and
**0.867 ms** for a precompiled cubin load + launch: **17.399 ms** saved per cold
request. Offline cubin construction was **173.004 ms**, amortizing after about
**10 cold launches**. The retained packet is
[`nvidia_sm120_aot_vs_jit_2026_07_30.json`](../../../../benchmarks/baselines/nvidia_sm120_aot_vs_jit_2026_07_30.json).

This closes the decision, not productization: a follow-on must add a versioned
SM120 cubin/fatbin artifact to the native package/runtime seam, preserve the
current NVRTC fallback for unsupported or stale artifacts, and execute-compare
the production matmul ABI before any selector promotion.

**Productization landed (2026-07-30).** `libtessera_nvidia_gemm.so` now ships
the first package-owned precompiled peer: the versioned
`tessera_nvidia_mma_f16_sm120_v1.cubin` beside the runtime image. Its canonical
`.cu` input is also generated into the library as the NVRTC fallback source, so
the AOT and JIT lanes have one kernel body, entry symbol, and physical
`A:u16[M,K], B:u16[K,N], D:f32[M,N], M,N,K` ABI. The loader admits the cubin
only on exact CC 12.0 and CUDA-driver >= 13000; absent, corrupt, incompatible,
or explicitly disabled artifacts use NVRTC. `TESSERA_NVIDIA_AOT_MODE=require`
is a strict deployment check and never falls back silently. Fresh-process RTX
5070 Ti execution proves forced-AOT, forced-NVRTC, and missing-artifact
fallback are numerically equivalent on ragged `17x31x9` f16 GEMM; the version
and canonical-source SHA are queried from the shipped C ABI. This is NVIDIA
runtime packaging only: no shared IR/ABI, selector, Apple, or ROCm contract
changed, so sibling backend plan changes are not applicable.

Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` moves the last 43
self-resolving test files onto the shared `tests/_support/compiler_tool.py`
driver contract, adds `--pass-pipeline=` inner-pass capability checking, and
folds `CompilerToolchain` onto one resolver and one capability check. NVIDIA is
**not applicable** for an architecture-specific reason: the NVIDIA lit and
compiler lanes drive the *separate* `tessera-nvidia-opt` binary through
`TESSERA_NVIDIA_OPT` / `CompilerToolchain.require_nvidia_opt` (the `%tnv`
substitution), which this resolver does not govern and which this change leaves
byte-for-byte untouched — `require_nvidia_opt` keeps its own `_tool_path`
lookup and its own skip. No CUDA registration, PTX or SM120 schedule, runtime
ABI, selector, or device evidence changed, and **no exact-device evidence is
claimed or required**. Should the NVIDIA lane later want the same
build-capability skip behaviour for `tessera-nvidia-opt`, that is a separate
follow-up owned by this plan, not a debt created here.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` validates that the shared
BF16 attention carrier and canonical forward/backward loop contracts can be
consumed by a second physical backend. ROCm now has exact ragged-GQA
bias+softcap+causal-window+dropout forward proof and deterministic five-entry
backward proof on gfx1151, with dedicated resident BF16 timing ratchets. This
is parity validation at the shared semantic boundary only. AMD BF16 WMMA,
LDS scheduling, HSACO packaging, HIP workspace and launch ABI, numerical
evidence, and timing do not transfer to CUDA; NVIDIA retains its own SM120
BF16 package and exact-device evidence requirements.

Cross-backend sync `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` is **closed**.
The shared lit resolver now accepts `TESSERA_OPT_BIN`, `TESSERA_OPT_PATH`, and
`TESSERA_OPT_CPP` after the canonical `TESSERA_OPT` override, and the validation
script forwards its selected binary through that contract. Exact gfx1151
verification proves the full ROCm driver, legitimate lean ROCm artifact
driver, conflict rejection, both named streaming-attention fixtures, the
seven-fixture filter, and the complete 50-test ROCm backend lit suite. This is
shared test/build infrastructure only; no CUDA registration, PTX schedule,
runtime ABI, device evidence, or selector changes.

Cross-backend sync `LSE-CHECKPOINT-CONTRACT-2026-07-27` lands the real shared
checkpoint vocabulary: explicit memref source/destination, SSA row offset,
identity, memory space, lifetime scope, cache policy, and read/write effects.
Default forward lowering no longer emits a destination-less save. ROCm
validates saved versus recompute on gfx1151 and retains the provisional
128+ policy, but the newer dual-clock packet is explicitly fail-closed on WSL:
HIP events are positive yet non-transferable, and FP16 at 256 is not a stable
saved winner. Bare-metal gfx1151 confirmation remains required. NVIDIA is
**follow-up required**: consume the same shared contract, measure its own CUDA
forward-store/backward-load package, and retain or replace its zero-workspace
policy using exact SM120 evidence. AMD WMMA, HSACO size, threshold, and WSL
host-wall results do not transfer.

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` makes ROCm gfx1151 the
first direct physical consumer of the shared tensor-valued attention backward
phase loops. NVIDIA remains **follow-up required** to validate the same
dQ/split-dK/dV/fixed-reduction contract and map it to a CUDA-owned package.
The AMD WMMA schedule, five-entry HSACO, HIP launch workspace, gradient
evidence, and host-wall timing do not transfer. No shared IR or NVIDIA
capability state changed in this ROCm-owned closure.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the deterministic split/reduced backward contract as tensor-valued
shared `scf.for` bodies with explicit dQ ownership, split dK/dV workspace
tensors, and ascending reduction. Registered shared score-bias and softcap
operations now preserve `softcap(scale*QK^T + bias)` inside the forward
KV-block recurrence, including rank-4 per-head bias. NVIDIA is **follow-up
required** to consume these phase operations through its SM120 package and
direct forward schedule. AMD HIP ABI code, HSACO, exact-device gradients, and
resident timing do not transfer.

Cross-backend sync `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified
split count, launch-owned workspace, block-loop metadata, ascending reduction
order, and canonical `softcap(scale*QK^T + bias)` semantics to the shared
carrier/oracle. NVIDIA is **follow-up required** to consume this form through
its SM120 schedule and validate dropout replay; AMD code and evidence do not
transfer.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The shared training spine now defines content-addressed logical-buffer lineage,
mutation identity, and typed Schedule→Tile contracts for Lion VJP,
factored/full Adafactor VJP, and sequence-mixer backward. **NVIDIA outcome:
follow-up required.** Existing SM120 packages remain NVIDIA-owned and do not
yet consume these exact artifacts. CUDA evidence and CUDA-owned buffer bindings
remain required.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` is not applicable
to NVIDIA physical execution. It adds a ROCm-owned five-entry HSACO and
gfx1151 split/reduced launch workspace without changing the shared launch
descriptor schema or canonical backward loop. AMD WMMA kernels, workspace
topology, exact-device gradients, timings, and selector state do not transfer.

The ROCm optimized-attention feature follow-up under
`ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` adds AMD-only deterministic dropout
replay and combined bias+softcap consumption to the gfx1151 WMMA schedule,
plus a host-wall resident performance ratchet. It changes no shared carrier,
ABI, NVIDIA Target IR, CUDA schedule, capability, or selector. NVIDIA parity
at the semantic carrier remains validated independently; AMD counter code,
HSACO evidence, and WSL timing do not transfer.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` removes the last active
shared and ROCm `#tile.buffer_ref` compatibility readers after their fixtures
migrate to `!tile.buffer`; the deprecated attribute is parser-only. NVIDIA was
already SSA-only, so its SMEM/TMEM schedule and evidence are unchanged. The
shared ReplaySSM lifecycle schema now keys Apple and ROCm resident ABIs while
preserving session-private ring ownership, flush/rollback, ordered submission,
and drain-before-release. MoE metadata now owns launch-lifetime workspace and
can bind a canonical NCCL/RCCL rank/device fingerprint. NVIDIA consumes the
same local descriptor as before; no CUDA schedule, selector, or timing changes.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` lands an
AMD-owned consumer, native HSACO package, descriptor, and exact gfx1151 proof
for the already-shared `tile.attention_kernel` contract, plus a direct
correctness consumer for `tile.attention_backward_kernel`. NVIDIA parity at the
shared semantic carrier remains validated by its existing SM120 forward and
backward packages. ROCm's wave32 WMMA descriptor, LDS allocation, HIP ABI,
resource counts, timings, selector boundary, and direct scalar recurrence do
not transfer to CUDA. The ROCm v2 benchmark's operation-total and resident
synchronized HIP host-wall domains do not replace CUDA-event or CUDA
end-to-end evidence. No NVIDIA plan state or exact-device claim changes.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` lands the AMD consumer of
the already-shared `!tile.buffer`, `!tile.async_token`, and
`!tile.pipeline_state` ownership vocabulary. It changes no shared operation,
type, verifier, ABI, or NVIDIA lowering. NVIDIA parity is therefore validated
at the existing SSA contract: WarpSpecialization continues to own SMEM/TMEM,
TMA/mbarrier, and architecture-specific pipeline mechanics. AMD LDS layouts,
waitcnt/s_barrier semantics, gfx1151 evidence, compiler timings, and selectors
do not transfer to CUDA or SM120; no NVIDIA follow-up is required.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` expands terminal
storage legalization without making sub-byte storage global. For `nvidia_sm120`,
the pass now proves the complete operation-specific consumer before stamping a
physical pack: packed load to ordinary store (explicit unpack/format
conversion), matching unscaled packed-load/store round trips, and packed
matmul whose A/B MMA descriptor agrees with the logical storage format.
Orphan or mixed-use packed loads, descriptor disagreement, arbitrary
operations, and the public shape-preserving Graph quantize/dequantize ABI stay
logical. The standalone empty-target transform remains available for explicit
IR inspection; named pipelines use the capability decision. Apple and ROCm
retain architecture-owned physical schedules; their evidence is not inferred
from SM120. The deprecated `#tile.buffer_ref` attribute is parser-only and no
active pass consumes it. No selector or timing disposition changes.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`.
The loop carries the FP32 output accumulator, running maximum, normalization
sum, producer and consumer `!tile.pipeline_state` values, and an absolute
boundary offset. Each block consumes K and V through typed async tokens; the
online update now takes V explicitly, while causal/window/ragged masking and
counter-based dropout consume the loop offset rather than replaying block zero.
NVIDIA TMA descriptor hoisting traces each block slice to its kernel argument
and retains typed coordinates plus logical source extents, enabling
out-of-bounds zero fill for the ragged tail. WarpSpecialization no longer emits
name-based `#tile.buffer_ref` or annotation-only `#tile.pipeline_state`
metadata, and Schedule→Tile consumes structured per-operand `#tile.layout`
directly. The SM90 structural pipeline is lit-green. Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` adds shared rank-4 batch/head
distribution and proves a direct ROCm consumer. A direct NVIDIA
Target-IR/runtime consumer of the shared loop remains open; gfx1151 LDS/WMMA
schedules, resources, wall timing, and selector evidence do not transfer.
Existing launch-level attention images and selectors are unchanged.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` is **landing**, owned by
NVIDIA under the `NVIDIA-E2E-2` continuation. The shared compiler now forms a
target-neutral M/N/K `scf.for` nest with FP32/INT32 loop-carried accumulation,
zero-pad ragged guards, structured copy layouts, asynchronous SSA
dependencies, and threaded `!tile.pipeline_state`. The SM120 launch-level
FP16/BF16/TF32 images serialize the same `tile_m/tile_n/tile_k` contract and
reject descriptor disagreement before NVIDIA materialization. Two exact RTX
5070 Ti SM120 runs each pass all 12 FP16/BF16/TF32 square, rectangular,
ragged-K, and fully fragment-misaligned rows with FP32 accumulation,
matmul→bias→activation→residual ordering retained by the shared epilogue
contract, warm image/descriptor identity, and numerical comparison. The
checked-in 12-row repeated-median packet discards the first complete call,
amortizes 1,000 resident launches and 50 complete calls, and retains 31
interleaved observations per cohort in both timing domains. Every row meets the
4% two-run WSL gate (maximum device-event delta 1.39%, maximum end-to-end delta
3.47%); shared FP16/BF16 uses 42 registers and 10 active blocks/SM, direct TF32
uses 38 registers and 24 active blocks/SM, and all rows retain zero local
memory and zero spills. INT8 and packed formats remain follow-on after the
ordinary loop is stable. WSL timing is selector-ineligible and no selector
changes in this slice.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` is **parity validated**
for NVIDIA. The only shared edit preserves the existing canonical
ragged-zero-fill guarantee across `tessera.matmul` → `tile.mma`; NVIDIA's
already-proven SM120 consumer and twelve-row packet are unchanged. AMD LDS,
wait/barrier, WMMA, HSACO resource, and gfx1151 wall-clock evidence do not
transfer. No NVIDIA route, capability, execution state, or selector changes.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: retired eleven
never-runnable CUDA13 pseudo-IR fixtures whose undefined `tessera_opt_built`
feature masked stale CLI options and unregistered operations. Core named
pipeline aliases now run in the ordinary LLVM lit lane; typed
Tile→NVIDIA→NVVM contracts remain owned by the NVIDIA backend lit suite.
`validate_nvcc_compile.py` now labels and runs its handwritten instruction
catalog strictly as a CUDA-toolchain probe, not evidence of Tessera emission.
The two integrated core+NVIDIA control-flow fixtures retain their precise
backend gate and still require the CUDA-enabled build on the NVIDIA host.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: shared
compiler-owner markers now report foreign compiler proofs as skipped with the
required Apple, CUDA, ROCm, X86, or AVX512 system and a per-system count. This
is test-harness observability only; NVIDIA compiler ownership, CUDA evidence,
and selector state are unchanged.

This is the execution plan for evaluating, repairing, and then restructuring
the CUDA compiler tests on the NVIDIA box. It complements
[`NVIDIA_AUDIT.md`](NVIDIA_AUDIT.md); it does not reopen completed sm_120 feature
work unless a test exposes a real defect.

Baseline state on the NVIDIA box (2026-07-15, commit `ecf9483f`):

- The repository collects **264 exact-device CUDA tests** under
  `pytest -m hardware_nvidia`: **246 correctness** cases and **18 measured
  performance** cases. Eight of the correctness cases also require external
  compiler tools.
- The required CPU PR lane now excludes hardware and measured-performance
  states while retaining host-free CUDA emit, selector, validation, rejection,
  registry, and source-contract tests.
- Live CUDA tests carry `hardware_nvidia`; measured tests additionally carry
  `performance`; compiler/toolchain crossings carry `compiler_tool`.
- The WSL NVIDIA host is an RTX 5070 Ti (UUID
  `GPU-5072cda5-509a-008c-93c8-dc06e105f307`, CC 12.0), driver 610.62, CUDA
  13.3.73, LLVM/MLIR 23, and Python 3.14.4. At collection it was idle;
  observed graphics clock/power were 375 MHz / 23.18 W with a 300 W limit.
- The compiler-artifact lit suite passed **19/19**. The exact-device
  non-performance lane passed twice: **224 selected, 0 failed, 0 errored, 1
  Apple-only skip** (54.783 s and 53.658 s). This covers the required
  execute/compare layer, but it does not constitute performance evidence.
- The serial measured lane passed **18 CUDA tests** (plus the same unrelated
  Apple-only collection skip; JUnit: `/tmp/nvidia-performance-ecf9483f.xml`).
  The production hot-path ratchet, device-resident event timing, convolution
  routes, and Tile/autotune selection were included. Verbose `ptxas` and
  `cuobjdump --dump-resource-usage` for the compiler-produced Tile kernels
  report zero spills: 36 registers and no shared/local memory for direct,
  GELU, and SiLU; 42 registers and 2 KiB static shared memory from `ptxas`
  (3 KiB in the cubin resource table) for shared-staging f32/bf16/bias-ReLU.
  Nsight Compute 2026.2.1 profiled the f16 device-resident GEMM proof: the
  19x13x29 case launches four one-warp blocks (2x2 grid), uses 40
  registers/thread and no kernel shared memory, and reports 50% theoretical
  versus 2.08% achieved occupancy. The low achieved value is expected for this
  deliberately tiny ABI fixture: four blocks cannot occupy all 70 SMs. It is
  launch-shape evidence, not a production-throughput claim.
- Nsight captures for the production hot-path, convolution-route, and Tile
  schedule tests are retained under `/tmp/nvidia-*-ncu.ncu-rep`. Hot-path
  GEMM/fused/attention use 40 registers/thread and no shared memory (the
  attention kernel has 22.92% theoretical occupancy); fp quantization uses 18
  registers/thread, no shared memory, and 74.36% achieved occupancy. The
  direct/shared convolution routes use 40/30 registers and 0/32 bytes static
  shared memory, respectively. The selected Tile candidates report direct:
  36 registers, 0 shared, 50% theoretical occupancy; shared: 42 registers,
  2.05 KiB shared, 83.33% theoretical occupancy. The measured achieved
  occupancies are shape-dependent and low for the micro-grid fixtures, so they
  are resource evidence rather than selector-retuning evidence.
- The hot-path ratchet intentionally fails when run under Nsight Compute:
  profiler replay raises the wall-clock samples above its uninstrumented
  repeated-median caps. Its ordinary serial run remains the timing proof; the
  Nsight run is resource-only and must never update or relax the ratchet.
- The serving benchmark completed 20-repetition device-event and end-to-end
  rows for ReplaySSM `1x128x64`/`1x256x128` and fused/staged paged-KV decode
  at 128/512/2048 tokens (`/tmp/nvidia-sm120-serving-test5.json`). A temporary
  candidate corpus records both timing domains for rectangular `128x256x64`
  and ragged `127x259x63` f16 GEMM: end-to-end selects shared Tile (0.506 and
  0.490 ms), while device timing selects direct Tile (0.00657 and 0.00703 ms).
  No committed selector or timing cap changed.
- Nsight reduction coverage passed 49 native cases: f32/f16 kernels use 28
  registers and 1.02 KiB static shared memory with 100% theoretical occupancy.
  MoE transport passed three native cases: gather/combine use 20 registers and
  no shared memory; grouped GEMM uses 40 registers and no shared memory. The
  two live MoE transport tests were missing `hardware_nvidia` and are now in
  the canonical exact-device collection; its host-only rejection test remains
  unmarked. Their repeated-median rows and resource evidence are now committed
  under the TEST-5 baselines described below.
- NVIDIA-TEST-4 now has a shared storage/accumulation tolerance contract for
  f32, f16, bf16, TF32, FP8, int8, and NVFP4 semantics. The shipped MMA and
  compiler-produced Tile proofs consume that contract; exact integer/NVFP4
  contracts remain bit-exact. The reduction matrix now also proves f16/f32
  non-finite propagation and rejects empty, rank-invalid, unsupported-storage,
  and unknown-operation contracts before launch. Two WSL exact-device runs
  recorded 243 tests with zero failures/errors (one expected Apple-only skip).
- NVIDIA-TEST-5 now productizes repeated-median reduction and MoE transport
  rows in `record_reduction_transport_baseline.py`: every route records both
  end-to-end and CUDA-event timing through the production generated kernel.
  Two 20-sample sm_120 runs were recorded, the committed ratchet baseline
  covers reduction sum/mean/max plus MoE dispatch/combine/grouped-GEMM, and
  the first expanded serial performance lane passed 19 tests (one expected
  skip). The wider corpus and parsed resource evidence have since landed.
- The TEST-5 D2 corpus now includes measured square `512x512x512`, rectangular
  `128x256x64`, and ragged `127x259x63` f16/bf16 GEMM rows in both timing
  domains, alongside fused GELU, forward attention, gated MLP, and convolution
  routes. Two 20-sample WSL runs were taken before retaining the second corpus;
  the initial end-to-end winners varied between runs, so no selector was
  promoted from that evidence.
  Serving was likewise refreshed from two 20-sample runs for ReplaySSM and
  fused/staged paged-KV at 128/512/2048 tokens, retaining device-event and
  end-to-end medians separately.
- **NVIDIA-TEST-5 is closed (2026-07-16).** Two fresh high-sample sweeps
  (50 end-to-end repetitions after 10 warmups; 200 device-event repetitions
  after 20 warmups) converge for all 20 retained D2 rows under the declared 3%
  noise policy. Every row is selector-eligible only because both runs share a
  near-winner consensus and the selected route has a committed resource
  fingerprint. Backward attention adds regular `1x8x128x64` and ragged
  `1x8x257x64` dual-domain ratchets. Parsed Nsight evidence records registers,
  static/dynamic shared memory, theoretical/achieved occupancy, and explicit
  local-load/store spill counters for GEMM/Tile, fused and forward/backward
  attention, convolution, reductions, MoE transport, paged-KV, and ReplaySSM.
  The backward VJP uses 48 registers and measurable local-memory traffic; this
  is retained evidence, not hidden by a zero-spill claim. All other selected
  rows in the resource manifest recorded zero local spill traffic. The final
  serial performance lane passed 20 tests with one expected Apple-only skip.
- NVIDIA-TEST-6 has begun with `tests/_support/nvidia.py` (with a retained
  `tests/unit/_nvidia_testutil.py` compatibility import): it centralizes
  CUDA-toolchain, MMA-runtime, and bare CUDA-host probes without conflating
  their skip semantics, and supplies a common native-provenance assertion.
  The MoE transport, reductions, paged-KV, and ReplaySSM families migrated in
  the first batch; 70 focused tests passed and the canonical device collection
  remains 243 nodes. This is NVIDIA-only test infrastructure; Apple and ROCm
  plan states are unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: NVIDIA exact-device parity is
  now validated on the RTX 5070 Ti after the shared LLVM/MLIR 23 migration.
  A clean `sm_120a` build required the MLIR bytecode interface include and the
  `NVVM::Barrier0Op` to `NVVM::BarrierOp` API migration. NVIDIA lit passes
  19/19, two stable collections contain the same 268 nodes, the host-free
  compiler-artifact proof passes, exact-device correctness passes 248/248
  twice, TEST-4/TEST-6 focused gates pass 190/190, and the isolated TEST-5
  lane passes 20/20. Explicit Tile tool paths now take precedence over stale
  build-tree binaries. ROCm receives only the LLVM 23 lit-shell compatibility
  update; Apple has no affected physical schedule or runtime contract.
- **NVIDIA-TEST-7 is closed as local WSL release ownership; GitHub runners are
  intentionally not used.** The release command exposes independent `cpu`,
  `compiler`, `device`, and `performance` layers, rejects overlapping runs with
  a host lock, writes a fail-closed status record, retains timestamped machine,
  JUnit, and baseline bundles, and keeps performance serial. The finalized
  all-layer invocation passed 410 host-free/shared-registry tests (one explicit
  skip), 20/20 lit, 1/1 compiler artifact, 268/268 correctness twice, and 20/20
  performance. Its retained bundle is
  `artifacts/nvidia-release/20260717T003224Z-18866bbb/all/`.
- The second batch removed the same local MMA-runtime probe from norm, softmax,
  matmul-ReLU, matmul-softmax, compiled KV-cache, forward/backward Flash
  Attention, and convolution tests. Their 89 focused exact-device tests passed
  on the RTX 5070 Ti. Specialized compiler and Tile availability probes remain
  local until their stronger capability contracts can be preserved explicitly.
- The third batch migrated control flow, DeltaNet, dequant GEMM, FP quant,
  local collectives, optimizers, positional encoding, and SSM to the shared
  MMA-runtime probe. It also classified their live CUDA tests with
  `hardware_nvidia` while leaving host-only negative tests unmarked. The 32
  focused tests passed; collection increased from 243 to 264 exact-device
  nodes (246 correctness, 18 performance).
- The next helper-deduplication batch replaced private ordinary MMA-runtime
  probes in linear attention, MLA decode, and sparse attention with the shared
  capability-specific helper. The E3 hand-tuned GEMM proof now uses the shared
  MMA-plus-PTX-launch predicate; Tile tool/runtime checks remain local because
  they prove a stronger compiler-path capability.
- The second physical relocation split the mixed NVIDIA MMA launch file into
  two host-free execution-matrix contracts and five exact-device launch/JIT
  proofs under `tests/device/nvidia/`. The mapped cohort passed 21 focused
  tests, 19/19 compiler lit plus its compiler pytest contract, exact-device
  correctness twice (246 passed, one Apple-only skip each), and serial
  performance (18 passed, one Apple-only skip).
- The third physical relocation moved the two device-only DSA sparse-attention
  proofs to `tests/device/nvidia/test_sparse_attention.py`. Its node map,
  focused execute/compare run, compiler artifact lane, two exact-device runs
  (246 passed, one Apple-only skip each), and serial performance lane (18
  passed, one Apple-only skip) all passed without changing the 264-node
  NVIDIA marker topology.
- NVIDIA compiler-artifact selection no longer relies on the
  `test_nvidia_*.py` filename pattern: the `compiler_nvidia` marker owns the
  CUDA artifact lane and its release-gate selection. `NvidiaDeviceSession`
  now frees all tracked buffers and destroys its stream even after a
  synchronization failure, and destroys a successfully-created timing event
  if its partner event cannot be created. Host-free fault-injection tests pass
  (2/2); the marker artifact lane passed and the real stream/event ABI fixture
  passed 15/15 on the RTX 5070 Ti.
- **NVIDIA-TEST-6 is complete (2026-07-16).** The closure audit found no
  remaining ordinary private MMA/PTX probe
  implementation: plugin and hot-path-ratchet compatibility names now delegate
  to shared predicates, while the Tile probe remains intentionally specialized.
  Running the hot-path ratchet immediately after the broad plugin matrix
  exceeded two f16 caps (512³ and 1024³); two isolated serial reruns both
  passed. The disposition is test-state contamination outside the canonical
  isolated performance lane, not a tolerance change or performance regression.
  An AST ratchet now rejects any future exact-device test under `tests/unit`.
  The final topology collects 333 NVIDIA nodes; compiler artifacts pass 20/20,
  exact-device correctness passes 313/313 twice, and the serial measured lane
  passes 20/20. New backward-attention and epilogue families landed directly
  in `tests/device/nvidia`, and the backward nodes are recorded in the
  executable post-migration map.
- The control-flow cohort is now accepted: source/rejection contracts remain
  host-free under `tests/unit`, while the bounded-control and runtime-binding
  execute/compare proofs moved to `tests/device/nvidia/test_control_flow.py`.
  Its mapped nodes passed focused validation, 19/19 compiler lit plus the
  NVIDIA compiler marker lane, exact-device correctness twice (246 passed,
  one Apple-only skip each), and serial performance (18 passed, one skip).
- The first device run exposed a product defect in Tile GELU: NVPTX could not
  select LLVM's `ftanh`; after its arithmetic lowering, SiLU exposed the same
  issue for `fexp`. Both now lower through a bounded Pade tanh expression, so
  no unsupported transcendental libcall is emitted. Focused Tile and related
  CUDA correctness tests passed **38/38** after the fix.

## Completion definition

This plan reaches `closed` only when all of the following are true on the
NVIDIA box:

1. Host-free CPU PR tests, NVIDIA compiler-artifact tests, CUDA device
   correctness tests, and CUDA performance tests run as separate commands with
   separate reports.
2. Every exact-device test proves `native_gpu` provenance and compares against
   the same numerical oracle used by the CPU/ROCm paths; no fallback earns a
   pass.
3. The full non-performance CUDA device matrix passes twice from a clean build.
4. Performance tests run serially after warmup and commit repeated-median
   kernel-only and end-to-end evidence. Timing under xdist is forbidden.
5. Tool, dtype, diagnostic, op, target, execution-state, and generated-doc
   registries remain green.
6. Duplicate/source-scan tests are removed only after an equal or stronger
   semantic, FileCheck, object/SASS, or execute/compare proof replaces them.
7. The NVIDIA release gate owns this lane and preserves logs plus machine
   identity for each proof run.

## NVIDIA-box preflight

Record this before interpreting any failure:

```bash
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total \
  --format=csv,noheader
nvcc --version
ptxas --version
python3 --version
git rev-parse HEAD
```

Required target is RTX 5070 Ti / compute capability 12.0. NVFP4/block-scale
tests compile the architecture-specific `sm_120a` target. Record driver, CUDA
toolkit, LLVM/MLIR, Python, GPU UUID, clocks/power mode, and whether another
process is using the device.

### Install LLVM/MLIR 23 on Ubuntu

Use the repository bootstrap on Ubuntu 24.04; it installs one matched LLVM,
Clang, LLD, MLIR, and Polly 23 toolchain from apt.llvm.org:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

For a toolchain-only manual installation, use a dedicated versioned source
file rather than replacing the distribution LLVM packages:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/apt.llvm.org.gpg

. /etc/os-release
LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}-23"
if ! wget -q --spider \
  "https://apt.llvm.org/${VERSION_CODENAME}/dists/${LLVM_SUITE}/Release"; then
  LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}"
fi
echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/${VERSION_CODENAME}/ ${LLVM_SUITE} main" \
  | sudo tee /etc/apt/sources.list.d/llvm-23.list >/dev/null
sudo apt-get update
sudo apt-get install -y \
  clang-23 lld-23 llvm-23 llvm-23-dev llvm-23-tools \
  mlir-23-tools libmlir-23-dev libpolly-23-dev

export LLVM_ROOT=/usr/lib/llvm-23
export PATH="$LLVM_ROOT/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

llvm-config --version
mlir-opt --version
mlir-tblgen --version
FileCheck --version
```

All four commands must report major version 23. Remove or disable any stale
pre-23 toolchain source selection from the build environment; keeping
multiple apt repositories installed is acceptable, but Tessera's CMake cache,
compiler executables, MLIR tools, and CMake package directories must all resolve
to `/usr/lib/llvm-23`.

Build the compiler and CUDA runtime from a clean NVIDIA build directory:

```bash
cmake -S . -B build-nvidia-cuda -G Ninja \
  -DCMAKE_C_COMPILER=/usr/lib/llvm-23/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-23/bin/clang++ \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_120a \
  -DTESSERA_BUILD_EXAMPLES=OFF
ninja -C build-nvidia-cuda tessera-opt tessera-nvidia-opt \
  tessera_nvidia_gemm tessera_runtime
```

Export explicit tool paths rather than relying on a previous build:

```bash
export TESSERA_OPT="$PWD/build-nvidia-cuda/tools/tessera-opt/tessera-opt"
export MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt
export PYTHONPATH="$PWD/python:$PWD"
```

Adjust `TESSERA_OPT` to the actual Ninja output reported by the build if the
generator places it under `build-nvidia-cuda/tools/tessera-opt/` differently.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and updates portable Tile/NVIDIA TableGen plus greedy-rewrite
compatibility. The shared sources compile in the LLVM/MLIR 23 ROCm build, and
NVIDIA exact-device parity is now validated independently on the `sm_120`
host. No CUDA execution status was inferred or promoted from the ROCm run.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-CALIB-1 | Validate T1 reuse/cache pruning against the committed sm_120 corpus | Supply evidence-backed sm_120 bandwidth/cache inputs, then compute per-family and per-shape rank correlations without reviving the rejected step-distance score. | The analysis records model version, corpus identity, rank correlations, and a retain/reject verdict; no new device run is required. |
| 2 | NVIDIA-RASTER-1 | Consume and measure the shared raster contract | Wire the emitter with row-major identity preserved, then sweep only on sm_120 with device timing and `ncu` L2 evidence. | Exact-device correctness remains unchanged and a measured raster-order/group decision is recorded without hardware-free arbitrary selection. |
| 3 | NVIDIA-AOT-1 | Decide whether the NVRTC lane needs a precompiled peer | Reuse the Apple AOT/JIT harness and its never-before-compiled-kernel cache control; first decide whether the expected cold-start use case justifies implementation. | A documented not-applicable/retain decision or a measured precompiled candidate with equivalent numerics and explicit offline-build amortization. |
| 4 | NVIDIA-TEST-1 | Establish a reproducible baseline | Run collection and each proof layer separately; save JUnit, skip reasons, duration report, machine identity, and the exact commit. Classify every failure as product defect, test defect, environment defect, or stale claim. | Two collections return the same node set; no unknown markers; every skip has an explicit unavailable capability. |
| 5 | NVIDIA-TEST-2 | Compiler-artifact layer | Run `check-tessera-nvidia` plus CUDA pytest files carrying `compiler_tool`; migrate private tool probes to `compiler_toolchain`; split artifact assertions from the eight tests that currently continue into device execution; replace large textual snapshots with named diagnostics, FileCheck, or focused IR/object invariants. | Clean build passes without a GPU; missing-tool simulation skip-cleans; no compiler test invokes a nonexistent path. |
| 6 | NVIDIA-TEST-3 | Exact-device correctness | Run `hardware_nvidia and not performance`; group failures by GEMM/Tile, attention, reductions/norms, control flow, KV/ReplaySSM, collectives, and ABI/conformance. Require native provenance and execute/compare. | Entire correctness matrix passes twice; fallback-injection negatives fail to earn native proof. |
| 7 | NVIDIA-TEST-4 | Numerical policy | Centralize dtype/op tolerances from accumulation/storage behavior. Add ragged, rectangular, boundary, non-finite, misalignment, and invalid-contract cases where absent. | f16/bf16/tf32/FP8/int8/NVFP4 cases use documented tolerances; no default zero-`atol` checks near zero. |
| 8 | NVIDIA-TEST-5 | Measured performance | Run `hardware_nvidia and performance` serially. Warm up compilation and caches; use repeated medians; measure kernel-only and end-to-end separately; record registers, shared memory, occupancy, spills, and selected route. | Stable baselines cover square/rectangular/ragged GEMM, fused epilogues, attention, paged KV, ReplaySSM, reductions, and transport. Each ratchet identifies the selected implementation. |
| 9 | NVIDIA-TEST-6 | Refactor and deduplicate | Move mature families toward `tests/compiler/`, `tests/device/nvidia/`, `tests/integration/`, and `tests/performance/nvidia/`. Consolidate repeated CUDA availability, compilation, launch, oracle, and cleanup code. | No central filename allowlist; no duplicated private CUDA probe/loader; process trees and device allocations clean up on failure. |
| 10 | NVIDIA-TEST-7 | Local release ownership | Own the NVIDIA-box release gate locally in WSL with a host concurrency lock and retained artifacts; GitHub runners are intentionally not used. Keep two-run device correctness required for NVIDIA promotion and performance serial. | A clean branch run reports NVIDIA host-free/shared registries, compiler artifact, device correctness, and performance independently and retains the fail-closed evidence bundle. |
| 11 | NVIDIA-LSE-1 | Consume and measure the real shared LSE checkpoint on CUDA | **Landing, SM120 P0 complete.** Compiler-owned f32 paired physical ABIs now carry `Q/K/V -> O,row_lse` and `dO/Q/K/V,row_lse -> dQ/dK/dV` through Tile operands, descriptors, bridge allocation/copies/argument order, runtime validation, and the native-package route. Exact RTX 5070 Ti proof covers oracle equality, saved-vs-recompute forward/backward equality, and malformed rank rejection. | At `[1,2,1,3,4,4,3]`, saved lowers backward event time (0.01716 vs 0.02777 ms) but the paired save/load e2e median loses (1.56149 vs 1.26335 ms); NCU/resource packet is retained in `benchmarks/baselines/nvidia_sm120_lse_checkpoint_2026_07_30.json`. Retain recompute default; repeat representative sequence/shape sweeps before promotion. |
| 12 | NVIDIA-E2E-1 | Canonical SM120 compiler spine | Under sync `E2E-SPINE-2026-07-18`, compose Graph/Schedule/Tile lowering with `LowerTileToNVIDIA(sm=120)`, NVVM/PTX/native-image packaging, and the existing register/invoke launch bridge. Prove f16 and NVFP4 first, including non-origin scale tiles and general-shape dispatch. | One canonical driver request returns a typed image artifact plus launch descriptor, registers and launches on `sm_120`, compares numerically, and retains compiler/ABI/device/resource evidence without a selector change. |
| 13 | NVIDIA-E2E-2 | Per-SM and operation breadth | Replace shared-alias/hardcoded target behavior with architecture-specific pipelines, then move supported CUDA families through the same typed image/launch seam. | Every enabled SM/family has the four-layer proof on its exact device or an explicit unsupported/planned terminal state; `sm_90` and `sm_100` are never inferred from `sm_120`. |

### High-risk NVIDIA-TEST-6 migration

**NVIDIA-TEST-6-HIGH — Relocate mature CUDA families without breaking the
proof contract.** Move mature compiler, device, integration, and performance
families toward `tests/compiler/`, `tests/device/nvidia/`,
`tests/integration/`, and `tests/performance/nvidia/`. This is high risk because
pytest node IDs, import roots, marker collection, CI selection, and retained
JUnit history can all change even if individual assertions still pass.

Before accepting the migration, record an old-to-new node map, preserve every
`hardware_nvidia`/`performance`/`compiler_tool` classification, prove that the
old paths have no duplicate collection, and run the host-free, artifact,
exact-device, and serial-performance layers. Do not combine this migration with
backend behavior, tolerance, or selector changes.

**Pilot evidence (2026-07-15, `landing`).** MoE transport is the first
relocated family. Its two native CUDA execute/compare nodes now live in
`tests/device/nvidia/test_moe_transport.py`; its host-free invalid-partition
contract remains in `tests/unit/test_nvidia_moe_transport_contract.py`. The
checked-in old-to-new map is `tests/device/nvidia/node_migrations.json`, and
`tests/unit/test_nvidia_test_location_migration.py` prevents restoration of
the old file or duplicate destinations. The second cohort applies the same
contract to the former mixed `test_nvidia_launch_execute.py`: two host-free
execution-matrix nodes remain under `tests/unit/`, and five native launch/JIT
nodes move to `tests/device/nvidia/test_launch_execute.py`. The combined roots
collect exactly **264** `hardware_nvidia` nodes (246 correctness, 18
performance). The two device-only DSA sparse-attention nodes are also mapped
to `tests/device/nvidia/test_sparse_attention.py`. Every relocated node
preserves its `hardware_nvidia` classification; none gained `performance` or
`compiler_tool` classification.

The compiler-artifact proof passed (19/19 lit and 1 compiler-tool pytest
contract), exact-device correctness passed twice (246 passed, 1 unrelated
Apple-only skip, zero failures/errors on each run), and the serial performance
lane passed (18 passed, 1 unrelated Apple-only skip). The second cohort
repeated those artifact, two-run correctness, and serial-performance proofs;
its executable node-map and retained host-free contracts passed (4/4). The
complete host-free PR command is **not an
NVIDIA-host acceptance gate** when it exercises Apple/ROCm compiler passes:
this WSL checkout's generic `build/` is intentionally NVIDIA-only, so 274
foreign-backend compiler tests cannot run here. This is not a relocation
failure or an NVIDIA-TEST-6-HIGH blocker. `APPLE-CI-2` and `ROCM-TEST-1` own
validation of their respective host-free compiler configurations on the correct
backend hosts; the NVIDIA host retains the focused host-free migration guard
plus its artifact and exact-device proof layers.

**Completion evidence (2026-07-16, `complete`).** The executable map now covers
**286** relocated node IDs. Mature execute/compare families are collected from
`tests/device/nvidia/`; paged-KV, ReplaySSM, and the MMA bridge are in
`tests/integration/`; and hot-path, Conv2D, MMA-symbol, and plugin timing
proofs are in `tests/performance/nvidia/`. The mixed plugin implementation is
shared through a non-discovered support module, while its 20 host-free
contracts, 53 native nodes, and 8 measured nodes are collected only from their
respective unit/device/performance entry points. The only remaining
`hardware_nvidia` references under `tests/unit/` are release-gate and
marker-policy structural assertions.

The final migrated plugin cohort passed its focused mapping/architecture guard
(93 tests), compiler artifacts (19/19 lit; one compiler pytest pass and one
hardware-excluded skip), exact-device correctness twice (246 passed and one
Apple-only skip each), and serial performance (18 passed and one skip).
Relocating Conv2D exposed an order-dependent product defect: automatic f32
dispatch admitted the explicit `im2col_tf32` candidate under a looser internal
tolerance. Automatic dispatch now selects only f32-accurate direct/shared
routes; explicitly requested TF32 performance coverage remains intact. The
post-fix exact-device matrix passed twice (246 passed and one skip each), and
the serial performance lane passed (18 passed and one skip). This is a product
correctness fix with retained before/after numerical evidence, not a tolerance
relaxation.

The final static audit finds no `hardware_nvidia` test function under
`tests/unit`; structural marker/release assertions remain host-free. The
expanded map contains 292 relocations plus 23 post-migration nodes. The final
four-layer proof is 20/20 compiler lit, 313/313 exact-device correctness twice,
and 20/20 serial performance on the RTX 5070 Ti. This closes the migration;
future native tests must land directly in device, integration, or performance
roots and satisfy the same AST/node-map ratchets.

## Canonical commands on the NVIDIA box

```bash
# 0. State/collection contract (currently 334 nodes)
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m hardware_nvidia --collect-only -q --no-header

# 1. Host-free PR contract, including CUDA emit/validation/rejection tests
python3 scripts/run_unit_tests.py --timeout=180 -q

# 2. Compiler artifacts without claiming device execution
ninja -C build-nvidia-cuda check-tessera-nvidia
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml=/tmp/nvidia-compiler-tool.xml

# 3. Exact-device correctness; run twice from the same clean build
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "hardware_nvidia and not performance" -q --durations=100 \
  --junitxml=/tmp/nvidia-device-correctness.xml

# 4. Measured lane: serial only
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m "hardware_nvidia and performance" -q -n 0 --durations=0 \
  --junitxml=/tmp/nvidia-performance.xml
```

Do not use `-x` for the first baseline: the complete failure topology is needed
to design the migration. After triage, use focused files for the edit loop and
rerun the complete layer before marking an item complete.

## Failure triage contract

For every failure, record:

- node id, proof layer, target/dtype/shape, seed, selected route, and native
  provenance;
- whether it reproduces alone, serially, and on the second clean run;
- compiler stdout/stderr and named diagnostic code;
- numerical maximum absolute/relative error and first failing index;
- kernel-only versus end-to-end latency for performance cases;
- register/shared-memory/occupancy/spill evidence when a kernel changes;
- disposition: fix product, fix test state, replace weak test, merge duplicate,
  or document an exact environment blocker.

Never relax a tolerance or timing cap solely to make the lane green. Recompute
it from dtype semantics or a stable repeated-median baseline, and retain the
before/after evidence.

## Initial family matrix

| Family | Representative coverage | Required follow-up on NVIDIA box |
|---|---|---|
| Tile/GEMM | compiler-generated SM120 fragments, shipped MMA symbols, ragged/grid GEMM, f16/bf16/tf32/FP8/int8, NVFP4 OMMA | Verify exact SASS/instruction family, lane maps, ragged stores, allocation cleanup, and kernel/device timing separation. |
| Fusion | bias, ReLU, GELU, SiLU, gated SwiGLU, matmul-softmax | Cross-check epilogue order and dtype accumulation against CUDA and ROCm shared oracles. |
| Attention | MHA/GQA/MQA, backward, sparse/DSA, window/bias/softcap | Separate compiler artifact from live execution; cover global decode positions and non-finite policy. |
| Reductions/norms | sum/mean/min/max, softmax, RMSNorm/LayerNorm | Validate non-power-of-two/ragged widths, NaN policy, large-offset variance, and dtype tolerances. |
| Stateful serving | paged KV and ReplaySSM async ring | Long decode, flush, rollback, rejection/backpressure, remapped pages, native provenance, and leak-free teardown. |
| Control/collectives | bounded for/if/while/scan and single-device collectives | Validate one-launch ABI, bad-shape rejection before launch, and explicit multi-rank deferral. |
| Performance | GEMM routes, convolution routes, device timing, hot-path ratchet | Run isolated and serial; record winner, resource evidence, kernel-only, and end-to-end rows. |

## ROCm-derived CUDA parity work

The completed ROCm work raised the proof standard for several features that
already exist on CUDA. These are CUDA audits and measured retunes, not literal
ports of AMD schedules. Share logical fixtures, ABI contracts, numerical
oracles, benchmark schemas, and decision rules across backends; keep physical
fragments and schedules architecture-owned.

In particular, an RDNA wave is not a CUDA warp, LDS is not evidence about
shared-memory behavior, VGPR pressure does not predict the CUDA register file,
and WMMA/MFMA winners do not select `mma.sync` or OMMA winners. Every production
selector change below requires fresh `sm_120a` measurements on the NVIDIA box.

| Order | ID | ROCm lesson and CUDA work | Current CUDA state | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-PARITY-TILE | Re-run the same logical portable-Tile fixture through the NVIDIA architecture-owned fragment selector. Cover direct/shared schedules, grid and ragged edges, supported f16/bf16/tf32/FP8/int8/NVFP4 forms, and bias/ReLU/GELU/SiLU epilogues. Add a CUDA fragment resource record containing registers, shared memory, occupancy, spills, and the selected SASS instruction family. | Compiler-generated SM120 fragments, layout oracles, and direct/shared execution tests exist. | Fixtures never author physical fragments; pack/execute/unpack/store matches the shared oracle; emitted instructions and resource rows match the selected `sm_120a` contract. |
| 2 | NVIDIA-PARITY-GEMM-RATCHET | Extend the hot-path recorder into a repeated-median schedule matrix covering square, rectangular, ragged, dtype, and fused-epilogue cases. Record kernel/device-event and end-to-end time separately, then capture registers, shared memory, occupancy, and spills before changing the production tile selector. | `record_hot_path_baseline.py` provides a useful but narrow latency ratchet. | A committed device-keyed baseline identifies every candidate and winner; two stable runs agree within the declared noise policy; no selector change lands without before/after resource evidence. |
| 3 | NVIDIA-PARITY-LEGACY-RETUNE | Re-evaluate older f32/tf32 GEMM, grouped GEMM, grouped SwiGLU, KV movement, and MoE transport now that the compiler and fragment selection are stronger. Compare compiled, shipped, and staged/direct candidates without conflating launch/transfer cost with kernel time. | Individual CUDA paths and hot-path rows exist, but there is no ROCm-equivalent wide retune corpus. | All candidates match one oracle; kernel-only and end-to-end winners are recorded independently; grouped and transport rows include launch collapse and achieved-bandwidth evidence. |
| 4 | NVIDIA-PARITY-ATTN-FWD | Apply the G6-B methodology to CUDA forward attention: evaluate occupancy-aware multi-warp CTA schedules with online softmax at D=128, plus ragged, causal/window, bias, softcap, and MHA/GQA/MQA cases. Do not assume ROCm's two-wave shape is the CUDA winner. | Compiled CUDA forward-attention paths and exact-device tests exist. | Candidate schedules match the shared oracle; traffic and resource evidence explain the winner; the selected route wins repeated-median kernel timing without regressing end-to-end timing. |
| 5 | NVIDIA-PARITY-ATTN-BWD | Apply the G6-C methodology to dK/dV backward. Measure the existing path against atomic and split-workspace/reduction candidates, including deterministic behavior and workspace limits. | Compiled CUDA backward attention is covered, but has not been re-ratcheted against the split/reduced design space. | Forward-derived gradients pass the shared tolerance matrix; determinism and workspace caps are explicit; resource, kernel-only, and end-to-end rows select the production route. |
| 6 | NVIDIA-PARITY-PAGED-KV | Re-prove the stable paged-KV ABI with non-identity/permuted pages, remaps, causal offsets, and boundary lengths. Compare direct resident page-table attention with staged/gather-to-FA using the same oracle and retain both timing domains. | Direct fused and staged paged-attention candidates plus an SM120 serving baseline already exist. | Every candidate consumes the same ABI and matches the same permuted-page oracle; device-event and end-to-end rows may choose different winners and the cache keys preserve that distinction. |
| 7 | NVIDIA-PARITY-REPLAY | Re-run CUDA ReplaySSM against the closure matrix exposed by ROCm: long decode, flush, rollback, speculative rejection, block submit, ordered async ring, backpressure, and teardown. Expand B/D/N/M shapes and record state traffic as well as latency. | CUDA is the reference persistent ReplaySSM implementation and has serving rows, but needs the wider proof and benchmark matrix. | All transitions match `SSMStateHandle`; rejected work cannot mutate committed state; ring ordering and cleanup survive stress; traffic plus kernel/end-to-end latency are committed. |
| 8 | NVIDIA-PARITY-EPILOGUE | Make the common Tile epilogue contract explicit for bias, ReLU, GELU, and SiLU. Check accumulator precision, operation order, optional bias/residual guards, ragged stores, and all supported storage dtypes against shared CUDA/ROCm fixtures. | CUDA emits fused epilogues and plugin tests cover representative forms. | One backend-neutral oracle drives both backends; every supported fusion executes natively; unsupported dtype/op pairs reject with registered diagnostic codes rather than silently de-fusing. |
| 9 | NVIDIA-PARITY-AUTOTUNE | Align CUDA and ROCm corpus schemas around device-keyed candidates, timing domain, compiler/resource fingerprint, cold/warm compile state, and cache behavior. Promote a winner only after the relevant correctness and schedule ratchets pass. | CUDA has autotune and serving corpus writers, but their evidence must be reconciled with the newer ROCm records. | Corpus validation rejects stale devices, compilers, resources, and timing domains; cold/warm behavior is reproducible; selector decisions cite a retained measurement row. |
| 10 | NVIDIA-PARITY-TRANSPORT | Close KV-movement and MoE-transport parity with direct/staged routes, ragged/grouped loads, bandwidth attainment, and launch-amortization measurements. Feed any winner into the legacy retune only after ABI and correctness closure. | CUDA transport operations exist but lack one consolidated exact-device performance proof. | Byte counts and achieved bandwidth are auditable; kernel-only and end-to-end winners are separate; awkward sizes and grouped routes match their reference without leaks or hidden host staging. |

### CUDA parity execution record

The parity queue uses ROCm's logical coverage and proof methodology, not its
physical schedules. CUDA owns warp/register packing, `HMMA`/`QMMA`/`IMMA`/OMMA
selection, shared-memory staging, barriers, occupancy limits, and every selector
winner. An AMD wave shape, LDS strategy, or VGPR result is never a CUDA default.

- **NVIDIA-PARITY-TILE — complete on sm_120a.** The architecture-owned SM120 fragment
  selector now describes f16 (f16/f32 accumulation), bf16, TF32, FP8 E4M3/E5M2,
  int8, and block-scaled NVFP4 separately. C++ lowering consumes the descriptor
  for physical input packing and per-lane register-count validation. The exact
  compiler path passes the shared numerical oracle for f16/bf16/TF32/FP8/int8,
  direct/shared grids, ragged edges, and bias/ReLU/GELU/SiLU. The reproducible
  13-row `nvidia_sm120_tile_fragment_resources.json` record retains cubin hashes,
  registers, shared memory, theoretical occupancy, spills, and observed SASS:
  `HMMA` for f16/bf16/TF32, `QMMA` for FP8, `IMMA` for int8, and block-scaled
  `OMMA` for NVFP4. Portable typed Tile now carries NVFP4's two logical UE4M3
  scale tiles. C++ consumes nibble-packed logical A/B storage, materializes the
  backend-owned scale selectors, emits real block-scaled inline PTX, assembles
  for `sm_120a`, and passes the non-uniform-scale numerical oracle without
  fixture-authored physical fragments. The resource row now comes from that
  typed compiler artifact rather than the original CUDA spike.
- **NVIDIA-PARITY-GEMM-RATCHET — measured complete; no promotion.** The
  device-keyed `nvidia_sm120_gemm_schedule_matrix.json` contains 34 exact-case
  rows spanning square, rectangular, ragged, f16/bf16, and bias plus
  none/ReLU/GELU/SiLU epilogues. Every row has two stable repeated-median runs,
  separate CUDA-event and rotated-interleaved end-to-end timing, complete
  per-candidate resource fingerprints, and an explicit 3% noise policy. The
  smallest ragged rows require 50 untimed device warmups to remove clock-ramp
  drift. The record intentionally leaves the production selector unchanged.
  CUDA 13.3's renamed local/shared spill-request metrics are normalized alongside
  the legacy Nsight metrics, and the synthesized fused fallback now has retained
  production-sized Nsight evidence.
- **NVIDIA-PARITY-LEGACY-RETUNE — stable complete; no selector promotion.** The
  device-keyed `nvidia_sm120_legacy_retune.json` now compares compiled exact-f32
  and shipped TF32 GEMM on square/ragged rows, one grouped GEMM launch against
  the retained per-expert decomposition, and a new grouped SwiGLU route whose
  four launches are independent of expert count against the legacy `4E` route.
  All candidates use one f32 oracle and retain separate event/end-to-end rows,
  byte and achieved-bandwidth accounting, launch counts, and linked resources.
  SwiGLU rows retain both the grouped-GEMM cubin fingerprint and the exact
  generated SiLU-gate registers, occupancy, and spill record.
  The final corpus uses production-scale 512-square and 509x773x257 ragged
  GEMM, 1024x384x256x5 grouped GEMM, and 512x256x384x8 grouped SwiGLU rows.
  Exact-route resident warmup occurs after allocation in the timed session,
  and two disjoint interleaved cohorts retain every device and end-to-end batch.
  All eight rows pass 3% (maximum device/end-to-end deltas 2.47%/0.77%). TF32
  and the launch-collapsed grouped routes win both retained runs, but this
  evidence intentionally leaves selectors unchanged.
- **NVIDIA-PARITY-ATTN-FWD — stable complete; CUDA 4-warp candidate leads kernel time.**
  CUDA-owned 4- and 8-warp CTA candidates now cover D=128 MHA, causal sequence
  1009, ragged GQA windowing, and MQA bias+softcap. Each warp owns one query,
  uses warp shuffles for QK, and keeps distributed online-softmax/PV state;
  this is not ROCm's two-wave LDS schedule. All rows match the shared oracle
  within `7e-8`. Both candidates use 56 registers with zero spills; modeled
  occupancy is 75% for four warps and 66.67% for eight. Four warps win CUDA-event
  timing on both retained runs for every case. Ten disjoint, sample-interleaved
  end-to-end batches remove run-order aliasing without sharing observations;
  all eight rows now pass 3% with maximum device/end-to-end deltas of
  0.22%/1.84%. Small end-to-end rows do not have unanimous winner consensus,
  so production selection remains unchanged.
- **NVIDIA-PARITY-ATTN-BWD — measured complete; atomic retained.** The atomic
  incumbent and deterministic two-part split/workspace/fixed-order-reduction
  candidate share one forward-derived oracle across D64 MHA, causal D128 MHA,
  and ragged windowed GQA. The split route is bitwise repeatable, rejects
  unsupported f16 storage, and enforces an exact one-extra-dK+dV f32 workspace
  cap (524,288 bytes on MHA rows; 134,144 on ragged GQA). All six candidate
  rows pass 3% with maximum device/end-to-end deltas of 0.36%/1.67%. Resources
  retain atomic 48-register/83.33%-occupancy and split dQ 48-register,
  dK/dV 56-register/75%-occupancy, and reduction 12-register/100%-occupancy
  fingerprints plus spill evidence. Atomic wins both timing domains in every
  case by a large margin, so `selector_changed` remains false.
- **NVIDIA-PARITY-PAGED-KV — correctness and timing complete; no promotion.** Both fused
  and staged routes now pass the same permuted-page oracle at lengths 1, 3, 4,
  5, 7, 8, 9, and 13, including non-monotonic logical indices and global causal
  offsets. The 13-row transport corpus covers 127/128/129/511-token boundaries
  with separate device/end-to-end keys, byte formulas, resources, and no
  selector change. Repeated event batches now remain inside one warmed resident
  session; all eight fused/staged rows pass the 3% two-run policy, with maximum
  device and end-to-end deltas of 1.89% and 1.85% respectively.
- **NVIDIA-PARITY-REPLAY — canonical state contract and correctness complete;
  timing characterization retained.** Exact-device
  tests cover long decode across flushes, rollback, speculative rejection,
  block submit, reset, ordered ring backpressure, rejected-submit immutability,
  and teardown over wider B/D/N shapes. The 10-row replay corpus spans five
  geometries and 16/64 tokens with traffic, resources, and both timing domains.
  Each runtime handle now carries the shared `tessera.replayssm.state.v1`
  descriptor: exact persistent device and pinned-host byte formulas, session
  lifetime with preserved initialization, ordered stream/event slot ownership,
  consumer-wait-before-release, and teardown draining. Span checks reject before
  CUDA submission. The CPU oracle is outside the end-to-end interval; each
  retained run has disjoint four-route batch medians with recorded out-of-band
  clock conditioning. All errors remain below `1.5e-8`. Under the WSL 4%
  foundation policy 5/10 refreshed rows satisfy both domains; the remaining
  small/multi-batch rows range from 4.05% to 8.75%. No selector decision consumes
  these unstable rows.
- **NVIDIA-PARITY-EPILOGUE — execution matrix complete.** `FusedRegion` is the
  backend-neutral bias/activation/residual/order oracle and now emits registered
  `E_FUSED_EPILOGUE_*` diagnostics for unsupported dtype/op/order and missing
  operands. The exact-device matrix now executes all 43 supported combinations
  over f32/f16/bf16/FP8 E4M3/FP8 E5M2, optional bias, no activation or
  ReLU/GELU/SiLU, and f32 residual-after-activation ordering. Accumulation is
  f32; low-precision residual, activation-before-bias, repeated activation,
  and unsupported dtype/op pairs reject with the registered diagnostics.
- **NVIDIA-PARITY-AUTOTUNE — strict admission complete; no promotion.** Corpus
  admission can require exact device, timing domain, compiler fingerprint,
  resource fingerprints, compile state, and cache state. The committed
  reproducibility record admits all 20 selector-eligible NVIDIA rows, rejects
  stale device/timing/compiler/resource mutations, and reproduces one kernel
  cache key across two cold builds and warm hits (about 0.05 ms warm lookup).
- **NVIDIA-PARITY-TRANSPORT — correctness, evidence, and timing complete.**
  The consolidated 13-row paged-KV/MoE/grouped corpus retains auditable traffic
  formulas, achieved bandwidth, launch-amortization keys, exact resources, and
  independent timing domains. MoE dispatch/combine now consume one canonical
  `tessera.moe_transport.v1` int32/fp32 descriptor with stable expert grouping,
  capacity/drop semantics, and dispatch-before-compute-before-combine ordering;
  grouped GEMM consumes canonical ragged sizes/offsets and retains empty experts.
  Local-device scope is explicit; multi-rank collective execution remains a
  separate backend/runtime item. Maximum oracle error is below `3e-7`; all 13
  rows pass the WSL 4% foundation policy. MoE CUDA-event samples retain one
  native allocation set across repeated batches, and the tiny routes use 101
  medians per run. No selector or legacy-retune winner is promoted.

- **NVIDIA-SM120-LOWP-PRODUCTIZATION — complete (2026-07-18).** The shipped
  CUDA ABI adds general-shape block-scaled NVFP4: packed E2M1 A/B, raw UE4M3
  scale views, M16/N8 grid dispatch, K64 accumulation, ragged zero fill, and
  pre-launch shape/view rejection. Fixed 16x8x64, multi-tile 33x19x129, and
  sub-tile 7x5x31 non-uniform-scale cases match the exact NVFP4 oracle on the
  RTX 5070 Ti. Native one-kernel TF32 and FP8 E4M3/E5M2 fused-epilogue,
  QK-softmax-PV attention, and gated routes now coexist with the composed
  candidates. Two fresh runs use 20 end-to-end medians and 100 CUDA-event
  repetitions per route. The cross-domain 3% gate promotes 11 of 18 retained
  shape/dtype rows; long attention and disagreement rows remain unpromoted.
  The linked 12-row cubin record reports 40-register fused/attention kernels,
  47–48-register gated kernels, 8/32 KiB attention dynamic shared memory,
  shape-dependent 22.92%/6.25% modeled attention occupancy, zero compiler spill
  storage, and the expected TF32 HMMA / FP8 QMMA SASS. Evidence:
  `nvidia_sm120_low_precision_native_{routes,resources}.json`.
- **Audit-document reconciliation — complete (2026-07-18).** This plan,
  `NVIDIA_AUDIT.md`, and `sm120-kernel-guide.md` now agree that mature SM120
  fragments lower for real, general NVFP4 dispatch is executable, and native
  TF32/FP8 transformer candidates exist. The plan remains `landing` only for
  unrelated architecture-specific follow-ons such as sm_90 WGMMA and sm_100
  tcgen05 exact-device proof; landed SM120 work is no longer described as open.

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16` changes the shared typed Tile
operand contract only. NVIDIA supplies exact-device materialization evidence;
Apple and ROCm do not inherit its physical schedule and record their outcomes
in their own plans.

Cross-backend sync `PR420-REVIEW-2026-07-17` corrects the NVIDIA-owned NVFP4
scale materializer to apply both `tile.view` origins using the declared
row-major A-scale and column-major B-scale layouts. A live `sm_120a` fixture
selects nonzero A-row and B-column scale tiles and matches the NumPy oracle;
the NVIDIA compiler lit suite passes 21/21. The SM120 Target IR selector also
accepts canonical `fp16` as the existing f16 fragment contract. This is a
correctness/dispatch repair only: no physical fragment, resource record,
timing row, or production selector changes. The same sync makes Ubuntu LLVM
repository setup install its probe prerequisites before first use; sibling
backend outcomes are recorded in their plans.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18` is NVIDIA-owned. It changes no
shared dtype spelling, portable Tile scale layout, backend-neutral epilogue
order, or generic autotune schema. Apple has no enabled NVFP4 cooperative-matrix
route and ROCm gfx1151 has no FP8/FP4 WMMA instruction; neither inherits CUDA
packing, HMMA/QMMA/OMMA schedules, resource values, timings, or selector rows.

Cross-backend sync `E2E-SPINE-2026-07-18`: NVIDIA owns **NVIDIA-E2E-1** and
**NVIDIA-E2E-2**. Shared code owns only the image/launch schemas and canonical
orchestration; NVIDIA retains PTX/SASS generation, physical fragments, launch
geometry, resources, and route selection. Existing NVRTC, shipped-library, and
PTX-register/invoke paths remain valid candidates while the typed spine lands.
Host-free IR/object evidence cannot promote an SM or selector, and exact-device
proof for `sm_90`, `sm_100`, and `sm_120` remains architecture-specific. The
completed E2E-SPINE-0 foundation records SM80 as lacking an exact registered
pipeline and SM100/SM120 as shared-builder aliases; it also corrects the Python
pass inventory to match that builder without changing CUDA runtime selection.
E2E-SPINE-1 adds the portable image/descriptor and rejection contract only;
PTX/cubin contents, warp schedules, launch geometry policy, resources, and CUDA
selectors remain NVIDIA-owned and unchanged until NVIDIA-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no CUDA hook and
does not reinterpret `nvidia_mma` or any shipped/NVRTC candidate; NVIDIA-E2E-1
still owns PTX packaging, `sm_120` registration/submission, numerical proof,
resources, cleanup, and the first Level-C row.

NVIDIA-E2E-1 is **complete**. The f16 slice makes an explicit canonical
driver request own the typed `tile.matmul_kernel`, runs the production
`LowerTileToNVIDIA(sm=120)` and NVVM/LLVM/PTX pipeline, validates the image with
`ptxas`, and returns the shared native-image plus exact A/B/D/M/N/K descriptor.
The descriptor registers and launches through the shipped PTX bridge on the RTX
5070 Ti; aligned `16x8x16` and ragged `37x29x23` rows match the f32 NumPy oracle.
The image retains compiler/toolchain fingerprints, cold/warm state, and ptxas
register/shared-memory/spill fields. This slice changes no production selector.
The same driver now selects a CUDA-owned general-shape NVFP4 descriptor with
packed E2M1 A/B, logical UE4M3 `scale_a`/`scale_b`, f32 output, and M/N/K. The
typed lowering owns M16/N8 origins, K64 accumulation, ragged zero fill, scale
word materialization, and guarded stores before LLVM 23 emits `sm_120a` PTX.
Exact RTX 5070 Ti rows `16x8x64`, `33x19x129`, and `7x5x31` match the block-scale
oracle; the multi-tile row uses nonuniform row/column scales to prove non-origin
scale views. Missing/malformed scales, wrong scale storage, bias, and malformed
launch shapes reject before CUDA submission. Both f16 and NVFP4 retain stable
cold/warm image identity and ptxas register/shared-memory/spill evidence. The
shared Tile verifier change is limited to the explicit eight-operand NVFP4
launch ABI; it transfers no CUDA schedule, layout, resources, or selector.

NVIDIA-E2E-2 is **closed for the available SM120 host**, with the unavailable
multi-GPU, SM90, and SM100 boundaries assigned the deferred terminal states
below. Its first dependency slice replaces the former
shared SM90 alias with exact SM90/SM100/SM120 Graph→Tile builders and registered
Tile→`tessera_nvidia`→NVVM producers. The exact target now reaches Tile IR,
the control-flow guard, async-copy lowering, and the target producer without
being rewritten to SM90. Hopper alone consumes the proven WGMMA and Hopper
FlashAttention markers; SM100 and SM120 retain target-tagged typed carriers for
architecture-owned lowering. Straight-line async copies mint typed completion
tokens, the matching wait retires them, and matrix consumers preserve those
edges through TMA lowering. Host WSL FileCheck proves the three distinct IR
routes; native SM90 and SM100 remain unsupported-by-evidence until exact-device
runs exist. No selector changes. That breadth statement described the first
landing slice and is now superseded by the implementation record below: SM120
canonical execution covers the complete matmul dtype matrix plus softmax,
reductions, fused epilogues, attention, paged-KV, ReplaySSM, and local MoE.

The next NVIDIA-E2E-2 family slice now gives static f16/f32 last-axis softmax a
canonical Level-C path. `tile.softmax_kernel` carries source/destination,
flattened Rows/K, `storage="f16"|"f32"`, `accum="f32"`, and `axis=-1`; the SM120
materializer emits a stable max-shifted row loop and target-native `nvvm.ex2`
instead of an unavailable NVPTX `fexp` libcall. LLVM 23 emits and ptxas
validates `sm_120a` PTX, while the typed descriptor registers and launches it
through the shipped CUDA-driver bridge. Exact RTX 5070 Ti proof covers shapes
`1x16`, `8x64`, `4x300`, and `2x3x48`, extreme logits, malformed output shape
rejection, stable cold/warm image identity, and resource/spill fields for both
storage types. f16 loads extend before the max/sum/normalization loops and
truncate only at output storage. This is a correctness-first 128-thread,
one-thread-per-row candidate. The existing cooperative CUDA-C route remains
selected. All four final canonical/production rows are stable in both timing
domains, and production wins both domains for the production-sized and ragged
cases.

The following NVIDIA-E2E-2 dtype-totality slice centralizes consumer-Blackwell
storage, math-mode, scalar/vector, Tensor Core, compiler, and runtime states in
`nvidia_dtype_contract.py`. Every canonical float storage type now has an
explicit row. CUDA 13.3 compile proof covers scalar/vector forms for fp64,
fp32, fp16, bf16, FP8 E4M3/E5M2, FP6 E2M3/E3M2, and packed FP4; TF32 remains
strictly an fp32 `math_mode`, never storage. Tensor Core Target IR/PTX rows now
cover the required TF32, bf16, fp16, FP8, FP6, FP4, and int8 families. The
canonical descriptor lane now executes BF16, explicit fp32-storage TF32 math,
FP8 E4M3/E5M2, and INT8 with int32 accumulation. FP64 m8n8k4 DMMA now owns a
distinct Tile lane map and f64 descriptor/bridge ABI; aligned and ragged RTX
5070 Ti rows match the f64 oracle with masked tails.
FP6 E2M3/E3M2 now assemble as `kind::mxf8f6f4`, m16n8k32,
UE8M0/`scale_vec::1X`; OCP/MXFP4 assembles as `kind::mxf4`, m16n8k64,
UE8M0/`scale_vec::2X`. Compiler-owned packed-memory Tile materializers,
five-buffer descriptors, CUDA-driver launch ABIs, and aligned/ragged numerical
proof now cover both FP6 encodings and MXFP4. In particular,
`fp4_e2m1` does not alias NVFP4: MXFP4's UE8M0 scale contract cannot reuse
NVFP4's UE4M3/`scale_vec::4X` scale words.
The shared MMA selector now requires explicit `math_mode="tf32"` for fp32 and
retains distinct `nvfp4` and `fp4_e2m1` K64 identities. No selector promotion
or production route changes.

The canonical dtype execution matrix records two disjoint, sample-interleaved
runs for square and ragged fp64/fp16/bf16/TF32/FP8/FP6/MXFP4/INT8 routes, with separate
CUDA-event and allocation/copy-inclusive timing, cold/warm image identity, and
ptxas register/shared-memory/spill fields. The retained 20-row collection
changes no selector. The final 31-sample, 10,000-device-launch and
50-end-to-end-launch run has 19/20 rows stable in both timing domains. The only
terminal miss is TF32 `256x256x256`: its device cohorts remain bimodal at 7.02%
while end-to-end is stable at 0.59%. That row is explicitly non-promoting; the
existing selector is retained rather than hiding the exact-device result.

The broader-family NVIDIA-E2E-2 reduction slice now carries
`tile.reduce_kernel(X,O,Outer,AxisExtent,Inner)` and an SM120-owned v2
materializer/descriptor ABI for f16/f32 sum, mean, and NaN-propagating max.
Normalized arbitrary axes and keepdims shape contracts execute through both a
single-owner serial schedule and a 128-thread cooperative shared-memory
candidate. Exact RTX 5070 Ti proof covers axes 0/1/2, keepdims on/off,
rectangular/ragged rank-3 inputs, f32 accumulation, non-finite values,
image/resource retention, and 42 numerical rows. The earlier last-axis record
remains historical evidence; the new comparative record applies the WSL 4%
foundation policy in both timing domains and changes no selector.

The canonical epilogue slice carries f16/bf16/TF32/FP8 E4M3/E5M2 bias,
ReLU/GELU/SiLU, optional f32
residual, and the explicit `matmul -> bias -> activation -> residual` order in
the Tile kernel plus launch descriptor. The CUDA materializer consumes distinct
bias/residual buffers and rejects unsupported dtype/order/shape contracts
instead of silently dropping epilogue semantics. The original 32 f16/bf16 rows
and the 48-case TF32/FP8 matrix pass exact-device execution. The comparative
record measures canonical single-kernel images against the existing production
composed routes, retaining both timing domains, cold/warm state,
image/resource fingerprints, spills, and raw disjoint cohorts. Production
selectors remain unchanged unless both domains select the same stable winner.

The first canonical attention slice adds a shared typed
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
f16/f32 storage, f32 accumulation/output, positive scale, and causal semantics.
The SM120 correctness-first materializer and four-buffer descriptor launch
through the shipped PTX bridge; exact RTX 5070 Ti proof passes 8/8
MHA/MQA, rectangular/ragged, causal/non-causal cases with zero spills. The
entry symbol includes the scale/causal semantic digest so incompatible images
cannot alias in the driver cache. Bias, window, softcap, dropout, and backward
are completed below. The retained eight-row
two-cohort baseline records CUDA-event and allocation/copy-inclusive timings,
cold/warm image identity, resources, and raw samples. A higher-amortization
rerun now has 8/8 rows within 3% in both domains. It remains historical
evidence; the final comparison below owns the production disposition.

The forward carrier now also owns optional dense f32 bias, signed left/right
window bounds, arithmetic softcap, and deterministic `lcg32_counter_v1`
dropout. These semantics participate in the image digest and descriptor
provenance. An exact-device advanced row proves causal+window+bias+softcap,
bitwise dropout replay, and malformed-bias rejection; the earlier 8-row
MHA/MQA matrix remains green. The f32 backward reference now also crosses the
compiler-owned seam through `tile.attention_backward_kernel` and a seven/eight-
buffer native descriptor. It assigns one dQ/dK/dV element to one thread,
performs fixed-order single-owner dK/dV reduction, requires
`deterministic=true`, and declares zero workspace. The exact-device GQA row
proves causal+window+bias+softcap derivatives, bitwise replay, descriptor-shape
rejection, and agreement with the shared Pade-softcap oracle. The final semantic
slice below adds matching f16 storage and dropout-mask replay.

The refreshed backward candidate matrix passes 6/6 exact-device oracle,
determinism, and workspace cases. All six atomic/split rows are stable in both
timing domains. Atomic wins both domains for MHA D64, causal MHA D128, and
ragged GQA; split/reduced remains the bitwise-repeatable option with one extra
dK+dV f32 workspace (134,144--524,288 bytes in the retained shapes). Production
already selects atomic, so the evidence retains that selector. The canonical
deterministic reference carrier is now landed; production selection continues
to be governed by the stable atomic/split corpus rather than the intentionally
serial reference materializer.

The paged-KV landing slice adds `tile.paged_kv_read_kernel` and a compiler-owned
f32-pages/i32-table direct descriptor ABI. Four exact-device boundary ranges,
two non-identity physical-page permutations, remap/reuse, and invalid-table
rejection pass. The existing 12-case fused/staged suite also remains green,
including causal offsets and page boundaries. The committed
`nvidia_sm120_e2e_spine_paged_kv.json` corpus compares canonical Tile-direct
against legacy CUDA staged gather at 128, 512-ragged, and 2048-ragged tokens.
It retains two repeated medians in both timing domains, cold/warm image and
cache state, registers, shared memory, occupancy, spills, and resource
fingerprints. This WSL foundation lane uses a 4% repeatability policy because
its graphics clocks are host-managed. All six candidate rows are accepted; the
legacy 2048 device-event row uses an explicit five-basis-point WSL margin at
4.02%, and margin-accepted rows are selector-ineligible. Timing-domain winners
also disagree at 512/2048, so the selector
remains unchanged. The SM120 foundation disposition is closed as retain-existing;
a future native-Linux controlled-host promotion attempt is a separate
hardware-environment follow-up, not an open migration dependency.

The stateful/MoE image slice adds compiler-owned Tile→NVIDIA→PTX packages for
ReplaySSM decode/flush and local f16/bf16/f32 MoE dispatch/combine/ragged
grouped GEMM.
The resident Replay handle no longer embeds those device kernels in its CUDA
host bridge: it loads the compiler-produced PTX functions while retaining the
session-persistent allocations, asynchronous ring, events, and ordering
contract. Compiler-owned MoE candidates launch through the generic descriptor
submission path. Exact RTX 5070 Ti tests cover
dispatch/combine numerical order,
zero-sized expert groups, ragged grouped GEMM, Replay transitions, persistent
workspace metadata, image identity, and resource retention.

The final comparative record contains 14 strict-stable rows for cooperative
softmax, four-warp forward attention, and local MoE dispatch/combine/grouped
GEMM. Every row retains two CUDA-event and allocation/copy-inclusive cohorts,
the discarded first lifecycle launch, 100-launch end-to-end amortization,
per-candidate clock conditioning, cold/warm image state, and exact resource
fingerprints. Production softmax and attention win both domains. MoE does not
produce cross-domain consensus across all three routes, so the existing MoE
selector is retained. ReplaySSM's higher-amortization 10-row matrix is now
10/10 stable in both timing domains. No selector changes.

The collective follow-on adds an explicit content-addressed rank/device
topology and a one-process/multiple-device NCCL executor for all-reduce,
all-gather, reduce-scatter, and grouped send/receive all-to-all. This host
exposes one CUDA device, so it proves deterministic topology/rejection and
records the two-device request as unavailable; it cannot supply the required
two-or-more-GPU numerical, topology, resource, or timing evidence. RCCL and
Apple mappings remain architecture-owned follow-ups. No collective or MoE
selector changes.

The remaining-dtype/reduction performance corpus uses production-sized square
and ragged TF32/FP8 fused epilogues plus f16/f32 arbitrary-axis reductions.
For every candidate it records first-use compilation/cache fill separately,
discards the first launch, and amortizes each device-event and end-to-end sample
over the next ten launches. Two disjoint time-interleaved 100-sample cohorts
retain raw samples, cold/warm or first/second-use state, image/resource
fingerprints, registers, shared memory, and spill fields. All 30 rows are
accepted under the WSL foundation rule: 29 pass the strict 4% gate and the
production fp16-mean reduction end-to-end row is explicitly margin-accepted at
4.099% under the user-approved 4.15% rounding bound. That row is
selector-ineligible. Seven strict rows have cross-domain winner consensus, but
the record changes no selector because stable consensus alone does not establish
a promotion policy or required material benefit.

The final SM120 semantic slice removes the remaining execution limitations.
The deterministic attention VJP now accepts matching f16 or f32 dO/Q/K/V and
gradient storage, accumulates in f32, and replays the forward
`lcg32_counter_v1` dropout mask from the semantic seed without a saved-mask
workspace. A compiler-owned `tile.paged_attention_kernel` consumes Q, K/V
pages, the i32 remap table, i64 logical token indices, and an explicit causal
offset in one fused descriptor; the offset is never inferred from allocation
capacity. MoE dispatch, deterministic combine, and ragged grouped GEMM now
accept f16, bf16, or f32 storage with int32 metadata, f32 combine weights, and
f32 grouped accumulation. Exact RTX 5070 Ti tests prove numerical agreement,
dropout bitwise replay, page remapping/causal boundaries, malformed metadata
rejection, and low-precision MoE execution. This shared Tile-carrier extension
transfers no CUDA schedule: Apple is not applicable because it owns a separate
resident paged-attention ABI and mature low-precision dispatch paths; ROCm
requires architecture-owned lowering before claiming these carrier variants.

Two hardware boundaries now have formal **deferred terminal** states for this
work item:

- exact two-or-more-GPU NCCL topology, numerical, resource, and timing proof is
  deferred because the available SM120 WSL host exposes one GPU;
- exact SM90 Hopper and SM100 datacenter-Blackwell Level-C evidence is deferred
  because neither exact target is available. Their compile-only Level-B
  artifacts do not inherit SM120 execution evidence.

Deferred hardware terminals do not authorize selector changes and do not hide
missing evidence. A future hardware follow-up must reopen its own exact-device
item under synchronization key `E2E-SPINE-2026-07-18`.

E2E-SPINE-3 is a shared-contract follow-up under the same synchronization key.
It is applicable to NVIDIA only as a family-granular evidence envelope around
the existing SM120 results: fixture identity, Level-C provenance, cold/warm
cache identity, benchmark metadata, and hash-sealed release-packet validation.
It changes no CUDA schedule, ABI, dtype capability, or selector. SM90, SM100,
and exact multi-GPU rows remain explicit hardware-deferred terminals and may
not inherit the SM120 packet.

The E2E-SPINE-3 exact-host recorder now packages all eight bounded SM120
families through compiler-owned image/descriptor seams: matmul, softmax,
reduction, fused epilogue, attention, paged-KV, ReplaySSM, and MoE. The six
formerly pending family rows add shared differential fixtures where needed,
prove cold/warm image and descriptor identity, retain selected route plus ptxas
resource fingerprints, and record independently conditioned repeated-median
device-event and allocation/copy-inclusive end-to-end rows. The hash-sealed
WSL RTX 5070 Ti packet is checked in against landed source commit
`9da32b78c37fc3bebf3f69d575e7b1eb4013a399`; all 16 timing rows pass the
unchanged 4% stability gate. Family-granular recording prevents one noisy
family from invalidating already-stable evidence while the final manifest
still seals one `(nvidia_sm120, sm_120a)` packet. No CUDA selector changed.

The LLVM-stage device-library follow-on makes CUDA `libdevice` an explicit
compiler dependency rather than accidental driver behavior. Native-image
identity now retains logical device-library name, content digest, and link mode
without serializing host paths. The SM120 packager fingerprints
`nvvm/libdevice/libdevice.10.bc` and uses `llvm-link --only-needed` whenever
translated LLVM IR retains an unresolved `__nv_*` call. A real `__nv_sinf`
fixture links through CUDA 13.3 libdevice, lowers with LLVM 23 `llc`, and
assembles with `ptxas -arch=sm_120a`. Intrinsic-only kernels retain an empty
linked-library set, while the available libdevice digest still participates in
the toolchain/cache fingerprint. This changes no runtime selector.

The CUDA floating-point follow-on separates three semantic routes: IEEE
arithmetic operators, function-specific CUDA libdevice calls, and explicit PTX
approximations. The shared softmax envelope now carries
`exp_mode="approx_exp2"` and `ftz=false`; SM120 accepts only that proven mode
and lowers it to `ex2.approx.f32`. The contract records PTX's full-range 2-ULP
bound, requires a nonzero near-zero comparison budget, and versions native
cache identity independently of `-O3`. It does not reuse the `__expf` accuracy
table for a different instruction and does not enable global fast math.
The semantic authority is NVIDIA's
[floating-point computation appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html);
instruction-specific accuracy comes from the
[PTX `ex2` specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

The CUDA Math API scalar/integer follow-on records representative integer math,
bit, packed-dot, numeric/bit-cast, and 2x16/4x8 packed-SIMD families. A CUDA
13.3 `nvcc -arch=sm_120a` fixture proves the documented symbols compile, while
this original synchronization point kept every Tessera Target-IR/runtime state
`planned`. `NVIDIA-PACKED-MATH-2026-07-25` first promoted a bounded subset; the
structured continuation recorded below now closes the listed bit, bit-cast,
and packed-SIMD families through typed Target IR and 27 exact-device cases.
The shared rounding vocabulary now represents CUDA's four conversion suffixes
RN/RD/RU/RZ exactly; nearest-away and stochastic modes cannot silently map to a
CUDA cast. Undefined signed-min absolute value, out-of-range float-to-integer
conversion, funnel-shift wrap/clamp, signedness, lane width, and saturation are
retained as contract boundaries. The later internal Tile route adds no public
Graph op or selector. Sources: [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html),
[integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),
[integer math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html),
[casts](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html),
and [packed SIMD](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

The PTX 9.3 truth audit now separates CUDA C++ storage spelling from physical
PTX typing. Fundamental storage rows are fp64 `.f64`, fp32 `.f32`, fp16 `.f16`,
and int8 `.s8`; BF16, TF32, FP8, FP6, FP4, and NVFP4 are alternate instruction
formats carried in same-width bit registers. Tensor fragments explicitly name
`.f64` or packed `.b32` operands. This corrects BF16 scalar/vector status to
`conversion_only` and prevents CUDA header types from implying fundamental PTX
types. PTX operand compatibility never performs automatic numeric conversion.
Direct `ptxas -arch=sm_120a` proof assembles the fundamental register surface
and rejects `bf16`, `tf32`, `e4m3`, and `u8x4` as register declarations.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It adapts the
shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, packages
HSACO, and submits through an exact gfx1151 HIP descriptor hook. NVIDIA's
`tessera_nvidia` lowering, `ex2.approx.f32` math contract, PTX ABI, SM120
schedule, resource/timing evidence, and selectors are unchanged. No AMD
wave/LDS or OCML behavior transfers to CUDA. ROCm's subsequent use of the
shared device-library record for its driver-selected OCML/OCKL/OCLC set is
parity validated at the schema boundary and requires no CUDA record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to NVIDIA target state. It adds no canonical dtype or alias and does
not change the SM120 PTX storage/Tensor Core contract, fragment ABI, runtime
readiness, or selector; it only prevents RDNA3.5 ISA formats from being
conflated with Tessera gfx1151 execution support.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. NVIDIA parity is validated at that
logical contract; NVFP4 and NVIDIA packed-weight/Tensor Core ABIs remain
distinct and backend-owned. No PTX capability, fragment ABI, runtime route, or
selector is promoted by the gfx1151 proof, and unsigned packed-4 remains
unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so CUDA schema parity is validated; no NVIDIA ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens its portable verifier to
admit bf16. At that synchronization point NVIDIA's backend-specific
materializer still accepted only f16/f32, so the ROCm five-argument HSACO ABI
transferred no CUDA claim. That historical NVIDIA boundary is superseded by
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25`, which owns its independent
`Outer/AxisExtent/Inner` PTX ABI, serial/cooperative-128 lowering, resources,
and exact-SM120 evidence without inheriting the ROCm schedule or selector.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. NVIDIA's existing direct PTX mapping remains parity validated; no ROCm
gather schedule, HSACO ABI, page-table validation evidence, timing, readiness,
or selector state transfers to CUDA.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. NVIDIA's typed PTX mapping remains
parity validated at the carrier boundary; no AMD gather schedule, HSACO ABI,
gfx1151 evidence, timing, readiness, or selector state transfers to CUDA.

The accompanying PTX memory contract records CTA/cluster/GPU/system scopes and
relaxed/acquire/release/acq_rel atomic semantics. Vector and packed memory
accesses are sets of scalar accesses in unspecified element order, not one
atomic unit; mixed-size races fall outside the model; `red` does not form an
acquire pattern; texture/`ld.global.nc` accesses are excluded; ordered CUDA
submission does not establish intra-kernel memory order. Sources:
[types and state spaces](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces-types-and-variables),
[instruction operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-operands),
and [memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

The first focused CUDA parity proof on the NVIDIA box is:

```bash
python3 -m pytest -q \
  tests/device/nvidia/test_tile_fragment_compiler_path.py \
  tests/unit/test_nvidia_fragment_layout.py \
  tests/integration/test_nvidia_paged_kv_native.py \
  tests/integration/test_nvidia_replay_ssm.py \
  tests/device/nvidia/test_flash_attention.py \
  tests/device/nvidia/test_flash_attention_backward.py

python3 benchmarks/nvidia/benchmark_serving.py \
  --shapes 1x128x64 1x256x128 \
  --tokens 64 --chunk 4 --slots 4 \
  --kv-tokens 128 512 2048 --heads 8 --dim 64 --page-size 16 \
  --reps 20 --output /tmp/nvidia-sm120-serving.json

python3 benchmarks/nvidia/record_hot_path_baseline.py --reps 20 --margin 2.0
```

The focused pytest command is a correctness loop, not a substitute for the
marker-separated full CUDA lanes above. Benchmark outputs under `/tmp` are
review artifacts only; update committed baselines or autotune corpora only
after two stable runs and an explicit before/after review.

## Next update

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
CUDA remains `native_gpu` with its existing event and end-to-end timing domains;
no PTX ABI, SM schedule, device evidence, readiness, or selector state transfers.
The x86 pilot consumes existing Tile softmax/reduction carriers without changing
their shared dtype or operation registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. NVIDIA inherits no
x86 ABI, vector schedule, host timing, readiness, or selector state. SM120
GQA/dropout and dtype breadth remain governed by NVIDIA-owned Target IR and
exact-device evidence; x86's narrower descriptor contract changes no CUDA row.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. NVIDIA parity is not applicable; no NVIDIA pipeline,
PTX ABI, schedule, capability, or selector changes. X86-E2E-2 subsequently
closed the remaining inventory and reassessed NVIDIA at each shared-contract
boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. NVIDIA parity is assessed at the carrier boundary only;
the AVX-512 ABI, CPU schedule/timing, 16K binary selector threshold, and exact
x86 evidence transfer no PTX implementation or CUDA selector claim. Existing
NVIDIA elementwise target and execution rows are unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. NVIDIA inherits no C ABI, null-operand convention,
32K selector threshold, CPU timing, PTX implementation, or CUDA selector
claim; NVIDIA target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
NVIDIA parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, PTX routes, and CUDA selectors
do not transfer. Existing NVIDIA rows remain unchanged.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. NVIDIA already owns
independent dtype, MMA, accumulator, PTX, and runtime contracts; no CUDA target,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, NVIDIA target capabilities, PTX ABIs, schedules, and
selector state are unchanged; NVIDIA parity is validated by the shared
attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; NVIDIA uses the
versioned apt LLVM 23 packages alongside CUDA 13. NVIDIA target semantics, PTX
ABIs, and selectors are unchanged, and the LLVM 23 compiler/lit build validates
host-free parity; exact-device claims remain NVIDIA-owned.

The collection contract, compiler-artifact, exact-device correctness, and
serial measured lanes now have recorded baselines. NVIDIA-TEST-5 and
NVIDIA-TEST-6 are closed; the requested attention, epilogue, and legacy-retune
parity records are stable without a selector promotion. Keep `plan_state:
landing` while unrelated implementation or exact-device follow-ons remain.
Move this plan to the NVIDIA archive only after every completion gate is met.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes the NVIDIA families as a **lead performance target** (Decision #28 — its
`wgmma`/`mma.sync` candidates set the ceiling and are never capped by the shared
mixer framework). It adds candidates under existing families, opening no new
NVIDIA-TEST item: channel-wise KDA/GDN decode → **NVIDIA-TEST-3/-5 KV/ReplaySSM**;
`sliding_window`/full mixer fwd + backward → **attention** (split/reduced dK/dV,
G6-C-style); chunkwise-scan inner GEMMs → **GEMM/Tile** (`wgmma` sm_90 / `mma.sync`
sm_120, preferably via the NVIDIA Tile IR lowering target); NVFP4/MXFP8 mixer GEMMs
→ **NVIDIA-TEST-4** numerical policy (this is the executing FP4 lane — sm_120
`mma.sync`, not `tcgen05`). Inherits the TEST-3 native-provenance / TEST-5
kernel-vs-E2E evidence contract unchanged. Direction pointer only; no NVIDIA gate,
route, or exact-device claim changes here.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. NVIDIA parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no PTX/CUDA implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, PTX/CUDA ABI, NVIDIA schedule, dtype capability,
execution row, or selector. NVIDIA parity is therefore not applicable.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the NVIDIA not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` hardens the LLVM 23 and
Linux TSAN host environment and repairs an Apple-only capability row. NVIDIA
parity is not applicable: no CUDA dtype, PTX ABI, schedule, execution row,
selector, or exact-device evidence changes.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus. NVIDIA fixture-schema parity
is validated, but the gfx1151 HSACO, HIP ABI, resources, timing, and
exact-device evidence do not transfer to CUDA. The ROCm-owned compiler lane
explicitly excludes `compiler_nvidia`; no NVIDIA capability, test ownership,
schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` closes shared Graph/Neighbors
verifier gaps and records the shared `sm_120` MMA selection in NVIDIA manifest
rows. Equal-tier candidates may use its analytical accumulator footprint only
after route-tier precedence. This is parity validated at the host-free
compiler/manifest boundary; it changes no PTX instruction schedule, automatic
selector promotion, CUDA ABI, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` makes compute dtype
legalization the default in NVIDIA named pipelines. Terminal storage
legalization remains intentionally opt-in for the generic value-level CUDA
route because it has no block-scale operand ABI. The later
`NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` continuation closes the physical
consumer gap for the scale-bearing NVFP4/MXFP4/FP6 launch envelopes and wires
StoragePackConsume after opt-in terminal legalization; at that point generic
INT4 and a sub-byte default remained unsupported. The executable row-major layout
materializer and guarded
dynamic launch are x86-only and transfer no PTX schedule, CUDA ABI, bucket
policy, selector, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. At that synchronization point the architecture-owned
Graph-cast materializer was open; it landed in the later NVIDIA continuation
recorded below. The pass stays opt-in and transfers no PTX layout, schedule,
selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to bucketed tensor-core routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no CUDA backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to bucketed
tensor-core routes; no CUDA ABI, schedule, selector, backward runtime, or
exact-device claim transfers. NVIDIA's architecture-owned Graph-cast consumer
is host-validated: after shared legality it accepts row/column-major/BHSD/NHWC,
removes the Graph marker, and carries the binding into `tile.async_copy`.
This changes staging metadata only and claims no PTX schedule or device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. This shared sync added no PTX/CUDA execution; the
architecture-owned affine backward ABI, runtime binding, and exact-SM120 proof
land in `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` below. It does not imply a
tensor-core schedule or selector promotion.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. The NVIDIA dynamic
affine materializer and backward runtime were still open at this sync and are
closed by the architecture-owned continuation below. The gfx1151 HSACO and
AVX-512 ABIs, schedules, and timings did not transfer to CUDA/PTX; no selector
promotion was inferred from sibling evidence.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. The
CUDA/PTX backward materializer and exact-device proof were supplied later by
the NVIDIA-owned continuation below; the gfx1151 two-kernel schedule, bitwise
evidence, and timing did not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. The then-open NVIDIA execution row is now closed by the CUDA/PTX
continuation below. Neither the gfx1151 HSACO ABI nor the AVX-512 ABI,
schedule, timing, or device evidence transferred; the NVIDIA implementation
retains its own descriptor and exact-device proof.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. NVIDIA parity is host-validated through the shared linalg contract.
All NVIDIA backend variants now execute Tile buffer reuse and materialize one
address-space-3 shared-memory arena with typed planned-offset views before
Tile-to-NVIDIA/NVVM lowering. Function-budgeted liveness-aware
rematerialization also runs in the shared production post-autodiff pipeline.
Exact CUDA/PTX shared-allocation assembly and occupancy were open at this
shared sync; the static and dynamic CUDA evidence is recorded in the NVIDIA
closeouts below. No selector promotion is implied.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. NVIDIA parity is host-validated
at the shared IR boundary. The gfx1151 HIP composition/module cache and
AVX-512 execution do not transfer to CUDA/PTX. The NVIDIA-owned compiled MSE
backward launch and exact-device evidence land in the continuation below; no
tensor-core training schedule or selector promotion is claimed.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized address-space-3 arena planning and a benchmark-fed
rematerialization cost contract. NVIDIA retains opt-in Graph layout assignment
through its existing materializer. The new MSE backward launch and numerical
proof are ROCm gfx1151-only. The architecture-owned CUDA VJP and dynamic
shared-allocation/occupancy proof land in the later NVIDIA closeouts; selector
evidence remains explicitly non-promoting on WSL.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. The architecture-owned CUDA/PTX backward materialization and
exact-device evidence land in the continuation below. The gfx1151 generated
HIP kernel, AVX-512 C ABI, boundary timing, caches, and selector state did not
transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. The NVIDIA CUDA/PTX materializers and exact-device evidence
land in the continuation below, including KL/JS and FP16/BF16 storage. The
gfx1151 and AVX-512 loss and optimizer ABIs did not transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. This sync
validated only the shared Graph/Linalg contract; the architecture-owned
CUDA/PTX fused materializer and exact-device evidence land below. gfx1151 HIP
and AVX-512 ABIs, cache identities, timings, and selector decisions did not
transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. At that point exact CUDA assembly,
resource, occupancy, and performance evidence were open and not inferred from
gfx1151; the NVIDIA-owned static/dynamic closeouts below now provide them.
No NVIDIA selector or default policy changes.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. The later NVIDIA packets provide
CUDA measurements; native-Linux policy selection remains deferred. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no PTX ABI, shared-memory allocation,
packed consumer, performance, or selector claim. The existing NVIDIA
architecture-owned layout consumer remains unchanged.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24` extends the
shared rematerialization corpus schema with softmax, RMSNorm, and MSE producer
families plus measured workload-budget decisions. The later NVIDIA packets
provide CUDA measurements; native-Linux policy selection remains deferred.
ROCm's packed
multi-arena LDS ABI, GELU normalization epilogue, and terminal-pack
dequant-GEMM consumer are architecture-owned; no PTX shared-memory ABI, packed
consumer, timing, selector, or support claim transfers. CUDA path-max and
general serialized launch expressions are closed below; physical packed
consumers remain governed by their own dtype rows.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24` extends the
shared rematerialization evidence schema to a measured four-layer workload with
softmax, RMSNorm, MSE, Huber, SmoothL1, and BCE instances. CUDA measurements
now land in the NVIDIA continuation below; controlled native-Linux policy
selection remains deferred. ROCm's
branch-path dynamic-LDS expression, binary normalization epilogues, and packed
elementwise/sparse/cache ABIs are architecture-owned; no PTX shared-memory
expression, packed ABI, timing, selector, or support claim transfers.

Cross-backend sync `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24` adds a shared
model/device-derived rematerialization budget contract with explicit override
precedence and bounded dynamic parameters. The exact CUDA-context capacity,
free-memory cap, reserve policy, parameter bounds, and measured packet now land
in the NVIDIA continuation below. ROCm's alias-aware nested/loop LDS slots and 40,208-byte
gfx1151 packet are architecture-owned; no PTX shared-memory expression,
occupancy, execution, or selector claim transfers.

NVIDIA-owned closeout `E2E-SPINE3-SM120-MEMORY-2026-07-24` completes the
static NVPTX boundary named by `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT`.
Tile lowering turns three logical 512-byte allocations with disjoint lifetimes
into one 1,024-byte address-space-3 arena at offsets `[0, 512, 0]`, reducing
the unreused 1,536-byte plan. LLVM 23 emits an exact 1,024-byte NVPTX shared
declaration; `ptxas` reports the declaration and the retained executable route
has matching tool-reported shared-resource accounting. Exact SM120 execution
compares that retained route with a register-rematerialized expression:
both return 42, have complete no-spill resource records and 100% theoretical
occupancy, and pass the 4% two-run gate in device-event and end-to-end domains.
The evidence is intentionally selector-ineligible. Dynamic/path-expression
arenas and model-level CUDA rematerialization policy remain separate follow-ups.

NVIDIA-owned continuation `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24`
closes the named SM120 execution gaps for dynamic-affine RMSNorm/LayerNorm
backward; MSE, MAE, Huber, SmoothL1, stable BCE-with-logits, KL/JS, and
class-index/label-smoothed cross-entropy backward; deterministic general
broadcast-gradient reduction; SGD, Momentum/Nesterov, Adam, and AdamW updates;
and fused loss-backward plus SGD/AdamW. FP32, FP16, and BF16 storage preserve
their physical dtype while every arithmetic/reduction path accumulates in FP32. The
architecture-owned path generates CUDA, compiles an immutable PTX image,
binds a launch descriptor, and executes through the shipped CUDA-driver
bridge. Public runtime artifacts and `@jit(...).native_backward()` use the
same descriptor seam. Exact SM120 tests cover dynamic/ragged extents,
transition boundaries, extreme logits, none/sum/mean cotangents, optimizer
state, fused-versus-composed numerics, invalid labels, cache identity, and
live resources.

The same continuation completes CFG-forwarded and locally computed dynamic
shared-memory sizing. A post-LLVM NVIDIA pass replaces runtime-sized
address-space-3 byte arenas with slices of an external NVPTX shared symbol,
colors mutually exclusive lifetimes into one slot, keeps simultaneously live
arenas in distinct aligned slots, and serializes checked constant, argument,
add, multiply, cast, select, and path-max launch expressions. The v2 CUDA
launch ABI evaluates the descriptor expression and passes the resolved byte
count to `cuLaunchKernel`. Exact SM120 execution proves the original
12,289/32,001-byte branch paths and a CFG-forwarded
`max(4096*4+17, 12289)` expression with a 16,416-byte allocation, rejects
undersized descriptors, and retains driver-JIT register/static/local-memory/
occupancy evidence.

The native bridge also reports total/free bytes from the same retained CUDA
context. The compiler policy caps usable capacity by current free memory,
marks static or explicitly bounded dynamic Graph IR model parameters, and
stamps reserve, gradient-copy, optimizer-state, and persistent-state inputs
consumed by the shared activation-rematerialization pass.

The checked-in 26-row repeated-median packet discards the first/JIT launch,
uses 1,000 device-event repetitions and 20 end-to-end iterations, and retains
cold/warm image identity plus artifact and resource fingerprints. Fourteen
rows meet the 4% two-run gate in both timing domains on this WSL2 host; twelve
retain explicit unstable dispositions, with the two tiny dynamic-shared probes
remaining especially host-noisy. Fused MSE+SGD and MSE+AdamW are stable and
beat their composed references in both timing domains, but their references
are not stable and WSL is not the controlled native-Linux promotion host.
Therefore no production selector changes. Rerunning this exact packet on
controlled native Linux is the remaining performance/selector boundary.

NVIDIA-owned closeout `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` extends the
compiler-owned SM120 Tile-image seam from FP16/FP32 to BF16 input storage with
FP32 accumulation and output for reduction, stable row softmax, and attention
forward/backward. The reduction contract covers sum, mean, max/min and
amax/amin aliases, arbitrary static axes, keepdims true/false, ragged extents,
NaN propagation, and both serial and cooperative-128 physical schedules.
Every BF16 descriptor carries a distinct typed ABI and two-byte input binding;
the CUDA-driver bridge retains four-byte FP32 result bindings. The canonical
Graph verifier, target capability, backend manifest, execution matrix, Tile
verifier, NVIDIA lowering, runtime registration, and launch bridge agree on
that boundary.

Exact RTX 5070 Ti SM120 execution proves 44 focused BF16 softmax, reduction,
and attention cases through MLIR to NVVM, immutable PTX image, descriptor, and
CUDA-driver launch. Compiler lit proves BF16 extension into FP32 arithmetic,
serial/cooperative reduction assembly, min lowering, and attention
forward/backward materialization. The checked-in 12-row reduction packet
compares both schedules across all six public kind spellings, discards the
first launch, amortizes 500 resident launches and 10 complete calls, and
retains two disjoint repeated-median cohorts, cold/warm image and entry-symbol
identity, numerical error, ptxas registers/shared-memory/spills, and live
driver local-memory/occupancy. WSL timing remains an explicit non-promotion
result: only 3/12 candidates meet the 4% gate in both timing domains and no
candidate has stable cross-domain winner consensus. Production selectors are
therefore unchanged; controlled native-Linux comparison remains required
before any schedule promotion.

The continuation of `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` closes the
remaining normalization envelope. Compiler-owned immutable PTX images now
execute unweighted RMSNorm/RMSNorm-safe and LayerNorm with f32, f16, or bf16
storage, f32 row statistics/accumulation, immutable nonnegative epsilon, and
same-storage output. Eighteen exact SM120 cases cover ragged/non-power-of-two
rows, multiple ranks, both normalization kinds, all three storage types,
resource/no-spill inspection, warm image/cache/descriptor identity, and FP32
oracles. Dynamic affine BF16 backward remains on the previously proven
training descriptor, so forward and backward now share the BF16 storage
boundary without claiming an affine forward fusion in this unweighted kernel.

The same synchronization point lands NVIDIA's first physical consumer of the
shared terminal packing descriptor. Canonical NVFP4, MXFP4, FP6-E2M3, and
FP6-E3M2 launch images carry `tessera.storage_pack`; NVIDIA lowering requires
the logical format, int8 container, packing factor (2 for four-bit, 1 for
six-bit), and format-defined signedness to agree with the selected
scale-bearing fragment ABI before generating packed byte/nibble loads.
Descriptor drift rejects in compiler lit, while exact general-shape/ragged
SM120 tests traverse the descriptor-driven loaders. Opt-in terminal storage
legalization now runs `StoragePackConsume` in the NVIDIA named pipeline.
The later capability expansion admits only the proven value-level decode,
unscaled round-trip, and matching packed-matmul def-use paths. It does not make
terminal packing the default for arbitrary FP4/FP6 values or for Graph-level
quantize/dequantize operations.

NVIDIA-owned continuation `NVIDIA-PACKED-MATH-2026-07-25` adds the missing
canonical signed-INT4 consumer. Its compiler-owned Tile image and typed CUDA
launch ABI accept only an int8 container, factor two, signed two's-complement
nibbles, low logical index in the low nibble, i32 accumulation/output, and no
scale or fused-epilogue operands. The correctness-first general-shape schedule
decodes packed A rows and packed B columns, guards ragged M/N/K edges, and
rejects container, factor, signedness, scale/operand, and epilogue disagreement
before PTX materialization. Exact aligned and ragged RTX 5070 Ti execution
matches an integer oracle, retains zero ptxas spills, and proves cold/warm
image and descriptor identity. NVFP4/MXFP4/FP6 keep their distinct
scale-bearing fragment semantics; no physical schedule is transferred between
those formats or from ROCm.

The 2026-07-25 continuation replaces the dictionary-only pack record with
portable `#tile.packed_format`, `#tile.scale_layout`, and
`#tile.packed_view` attributes plus generic `tile.packed_load`/
`tile.packed_store`. The contract records logical bit width independently of
the int8 container (notably FP6 factor one), signedness/encoding/lane order,
packing axis/strides, alignment/offset, and an explicit scale operand/layout.
SM120 lowering decodes signed INT4, FP4/NVFP4, and both FP6 encodings, applies
origin-aware scale indexing for non-origin views, guards ragged bounds, and
supports unscaled packed round trips. Descriptor, axis, scale, alignment, and
store disagreement fail closed. Terminal marking is capability-filtered by
target + operation + format + available consumer; unsupported operations stay
logical rather than inheriting launch-envelope support.

The generic value-level consumer now also owns a CUDA runtime ABI rather than
ending at compiler lit. Exact RTX 5070 Ti execution covers non-origin/ragged
NVFP4 with explicit UE4M3 scale binding, signed INT4, and FP6-E2M3/FP6-E3M2
with explicit UE8M0 scale binding. All four cases match their format oracles,
retain zero ptxas spill bytes, and reproduce cold/warm image plus launch
descriptor identity. Source/scale byte extents are launch scalars checked
against the serialized physical view; the bridge rejects negative origins,
nonpositive extents, overflow, or buffer/descriptor disagreement. Generic
packed stores retain host-free compiler round-trip proof; no unsupported
format is default-enabled by this runtime addition.

The checked-in SM120 packed-storage packet measures the production-sized
`4097x4099` ragged view for signed INT4, NVFP4, FP6-E2M3, and FP6-E3M2. Each
device observation amortizes ten resident launches, each cohort retains seven
device-event observations and ten end-to-end submissions, and the first
allocation/copy/JIT-inclusive submission is discarded. Every row reproduces
its compiler image, launch descriptor, cache fingerprint, and 18-register,
zero-local-memory resource record. On this WSL host all four rows meet the 4%
two-run gate in both timing domains: the device-event deltas are 0.62%, 0.57%,
0.32%, and 0.08%, while the end-to-end deltas are 0.27%, 0.09%, 3.46%, and
0.09%. WSL evidence remains selector-ineligible, so every row records an
explicit retain disposition. No selector or terminal-legalization default
changes from these timings.

CUDA Math now crosses a registered `tessera_nvidia.cuda_math_kernel` seam
instead of extending an open five-pointer string dispatcher. Its closed
verifier covers scalar `brev`, `prmt`, `clz`, `ffs`, `popc`, numeric and
bit-preserving f32/i32 casts, packed 2x16 and 4x8 signed/unsigned wrapping or
saturating arithmetic, byte absolute difference, and both lane-mask and
predicate-bit comparisons. All 27 exact RTX 5070 Ti cases launch, match
bit-exact/numerical oracles, retain zero spills, and reproduce warm image and
descriptor identity. No production selector changes.

The structural continuation registers `!tile.buffer`, `tile.alloc`,
`tile.dealloc`, `!tile.pipeline_state`, `tile.pipeline_init`, and
`tile.pipeline_advance`. It now also registers SSA TMA descriptor, mbarrier,
mbarrier-token, TMEM, and TCGen05 types/operations. WarpSpecialization allocates
SMEM/TMEM handles in the parent region, threads them into staged copies and
consumers, threads producer/consumer pipeline states, and deallocates only
after CTA synchronization. AsyncCopy lowering emits registered TMA descriptor
and copy operations; descriptor deduplication assigns slots, creates one SSA
mbarrier, binds it to every copy, and threads copy completion tokens into the
typed wait. FlashAttention emits typed arrive/try-wait token chains.
Barrier-reuse legality passes on this real WarpSpec output. NVIDIA no longer
emits or consumes name-based `#tile.buffer_ref`; the shared reader remains only
for Apple/ROCm migration fixtures. Annotation-only `#tile.pipeline_state`
compatibility metadata is rejected, and the structured Schedule→Tile layout is
consumed directly. TCGen05/TMEM has host-free structural and SM120 fail-closed
proof only; exact execution remains SM100-owned and cannot be inferred from
consumer Blackwell.

Cross-backend sync `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds ROCm-owned
Adam/AdamW and KL/JS physical backward execution plus a ROCm normalization
softcap epilogue; none of those HIP kernels, gfx1151 timings, or selector
evidence transfers to CUDA. The shared change is the target-neutral,
serializable dynamic-local-memory expression field on `LaunchDescriptor`.
NVIDIA's existing SM120 add/multiply/path-max/alignment probe now consumes
that field and retains its CUDA-owned PTX, launch-v2, resource, and exact-device
evidence. No NVIDIA execution row or selector changes.

Cross-backend sync `ROCM-LION-BACKWARD-2026-07-27` adds only the ROCm-owned
physical consumer of the already-shared Lion stop-sign VJP policy and extends
the gfx1151 operation-total benchmark packet. HIP code objects, AMD launch ABI,
and WSL timings do not transfer to CUDA. NVIDIA remains follow-up required for
an architecture-owned compiled Lion backward materializer; no SM120
capability, execution row, PTX schedule, or selector changes.

**NVIDIA follow-on update (2026-07-30):** the CUDA-owned Lion materializer now
has landed as `nvidia_lion_bwd_compiled`: an SM120 PTX package with an explicit
eight-buffer ABI (`p/g/m`, two output cotangents, and `dp/dg/dm`), CUDA bridge
layout, runtime executor, execution-matrix row, f32 oracle, and RTX 5070 Ti
device validation. It implements the shared stop-sign VJP without importing an
AMD code object or schedule. This is a correctness-first 128-thread package;
no optimizer selector changed and a timing packet remains a separate SM120
measurement task.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` emits a shared
unique-clock warmup/steady/cooldown dependency order after pipeline legality.
At this synchronization point CUDA runtime consumption and collective overlap
remained NVIDIA-owned follow-up; the immediately following
`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record supersedes that structural
gap with the shared runtime consumer. The carrier itself changes no SM120
capability, PTX schedule, selector, or exact-device claim.

Cross-backend sync `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` supplies a shared
runtime consumer for emitted 1F1B steps with independent collective transport,
and makes measured schedule records alter the actual Schedule/Tile M/N/K,
warp-count, and stage attributes after target/evidence validation. NVIDIA's
named pipelines now default layout assignment on because the architecture-owned
Graph-cast materializer immediately consumes the markers; focused structural
proof covers ordering. A real CUDA multi-rank transport packet and measured
SM120 selector application remain NVIDIA-owned exact-device follow-ups.

The same sync replaces DeltaNet-family finite-difference reverse mode with an
analytic carried-state recurrence and explicit reverse-token scheduling.
Directional-derivative fixtures validate shared semantics only; CUDA backward
packaging and device scheduling remain follow-up required. ROCm factored
Adafactor HSACO, HIP capacity query, gfx1151 numerics, and WSL timing do not
transfer to CUDA.

Cross-backend sync `CORE-PRODUCTION-EVIDENCE-2026-07-27` serializes collective
descriptors on emitted pipeline steps and binds OptimizerShard ownership
transitions to the shared runtime. NCCL remains the CUDA-native executor, but
this ROCm-host continuation contains no real multi-rank CUDA packet and changes
no SM120 selector. The gfx1151 Adafactor adjoint and two-entry DeltaNet
reverse-chunk HSACO (later superseded by the five-entry AMD package) are AMD
schedules and do not transfer. CUDA sequence-mixer
backward packaging and a refreshed measured selector packet remain
NVIDIA-owned exact-device follow-ups.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` adds the
exact modified-Delta normalization VJP to physical ROCm and AVX-512 backward
paths and proves an affine parallel chunk-composition algorithm for
`erase=false`. This changes shared algorithm evidence, not CUDA execution. The
five-entry gfx1151 HSACO, AMD workgroup schedule, and WSL resident timings do
not transfer to SM120. CUDA sequence-mixer backward packaging, nonlinear/erase
chunk scheduling, and a refreshed exact-CUDA-host selector packet remain open.

**CUDA DeltaNet reverse architecture (2026-07-30):** NVIDIA owns a separate
four-stage package: (1) fp32 state checkpoints per `(batch, head)` trajectory;
(2) an erase-free affine chunk-summary/prefix path; (3) a serial nonlinear or
`erase=true` checkpoint-fill path; and (4) reverse-token gradients with unique
`(batch, head)` ownership for Q/K/V/gate/beta/decay. The modified-normalization
derivative is part of stage 4, not a reuse of an AMD or AVX implementation.
The current CUDA forward recurrence is intentionally unchanged. Do not add a
`nvidia_deltanet_bwd_compiled` execution row or selector candidate until this
CUDA package has exact f32 numerical tests plus SM120 device-event and
end-to-end timing cohorts; ROCm/x86 packets are non-evidence for that decision.

**Implementation update (2026-07-31):** the versioned v2 CUDA package now
keeps its fixed 13-buffer/10-scalar ABI while executing the validated bounded f32
`Dqk,Dv <= 8` reverse recurrence. It has CUDA-owned analytic gate, beta, and
decay derivatives, followed by serial replay for `erase` and modified-update
normalization; direct-package and public JIT exact-device tests compare all six
gradients against the shared oracle. This is correctness evidence only: no
execution-matrix promotion or selector choice has been made until the required
SM120 timing and counter packet exists. Apple, ROCm, and x86 are **not
applicable to this physical package**; their differently scheduled packages
and evidence remain sibling follow-ups rather than CUDA proof.

**Timing/NCU packet (2026-07-31):**
[`nvidia_sm120_deltanet_backward_2026_07_31.json`](../../../../benchmarks/baselines/nvidia_sm120_deltanet_backward_2026_07_31.json)
records two repeated CUDA-event and end-to-end cohorts at `[B,H,S,Dqk,Dv] =
[1,2,16,8,8]`: plain `0.64318 / 1.53015 ms`, affine `0.76717 / 1.54674 ms`,
and erase+modified serial-fill `0.92778 / 1.69752 ms`. All variants match the
oracle (maximum errors `7.45e-09`, `1.86e-09`, and `1.35e-04`). The packet
also records live resources (255 registers/thread, 1328 B local/thread, two
active blocks/SM) and two NCU L2/DRAM cohorts. Only serial-fill counters were
stable (94.77/94.70% L2 and 779,776/715,008 DRAM B); plain and affine counter
cohorts diverged materially, so they are evidence of collection instability,
not a schedule result. The packet changes no selector: there is one
correctness-first implementation and no controlled alternative to promote.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: follow-up required.** 8 files under this backend reference `FragmentType` / `!tile.fragment`. Two obligations, both already scoped in [`W1_1_TYPING_DESIGN.md`](../../compiler/W1_1_TYPING_DESIGN.md):

* **Step 2b (blocking for the motivating GEMM).** `NVIDIALowering.cpp` requires the accumulator's direct defining op to be `FragmentZeroOp` (3 sites). A K-loop accumulator is an `scf.for` iter-arg with no defining op, so a typed K-loop will verify and still fail to lower until this is block-argument-aware. Needs a per-backend *lowering* fixture, not just a verifier one.
* **Step 3.** `GenerateWMMA*`-equivalent producers migrate to the typed form one PR at a time.

`family` is a type parameter partly because `NVIDIALowering.cpp` gates on it before matching (m, n, k, dtype) to an `mma.sync` variant — leaving it on the attribute would have let a fragment packed for one family feed an op selecting another.

No exact-device evidence in this PR; none required, since no generated code changed.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: follow-up required — and larger than previously recorded.**

Same finding as ROCm under this key. `NVIDIALowering.cpp` synthesizes zero constants for the accumulator (`operands.append(4, zero)` for f32, and the f16/s32 equivalents) and never reads `mmaData[2]` as a value. So step 2b is accumulator threading plus an `scf.for` region-signature conversion, not a relaxed check — and relaxing it alone would emit a silently wrong GEMM.

No sm_120 device evidence in this PR; no generated code changed. When 2b lands, its gate needs numerics on real hardware for the same reason ROCm's does.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: follow-up required — this backend owned the defect.**

`NVWGMMALoweringPass` now refuses such an mma with `NVWGMMA_ACCUMULATOR_DROPPED` and calls `signalPassFailure()`. The check lives in the PASS BODY, not the pattern: a pattern that emits an error and returns `failure()` only declines to match, so the diagnostic printed while the tool still exited 0 and the pipeline continued — an error that does not fail compilation is a warning in disguise, the same fail-open shape as the bug.

The two-operand lane is untouched: `nvwgmma_lowering.mlir`'s emitted call is byte-identical before and after (diffed, not assumed).

**Follow-up:** W1.1 step 2b threads the accumulator for real, which needs an `scf.for` region-signature conversion. This guard is replaced by lowering then, not relaxed. No sm_120 device evidence here; no working codegen changed, only a refusal added.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: not applicable — architecture-specific reason.** The changed sites are all `_RocmCompiledUnavailable` raise points inside ROCm compiled-lane hsaco builders. NVIDIA's compiled lanes do not raise that exception and are untouched.

**Follow-up worth recording, not created here:** NVIDIA has no equivalent failure/envelope split on its own compiled paths, so the same masking may exist there. Establishing that needs an sm_120 host, which this box is not — asserting it either way from here would be the guesswork this thread has been eliminating.

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: follow-up required — recorded, not fixed here.** The equivalent NVIDIA seam is worse, not merely missing: `NVWGMMALoweringPass` lowered a `tile.mma` carrying an accumulator to a two-operand call and dropped it (`NVWGMMA-ACCUMULATOR-GUARD-2026-08-03`). That is guarded to fail closed; threading it for real is W1.1 step 2b on this backend and needs an sm_120 host for the numeric gate that ROCm just got.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: follow-up required — refuses the bounded form, and the refusal is UNVERIFIED on this box.**

`NVIDIALowering`'s fragment materializer emits an unguarded load, so ignoring `(rowBound, colBound)` would read past the edge of a ragged matrix. It now emits `NVFRAGMENT_BOUNDED_VIEW_UNSUPPORTED` naming op and target (Decision #21) instead of folding the case into the generic arity message, which would have read as "malformed IR" for IR that is well-formed and merely unsupported here.

**Explicitly not verified:** the NVIDIA dialect is off by default in this build, and neither `--tessera-lower-to-gpu` nor `--tessera-nvidia-pipeline-sm120` reached the materializer with a bounded view on this host — no diagnostic, no error. The code path is written and compiles; it has not been executed. Verifying it needs a build with `-DTESSERA_ENABLE_CUDA=ON` and, for the numeric half, an sm_120 host.

Until this materializer grows masking, a portable producer must emit the 3-operand form; only ROCm can consume the bounded one.

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

**ROCm has implemented and measured the shared answer.** `tile.view` now accepts
an optional precomputed `linear_base`, and the ROCm producer uses it for A-base
hoisting and sibling-B address sharing. At 2048^3 the final rebuilt ratio is
0.711x (15.45/21.74 TFLOP/s), still insufficient for promotion. The
remaining ROCm evidence points to fragmented load scheduling and excess waits.

**Outcome for NVIDIA: FOLLOW-UP REQUIRED.** This backend consumes `tile.view`
and `tile.fragment_pack` (8 files each), so it must consume the optional base
form correctly or fail closed when a producer selects it. No exact-device
evidence from an NVIDIA host is claimed. This is also the owner of W1.1's final two untyped C++ `tile.mma`
construction sites: both are tensor-valued producers in
`TileIRLoweringPass`. They must migrate together with NVIDIA's typed
tensor-to-fragment materializer and accumulator-threading gate; ROCm/x86
hardware cannot validate that physical decision.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for NVIDIA: FOLLOW-UP REQUIRED.**
The sm_120 fragment materializer fails closed on this valid shared form and
still requires a static leading dimension. No CUDA-enabled build or sm_120
device evidence is claimed from this ROCm/x86 host.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **NVIDIA
outcome: follow-up required.** SM120 packaging still consumes `GraphIRModule`
and synthesizes NVIDIA-owned Tile, so its lineage truthfully records the Graph
fork and remains incomplete. This change does not migrate the two untyped
`tile.mma` producers, accumulator threading, bounded/dynamic views, PTX, or a
physical schedule; no SM120 evidence is claimed. Those remain NVIDIA-owned
gates after the shared x86/ROCm vertical slice establishes the consumer API.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

Shared Graph→Schedule→launch-Tile lowering is now real for the initial x86-f32
and ROCm-f16/f32 matmul instances. **NVIDIA outcome: follow-up required, not
validated by this slice.** SM120 is deliberately outside the first bounded
dtype/descriptor selector and therefore fails closed rather than inheriting
another architecture's schedule. NVIDIA packaging still synthesizes its Tile
program from Graph IR. Its later consumer must accept the canonical scheduled
artifact only after the final two untyped `tile.mma` producers, accumulator
threading, and dynamic/bounded view gates pass in a CUDA-enabled SM120 lane.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The shared package boundary is now concrete: a validated
`ScheduledMatmulArtifact` carries exact Graph, Schedule, and launch-Tile text
plus content identities into x86 and ROCm physical consumers. **NVIDIA outcome:
follow-up required.** No CUDA code changed and no SM120 evidence is claimed.
The NVIDIA consumer remains blocked on its two untyped `tile.mma` producers,
accumulator threading, dynamic/bounded view support, and CUDA-enabled SM120
validation before it can adopt this boundary without recreating Graph intent.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

Schedule/Tile matmul now distinguishes instruction-tile shape from an
architecture-owned macro tile and carries that value through artifact identity
and launch provenance. **NVIDIA outcome: follow-up required.** This generic
contract is applicable, but no CUDA lowering, SM120 schedule, PTX, selector, or
device evidence changed. NVIDIA must choose its own macro tile after its two
untyped producers, accumulator threading, and bounded/dynamic view gates land;
the gfx1151 32x64 decision does not transfer.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The shared spine now has content-addressed `schedule.softmax` and
`schedule.reduce` SSA edges and atomically lowers the bounded canonical f32
contracts to launch-level Tile artifacts. **NVIDIA outcome: follow-up required
on a CUDA-enabled SM120 host.** Existing NVIDIA physical Tile softmax/reduction
lowering is unchanged, but SM120 packaging still synthesizes its Tile program
from `GraphIRModule`; no PTX, cubin, descriptor, schedule, selector, or device
claim changed. Its consumer must accept the exact scheduled artifact and run
the established SM120 numerical/performance gates. x86/gfx1151 schedules and
evidence do not transfer. Canonical Graph reduction currently excludes
mixed-output and keepdims forms; widening that shared contract is separate
from adopting this first f32 boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

The shared spine now defines a content-addressed `schedule.attention` edge and
one launch-level Tile artifact for the bounded x86/gfx1151 instances.
**NVIDIA outcome: follow-up required on a CUDA-enabled SM120 host.** Existing
SM120 forward packaging still synthesizes an NVIDIA-owned Tile program from
Graph IR; no PTX, cubin, LSE policy, schedule, selector, or device evidence
changes here. NVIDIA must define its own schedule/LSE instance, consume the
exact artifact, and run its forward numerical/performance gates. x86 and
gfx1151 policy or evidence does not transfer.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

The shared spine now defines a content-addressed three-result
`schedule.attention_backward` program carrying dQ, split-dK/dV, fixed reduction,
workspace, and LSE checkpoint identity. **NVIDIA outcome: follow-up required on
a CUDA-enabled SM120 host.** Existing SM120 backward packaging remains
NVIDIA-owned Graph-to-Tile synthesis. NVIDIA must define its own LSE identity,
consume this exact artifact, and validate MHA/GQA/MQA plus modifier/ragged
coverage before claiming parity; x86/gfx1151 schedules and evidence do not
transfer.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The shared orchestration direction now has a concrete typed configuration:
family, input artifact level, output artifact level, architecture, Tile
producer, Target-IR consumer, and backend code generator. **NVIDIA outcome:
follow-up required.** CUDA must define its own SM-specific family plugins around
the canonical Schedule/Tile artifact and NVVM/PTX/cubin code generator; this
change does not transfer gfx1151 scheduling or AMD wait semantics and supplies
no CUDA-enabled or SM120 evidence. The existing Graph-owned synthesis lane is
unchanged. NVIDIA accepts the shared strict-boundary policy (no surviving Tile
or Target IR, undefined result, or contract-marker symbol), but enforcement in
the NVVM/PTX/cubin pipeline remains CUDA-owned follow-up.
ROCm has now retired its final generic runtime pass-name helper. **NVIDIA
outcome: follow-up required:** CUDA family plugins must likewise expose a
closed semantic registry rather than an arbitrary pass option. No PTX/cubin,
SM schedule, selector, or device evidence changes here.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

The shared `schedule.spectral_program` contract now hashes packed-real fusion
topology and N/2 child identity. **NVIDIA outcome: follow-up required on a
CUDA-enabled host.** No NVVM/PTX/cubin consumer changed. NVIDIA must select its
own real-transform plan and carry the exact v5 artifact through its physical
package; Zen 5 and gfx1151 schedules, workspaces, and evidence do not transfer.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`): typed `!tile.async_token` SSA is production; legacy
grouping keys are the declared envelope, optional and conservative on absence.
New shared diagnostic `TILE_ASYNC_STAGE_NEGATIVE`. **NVIDIA outcome: parity
validated at the core-IR level.** The typed-token SSA edge is the NV warp-spec
production model (`TileIRLoweringPass::emitAsyncCopy`,
`tessera-warpspec-legality`); the previously contradictory required-stage
verifier was the one rejecting production NV Tile IR, and
`phase2/pm_verify_async_token.mlir` (red at baseline) now passes. No
NV-device-lane (`tessera-nvidia-opt`) fixtures changed.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Follow-up required.** The retyped family is NVIDIA's vocabulary:
`tile.mbarrier.wait` gains an optional `!tile.mbarrier_token` segment (the
operand-segment ABI is now barrier/token/dependencies), `tile.tma.copy_async`
and the wait grow fail-closed gates-on-nothing verifiers, the keyless legacy
wait is an explicit `tile.retire_all` marker (stamped by AsyncCopyLowering,
resolved — or preserved when no completion tokens exist — by
NVTMADescriptorPass), and `--tessera-tile-dataflow-legality` runs in every
NVIDIA pipeline after the post-NVTMA legality blocks. Host-free evidence:
full lit 324/0 including the pipeline-alias and NVTMA fixtures. **Open on
this queue:** sm_120-host revalidation of the NVTMA pipelines and any
SM90/Hopper device proof (Phase G/H); the barrier-at-birth emission
restructure is the tracked follow-on (compiler_enhancement.md §5.2.1).

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**Follow-up required — no NVIDIA lane exists for any of the ten.** No
`tessera-nvidia-pipeline-{sm90,sm100,sm120}` stage, PTX emitter, or backend
manifest row consumes either family; the declared tier is the Python
reference. GAME_THEORY_PLAN.md G5 names the per-target arbiter registration
(SubsetZetaRegion beside SpectralFFTRegion, `boltzmann_value` on the
online-softmax emitter) as the NVIDIA-side entry point for the lattice family;
the solver's entry point is the PDE op-set admission, not G5. No sm_120 host
revalidation was run for this registration and no device evidence is claimed —
nothing here changes generated code.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**NVIDIA outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions consumed only by `compiler/emit/apple_msl.py`; SM120
reduction lowering is untouched. No CUDA schedule, PTX path, ABI, or exact-device
claim changes. The NaN-propagation contract is worth noting for any future
`emit/nvidia_cuda.py` reduce emitter: CUDA's `fmaxf` suppresses NaN the same way,
so a naive port would reintroduce the same divergence from the declared
`nan_mode = "propagate"`.

Also recorded for coordination: PR #571 admits Apple GPU into the shared
scheduled reduce contract (`scheduled_kernel.py`, last axis only) and closes
APPLE-DEVICE-EVENT-1 by giving the Apple MPSGraph BMM route an owned command
buffer. Both are Apple-guarded — the shared `scheduled_kernel` gate adds an
`apple_gpu` branch beside the existing x86/ROCm ones and changes neither — and
the runtime edit is in `apple_gpu_runtime.mm`, which no sibling links.

## APPLE-ATTN-BWD-PERF-1-2026-08-16 — backward row-prepass assessment (PR pending)

**Outcome: not applicable to this architecture — no shared contract changed.**
Apple's attention-backward dK/dV split kernel became key-parallel by finally
implementing the `row_prepass` stage that
`attention_contract.plan_attention_backward_workspace` **already declares**
(`row_lse`, `row_delta`, consumed by `dkdv_split` and `dq`). The change is
confined to Apple-private MSL in `apple_gpu_runtime.mm` and its dispatch; no
shared IR, Schedule contract, workspace plan, dtype, ABI, or sibling schedule
was modified. The declared schedule is unchanged — still `split_count = 2` with
ascending `reduction_order = (0, 1)`; only Apple's thread mapping changed, which
is architecture-owned.

**Worth reading anyway, because the shape of the bug is portable.** The declared
workspace stage existed and had no consumer, so every kernel recomputed the row
statistics inline. That is not merely duplicated work: because the statistics
reduce over the whole key axis, an inline-recompute kernel must own an entire
query stream, which capped the dK/dV split at one thread per (partial, KV batch)
— 4 threads at `B1 Hq4/Hkv2 S64`. Implementing the declared prepass raised it to
`2 * kv_outer * Sk` and moved backward from 195 ms to 6.0 ms (~32x), while
keeping single-owner determinism. Any backend whose backward recomputes row
statistics inside its dK/dV kernel should check whether it has the same
structural cap before attributing slowness to memory or instruction mix.

## APPLE-NORM-VJP-1-2026-08-16 — Apple native-VJP registration assessment

**Outcome: not applicable — no shared contract changed.** Apple registered as a
Target consumer in the existing native-VJP normalization plugin and added its own
Metal MSL VJP kernels. The shared registry schema, the `_parse_compiled_norm_backward`
contract, the Graph/Schedule contracts, and every sibling executor are unchanged;
the new code is Apple-private (`apple_gpu_runtime.mm`, one runtime executor, two
`execution_matrix` rows behind one new Apple executor id). This target's own
normalization VJP rows, evidence and selectors are untouched.

Recorded because the *scoping* lesson is portable: the item was scoped as
"registry wiring", but the target had no normalization-backward ABI at all, so
the entry would have been an unconsumed declaration (Decision #29). Any backend
asked to join this boundary should confirm it owns an executable VJP before
declaring a consumer.

## APPLE-NORM-VJP-2-2026-08-16 — reduced-precision normalization VJP assessment

**Outcome: not applicable — no shared contract changed.** Apple's normalization
VJP gained f16/bf16 storage: four new Apple-private C ABI exports
(`tessera_apple_gpu_{rmsnorm,layer_norm}_bwd_{f16,bf16}`), their MSL, the
non-Darwin stub returning 0, and Apple's own ABI registry rows. The shared
native-VJP registry schema, `_parse_compiled_norm_backward`, the Graph/Schedule
contracts, the execution-matrix executor id, and every sibling executor are
unchanged. This architecture's normalization VJP dtype support, evidence and
selectors are untouched.

Two portable notes, recorded because both are cheap to get wrong:

1. **Store rounding is part of the numeric contract.** The kernel accumulates in
   f32 and stores back at the operand's own dtype, and bf16 stores use
   round-to-nearest-even. Truncating instead would bias every gradient in one
   direction — a systematic error, not noise. Any sibling adding reduced-precision
   gradients should match its own runtime's established rounding convention
   rather than defaulting to a shift.
2. **A tolerance derived from storage epsilon is what proves the accumulator.**
   Measured error is ~0.5-0.65 ulp of each format's epsilon, i.e. one rounding on
   store; storage-format accumulation would land near `sqrt(cols)` ulps. A test
   asserting against that second level catches a silently dropped f32
   accumulator, where a hand-picked loose tolerance would absorb it.

A third note is Apple-specific but the shape recurs: adding an export obliges
updating the dylib **freshness gates**, or a runtime built between two landings
exports the older names, passes the staleness check, and then fails at the call
site — which reads as a broken consumer rather than the stale build it is.

---

## Cross-backend record — MC1 matrix-function family (PR #596, 2026-08-20)

**Owning item:** MATRIX-CALCULUS-MC1 · **synchronization key:** `MC1-LINALG-FAMILY`

**Shared contracts changed.** Ten public op registrations (`det`, `logdet`,
`inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`, `norm`); two
new Graph IR lowering kinds (`linalg_function`, `linalg_multilinear`) with four
new shape rules (`matrix_scalar`, `vec`, `kron`, `eigh`); numeric-policy
entries for both kinds; two diagnostic codes (`E_LINALG_CONTRACT`,
`E_METRIC_CONTRACT`); the `degeneracy_policy` semantic key extended to
eigenvalue gaps and to the nuclear norm's rank condition.

**Outcome for this backend: `not applicable — no lane attempted or implied`.**

The family landed as a *derivative contract* — closed-form VJP+JVP pairs under
the AD law sweep — because that was the gap in the AD stack. No Tile IR
lowering, Target IR op, or kernel was added for any backend, and none is
implied: every one of the ten is registered `backend_kernel: partial` /
reference tier, and each is listed in the Apple no-lane golden and the
single-GPU closeout classifier as an *intentional* reference-only decision with
a stated rationale, so none of them appears on this backend's promote queue as
a phantom blocker.

**What a sibling picking this up would need to decide, per op.** The reference
implementations delegate to LAPACK through numpy, so a native lane is a
vendor-library question rather than a codegen one for `eigh`/`inv`/`solve`
(rocSOLVER / cuSOLVER / Accelerate / MKL), and a "probably never worth it"
question for `det`/`logdet`/`trace`/`matrix_power`, which reduce a small matrix
to one number. `kron` and `vec` are layout/contraction shaped and would ride
existing lanes if anything.

**Numeric policy this backend must match if it does build a lane.**
`linalg_function` and `linalg_solver` declare **f64 accumulate with f32 storage
admitted** — the same conditioning-sensitive policy `linalg_decomposition`
already carries, because `det`/`logdet`/`inv`/`matrix_power` carry a factor of
the condition number. `linalg_multilinear` (`trace`, `kron`) declares the
ordinary f32 accumulator, since neither carries one. A lane that accumulates in
storage precision would silently lose the digits these rules exist to keep.

**Degeneracy contract that travels with the rules.** `eigh`'s eigenvector
coupling `1/(w_j - w_i)` and `svd`'s `1/(s_j^2 - s_i^2)` have no limit at a
crossing; both fail closed under the declared `degeneracy_policy` rather than
emitting `inf`/`NaN`. Any native lane must reproduce that refusal, not paper
over it with an epsilon — a damped coupling returns a finite, plausible, wrong
gradient, which is the failure mode the key exists to prevent.

**Exact-device evidence: none, and none claimed.** Everything in PR #596 was
validated on the host-independent Python reference lane on an M1 Max, where no
AMD GPU, no CUDA device and no AVX-512 exist. No parity claim is made for any
backend.

**NVIDIA-specific note.** cuSOLVER covers `eigh`/`inv`/`solve` directly; the
open design question is whether they enter through Target IR as `abi_call`-style
delegated work (Decision #19's x86 precedent) so the arbiter can still tell
compiler-generated work from delegated work.
