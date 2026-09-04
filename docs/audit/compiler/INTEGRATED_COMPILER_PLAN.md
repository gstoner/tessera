---
last_updated: 2026-08-24
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

## Central development plan — 2026-08-17

This is the executable center of the compiler, TSOL, and compiler-tools work.
The wave tables and scoped plans below remain the detailed design record, but
they do not create parallel queues. Generated gap counts are inventories: a
backend-kernel, direct-test, ABI, or benchmark count is not a list of equally
urgent implementations.

### Current baseline

- The public API, frontend disposition, Graph registration, Schedule lowering,
  runtime dispatch, batching, transpose, and lowering-rule axes are closed in
  the generated dashboards. Tile, Target, sharding, direct proof, ABI, and
  benchmark evidence remain mixed.
- All 51 canonical TSOL operations have registered math, shape, dtype/layout,
  lowering, and explicit VJP/JVP dispositions. TSOL work therefore starts at
  physical consumption, sharding, policy breadth, and evidence—not at another
  semantic inventory.
- `ScheduleObject` SO-1 is complete and SO-2 is bounded/landing. The inferred
  action DAG and scalable list scheduler exist, but most physical producers do
  not consume them yet.
- The profiler has provider, timing, packet, and reporting contracts. Native
  ROCprofiler/CUPTI/Metal/perf evidence and the C++ `tprof` surface are not all
  release-grade runnable surfaces yet.

Use
[`generated/compiler_progress.md`](../generated/compiler_progress.md),
[`generated/tsol_coverage.md`](../generated/tsol_coverage.md), and
[`generated/surface_status.md`](../generated/surface_status.md) for live counts.
This section owns what to do with those counts.

### Ordered queue

| Order | Work item | Deliverable | Acceptance gate | Depends on |
|---:|---|---|---|---|
| 1 | **E2E-REAL-6F — optimizer VJP authority complete; exact family/target certificates landing** (x86 + gfx1151 packets landed 2026-08-26) | Every successful family-plugin launch emits a content-addressed `tessera.native_vjp_execution.v1` certificate. Runtime-origin `tessera.runtime_physical_execution.v1` attestations distinguish actual silicon launches from test doubles; only the former count as `exact_device`. Target-owned packets execute all declared families in one process and compare the observed family/target set exactly with the live registry. The three former optimizer `JitFn` helpers remain deleted. | The AVX-512 packet covers all 10 x86 family rows and the gfx1151 packet covers all 13 ROCm rows, including SGD, Momentum, Nesterov, Adam, AdamW, Lion, full/factored Adafactor, sequence mixer, and selective SSM against independent oracles. Eight sibling rows remain blocking: seven SM120 families and Apple normalization. Runtime receives no source Graph or operation dictionary; sibling evidence never transfers. | E2E-REAL-6E state-lineage package. |
| 2 | **W4-PRODUCT-1 — executable multi-block regions** | The bounded arbitrary-CFG compiler boundary, per-slot dynamic saved-value envelopes, companion logical-shape tapes, mixed-state SAVE/HYBRID tapes, and nested canonical bodies are landed. Exact polynomial specialization guards remain outside Presburger proofs and require complete witnesses. Compiler-generated replay-safe assertions are admitted; mutation, unkeyed RNG, I/O, alias-sensitive work, and ordered collectives remain fail closed pending operation-owned recorded-product ABIs. The gfx1151 irreducible-state-machine row landed 2026-08-21: `--generate-rocm-state-machine-kernel` lowers a paired `bounded_state_machine_v1` function (forward AND generated backward) to one per-thread device kernel — per-element program counter, structured-CFG digest stamped on the gpu.func, `cf.assert` bound check host-enforced through a STATUS buffer — with both entry paths of a two-entry SCC executing on gfx1151 against the analytic oracle (`test_rocm_state_machine_exec.py`). The sibling x86 row landed 2026-08-21 as well: the same paired functions compile through `tessera_jit` (tessera-to-linalg → elementwise-to-linalg → one-shot-bufferize → loops → LLVM → ORC JIT) and execute natively on the AVX-512 host — both entry paths, forward + backward, digest/residual-policy bound, native `cf.assert` bound trap, proof-of-execution counter (`test_x86_state_machine_exec.py`). One physical packet family with admissible effects landed 2026-08-25 (W4-EFFECTS-1 slices E1-E5, PR #630). **The bounded `tessera.control_scan` reverse landed 2026-08-26:** statically bounded symbol-body scans are normalized through the canonical SCF lowering inside the paired pass and use the proven reverse-loop/tensor-slice rules; payload, dynamic, malformed, and unlowerable forms retain `AUTODIFF_CONTROL_SCAN_UNSUPPORTED`. | Existing region verifier/paired VJP fixtures stay green; padded tape bounds must never replace logical extents; native x86/gfx1151 numerical rows must bind the exact CFG and residual digests before physical execution is claimed. | W2.1 dataflow, W2.2 effects, current bounded W4 carrier. |
| 3 | **SO-3 + W5.2e-PRODUCER-1 — one schedule authority closed 2026-08-24.** The MegaMoE producer likewise consumes `infer_action_dag` with the hand-authored plan DAG demoted to a fail-closed coverage oracle (#31), and adopting it EXPOSED three over-conservatisms in the shared inference that made any pipeline containing transport infer a total chain — ordered collectives serialized against all local work, registered effectful producers discarded their declared `aliasing="none"`, and every effectful/pure pair was assumed memory-dependent — each corrected with the sound rule and pinned by tests (a 12-action MegaMoE plan went from all 66 edges of the complete order to 36, still oracle-covering, extras collective-ordering only). Its schedule digest binds the plan's own artifact digest. **Two follow-ups closed 2026-08-25:** the interleaved generator emitted backward passes in ASCENDING virtual-stage order (`bwd_clock = fwd_clock + p*v`, a constant offset), the opposite of gradient flow — the stage term is now mirrored, so gradients run from the last virtual stage back to the first with the makespan bit-identical (`2*p*v + m - 2`), the backward gradient edge is expressible again and is REQUIRED rather than excused, and the same work found that DECOUPLED schedules were carrying fabricated cross-stage edges despite being defined by having none. And the spectral policy is now bound by a hash CHAIN rather than a comparison: `sha256(tessera.schedule_payload)` must equal the module digest, and that payload must name `object_id "spectral:<sha256(tessera.spectral_semantic)>"`, so the consumed attributes and the declaration vouching for them can no longer be co-edited. | Pipeline lowering now requires the digest-bound `tessera.pipeline_schedule.v1` carrier, stamps its Schedule Object digest on functions and communication ops, and has no scalar-plan/options reconstruction path. The spectral producer represents fused stages as registered Graph actions, consumes `infer_action_dag`, binds roles/resources into the content-addressed Schedule Object, and requires that digest at Schedule→Tile. MegaMoE R3 likewise consumes inferred registered-semantics edges; its former handwritten DAG is only a fail-closed coverage oracle and additional conservative collective edges remain reported. | Inferred spectral and MegaMoE DAG tests preserve reasoned edges, roles, and resource vectors; stale/missing digest carriers fail closed; focused schedule/producer suite passes (225 tests plus dedicated carrier tests); compiled spectral numerics are unchanged across the complete Zen 5 AVX-512 and WSL gfx1151 suites (42 tests). | SO-1/SO-2 and W5.2e inference. |
| 3b | **NUMPOL-CARRIER-1 — the S5 generalized `numeric_policy` carrier (owned 2026-08-24; steps 1–2 landed 2026-08-25 — schema + reduction-family carrier, see the status note under this table)** | One carrier design for storage/accumulator/math-mode that survives Schedule and Tile IR beyond MMA fragments (pointwise, reduction, and butterfly chains), plus the Decision #32 boundary verifier that FAILS on silent loss instead of recording it. Builds on the landed W1.1 `!tile.fragment<…, acc, …>` accumulator carrier (typed ROCm route) as the worked reference. Four mandating consumers: CAKE (#32's original derivation), game-theory §6 (fusion is a correctness feature — the zeta intermediate must not round through fp32), PDE §III.4 (interim `tessera.info_loss` records retire), and AD-JET-IR-1 (coefficient/cotangent policy, W6.3 §2.3). FORGE §1.3 supplies the measured acceptance target: the fused-epilogue fp32-accumulator realizability verdict (913× → 1.1× → 1.0× purely as a function of accum × state dtype) must be decided by the carried policy, not a special case. | Carrier attribute round-trips Schedule→Tile with a lit-verified boundary check per crossing; a lowering that drops the policy fails closed with a named diagnostic (#21a/#32); the W1.1 fragment path re-expressed as an instance of the general carrier without behavior change (bit-identical existing gfx1151/x86 outputs); the PDE `tessera.info_loss` interim records replaced by carrier facts; dashboard row tracks per-boundary coverage. May proceed in parallel with Orders 3 and 5 (orthogonal IR-carrier work; the schedule authority does not consume the policy). | W1.1 fragments (landed); Decision #32; #21a semantic-key discipline. |
| 4 | **LAYOUT-ALG-1 L4 — physical layout decisions closed 2026-08-24** | L3 factorization/residency and SO-4 proof attachment are implemented. Mixed-radix/static tuple products, SM120 dynamic strided typed+macro routes, gfx1151 bounded-dynamic execution, the four x86 core GEMM index families, and every reachable Apple MSL rank-2 template consume shared authority. Dynamic non-separable tuple codomains remain fail closed. | Existing raster/index outputs remain bit-identical; unresolved layouts fail closed; materialization proof covers alias, capacity, and lifetime; Apple7 canonical and fused-cooperative cohorts pass exact-device; no architecture's schedule is promoted by another architecture's evidence. | Current L1/L3/L5 authority and architecture-owned device proofs. |
| 5 | **W5.4-RESHARD-1 — executable placement closed 2026-08-24.** | Fixed-point placement now derives exact mesh-sized local result types and inserts explicit reshard SSA at the consumer boundary, including registered `tessera.slice` local shards rather than fake same-shaped collectives. Plan digest, subgroup, matching rounds, and nested region path survive Graph→Schedule→Tile. A deterministic mock-mesh executor consumes that SSA directly. | 13 focused placement tests plus 76 shared Schedule/Tile/collective tests pass. Nondivisible shapes, mesh/subgroup mismatch, unknown placement, and sibling-region escape fail closed. `local_shard`, `all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all`, and `collective_permute` execute numerically with an explicit movement trace. Native transport remains a separate evidence gate. | Orders 2–4. |
| 6 | **DIST-NATIVE-1 — real multi-rank execution** (bounded MPI slice landed 2026-08-26) | Schedule→Tile now preserves communicator/subgroup identity and Schedule artifact hashes; the content-addressed Target package dispatches explicit all-reduce, all-gather, reduce-scatter, all-to-all, and collective-permute SSA to one process-owned MPI rank. The f32/SUM/axis-zero envelope binds world/rank ownership, ordered subgroup communicators, collective ordinals, reshard identity, tensor dtype/shape, topology, and artifact digest without a mock fallback. Next extend the packet beyond two ranks and nonparticipant subgroups, then separately add NCCL/RCCL process-rank ownership and OFI/SHMEM. Keep ROCm LSA, GIN/RMA, Copy Engine, and gfx1250 DDA as independent advanced gates. | The checked-in packet executed under bundled MPICH 4.1.2/`mpi4py` with two real processes: all five compiled SSA numerics and reordered subgroup communication pass. Cross-rank order, subgroup, artifact/topology digest, dtype, and shape mismatches fail during shared admission before data transport. Required remaining gate is a multi-rank packet with proper-subgroup nonparticipants; no NCCL/RCCL or mock result is promoted by this slice. | Order 5. |
| 7 | **AD-TSOL-STFT-BWD-1 — bounded native spectral products closed 2026-08-26** | Independent content-addressed AVX-512 and gfx1151 STFT/ISTFT packages accept contiguous batched final-axis aligned n=16 and ragged mixed-radix n=18 f32, f16, and bf16 storage through explicit two-byte ABIs with f32 accumulation; spectra remain complex64. The gfx1151 stored-bin adjoint is a direct DFT kernel independent of both the x86 packed-C2R implementation and the ROCm forward FFT. Uncentered, onesided, explicit-hop, `n_fft == window`, and uncropped ISTFT are the complete bounded policy. | Both architectures match the independent Python VJP. gfx1151 additionally executes the native forward/adjoint inner-product identity and emits runtime-origin exact-device certificates. Unsupported length, policy, contiguity, dtype, shape, numeric policy, or artifact digest fails before launch. No evidence transfers to Apple/NVIDIA. Centering/cropping, arbitrary axes/strides, and broader lengths move to Order 8. | Order 1 plugin boundary and existing spectral VJP carriers. |
| 8 | **TSOL-POLICY-PHYS-1 — complete the spectral policy envelope (8a/8b landed; 8c/8d/8e x86 landing 2026-08-26)** | 8a/8b physically adopt centered/cropped policy and arbitrary logical axes on x86/gfx1151. The landing x86 v7 package consumes true runtime stride descriptors and implements explicit `n_fft >= window`, odd lengths, and one-sided/full-complex forward and reverse; the reverse artifact binds the stride ABI. Streaming state has digest-chained policy/window/tail/counter/parent lineage. Remaining: the corresponding gfx1151 physical package and exact-device proof, per-batch broadcasting, and physical streaming/chunk execution. | The x86 packet passes independent NumPy/Python-VJP oracles for non-contiguous axis-1/axis-2 n=20/window=15 full spectra plus existing low-precision and bounded rows. The lineage oracle rejects tail, counter, parent, policy, and window drift. These results transfer no gfx1151/CUDA/Metal evidence and do not promote 8f/8g physical rows. | Order 7. |
| 9 | **TSOL-SCALE-1 — ND and large transforms** | Add 2D/ND, batched nontrivial-stride, large-transform six-step/Bailey, and large-prime execution behind selector-visible algorithm identities. | Correctness across prime/composite/ragged shapes; plan/twiddle/workspace cache identity; packed-vs-full and library comparisons; retain/promote/reject per architecture. | Order 8 and evidence tooling below. |
| 10 | **TSOL-SHARD-1 — distributed TSOL contracts** | Close the 18 partial TSOL sharding rows by routing spectral, solver, sparse/segment, and layout families through W5.4 placement and explicit reshard operations. | Registry totality, mock-mesh numerical tests, and no claim of native transport before exact multi-rank proof. | Orders 5–6. |
| 11 | **TSOL-PHYS-TAIL-1 — high-value physical families** | Use per-target maps—not the all-target aggregate—to promote the remaining high-use solver, sparse/segment, and layout rows. Start with `tri_solve` and PDE/solver consumers, then sparse/segment families; compose coalition-lattice operations through the shared butterfly rather than creating one-off emitters. | Independent x86 and gfx1151 correctness/performance verdicts; dtype/shape envelopes; no reference execution relabeled as native; Apple/NVIDIA remain separate consumers. | Orders 3–5 and the relevant scoped family plan. |
| 12 | **TPROF-NATIVE-1 — trustworthy clocks and sampling** | Graduate ROCm dispatch/activity/counter/PC-sampling capture and x86 perf/IBS symbol correlation; retain independent host, device/event, and profiler clocks with validity/provenance. | Fresh-process collection; no clock substitution; clean-image/instrumented-image overhead comparison; bare-metal gfx1151 and Zen 5 calibration packets. WSL packets remain regression-only. | Can run in parallel with Orders 1–8; blocks selector promotion. |
| 13 | **EVIDENCE-PACKET-1 — one selector packet contract** | Make every physical benchmark emit the same schema for semantic/artifact/image digests, dtype/shape/policy envelope, resources, clocks, warm/cold state, samples, source cleanliness, and eligibility. | Schema validator and replay test; packets with missing native clocks, dirty sources, mismatched digests, or unsupported architectures are promotion-ineligible rather than silently downgraded. | Order 12. |
| 14 | **COMPILER-DEVEX-1 — runnable compiler tools** | Promote `tprof` and `tessera-opt` from `compile_only` only after installed-driver smoke, dialect/pass inventory, negative target-load tests, and reproducible integration/numerical entry points exist. | Clean-build host smoke, `--version`/build provenance, positive and fail-closed fixtures, and generated surface-status promotion through its owning registry. | Orders 1–4 and 12. |

**NUMPOL-CARRIER-1 status (updated 2026-08-26) — the generalized carrier
ceiling is closed; architecture-specific arithmetic remains evidence-gated.**

*Step 1 — the policy gets a schema.* The row assumed a payload that could be
carried. Measured first, and it could not be: `numeric_policy` was a bare
`DictionaryAttrBase` whose ODS predicate checked only "is a dictionary". Five
malformed policies were all ACCEPTED (exit 0) while the documented
TF32-as-storage violation correctly failed — so the pass was running and simply
had nothing to say about a typo'd key, a non-string value, a nonexistent dtype,
a `math_mode` that reduces nothing, or an accumulator narrower than its
storage. The typo is the sharpest: `getAs<StringAttr>` returns null for a
misspelled key exactly as for an absent one, so the op carried a policy that
looked like it stated an accumulator contract and stated none — Decision #21a's
scar in a fresh place. Seven diagnostics now refuse each case, registered and
cross-checked.

The narrowing rule is refused rather than warned, on measurement: at a fixed
dtype-bit budget the narrowing policy is strictly dominated by the one that
swaps storage and accumulator (25.8x for the fp16/fp32 pair, K=4096 dot product
vs an fp64 reference), and with a narrower accumulator the wider storage is
BIT-IDENTICALLY unobservable. There is no program for which it is the right
answer. Comparison is on significand bits, not width, so fp16-storage
accumulating into bf16 is caught.

*Step 2 — the reduction family carries its accumulator.* The gap was worse than
"metadata is lost". `{storage="bf16", accum="fp32"}` on rmsnorm / softmax /
layer_norm lowered to `arith.addf … : bf16`: the emitted code CONTRADICTED the
declared contract on the very op that performs the accumulation. Executed on
this box through `--tessera-to-linalg` → LLVM → native object, a 4096-wide
softmax row summed to **1.466** — a 47% violation of the function's defining
property — versus **1.000169** once the accumulator is honoured. The residual
is bf16 storage rounding of the output, which is what the policy asked for.
Cast placement was a measured fork: truncating the REDUCED value back to
storage leaves 5.6e-04 where truncating only the RESULT reaches 1.7e-06 (326x),
and the latter is the faithful reading of #15a, where storage is the dtype of
the tensor and the tensor is the result. With no policy the emitted IR is
byte-identical — a carrier that widens unasked is a global dtype promotion, not
a carrier.

The Graph→Linalg crossing is now bracketed by the Decision #32 record/verify
pair and declares `represented_in_type` / `re_expressed`. It was NOT bracketed
before: the pair straddles only Graph→Tile in the production pipelines, which
is why this drop was silent. The verifier then caught a defect in the
declaration logic itself (a function lowering one of two policy-carrying ops
loses a value while the name survives — `re_expressed`, not
`represented_in_type`), which is the mechanism working.

*Open, and item (a) is not what the row assumed.* Measured 2026-08-25 while
scoping it: for the MMA path the policy **already survives to the bottom**.
`TileIRLoweringPass` forwards it onto `tile.mma`, and `TileToROCM.cpp` carries
it onto the `tessera_rocm.*` ops — but all three of its uses are
`copyAttrIfPresent`. It is **forwarded, never read**. The accumulator that
actually reaches the hardware is inferred independently in
`GenerateWMMAGemmKernel.cpp` as `fragmentAcc = T.isInt ? "i32" : "f32"`, in a
file that does not mention `numeric_policy` at all.

So there are two sources of truth for one fact, and the declared one loses.
They agree today only because every real program uses fp32. A policy of
`{storage="bf16", accum="fp16"}` — legal, and now schema-checked — silently
gets f32 on gfx1151, and `accumulatorProbeDtypes` returns an EMPTY candidate
set for any accumulator that is not `i32`/`f32`, so nothing downstream could
honour it either. This is not hypothetical hardware: the in-repo ISA archive
records `V_WMMA_F16_16X16X16_F16` on RDNA 3.5 alongside the f32-accumulate
forms, so an f16 accumulator is a real gfx1151 capability the compiler cannot
currently be asked for.

That makes (a) a **Decision #29** item rather than a plumbing one — the
carrier is built and the consumer is missing, which is the case #29 calls
worse than a missing declaration because it reads as a closed contract in
review.

**(a) is now closed for the consumer half, and the codegen half is deferred on
measurement (2026-08-25).** `GenerateWMMAGemmKernel` reads
`numeric_policy.accum` and refuses with `ROCM_WMMA_ACCUM_UNSUPPORTED` when it
names an accumulator this path does not provide, rather than substituting f32
and reporting success for a different computation. Every existing program is
untouched: a real before/after control — capture, revert the file, rebuild,
recapture — showed the generated output **byte-identical across all 65 ROCm
fixtures**, and the gfx1151 device lanes come back at the recorded baseline.

Wiring the f16-accumulate WMMA itself is deferred, and the reason is a
measurement rather than effort. Its ROCDL form is `(v16f16, v16f16, v16f16) ->
v16f16` with an `opsel` bit — a different accumulator ABI, not a parameter
swap. What it buys is half the accumulator VGPR footprint; what it costs,
measured on 16x16x16-tiled GEMMs against an fp64 reference, is 5212x (K=64)
rising to 7856x (K=4096) relative error. That is a real choice for short-K
inference shapes and a bad one for training — which is precisely the kind of
decision a declared policy should make and a storage-dtype inference cannot,
so the interface is the deliverable and the codegen follows a measured need.
Revisit when an occupancy-limited kernel shows the accumulator is the binding
constraint, the same basis on which Decision #26a deferred the AIR emitter —
not on the grounds that the instruction exists. (b) `math_mode` still has NO
MLIR consumer: measured, it appears in C++ only inside a rejection message,
while `mma_selector.py` / `nvidia_dtype_contract.py` consume it on the Python
side. A semantic key with no consumer below the frontend is Decision #29's
case, and it is the NVIDIA half of this row. (c) **Closed.** The FORGE §1.3
realizability verdict lives in `compiler/precision_realizability.py`, decided
from the carried policy plus the state dtypes. Reproducing §1.3 first caught a
modelling error of my own: rounding the gradient to the MASTER dtype is a
no-op for fp32 masters, so all three rows came back 1.0x. The gradient's
storage write is bf16 in mixed precision, and that write is what the fusion
removes. Corrected, the structure reproduces — 208x / 1.2x / 1.0x against the
assessment's 913x / 1.1x / 1.0x: the masked rows match closely, the unmasked
row agrees in kind but not magnitude. **That asymmetry set the interface.**
The oracle puts a number on the question it can answer soundly — is the
benefit masked? — and refuses to number the unmasked case, whose size depends
on gradient distribution and step count a compiler does not know. A diagnostic
promising 913x and delivering 208x would be worse than one saying "large;
measure it". Its tests re-derive the table by RUNNING the training loop, so a
wrong oracle fails even when it agrees with the write-up.

(d) **Not what the row assumed.** No pointwise op and no spectral/butterfly op
declares `numeric_policy` at all, so there was nothing to carry. For pointwise
that is correct: there is no accumulation, so `accum` is meaningless. For the
butterfly chain it was hiding something. The spectral scheduler emitted
`numeric_policy = "f32;ortho"` — a **StringAttr** holding a private
semicolon-delimited encoding, under the name of a well-defined DictionaryAttr.
Since `getAttrOfType<DictionaryAttr>` returns null for a wrongly typed
attribute exactly as for an absent one, that contract was invisible to the
schema validator and to every accumulator consumer — and it is not a #15a
policy in the first place: its value can be
`"deterministic_f32_ascending_frames"`, a reduction-ORDER contract rather than
a dtype. Renamed to `tessera.spectral_accumulation` /
`tessera.spectral_normalization` (349 spectral tests unchanged; the schedule
digest is computed separately, so identities are stable), and
`NUMERIC_POLICY_NOT_A_DICTIONARY` now refuses the wrongly typed case so the
collision cannot recur. Ops declaring the attribute in ODS were already
covered by its constraint; the gap was the discardable case, which is exactly
where the spectral contract lived.

**Row 3b carrier ceiling closed 2026-08-26.** Scheduled compound spectral and
butterfly programs now carry a real digest-bound
`numeric_policy={storage,accum}` dictionary through Schedule→Tile. The child
FFT schedule already declares `accum="f32"`; the parent now binds the same
choice in its semantic preimage, validates storage spelling, requires fp32
accumulation, and copies the dictionary onto the Tile kernel. The separate
`tessera.spectral_accumulation` field remains the deterministic reduction-order
contract. A missing, narrowed, wrongly typed, or digest-inconsistent policy
fails closed with the registered numeric-policy diagnostic. This closes the
general carrier ceiling, not every architecture's arithmetic implementation:
SM120 `math_mode` consumption and exact-device proof remain NVIDIA-owned, and
the measured/deferred gfx1151 f16-accumulate WMMA choice remains a separate
codegen policy row.


**E2E-REAL-6F: the deletion gate had no measurement (2026-08-25).** Order 1
ends "the three former `JitFn` compatibility helpers are deleted", and
MASTER_AUDIT §1 permits that only after differential execution covers each
migrated family. Measured: **nothing enumerated that coverage.** The
certificates live in per-`JitFn` dicts
(`_frontend_differential_certificates`, `_frontend_nonreexecuting_certificates`)
and die with the instance; no generated doc, registry, or test listed the
families. The gate could not be evaluated at all — not because the work was
missing, but because the evidence had nowhere to live, and a condition nobody
can evaluate is not a gate.

`generated/frontend_authority_coverage.md` is now that evidence, derived from
the `register_native_vjp_plugin` declarations rather than asserted, so a family
added without one does not silently appear covered — it does not appear.
**13 families**, 37 owned Graph ops: 7 `pure_only`, 5
`non_reexecuting_state_lineage`, 1 `zero_dropout_attention`, and **0 with no
certification path**. The policy split is load-bearing rather than
bookkeeping: `certify_frontends` proves equality by RE-EXECUTING the source, so
for a state-mutating family the second run is a different program and such a
certificate would compare two things never meant to be equal — and would pass.
A drift test pins that the five stateful families use
`certify_frontends_non_reexecuting`, that every named certifier exists (a
certifier named but absent would make every row citing it read as proven), and
that a family with an unhandled policy is reported as blocking — that last one
a control, since without it the whole gate passes vacuously on a registry that
happens to be clean.

Scope, so the row is not over-read: this proves the gate is **evaluable** for
all 13 families and that each declares a whole Graph→Schedule→Tile→target
spine. Successful plugin launches now record concrete evidence as
content-addressed `tessera.native_vjp_execution.v1` certificates in a
process-level family registry rather than only in a `JitFn`. The first newly
closed optimizer gap is factored Adafactor on AVX-512, whose certificate binds
`topology=factored` and the frontend/state/Schedule/Tile/Target identity.
The exact-target gate now enumerates 31 family/target rows. Target-owned
all-family packets cover all 10 x86 rows and all 13 ROCm rows using
runtime-origin physical attestations; mocked launch results remain
`runtime_unattested`. The remaining eight blocking rows are seven SM120
families and Apple normalization, each requiring its own hardware packet.

**W4-PRODUCT-1: `tessera.control_scan` is the one control primitive with no
reverse rule (measured 2026-08-25).** Bounded `if`, counted `for`, and
canonical bounded `while` all differentiate through the scf region machinery.
Scan did not, and said only that some interface was missing —
`AUTODIFF_OP_NOT_DIFFERENTIABLE`, true and unhelpful. All four
`tessera.control_*` ops carry `[Pure]` and nothing else; the reverse machinery
lives on the scf ops, and scan has no path there.

**The mathematics is settled, not open, and is now recorded executably**
(`tests/unit/test_control_scan_vjp_contract.py`). For
`(c_{t+1}, y_t) = body(c_t, x_t)` the reverse recurrence is
`(cbar_t, xbar_t) = body_vjp(c_t, x_t; cbar_{t+1}, ybar_t)` for `t = T-1..0`,
so **the adjoint of a scan is a scan** over reversed t, carrying the carry
cotangent and consuming `(c_t, x_t, ybar_t)`. Checked against central
differences on a nonlinear body: max absolute error 4.7e-10 on both the init
and the `xs` cotangents. The rows include a control that must fail (a rule
that forgets the per-step `ybar` — the likeliest slip, since the stacked
output is the operand a for-loop rule does not have), a row proving the
per-step carries are a CORRECTNESS requirement rather than a memory
optimisation (evaluating `body_vjp` at the final carry throughout is wrong by
>1e-3), and a row proving RECOMPUTE reproduces SAVE bit-identically.

**What blocks it is structural.** The reverse scan needs the BODY's paired
backward — a companion function this pass generates — and a residual tape of
the intermediate carries, which the forward scan does not stack.
`AdjointInterface::buildAdjoint` receives only an OpBuilder positioned at the
forward site and is contractually limited to emitting ops there, so it can
create neither. The rule therefore belongs beside the scf region handling in
`AutodiffPairedPass`, where companion functions and residual policies already
live. `AUTODIFF_CONTROL_SCAN_UNSUPPORTED` now says exactly that, with the
recurrence and the SAVE/RECOMPUTE/HYBRID connection in its notes, so the next
implementation starts from the contract instead of re-deriving it — and does
not put the rule in the wrong place.

Not claimed: the compiler rule itself. This slice makes the gap named,
bounded, fail-closed, and oracle-backed; the paired-pass implementation and
its device rows are the next block.

**AD-TSOL-STFT-BWD-1: the bounded AVX-512 and independent gfx1151 packages
landed (updated 2026-08-26).** The starting measurement was:

| | forward | tangent (JVP) | adjoint (VJP) |
|---|---|---|---|
| x86 AVX-512 | native (`tessera_x86_stft_f32` / `istft_f32`) | native (`tessera_x86_istft_jvp_f32`) | **none** |
| gfx1151 | native bounded STFT/ISTFT | none | native direct stored-bin package |

The initial asymmetry was forward-yes / tangent-yes / adjoint-no on x86, and
nothing at all on ROCm. The x86 native VJP plugin now owns content-addressed
STFT/ISTFT packages for contiguous last-axis, uncentered, onesided
f32/complex64, explicit positive hop, `n_fft == window`, and uncropped ISTFT
length. It returns both signal/spectrum and window cotangents and supports
backward, forward, and ortho normalization. Odd packed-R2C ISTFT, altered
lineage, unsupported dtype/shape/policy, non-x86 targets, and the unwired
generic `TileToX86Pass` STFT kind remain explicitly fail-closed. gfx1151 still
owns a separate direct stored-bin DFT reverse kernel and the existing native
forward package. It does not reuse x86 output or the ROCm forward FFT.

Both operators are LINEAR, so the VJP is the adjoint and the contract is
checkable exactly rather than approximately. Recorded in
`tests/unit/test_stft_adjoint_contract.py`: the reference VJPs satisfy
`<STFT(x), Xbar> == <x, STFT^H(Xbar)>` to 0.0 and 4e-16, the COLA overlap-add
identity holds to 4.4e-16, and the adjoint from first principles reproduces
the reference to 1.4e-15 —

    STFT^H(Xbar)[n + tH] += w[n] * Re( sum_f Xbar[t,f] * exp(+2i*pi*f*n/N) )

each STORED bin counted once.

**Implementation strategy, decided by measurement:** the STFT backward uses
the stored-bin adjoint directly, avoiding accidental Hermitian doubling. The
ISTFT backward is the exact transpose of normalized overlap-add and reuses
the packed R2C primitive for the frame cotangent, with DC/Nyquist counted once
and interior bins twice. The equivalent compositional oracle remains pinned:
`irfft` reconstructs the Hermitian pair, so halving interior bins makes
`N * irfft` reproduce the per-frame STFT adjoint to 1.4e-15; omitting that
correction is wrong by 127%.

**Two shortcuts refuted, so the next implementer does not spend a day on
them.** Reusing the existing forward ISTFT kernel would have made the package
nearly free, and it cannot: ISTFT is not `STFT^H` up to any global scale
(best-fit residual 0.968 for backward/forward/ortho), nor after undoing the
COLA window-sum division (0.887). The division is pointwise and the windowing
differs; the adjoint is a different program.

Claimed now: bounded independent AVX-512 and gfx1151 implementations,
content-addressed Schedule→Tile lineage, exact host/device numerical proofs,
and explicit low-precision storage ABIs. On gfx1151, aligned n=16 and
ragged/batched n=18 match the independent Python VJP for f32/f16/bf16 and the
native forward/reverse pair satisfies the inner-product identity. Not claimed:
full-spectrum policy, true non-contiguous strides, lengths outside {16,18},
generic `TileToX86Pass` emission, Apple/CUDA packages, or selector-grade
performance. Order 8a adds native centered constant/reflect padding and
centered/explicitly cropped ISTFT for n=16/n=18 on both proven architectures,
with policy identity, low-precision storage, independent VJP, and exact-device
checks. Order 8b carries arbitrary normalized logical axes through
Schedule→Tile as `outer`/`inner` indexing and proves nontrivial `(2,3)` native
forward/adjoint packets independently on AVX-512 and gfx1151 for C-contiguous
storage. The x86 v7 package now consumes true runtime stride descriptors and
proves full spectra plus explicit n=20/window=15 forward/reverse against
independent oracles; its reverse Schedule→Tile hash binds the stride ABI.
gfx1151 still needs the equivalent architecture-owned implementation and
exact-device packet. Per-batch broadcasting and physical streaming remain
open. Streaming state lineage itself now binds policy, window, tail, counters,
and parent state, with tamper tests. No sibling-backend evidence is inferred.
### Architecture expansion after the shared gates

- **ROCm:** gfx1151 owns the first exact-device correctness and calibrated
  performance packets. gfx1200, gfx1250/MI455X, and gfx1251/MI430X remain
  fail-closed until their own Target operations, dtype paths, packages, and
  device packets exist. FP8/BF8, IU8, sparse SWMMAC, FP4/MX scaling, and
  gfx1251 FP64 WMMA remain separate typed-execution work, not capability-table
  promotion.
- **x86:** AVX-512 owns the no-async Schedule Object proof and clean Zen 5
  packets. AMX remains a separately access-gated lane.
- **Apple and NVIDIA:** consume the same shared Graph/Schedule/Tile contracts
  through architecture-owned plugins. NVIDIA's remaining typed-fragment and
  barrier-at-birth work precedes its SO-2/SO-4 and TSOL promotion; Apple keeps
  independent Metal policy and evidence decisions.

### PR and promotion rules

1. A PR changes one shared boundary or one architecture's physical evidence,
   not both unless the exact hardware is available in that same validation
   environment.
2. Every migrated family declares its Graph, Schedule, Tile, and Target owner;
   package construction cannot move back into `JitFn` or runtime dispatch.
3. Every schedule/layout optimization lands first as rank/prune-only and keeps
   the old path as an oracle. Selector authority requires clean exact-device
   packets from `EVIDENCE-PACKET-1`.
4. Shared IR, ABI, dtype, numerical policy, benchmark-schema, or runtime changes
   update all four backend queues with parity, follow-up, or not-applicable
   outcomes.
5. Generated dashboards are regenerated by their owning generator. Plans link
   to them and do not hand-maintain competing totals.

**E2E-REAL-6F optimizer migration is complete; the fleet certificate gate is
landing:** the remaining physically owned optimizer VJPs share the
non-reexecuting state-lineage plugin boundary, and x86/gfx1151 now have total
family packets. Seven NVIDIA rows and one Apple row remain independent,
blocking exact-device follow-ups. Orders 12 and 13 may
proceed independently because they are evidence infrastructure; they do not
authorize promotion until the corresponding architecture packet exists.

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
   **Gate ACTIVE (2026-08-20):** a migrated differentiable family also
   clears the Law-3 adjoint sweep (`⟨Jv,u⟩ = ⟨v,Jᵀu⟩` at dimension-scaled
   probe counts) before its duplicate authority is deleted. The oracle
   exists and is swept: AD-LAW-1's spec growth closed
   ([`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) §7) — 300
   tensor pairs + the full geometric registry pass, the `vjp_only` class
   is empty, and the only unswept rows are eight rule-capability gaps
   pinned with named reasons (`_OPEN_UNSWEEPABLE_RULES` in
   `tests/unit/test_autodiff_laws.py`); a family touching one of those
   ops fixes the rule first. The check completes the existing pointwise
   numeric identity on the transpose axis; it does **not** replace
   per-family derivative oracles, since a matched-wrong JVP/VJP pair
   satisfies the adjoint identity on every probe.
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
| 5B | **E2E-REAL-5B — canonical attention backward** — **bounded physical carriers implemented 2026-08-05; public x86/gfx1151 plugin authority landed under E2E-REAL-6C on 2026-08-17.** | `schedule.attention_backward` is one content-addressed three-result dQ/dK/dV edge. Its digest binds the tensor-valued dQ loop, two-way dK/dV split, ascending reduction, launch-owned workspace, modifiers, and architecture-owned LSE checkpoint identity. `ScheduledAttentionBackwardArtifact` carries the exact program Tile text into the x86 saved-LSE ABI or the gfx1151 five-entry package without Graph re-entry. The native-VJP plugin now binds the tracer parent, Schedule, Tile, native image, and aggregate artifact identities before runtime. | Production Graph→Schedule→Tile and both physical lowerings pass. Public rank-4 exact x86 and WSL-visible gfx1151 tests cover ragged GQA causal gradients, and the 67-test affected ROCm cohort proves bottom-right ragged causal/window alignment across generated forward/backward kernels and the shipped ABI. The rank-3 `multi_head_attention` wrapper remains explicit compatibility work; gfx1200/gfx1250 fail closed; NVIDIA and Apple need independent plugin consumption and owning-host evidence. |
| 5C | **E2E-REAL-5C — stateful/training families** — **bounded Lion VJP, factored/full Adafactor VJP, and sequence-mixer backward slices implemented for x86/Zen 5 and ROCm/gfx1151, 2026-08-05.** | `tessera.state_buffer_lineage.v1` content-addresses logical buffer name, role, static shape, dtype, version, access, parent identities, and mutation policy independently of host object or device address. Typed `schedule.lion_vjp`, `schedule.adafactor_vjp`, and `schedule.sequence_mixer_backward` operations each lower to exactly one `tile.training_kernel`; the runtime consumes that exact artifact and does not retain or reconstruct Graph-op metadata. Adafactor binds factored row/column and full-state topologies. Sequence-mixer identity includes checkpoint, chunk-summary/prefix/fill, reverse phases, workspace, and fresh dQ/dK/dV/dgate/dbeta/ddecay outputs. | Schedule/Tile tamper and structural tests pass; exact Zen 5 and WSL-visible gfx1151 Lion, Adafactor, and gated/modified DeltaNet backward cohorts pass without fallback. gfx1200/gfx1250 remain fail-closed pending profiles and exact-device evidence. NVIDIA and Apple require architecture-owned consumers and owning-host validation. Broader stateful/training families remain under this row; these three bounded migrations are no longer open work. |
| 5D | **E2E-REAL-FFT — canonical spectral FFT** — **typed artifact boundary, persistent ROCm package, and the second x86/gfx1151 performance slice implemented; hardware follow-ups remain landing, 2026-08-05.** | `schedule.fft` content-addresses mode, shape/axis, direction, normalization, storage/accumulation, algorithm, radix policy/sequence through radix 17, Bluestein size, workspace policy, residency, twiddle policy, kernel family, and launch size. x86 caches Bluestein plans and owns native AVX-512 mixed-radix Stockham codelets. gfx1151 loads a prebuilt versioned shared image whose bounded persistent plan is keyed by the exact Tile digest; Bluestein owns four M buffers including an immutable transformed chirp. Rader remains candidate-only, Bailey is rejected, and gfx1151 fused LDS remains a separate candidate. | Production `tessera-opt`, native x86 images, and `libtessera_spectral_rocm.so` rebuild. ROCm persistent plans are 1.24x--1.45x faster than legacy per-call allocation at N=257/509/1009 in synchronized WSL host-wall timing. x86 cached Bluestein is 1.57x--1.76x faster and mixed radix wins 12/13 shapes. HIP events still return zero and rocprofv3 emits no WSL timestamps, so fused LDS remains experimental pending bare-metal evidence. gfx1200/gfx1250 fail closed. |
| 5E | **TSOL-ROCM-E2E-1 — compound spectral programs** — **typed Schedule→Tile carriers plus expanded x86/Zen 5 and ROCm/gfx1151 consumers implemented 2026-08-06.** | `schedule.spectral_program` content-addresses child FFT digests, bounded specialization template and exact shape, arbitrary axis, storage, normalization, layout, pad/crop, window/hop/frames, workspace, accumulation, native entry, and mutation lineage for all five compound spectral ops. Both runtimes consume one exact Tile artifact without Graph re-entry. Native package ABI v4 owns forward/ortho scaling, f16/bf16 conversion around f32 accumulation, and host-side arbitrary-axis pack/unpack. The HIP image exports one compiled architecture and stale cross-architecture images fail closed. | 36 combined Zen 5 contract/package/evidence tests and 15 exact gfx1151 package tests pass. Each architecture owns a 30-row full-family packet covering all five operations, seven digest-changing bounded specializations, every physical policy, and combined dynamic-axis-reduced-storage-ortho execution. x86 timing is selector-eligible; gfx1151 timing is synchronized WSL host wall and remains selector-ineligible. Separately stamped gfx1200/gfx1250 ABI-v4 packages cross-build, but their profiles remain `build_only`/fail-closed pending architecture-owned schedules and exact-device evidence. Bare-metal gfx1151 device events and Apple/NVIDIA physical consumers remain follow-ups. |
| 5F | **ROCM-MATH-EVIDENCE + MATH-PHYSICAL-2 — stable statistics, boundary semantics, and physical math efficiency** — **gfx1151 and Zen 5 bounded slices implemented 2026-08-06.** | ROCm var/std use centered parallel Welford; unary/binary codegen preserves difficult IEEE/NumPy domains; generated HIP math modules are process-cached by family/chip/op/dtype; and x86 arithmetic scans use an evidence-selected AVX-512 Hillis--Steele prefix while extrema retain their faster scalar recurrence. Binary physical packages reject mixed input storage. | Exact gfx1151 math passes 579 tests across fp32/fp16/bf16 storage; exact Zen 5 math passes 167 tests. The gfx1151 module cache improves seven f32 host-wall medians by 1.46x--3.58x. Paired Zen 5 `cumsum`/`cumprod` improve 1.48x/1.47x. ROCm timing remains selector-ineligible under WSL; sibling GPU backends require owning-device validation. |
| 5G | **TSOL-CONTRACT-GENERALIZE + X86-WELFORD-PARITY** — **shared contract, x86 Welford parity, and x86/gfx1151 TSOL policy expansion implemented 2026-08-06; packed fusion and DCT-I/III/IV landed 2026-08-08.** | `tessera.scheduled_spectral.v5` separates bounded template identity from exact physical specialization and carries dynamic bounds, arbitrary axes, fp32/fp16/bf16 storage, backward/forward/ortho normalization, explicit DCT-I/II/III/IV identity, and hashed fusion topology through verified Schedule→Tile lowering. Even-length compound paths bind packed N/2 children; gfx1151 v6 folds Hermitian work into fused LDS. The causal streaming-STFT policy content-addresses overlap state and fails closed for centred streaming without lookahead lineage. | Native x86 and gfx1151 correctness suites pass. Historical v5 packets remain historical; v6 promotion requires fresh clean Zen 5 and bare-metal gfx1151 evidence. Physical adoption of centred/n-FFT/full-spectrum/output-length STFT policies remains open. gfx1200/gfx1250 execution stays fail-closed; CUDA/SM120 and Apple/Metal physical consumers remain architecture-owned follow-ups. |
| 6 | **E2E-REAL-6 — delete duplicate authorities — active family-migration cohort (updated 2026-08-17).** | Ordinary pure straight-line tensor calls cache tracer-owned canonical Graph IR; native forward and reverse products consume that module rather than decoration-time AST Graph. Pure families use a cached content-addressed topology/numerical differential. Stateful Lion, factored/full Adafactor, and causal sequence-mixer backward use the complementary structural non-reexecuting certificate: the tracer executes once, the legacy candidate is never replayed, and physical numerical authority is bound into the package with versioned state/workspace lineage. Native-product plugins now own normalization, bounded compound spectral products, canonical rank-4 attention reverse, and those bounded x86/gfx1151 stateful reverse families. `JitFn` has no x86/ROCm package construction for them. The rank-3 MHA wrapper, sibling-architecture stateful packages, and other unmigrated families remain explicit compatibility work. `_OpExtractor` remains only as the named candidate/CPU-oracle substrate until those migrations close. | Fresh LLVM/MLIR 23 x86 and combined ROCm+x86 builds remain the host-free gate. Exact WSL-visible gfx1151 tests cover the migrated numerical families. E2E-REAL-6E adds one-execution, frontend-certificate, source/state/Schedule/Tile lineage, flat Adafactor topology, and runtime-no-Graph proof for Adafactor and sequence mixers on x86/gfx1151. Clean selector evidence and sibling architecture packages remain independent. Global Decision #31 remains open for remaining effectful families and backward-helper extraction. |

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

> **W1 status — 2026-08-18. Target-specific closure remains.** Verified against
> the changed tree.
>
> | item | state |
> |---|---|
> | W1.2 shape-rule registry | ✅ complete |
> | W1.3 metadata boundary verifier | ✅ complete |
> | W1.4 GA grade threading | ✅ complete |
> | W1.1b semantic `$kind` | ✅ closed sets are declaration-owned across Graph, Neighbors, Apple, NVIDIA, and ROCm; open symbol names remain deliberately open |
> | W1.1 Tile IR typing | 🟡 typed shared fragment contract closed, including bare-fragment rejection and CAKE TCGen05 row 8; NVIDIA producer/Target proof remains |
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
| W1.1 | **Shared fragment permissiveness removed; architecture closure remains.** `MMAOp::verify()` no longer chases producers or admits bare `!tile.fragment`; the historical spelling only parses to produce a named migration diagnostic. The explicitly retained tensor value lane is rank/arity/accumulator checked and belongs to migration into `tile.matmul`, not to the fragment ABI. CAKE row 8 now requires typed role-a/role-b `family="tcgen05"` fragments and descriptor agreement. ROCm's typed route and performance closure remain as previously recorded. **Open:** migrate NVIDIA's final tensor producers and prove its Target lowering on SM120; remove the temporary tensor value lane after those producers move. gfx1200/gfx1250 remain fail closed pending architecture-owned profiles and evidence. | IR Stack §U1 + Target §X2 | landing |
| W1.1b | **Closed for semantic selectors on 2026-08-18.** The measured scope was broader than the old “14 ROCm kinds” wording: closed operation-specific constraints now own Graph reduction/loss/spectral selectors; ROCm spectral, math, pack, reduction, scan and related selectors; Apple KV-cache kinds; NVIDIA CUDA-math kinds; and the canonical Neighbors topology kind. The topology constraint was placed on the core registered op as well as the standalone component after validation exposed that the standalone `.td` is not the registered authority. Unknown kinds now fail at IR verification. Open caller-chosen symbol names remain deliberately free strings. | Target §X3 | closed |
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
| W2.3 | **Region-aware and whole-program memory activity landed.** Reverse AD consumes W2.1 activity and `stop_gradient` semantics instead of its private walker. Active structured operations propagate to explicit operands and implicit captures. A backward whole-function memory fixed point now retains preceding writes that may feed an active read, including result-less writes and conservative cross-block dependencies; the inspection pass annotates result-less operations too. `RegionAdjointInterface` owns admitted internal block activity. | Autodiff D3 | closed |
| W2.4 | **Production relational legality authority consolidated 2026-08-18.** Fragment/TCGen05 and semantic-kind local invariants are binding ODS/type constraints, and WarpSpec divergence derives only from registered `schedule.warp` ancestry. `TileDataflowLegalityPass` now owns the shared pipeline, WarpSpec, barrier-reuse, and derived token/dataflow implementations in explicit pre-lowering and final stages. The old CLI pass names are compatibility wrappers over those same implementations and are not production authorities. `IRContractLegality` and Graph/layout checks remain separate level authorities. Remaining cleanup is deletion of the wrappers after downstream users migrate, not another legality algorithm. | IR Stack §U2 | shared objective closed; wrapper retirement open |
| W2.4a | **CAKE typed Tile sync/memory + SO-2 role surface.** Phase 1 §5.1–§5.4 is closed: typed waits/tokens, loop-carry provenance, registered sync vocabulary, hatch deletion, and production NVIDIA legality wiring. On 2026-08-16 the Phase 2 role carrier added loop-carry-safe `!tile.role`, role-bearing pipeline/mbarrier ownership, a ROCm role-producing and role-consuming wave/LDS path, and explicit x86 `no_async_noop`. The plan-named gfx1151 §5.5 cohort passed **8/8**. On 2026-08-18 WarpSpec stopped recognizing arbitrary `tile.warp_role`/`tile.warp_guard`/`tile.wg_id` ancestor markers, and CAKE row 8 became a typed TCGen05 fragment contract with positive and negative host-free tests. **Open:** NVIDIA barrier-at-birth emission and SM120 exact-device proof; no CUDA performance claim is implied by the host-free row-8 verifier. | CAKE §5–§6 | gfx1151 gate closed; shared row 8 closed; NVIDIA producer proof open |

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
| W3.1 | **Delayed/symbolic tracing foundation landed.** Pure programs with fully static tensor annotations now abstract-trace directly into canonical Graph IR during JIT emission, preserving Python argument names and bypassing `_OpExtractor`. Their AST module is materialized lazily only when an explicit differential/non-reexecuting certificate requests the compatibility oracle. Concrete tensor signatures remain tracer-owned. `_OpExtractor` is still required by the non-executing effect classifier and by dynamic/untraceable compatibility families; deletion therefore depends on symbolic-dimension tracing plus registered-effect discovery that does not execute user code. | Frontend §U1 + IR Stack §U3 | landing |
| W3.2 | **Superseded in delivery shape by E2E-REAL-0 through E2E-REAL-5.** Build and register a real Schedule dialect, preserve Graph SSA under schedule decisions, lower one scheduled matmul to the launch-level Tile ABI, make x86/ROCm packages consume that artifact, then migrate families. The Python spine is the differential oracle. This is three boundaries plus bufferization/package API work, not a 3-week convergence edit. | IR Stack §U3 + Target §X5 | re-estimate after the matmul vertical slice |
| W3.3 | Split the Tile dialect by level: primitives stay `tile.*`; whole-kernel ops → Graph IR / `tessera.kernel.*`; domain ops → `tessera_ebm`; `svd`/`qr`/`cholesky`/`lu` → linalg solver | IR Stack §U4 | 2w |
| W3.4 | **Native reverse dispatch decomposition landing.** Loss VJPs use explicit Graph/Schedule/Tile/Target plugins on their proven x86/gfx1151/SM120 pairs; KL/JS remains gfx1151-only. ROCm matmul composition and selective-SSM backward resolve through plugins. The verified SM120 Lion and causal DeltaNet packages are now selected by those family plugins, so unmatched NVIDIA and ROCm operations fail closed instead of entering target-wide dispatch. The six unreachable loss/ROCm/NVIDIA compatibility helper bodies have been deleted. The CUDA executors remain architecture-owned and do not inherit x86/gfx1151 lineage or performance evidence. JitFn/public-module decomposition and general forward-package ownership remain open architectural cleanup. | Frontend §U5–U6 | landing |
| W3.5 | **General shared IFT execution plus typed physical child-composite landed.** `NewtonAutodiff` accepts arbitrary typed residual bodies and emits value-producing IFT VJP/JVP functions with explicit matrix-free GMRES/CG convergence policy. The execution oracle and physical parent use restarted GMRES with true-residual checks; the parent hashes residual, solution JVP/VJP, and parameter JVP/VJP children and executes them on AVX-512/gfx1151 without materializing a Jacobian. The compiler generates all five children for pointwise, sum/mean, rank-2 reduced-storage matmul/transpose, distinct parameter/solution spaces, bounded-dynamic dimensions, explicit mixed-storage widening, statically bounded `control_for`, and pure scalar `if`/bounded-`while` predicate replay. Expanded 30-sample WSL correctness packets are committed for AVX-512 and gfx1151. Remaining: Apple/NVIDIA consumers, broader Krylov packets, non-pure/vector predicates, and clean selector-grade timing. | Autodiff §B8 + OT R2 | landing |
| W3.6 | Batched operands in `ExpandProductTable`; connect `RotorSandwichFold`'s marker to a consumer | GA/EBM §1.3 | 2w |
| W3.7 | **ROCm reverse-package ownership closed for the current single-op registry; forward ownership remains.** Every currently admitted ROCm native VJP resolves through an explicit family plugin, including losses, matmul composition, and selective SSM; unmatched families fail closed. Registered C++ MLIR→ROCDL/HSACO remains the canonical native spine. Forward package construction and dead compatibility-method deletion remain part of E2E-REAL-6, not a reason to retain a general ROCm backward dispatcher. | Target §X6 | landing |

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
| W4.1 | **Tracer-owned multi-block carrier and bounded arbitrary-CFG consumer landed.** `for`/`if`, bounded `while_loop`, and `control_scan` retain nested bodies and recover a content-addressed basic-block graph with explicit branch/yield/backedge values, merge arguments, entry, and exit. A deterministic Tarjan analysis classifies source carriers as acyclic, reducible, or irreducible. The paired pass keeps the compact canonical `scf.if` form for four-block diamonds and lowers every other bounded pure native `cf.br`/`cf.cond_br` graph—including true two-entry irreducible SCCs—to a typed program-counter state machine. Each block argument owns a distinct state slot; CFG and Presburger identity survive. Unbounded graphs, effects, unsupported nested regions, malformed edge ABIs, and absent execution bounds fail closed. | landing |
| W4.2 | **Typed Presburger and nonlinear-witness boundaries landed.** Python emits a versioned, content-addressed coefficient-vector carrier (`eq`/`ge`/`mod`) before compilation. `SymbolicDimEqualityPass` consumes it with MLIR `IntegerPolyhedron`; modular rows introduce existential quotient locals. Attaching a system re-digests the structured CFG and stamps the same typed-system identity on every nested block. Polynomial relations use a separate content-addressed carrier, require complete concrete dimension witnesses, use checked exact integer evaluation, and never become affine scheduling facts. Remaining: richer automatic source-CFG capture and native Presburger queries over general multi-block transformations. | landing |
| W4.3 | **Explicit residual SSA execution and bounded physical products landed.** `RegionAdjointInterface` differentiates `scf.if`, positive-step counted `scf.for`, canonical bounded `scf.while`, lowered `control_scan`, and the bounded arbitrary-CFG state machine. Dynamic saved state now has a total per-slot maximum envelope, bounded data tape, runtime bound assertion, and companion shape tape; backward slices by recorded logical extents rather than padded bounds. SAVE/HYBRID carry exact checkpoint state. Compiler-generated extent assertions are the first replay-safe observational effect; user-marked mutation/RNG/I/O/collective work cannot bypass registered effects. The tracer-owned CFG carrier also recovers variadic branch merge state. Existing x86/gfx1151 packets remain correctness-only. Remaining: operation-owned products for admissible effects, broader automatic Python CFG recovery, and clean Zen 5/bare-metal gfx1151 timing (BOTH exact-device irreducible-state-machine rows landed 2026-08-21 — gfx1151 per-thread device kernels via the canonical executable pipeline, and native x86 execution via tessera_jit; each binds the CFG digest and enforces the step bound). | landing |

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

### MATH-SOURCE-WORKSTREAM-1 — host-free reference-lane completeness

Owned by [`MATH_SOURCE_WORKSTREAM.md`](MATH_SOURCE_WORKSTREAM.md), derived from
two tensor-calculus texts and a deep-learning theory book supplied 2026-09-01.
**It is recorded here rather than given an Order row because it does not compete
with the ordered queue**: every item is host-free reference-lane or contract
work, while Orders 1-14 are physical-execution evidence. It contends for no
queue position, no box, and no architecture proof.

MSW-1 landed 2026-09-01 and is the reason the rest is queued: `grad(grad(f))`
and `jacrev(grad(f))` were returning a **silent zero gradient** for functions
with a nonzero second derivative. `_ACTIVE_TAPE` is a single contextvar, so an
inner `tape()` shadows the outer completely; the outer then walks a tape the
differentiated input never reached, and `grad`'s zero branch read "no
cotangent" as "constant in this argument". Every existing guard checked the
OUTPUT side; nothing checked the input side. The forward-mode twin was already
caught via `active_jvp_trace()`, so the fix restores symmetry rather than
inventing a rule. This is a fail-open path of exactly the class W0.5/W0.8 are
listed as "what never to cut".

Two boundaries this workstream must respect, both already owned elsewhere:
MSW-5 (coordinate-aware field calculus) routes through
**PDE-STENCIL-FOUNDATION-1**, whose "none is manufactured by legalization" rule
is the precedent it applies to `ga/calculus.py`; and MSW-2's exact higher-order
path reaches for the capability **W6.3 / AD-WEIL-1** owns, without claiming it.
MSW-7 may not add a `tessera.einsum` ODS op ahead of a consuming pass (#29) or
as a second lowering authority (#31).

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
   binding fails closed with a named diagnostic and ships **no layout-evaluation
   fallback path**, because a fallback is a second implementation in disguise.
   The fixed rank-2 source-text template used by host-free MSL emission is an
   explicitly bounded exception: it does not evaluate a layout, has only the
   two ABI orders, and is checked against the native plan. Exhaustive proof
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

**2026-08-24 L1/L3/L4/L5 checkpoint:** the C++ authority transports nested
layout trees, implements scalar composition, complement/inverses, coalescing,
product/divide variants, slice-with-offset, bounded dynamic residues, exact
layout factorization, and `cosize`-based residency. Schedule Object v2 binds
the resulting layout digest, factorization, capacity, residency, alias set,
and lifetime proof. Tile IR materializes mixed-radix basis maps and represents
a static tuple codomain as a product of independently proven scalar composed
layouts; dynamic/non-separable tuple codomains remain fail closed. NVIDIA,
ROCm, and x86 re-run the proof before target address emission. The x86 consumer
handles static, bounded-dynamic, nested mixed-radix, and static tuple-product
scalar maps with runtime guards and exact LLVM execution on AVX2 and AVX-512.
SM120 has both the narrow
typed and alignment-safe dynamic macro-CTA strided routes exact-device proven;
gfx1151 has a bounded-dynamic compact package proven on Radeon 8060S. The
shared rank-2 authority now serves CUDA, ROCm, the x86 f32/f64/bf16/u8s8 core
GEMMs, and the reachable Apple MSL families, while the shared raster generator retains
its exhaustive equivalence gates. x86 odd-shape and vector-tail cases pass on
the AVX-512 Ryzen AI Max+ 395 host, and incompatible AVX2 hosts reject the image
before loading. Apple consumes the plan through the versioned no-fallback
native ABI: all reachable steel, fused matmul, normalization, attention, and
gated-matmul MSL templates preserve their established arithmetic. All 20
raster choices pass host-free, canonical GEMM passes 3/3 including a ragged
edge, and fused cooperative-matrix pointwise/reduction passes 38/38 on M1 Max.
This closes the physical-consumer tail and LAYOUT-ALG-1; unsupported dynamic
non-separable tuple codomains remain fail closed rather than becoming an Apple
schedule claim.
3. **L3 — the `⊑` decision procedure — implemented.** FORGE W1/W2's legality query and
   residency check implemented as layout factorization and `cosize`. The lattice
   stays the declared interface; the algebra makes `block` precise instead of a
   promise. Closes the S9 implementation gap.
4. **L4 — emitter index math — landed on CUDA/ROCm/x86/Apple.** Consolidates the four hand-written block-index
   emitters and the hardcoded `A[row*K+k]`/`B[k*N+n]` templates onto the shared
   algebra. Acceptance is **bit-identical output** for every currently reachable
   `(raster_order, raster_group)` — a pure-refactor proof. Choosing a measured
   non-default raster is explicitly **not** in this item: per ROCM-CALIB-1 that
   is architecture-owned and blocked on a correlation/retain verdict, and it
   does not transfer between targets.
5. **L5 — MLIR carrier.** The bounded seed is implemented as
   `#tile.composed_layout` plus `tile.materialize_composed_layout`, preserving
   nested static/dynamic scalar-affine residue while delegating proof to L1.
   Mixed-radix basis maps and static separable tuple codomains are physical on
   SM120/gfx1151; dynamic/non-separable tuple codomains remain carrier-only. Folding the proven
   static subset and any future integration into `#tile.layout` must still use
   a `MaybeStaticTypeInterface`/fold-static boundary rather than duplicate the
   algebra. The Decision #31 ordering caveat in
   [`W1_1_TYPING_DESIGN.md`](W1_1_TYPING_DESIGN.md) continues to apply to
   `#tile.mma_desc`.

Independent of the above and not blocking it: a negative-scoped
`tessera-target-opt` driver that registers the Target IR dialects without
`tessera`/`tile`, making Tile IR leakage a parse error. `test_target_ir_contract.py`
parses and verifies every emitter and golden, but registers everything, so it
cannot falsify a *leakage* claim — the same argument as Decision #19's standing
lesson that a host with the ISA cannot falsify a host-portability claim.

### FRONTEND-IR-MEDIUM-1 — the IR as the single medium of record

Bound 2026-09-03 from
[`FRONT_END_LOWERING_ASSESSMENT.md`](FRONT_END_LOWERING_ASSESSMENT.md), which
owns the finding and the KGEN-comparison rationale. Global ordering and
promotion authority remain here.

The finding that justifies an ID rather than a reference: **Tessera's system of
record is the Python object graph, and the MLIR text is a lossy projection of
it.** Measured on the checked-in `tessera-opt` (host-independent): symbolic
shapes do not survive to the parser (`Tensor['M','K']` → malformed
`tensor<?x?x?>` → parse error), `loc` is absent from the emitted text entirely,
`numeric_policy` is opt-in rather than universal, and `provenance`/route/arbiter
decisions live in Python descriptor objects (46 modules) rather than as IR
attributes. `tessera-opt` optimizes the projection; the Python spine reasons
over the record; the two do not share a memory. This subsumes the Apple
"two compilers" and Python-packager seams as symptoms. It also reads
E2E-REAL-6's "one compiler authority" one level stronger — from *one frontend*
to **one medium**: every fact a pass reads is an attribute on the IR the C++
passes see (Decision #29's consumer rule, run in reverse).

Three sub-tracks, sequenced smallest-blast-radius first:

1. **Down-payment (independently landable now).** (i) **Landed 2026-09-03 — scope
   corrected on implementation:** the front door already recorded the
   value-lane failure (`apple_value_target_ir_error`, S4); what was missing was
   a *named* reason. `graph_ir.unresolved_element_type_diagnostics` now emits
   `GRAPH_IR_UNRESOLVED_ELEMENT_TYPE` (Decision #21a) per unresolved
   argument / result / op type and the driver's value lane consults it before
   rendering, so the recorded reason names the argument and the missing
   semantic key instead of the parser's symptom; renders are byte-unchanged and
   `index` / `i1` / `!tessera.*` handles are never flagged. (ii) **Landed
   2026-09-03:** the tracer records the user's call-site span on every op
   (`trace._user_source_span`, first frame outside the package) and the
   canonical render emits a repo-relative `loc("file":line:col)` — relative so
   the content-addressed canonical text stays host-independent — verified
   through `tessera-opt -mlir-print-debuginfo`. The AST `_OpExtractor` still
   emits no `loc`, so its deletion no longer regresses Decision #13.
   (iii) **Landed 2026-09-03.** The symbolic→concrete elaboration boundary,
   the region-privilege (Decision #2) and the `ConstraintSolver` (Decision #4)
   drops are declared through the **existing** W1.3 metadata-obligation pass
   (`--tessera-verify-metadata-obligation`), not a new verifier (Decisions
   #29/#31), using `not_yet_carried:FRONTEND-IR-MEDIUM-1`. The C++ verifier is
   **unchanged**.

   *One correction to how this was specified, recorded because it is the whole
   trick.* The instruction "declare them through the existing pass" is not
   directly executable: that pass takes its `before` from
   `--tessera-record-metadata`, which walks MLIR, and these three facts never
   enter MLIR at all — so the pass can never record them, and a bare
   declaration is refused as `METADATA_OBLIGATION_STALE_DECLARATION` (measured
   first, on all three names). What makes it work is that the verifier decodes
   the snapshot **unfiltered**: it is not restricted to the four attributes
   `isTrackedName` collects. So the **frontend writes the snapshot itself** —
   the frontier is the one boundary whose `before` lives in Python —
   and the verifier then behaves exactly as it does for `numeric_policy` at the
   Tile boundary. `graph_ir.declare_frontier_debt` stamps both halves from
   `JitFn`'s decoration-time `IRArg`s and solver; `tests/unit/
   test_frontier_metadata_obligation.py` (11) covers it, plus a positive and a
   negative lit fixture. Verified end to end on real `@tessera.jit` output: the
   emitted module passes `--tessera-verify-metadata-obligation`, and deleting
   any one declaration fails it with `METADATA_OBLIGATION_SILENT_DROP`.

   Deliberately narrow: nothing is stamped when the frontend held none of these
   facts, and a fact the IR still carries is never declared — both would be the
   unconsumed declaration of Decision #29, and the second silently licenses a
   real future drop.

2. **Pre-elaboration parametric optimization (perf + rigor).** Elaboration is
   entirely pre-MLIR today, so `tessera-opt` only sees concrete instances and
   there is no tier that optimizes the parametric recipe once before it is
   stamped per shape bucket. Build one on the existing `PresburgerSystem`
   substrate carried through `structured_cfg.py`. Arbiter payoff (Decision #28):
   buckets compared as instances of one optimized recipe, not N independently
   lowered programs, so a bucket-to-bucket regression is a real difference. Also
   retires the Decision #29 producerless consumer — `SymbolicDimEqualityPass`
   consumes `tessera.dim_names` that the frontend cannot currently emit
   parseably.

3. **Raising / idiom recognition (algorithmic).** No pass lifts a hand-written
   loop nest back to `tessera.matmul` / `tessera.flash_attn`, so user-written
   math cannot reach the arbiter's Tier-3 hand-tuned candidates. Add a raising
   path so arbitrary user code has an on-ramp into the high-performance tier —
   the inverse of `TesseraToLinalgPass`, composing with sub-track 2 (a raised
   parametric idiom is optimized once and elaborated per bucket).

**2026-09-04 implementation update — first rank/prune stage only.**
`parametric_recipe.prepare_recipe` now runs the existing native symbolic
verifier, canonicalizer and CSE on a typed, dynamic-shape Graph IR recipe before
bucket analysis. It retains the input MLIR oracle and content-addresses the
optimized program, Presburger system and compiler binary. Complete positive
integer witnesses may be pruned by the typed constraints; all survivors remain
promotion-ineligible. `JitFn.rank_parametric_buckets` is the explicit opt-in
consumer. A duplicate-matmul fixture proves native CSE occurs before buckets.
The SymbolicDimEquality consumer now reads the frontend's argument-local
`tessera.dim_names` as well as the legacy function array; positive and negative
parser-backed fixtures cover the connection.

`loop_idioms.recognize_matmul_loop` provides the first raising analysis:
two matching f32/f64 rank-2 arguments, fresh NumPy zero output, and an exact
complete i/j/k multiply-accumulate nest. The source loop is retained as the
oracle; the candidate contains the existing `tessera.matmul` operation and
feeds `prepare_recipe`. Unsupported source is refused. This is an explicit
compatibility analysis, not a second execution authority. Reassociation and
native numerical validation remain promotion requirements.

**Still open:** instantiate the optimized native program per bucket, wire its
identity and measured candidates into the arbiter, recognize tracer/MLIR loop
regions beyond this narrow source form, and add a separately proved attention
recognizer. No Tier-3 selection or backend execution support is claimed.

The retained AST frontend now emits file/line/column locations (virtual source
identity for explicit source that differs from the file); cache keys include
source location. `validation_tree.py` binds external expected file hashes and
rejects changes during a gate. The validation spine uses it, and generated
coverage has a resolve-authored-inputs-then-regenerate workflow documented in
[`VALIDATION_SPINE.md`](../../spec/VALIDATION_SPINE.md).

Acceptance is staged: sub-track 1 lands under existing frontend/driver fixtures
plus a fail-closed dtype fixture and a `loc`-round-trip check; sub-tracks 2 and
3 are rank/prune-only first and keep the current path as an oracle per the PR
rules above. No device evidence is implied — every probe in the assessment is
host-independent.

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
| W5.2e | **Automatic dependence-edge generation — conservative parity gate landed; selector proof remains open.** `infer_action_dag` consumes a fresh W2.1 Graph snapshot and total R2 resource vectors. It emits explicit edges for SSA producers, overlapping or unknown alias sets, value-scoped memory dependence, mutation/state/I/O, registered stochastic identity, ordered collectives, unknown effects, and region boundaries. Unknown alias facts now carry an explicit reason rather than relying on a side effect of `may_alias`. `compare_inferred_action_dag` requires generated edges to cover every edge in an existing hand-authored R3 fixture; additional conservative edges are reported separately. Pure SSA matches the current fixture exactly, and opaque dataflow is serialized. `CompositionCandidate.from_graph` remains the physical-family entry point. **Open (MegaMoE adopted 2026-08-24 — inferred edges with the hand DAG as fail-closed coverage oracle):** adopt it in the remaining non-JVP physical producers (spectral next) and collect clean calibrated Zen 5/bare-metal gfx1151 packets before scalar measured latency may select a retained schedule. WSL and analytical vectors remain prune/rank-only. | W2.1 + TileRT R3/R4 | landing |
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
| W6.3 | **Taylor/jet mode over Weil algebras — design landed, estimate supplied.** [`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md) owns the design and acceptance detail: one `DerivativeContract` datum per primitive evaluated under an explicit `DifferentialAlgebra` codomain parameter, so `Dual`, `TruncatedJet(k)` (Weil, dim k+1 vs nested 2ᵏ), `CliffordTangent`, `OperatorTangent`, and `TaylorModel` are instances of one interpreter rather than parallel registries. That protocol is the generic algebra representation this row asked for; the W6.4 shared-substrate hypothesis becomes a falsifiable acceptance criterion (run `Cl(3,0)` over the same multiplication-table substrate with `ga/signature.py` as oracle) rather than a sequencing assumption. Correctness is executable algebraic law (adjoint, homomorphism-exact-by-nilpotency, jet-vs-nested differential proof) with a machine-checked math harness per `CORE_SUBSTRATE_VIEW.md` §0.1. Slices: **AD-LAW-1** (~1–1.5w, host-free, no dependencies — law oracles over today's registries; may start in parallel like Orders 12–13), **AD-WEIL-1** (~3w, host-free), **AD-JET-STRUCT-1** (~4w), **AD-JET-IR-1** (~5–6w, research-adjacent; depends on W4-PRODUCT-1, LAYOUT-ALG-1 L1/L5, real `batching_rule`s, and the S5 `numeric_policy` carrier), plus **AD-OPERATOR-1** (~2w, independent — absorbs S8's unowned implicit-diff strict-complementarity item) and consumer-gated **AD-CERT-1**. No hand rule is retired before its jet-vs-nested proof is green (Decision #31 ordering). | Autodiff D6 | ~10w excl. AD-JET-IR-1 (open) |
| W6.4 | Table-driven GA kernel synthesis via `emit/`; then PGA `Cl(3,0,1)` | GA/EBM §2.3–2.4 | 5w |

**Exit:** a defensible "exceeds SOTA" claim with a benchmark behind it — sparse
Jacobian scaling `O(colors)` not `O(rows)`; order-`k` derivatives sharing the
tuned GA kernels. W6.3's half of that exit is sharpened by its plan to a
measured curve: order-`k` directional derivatives of a real workload via fused
jets versus nested forward-over-forward on the same hardware and numerics
budget. If the curve does not separate by `k = 3`, the IR descent is not worth
its cost and the program stops at the reference + law infrastructure — a
falsifiable stop condition, not an open-ended track.

> W6.4 can supply useful table-lowering machinery, and W6.3's generic algebra
> representation and AD semantics are now specified in
> [`AUTODIFF_NEXTGEN_PLAN.md`](AUTODIFF_NEXTGEN_PLAN.md). **The reuse
> hypothesis is still a hypothesis to prove, not a sequencing-based cost
> reduction** — it is now an explicit AD-WEIL-1 acceptance criterion
> (`Cl(3,0)` over the shared multiplication-table substrate, `ga/signature.py`
> as oracle), so it is settled early and cheaply rather than assumed.

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

### Engineering follow-through — 2026-09-04

Published snapshot: [PR #721](https://github.com/gstoner/tessera/pull/721).
The following work remains independently gated; publishing the snapshot does
not close native/compiler evidence gaps.

| Owner | Current action and acceptance |
|---|---|
| APPLE-DISPATCH-WEDGE-1 / telemetry | Scoped telemetry capture restores prior process state, including nested/error paths; two device-clock tests adopted it. An opt-in pytest order tracer records transitions after teardown without resetting them. The original process latch and downstream MoE failures still require the triggering order and exact Metal reproduction. |
| MSW-9 | Program-pair native evaluator adapter and separate reference composition/identity law family implemented. Graph IR fragment inventory and the actual fusion consumer remain open. |
| FRONTEND-IR-MEDIUM-1 | PR #721 contains native pre-bucket CSE and narrow source-loop recognition in rank/prune mode. Native instantiation, arbiter consumption, tracer/MLIR raising and attention remain open. |
| APPLE-DISPATCH-WEDGE-1 | Metal 4 direct wait now stamps timeout kind/message. Local runtime compilation and WSL source contracts are separate from the outstanding exact-device re-seal. Lowp MoE remains opt-in without a measured ledger row; cooperative rewrite remains open. |
| DISPATCH-BREAKER cross-backend | CUDA and HIP waits remain unbounded. A replacement must poison the owning context on timeout and retain every outstanding buffer/module/event; adding a polling deadline followed by current synchronous cleanup would still hang or free live storage. First slices should target one owning bridge each, with injected not-ready/error cases before GPU proof. |
| IKF-1 | Bind `INTRA_KERNEL_FEEDBACK_PLAN.md` here. P0 extends the existing D2 gfx1151 timing probe; P2 waits for green clocks and consumes Schedule-Object region identity. P1 host schema/math is independent; no IR instrumentation is authorized by a scalar clock sample alone. |
| Toolchain evidence | Assertion-enabled LLVM remains an explicit fleet proof gap. Do not interpret release-build negative tests as assertion-enabled contract falsification. The sandbox/Metal mechanism remains unproven; no claim of its root cause is made. |
| Apple cross-run decision | Preserve mean ± t·sd and incumbent-biased refusal. A robust alternative must be evaluated on recorded clean/outlier cohorts with unchanged promotion semantics before adoption. |
| Generated coverage | Preserve Decision #26 committed primary evidence. Continue regeneration from merged authored input via the owning generator. Moving evidence to CI-only artifacts requires a replacement immutable evidence contract, not merely deleting the files. |

All numerical/backend promotion remains architecture-owned. The immediate
reproduction priority is telemetry attribution; a scoped cleanup proves a
state leak is fixed but does not establish the root cause of the reported
order-dependent failure.
