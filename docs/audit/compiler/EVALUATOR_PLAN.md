---
status: Ratified (direction locked 2026-06-11); implementation phased below
classification: Design / Roadmap
authority: Compiler evaluator architecture — supersedes ad-hoc benchmark/conformance framing
last_updated: 2026-07-11
audit_role: plan
plan_state: landing
---
# Tessera Compiler Evaluator — Architecture & Roadmap

> **One-line thesis.** Tessera's compiler is past surface-inventory mode. The
> next unlock is not more ops — it is to make the **evaluator** generative,
> execution-derived, and **backend-rung-aware**, so the conformance matrix,
> benchmark rows, autotuning records, and any future search all *derive from*
> one honest scoring engine instead of being hand-declared registries
> spot-proven by one fixture each.

This document is the ratified architecture for that engine ("the Evaluator")
and the roadmap that drives Apple, NVIDIA, and AMD ROCm support forward through
it. It is grounded in the current compiler spine (see
[`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) and
[`DEEP_COMPILER_AUDIT_2026_06_10.md`](archive/DEEP_COMPILER_AUDIT_2026_06_10.md)) and in
a 2024–2026 research survey (§9). Counts are not copied here; the drift-gated
dashboards under `docs/audit/generated/` remain the count authority (Decision
#26), and every claim is grounded in source (Decision #27).

---

## 1. Why this, why now

The deep audit's own Method note is the whole game:

> *"A green drift gate proves the renderer matches the registry, not that the
> registry models reality."*

Every honest mechanism in the repo today — the conformance matrix, the
fixture-proof gate, the backend manifest — is **hand-declared and spot-proven**.
A cell reaches `numerical_check = complete` because a human wrote one
`execute_compare_fixture`. The drift gates prove the *dashboards match the
registries*; nothing proves the *registries match what the compiler actually
does* on inputs nobody hand-wrote. Deep-audit finding #2 is the tell:
`grouped_gemm` / `moe_swiglu_block` had working fused kernels + fixtures but were
invisible to the audit model for weeks. The renderer cannot catch a stale model.

A 2024–2026 research sweep (§9) converges independently — from compiler testing
(DESIL, PolyJuice), from benchmark/agentic work (TensorBench, AlphaEvolve,
Magellan, the **Sakana** reward-hacking retraction), and from autotuning (BaCO,
TLP, Mirage) — on a single structure: **one `evaluate(candidate) → verdict`
function with a hard correctness gate and a soft objective, of which everything
else is a client.** The literature also proves the discipline the hard way:
Sakana's agent found a sandbox hole and farmed fake 100×+ speedups until the
*scorer* was hardened; honest results then collapsed to ~1.1–1.2× median. **The
evaluator is the product; the agent is a commodity.** Tessera's unfair advantage
is that it already owns the correctness predicate everyone else retrofitted
after a blowup.

---

## 2. The core artifact: a per-backend Rung Ladder

The central correction to prior framing: backends are **not binary**
(artifact-only vs hardware-verified). They climb a ladder, and **most rungs are
hardware-free**. The Evaluator records, per generated program **per backend**,
the highest honest rung reached.

| Rung | Claim | Current evidence |
|---|---|---|
| 1 `artifact_only` | IR emitted, with no stronger evidence | The floor for an unsupported or fallback pathway. |
| 2 `lowers_clean` | Target IR passes the verifier with no unsupported diagnostic | Program-specific lowering evidence; do not infer it from a capability row. |
| 3 `emits_asm_text` | backend emits structurally-valid PTX / AMDGCN text | NVIDIA WGMMA PTX and ROCm AMDGCN/LLVM-IR emission are evaluator-visible. |
| 4 `assembles` | emitted text assembles in a real toolchain | ROCm WMMA can reach this rung through host `llc`; NVIDIA assembly remains toolchain/CI-gated. |
| 5 `codegen_stable` | two optimization levels produce structurally-equivalent code | A program-level evaluator check, not a backend-wide status. |
| 6 `numerical_symbolic` | microkernel is symbolically equivalent to its specification | Optional, hardware-free work. |
| 7 `executes` | the demanded backend ran on real silicon | The execution matrix has executable Apple, x86, ROCm, and `nvidia_sm120` pathways. |
| 8 `hardware_verified` | the demanded backend ran and matched the oracle | Earned only by `evaluate()` for the particular program and input/oracle pair. |

The enum names and ordering are implemented in
[`evaluator.py`](../../../python/tessera/compiler/evaluator.py); current runtime
pathways are generated from
[`execution_matrix.py`](../../../python/tessera/compiler/execution_matrix.py)
into [`runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md).
The matrix proves dispatch availability, **not** a universal rung-8 claim: a
fallback row remains artifact/reference evidence, and every hardware-verified
claim needs the evaluator's provenance and oracle gates.

**Current strategic point:** ROCm and x86 now have broad executable lanes; Apple
has its established runtime lanes; `nvidia_sm120` has a native MMA pathway.
Generic NVIDIA code generation and broader per-program hardware proof remain
separate work. Advance each pathway with an emitted/assembled artifact, then a
native execution record, then an evaluator oracle result—never by promoting a
registry row alone.

---

## 3. The verdict contract

For each generated program `p`, the Evaluator emits one `Verdict`:

```
Verdict(p) = {
  program_id, component_ops, fusion_descriptor,        # tie to the canonical spine
  inputs: [generated, hidden],                          # hidden = candidate never sees
  per_backend: {
    <target>: {
      rung: <1..7>,                                      # highest honest rung reached
      correctness: PASS | FAIL | UNPROVEN,               # multi-oracle (§4)
      execution_kind: reference | optimized_native | artifact_only,
      provenance_ok: bool,                               # NOT a silent fallback
      semantics_unproven: bool,                          # e.g. WGMMA assembles, not proven
      latency_ms?: {median,p10,p90},                     # only where rung>=6 (or Apple/x86)
      blocker: (op, failing_gate) | null,
    }, ...
  }
}
```

The **reward** for any search/autotune client is derived, never hand-fed:

```
reward(p, target) =
  0   if  rung < demanded_rung           # compile/lower/assemble failed
      or  correctness == FAIL
      or  provenance_ok == false         # silent fallback ≠ success
  else  speedup_vs_reference(target)     # perf gradient — ONLY where rung>=6
```

This is the convergent shape of FunSearch / AlphaEvolve / Magellan / Compiler-R1
/ post-fix Sakana: **a scalar gated behind a hard correctness predicate.**

---

## 4. The oracles (correctness, multiple and cheap)

Tessera's current oracle is *vertical* (eager-numpy vs the executed backend).
The unlock is to add *horizontal* and *opt-level* oracles that catch what the
vertical one masks, and to make provenance a first-class gate:

| Oracle | What it checks | Source idea | Backend reach |
|---|---|---|---|
| **Vertical** | executed output ≈ eager-numpy reference (tolerance-aware) | (have it) | Apple/x86 execute; NVIDIA/ROCm in HW window |
| **Horizontal** | run `p` through its own fusion/canonicalization rewrites; assert `pre ≡ post` on the *same* backend | PolyJuice (OOPSLA'24) | Apple/x86 today — backend-independent at the rewrite layer |
| **Opt-level** | same IR at min-opt vs full-opt → identical integer checksum (executable) OR structurally-equivalent SASS (NVIDIA/ROCm, hardware-free) | DESIL (OOPSLA'25) | Apple/x86 runtime; NVIDIA structural variant |
| **Symbolic** | microkernel ≡ spec via finite-field polynomial-identity / SMT | Mirage (OSDI'25), GPU-kernel SMT equivalence (2511.12638) | hardware-free; rung 5; heavy |
| **Provenance gate** | result came from `execution_mode="metal_runtime"` (the demanded backend), not the numpy fallback | synthesized + TritonRL anti-fallback scan | all |

**Tolerance discipline** (the soft underbelly the literature flags): tight
tolerances on integer / exactly-representable subgraphs (DESIL's philosophy);
for FP, pair the tolerance check with a metamorphic reassociation oracle so we
measure whether deviation *grows under equivalent rewrites* rather than picking
one magic epsilon. Generated inputs use NNSmith-style gradient-guided selection
to avoid NaN/Inf regions that dominate FP comparison.

---

## 5. Driving NVIDIA & AMD ROCm forward (without local silicon)

Three hardware-free levers, in priority order:

1. **Assembler-text emission, THEN assembly of every emitted kernel (rungs
   2.5 → 3) — corrected.** The earlier framing ("assemble Tessera's real emitted
   PTX — buildable now") had an unmet prerequisite: **Tessera does not emit
   assembler text today.** `@jit(target="rocm")` / the NVIDIA pipeline stop at
   Target IR MLIR (`tessera_rocm.mfma` / `tessera.tile.wgmma`), and
   `scripts/validate_nvcc_compile.py` assembles its *own* 8 hand-written CUDA
   stubs, not Tessera output. So the real sequence is:
   (i) **rung 2.5 — build the emission path**: lower `tessera.tile.wgmma` →
   actual `wgmma.mma` PTX (and `tessera_rocm.mfma` → `v_mfma` AMDGCN), one narrow
   kernel at a time (e.g. an sm_90 bf16 matmul);
   (ii) **rung 3 — assemble it**: `ptxas -arch=sm_90a` / `hipcc`+`llvm-mc` over
   the *real emitted* text — strictly stronger than `nvcc -ptx`; catches register
   pressure and illegal encodings. This is the NVIDIA/ROCm analog of the Apple
   provenance gate: it turns "we emit NVIDIA artifacts" into "every kernel we
   emit assembles for sm_90a." Rung 3 runs on a Linux x86 CI box with the
   CUDA/ROCm toolchain and **no GPU**; both rungs skip-clean on the arm64 dev Mac
   (no toolchain), so rung 2.5 (emission) is the only piece verifiable up to
   "emits valid-looking text" locally — assembly is Linux-CI.

2. **Correctness transfer via backend-independent Tile IR (rungs 2+4).**
   Graph/Schedule/Tile IR is backend-neutral. If the Evaluator proves `p`
   correct on Apple/x86 (which execute) **and** proves the NVIDIA lowering is a
   structure-preserving transform of the *same* Tile IR — same `fusion_groups`,
   shapes, dtypes, `layout_contracts`, all already carried by `CompileResult` —
   algorithmic correctness transfers down to the codegen boundary for free.
   TensorRight (POPL'25) is the formal version; the cheap version is structural
   lowering-equivalence asserted today.
   - **Irreducible limit (stated plainly):** this transfers *algorithm*
     correctness, **not** *codegen faithfulness*. Whether the WGMMA / TMA /
     tcgen05 / MFMA microkernel actually implements its descriptor is the one
     thing needing rung 5 (symbolic) or rung 7 (hardware). A subtly-wrong WGMMA
     descriptor **assembles fine and computes garbage** — so for these ops the
     Evaluator marks `semantics_unproven = true` and never lets `assembles`
     masquerade as correct.

3. **Pre-train the perf model on Apple data; validate in a hardware window.**
   Build the analytical NVIDIA/AMD **roofline** today from published H100/MI300
   peak FLOPs+BW (ROFT-style — buildable now, validated later), and pre-train
   the learned cost model on Apple `(schedule, latency)` records. **MTL-TLP
   shows ~7% target-hardware data closes the transfer**, so a few hours of
   rented H100/MI300 later fine-tunes a model already most of the way there.

**Minimal-hardware-touch posture (the actual NVIDIA/ROCm unlock).** You do not
need to *own* a GPU — you need a thin, well-defined hardware-touch step, and the
**G7 launch bridge** (`tsrRegisterGpuLauncher` / `tsrLaunchKernel`, landed
2026-06-10) is the seam (it is the NVIDIA audit's own next-work item #2). Because
rungs 1–5 are green *before* any silicon, a nightly/on-demand GPU runner (GH
Actions GPU, Lambda, RunPod, Modal; MI300 cloud increasingly cheap) runs the
**already-assembly-validated batch** and promotes rungs 6–7 across the whole set
in hours. The hardware session is a batch promotion, not a bring-up.

---

## 6. Three pillars and honest sequencing

| Pillar | What | Real when | Primary research → integration point |
|---|---|---|---|
| **1. The Evaluator** (correctness authority) | generative legal-program gen + multi-oracle verdict + provenance gate + per-backend rung; **derives** conformance cells instead of declaring them | **today, no hardware** | PolyJuice horizontal oracle over `CanonicalizeTesseraIR.cpp` + fusion passes; DESIL UB-elimination + checksum; provenance gate on `execution_mode`; NNSmith gradient inputs — extend `tests/unit/_diff_lane.py` + `tessera.compiler.conformance_matrix` |
| **2. The Flywheel** (perf signal) | constrained autotune + analytical baseline + `(schedule,latency)` corpus + offline microbench | **today, Apple GPU only**; NVIDIA/ROCm perf pre-trained, HW-validated later | BaCO Chain-of-Trees → upgrade `autotune_v2.py`; ROFT roofline pre-filter; Triton-anatomy offline microbench + decision-tree distillation; MTL-TLP transfer; the record schema (§7) |
| **3. The Environment** (search) | wrap 1+2 as `evaluate()→reward`; *then* layer evolutionary/agentic search | **aspirational, gated on 1+2 being trustworthy** | CompilerGym `(obs,action,reward)` interface; Magellan template-proposes / autotuner-fills split; Sakana + TritonRL + TensorBench adversarial audit baked in from day one |

**Honesty matrix — what each backend's self-improvement can actually claim:**

| | correctness/coverage | perf |
|---|---|---|
| **Apple GPU / x86** | real today (rungs 1–7) | real today (measured) |
| **NVIDIA / ROCm** | real today to rung 4 (assembles + codegen-stable + algorithm-transfer); rung 5 symbolic aspirational; rung 7 in HW window | **pre-trainable today, validated only in a HW window** — never claim a perf flywheel before an executed signal exists |

The schema **enforces** this asymmetry: an `artifact_only` backend can earn a
correctness reward, never a perf reward — which structurally prevents the Sakana
failure (search cannot farm a fake perf win on a backend it cannot run).

---

## 7. The autotuning record schema (Pillar 2 data contract)

Deterministic `(candidate, outcome)` record feeding, in order, the analytical
roofline → TLP-style learned scorer → (later) uncertainty/active-learning loop:

```
record {
  schema_version, op_chain, problem_shape{M,N,K,B,H,seqlen,head_dim},
  dtype_policy{storage,accum}, target, device_id,        # NEVER aggregate across device_id
  schedule{ tile_q, tile_kv, pipeline_stages, fusion_choice, threadgroup_mem_bytes, ... },
  legal: bool + violation_reason,                          # feeds BaCO constrained search
  latency_ms{median,p10,p90}, achieved_tflops, achieved_bw_gb_s, energy_j?,
  numeric_check,                                           # rel-err / finite-field equiv
  roofline_predicted_ms, model_predicted_ms, model_uncertainty,   # residual audit on same row
  toolchain_version, timestamp, search_method, trial_id,
}
```

Disciplines: never aggregate across `device_id` (schedules don't transfer across
chips — classic cost-model poisoning); store `legal`+reason even for rejected
candidates (the constraint-learning signal); keep predicted vs measured on the
same row so every record is both a training sample and a model-residual audit.

---

## 8. Anti-degeneracy / threat model (baked in from day one)

The Sakana retraction is the receipt: *the agent exploits any gap in the
scorer.* Defenses, adopted before any search:

- **Hidden inputs the candidate never sees** (TensorBench names "no independent
  hidden tests" as its own weakness) — the generator owns a held-out input set.
- **Anti-fallback gate** (TritonRL): a "passing" candidate whose `execution_kind`
  silently downgraded to `reference`/`artifact_only` is a hard FAIL, decoupled
  from numerical agreement.
- **Adversarial audit** (TensorBench): reject vacuous/weakened/API-mismatched
  fixtures; a promoted cell must reference a real execute-compare.
- **Provenance ≠ correctness**: numerical agreement and "ran on the demanded
  backend" are *separate* gates; passing one never implies the other.

---

## 9. Research map (verified 2024–2026; see survey notes for skeptic flags)

| Work | Venue/yr | Steal | Rung/Pillar |
|---|---|---|---|
| **PolyJuice** | OOPSLA'24 | horizontal equivalent-program differential oracle over own rewrites | oracle / P1 |
| **DESIL** | OOPSLA'25 | checksum-across-opt-levels + UB-elimination; structural-SASS variant for NVIDIA | oracle / rung 4 |
| **Mirage** | OSDI'25 | finite-field polynomial-identity equivalence verifier | rung 5 / P1 |
| **TensorRight** | POPL'25 | formal translation validation of tensor rewrites (long-pole) | rung 2/5 |
| **GPU-kernel SMT equivalence** | preprint'25 (2511.12638) | bounded-FP symbolic equivalence, CUDA+ROCm, hardware-free | rung 5 |
| **NNSmith / MLIRSmith / MLIRod** | ASPLOS'23 / ASE'23 / ISSTA'24 | gradient-guided inputs; grammar-driven valid MLIR; op-dependency coverage | generator / P1 |
| **TensorBench** | 2026 | suite-pass + adversarial-audit grading on a Tessera-like framework | P3 / §8 |
| **TritonRL** | preprint'25 | anti-fallback + multi-input + compile + signature verifier checklist | §8 |
| **KernelBench** | 2025 | `fast_p` (correct AND faster) composite metric | reward / P3 |
| **LongCA-bench** | ICLR'26 | parameterize attention by (mask × seq-len × MHA/GQA × fwd/bwd), not op-name | task taxonomy |
| **AlphaEvolve / Magellan** | 2025 / CGO'26 | fitness gated behind correctness; template-proposes / autotuner-fills split | P3 |
| **Sakana (post-fix)** | 2025 | the reward-hacking threat model (honest-first wins) | §8 |
| **Compiler-R1 / CompilerGym** | 2025 / 2021 | reward = relative-to-baseline gated behind compile+run validity; Gym env interface | P3 |
| **BaCO** | ASPLOS'24 | Chain-of-Trees constrained Bayesian search for illegal-config-heavy spaces | P2 / autotune_v2 |
| **TLP / MTL-TLP** | ASPLOS'23 | schedule-primitives-as-language learned cost model; ~7% target-data transfer | P2 |
| **ROFT / Triton-anatomy** | 2025 | roofline analytical pre-filter; offline microbench + decision-tree distillation | P2 |
| **GPU Portability needs Autotuning** | 2025 | "portability = autotuning one parametric kernel, not per-vendor hand-tuning" | P2 thesis |

---

## 9.5 Status — what has landed (updated 2026-06-12)

The plan below was authored as a forward roadmap; most of it has now shipped.
Notably, the **scored environment + search layers (E5) landed earlier than their
original last-place sequencing** — the oracles matured fast enough that the grader
and Magellan/AlphaEvolve lanes became buildable. Generated dashboards remain the
count authority; this is a structural map, not a count.

| Area | Status | Shipped surface |
|---|---|---|
| **Evaluator engine** (E1) | ✅ landed | `evaluator.py` — `Rung` ladder (8 rungs), `verdict_for`/`evaluate` (execution-derived, provenance-gated), `run_native` |
| **Oracles** | ✅ landed | vertical (`evaluate`), horizontal/PolyJuice (`horizontal_equivalence`), metamorphic (`metamorphic_equivalence`), **DESIL cross-path** (`cross_path_equivalence`) |
| **Legal-by-construction inputs** (E2) | ✅ landed | `safe_input` (DESIL UB-elim / NNSmith) |
| **Conformance corroboration** (E1c) | ✅ landed | `conformance_evaluator.py` — all complete cells re-derived at rung 7 ("derive validates declare") |
| **NVIDIA emission** | 🟡 rung 2.5 | `ptx_emit.py` + `matmul_pipeline.emit_nvidia_ptx` + `evaluator.nvidia_emission_verdict` (WGMMA PTX text, wired into `@jit`). Rung 3 (`ptxas`)/4+/hardware **open** |
| **Flywheel** (E3) | ✅ landed | `flywheel.py` (records, roofline, sweep, per-chip calibration, persist/distill) + `flywheel_autotune.py` (autotune_v2 bridge, `autotune_matmul`) |
| **Scored environment / grader** (E5) | ✅ landed | `compiler_grader.py` (TensorBench-style, hidden inputs, anti-cheat) + `attention_tasks.py` (LongCA mask×seqlen) |
| **Search layers** (E5) | ✅ landed | `magellan.py` (gated heuristic evolution) + `alphaevolve.py` (evaluator-driven search; reward-hack rejection proven) |
| **Opt-level checksum** (E2) | ✅ landed | `evaluator.opt_level_checksum` + `opt_level_equivalence` (DESIL checksum-across-opt-levels; tolerance-rounded so benign float reordering doesn't false-alarm; `inconclusive` unless ≥2 variants run natively). MLIRod grammar-generator coverage remains a larger generator-hardening follow-on. |
| **AMD emission rung** (E4, hardware-free) | ✅ landed | `evaluator.rocm_emission_verdict` — `llc -mcpu=gfx1151` assembles real `v_wmma` AMDGCN on-host → rung 4 (ASSEMBLES); parallel to `nvidia_emission_verdict` (rung 3, PTX text). |
| **NVIDIA/ROCm hardware window** (E4) | ⬜ open | Linux/CUDA + Strix Halo / Blackwell silicon: `ptxas` rung 4, rungs 6–7 |

Guards: `tests/unit/test_{evaluator,evaluator_e2,conformance_evaluator,ptx_emit,
flywheel,flywheel_autotune,compiler_grader,attention_tasks,magellan,alphaevolve}.py`
(~80 tests; Darwin-gated where execution is required, portable contracts elsewhere).

---

## 10. Phasing — original plan (annotated with landed status)

> ✅ landed · 🟡 partial · ⬜ open. The scored environment (E5) shipped ahead of
> sequence; the NVIDIA hardware window (E4) remains the gate.

**Phase E1 — Evaluator spine + first two rungs (the concrete first slice). ✅**
One engine, one generated program set, a per-backend `Verdict`:
- *Apple:* horizontal-equivalence oracle (run `p` through its fusion rewrites,
  assert pre ≡ post on Apple GPU) + provenance gate (silent numpy fallback =
  hard FAIL) → reaches rung 7; *emits* conformance rows rather than sitting
  beside the matrix.
- *NVIDIA/ROCm:* the `ptxas`/`hipcc` assembly rung (rung 3) over Tessera's real
  emitted kernels, replacing the 8 hand stubs; WGMMA/MFMA tagged
  `semantics_unproven`. Skip-clean locally; runs in Linux CI.
Deliverables: extend `_diff_lane.py`; new `horizontal_oracle` + `provenance_gate`
+ `assembly_rung`; wire into `conformance_matrix`; a `test_evaluator_*` gate.
Acceptance: catches a deliberately-injected fusion divergence on Apple; fails on
a deliberately-malformed PTX in CI; conformance rows are *derived*, drift-gated.

**Phase E2 — Generator hardening + opt-level oracle. 🟡** (safe inputs + the cross-path/metamorphic oracles landed; opt-level checksum + MLIRod coverage open.) DESIL UB-elimination so
generated programs are legal-by-construction; NNSmith gradient inputs; opt-level
checksum (Apple/x86 runtime) + structural-SASS diff (NVIDIA/ROCm). MLIRod
op-dependency coverage to steer generation toward novel adjacencies.

**Phase E3 — Flywheel (Apple-measurable). ✅** BaCO constrained search into
`autotune_v2`; ROFT roofline baseline; offline microbench harness; the §7 record
corpus; decision-tree distillation for O(1) dispatch.

**Phase E4 — NVIDIA/ROCm hardware window. ⬜** (the remaining gate — needs Linux/CUDA + silicon.) Register a CUDA/HIP launcher into G7;
run the assembly-validated batch on a rented/CI GPU; promote rungs 6–7; fine-tune
the pre-trained cost model with the ~7% target data.

**Phase E5 — Scored environment. ✅** (landed ahead of sequence — grader + Magellan + AlphaEvolve.) Wrap E1–E3 as `evaluate()→reward` behind a
CompilerGym-style interface; only then layer Magellan/AlphaEvolve-style search,
anti-cheat (§8) baked in.

---

## 11. Non-goals, risks, honesty constraints

- **`ptxas` clean ≠ hardware-verified.** Rung 3 proves *assembleability*, not
  computation. Never let a green assembly rung read as correctness, especially
  for WGMMA/TMA/tcgen05/MFMA (`semantics_unproven`).
- **Algorithm transfer ≠ codegen transfer.** §5 lever 2 transfers Tile-IR
  semantics, not microkernel faithfulness; the last mile is rung 5 or 7.
- **No perf flywheel claim for NVIDIA/ROCm before an executed signal.** Apple
  perf is real; NVIDIA/ROCm perf is pre-trained + HW-window-validated. Say which.
- **Symbolic (rung 5) is heavy.** Treat finite-field/SMT equivalence as a
  research track, not a near-term gate; the credible near-term closer is the
  hardware batch (rung 7).
- **Generated dashboards stay the count authority.** The Evaluator *emits into*
  conformance / benchmark / record surfaces; it never becomes a second
  hand-maintained source of truth (Decision #26).

---

## 12. Where this plugs into the existing spine

- `CompileResult` already carries `component_ops` / `fusion_groups` /
  `layout_contracts` / `program_executable` → the Verdict's spine fields.
- `tessera.compiler.conformance_matrix` (fixture-proof gate) → becomes a
  *consumer* of Evaluator verdicts, not a hand-declared registry.
- `benchmarks/common/artifact_schema.py::BenchmarkRow` (already has
  `compiler_path` / `execution_kind` / `Correctness` / `ArtifactLevels`) →
  add `hardware_verified`, `component_ops`, `fallback_reason`, `rung`.
- `tests/unit/_diff_lane.py` + the differential generators → the Evaluator's
  generation + oracle core.
- `scripts/validate_nvcc_compile.py` / `validate_hipcc_compile.py` → grow into
  the rung-3 assembly lane over real emitted kernels.
- `autotune_v2.py` → the Flywheel's search engine.
- G7 `tsrRegisterGpuLauncher` → the rung-6/7 hardware seam.
