# Tessera Audit Master

**Last updated:** 2026-06-22

> **Reconciled 2026-06-22:** the multi-op compiler-metadata P1 is **closed** —
> component-aware metadata (`component_ops`, `program_executable`,
> `component_blockers`, `effects`, `shape_envelope`, `layout_contracts`,
> `fusion_groups`, `outputs`) is derived per-component and carried to
> `fn.runtime_artifact().metadata` (verified by direct inspection + 57 locking
> tests), and fusion dispatch is authoritative (Phase 0 seam closed). The
> follow-on Phase 1 has begun landing too (Graph-IR folders/canonicalizers on 7
> ops; `LayoutAssignmentPass`). The `batching/sharding` long-tail counts below
> were refreshed against
> [`generated/s_series_status.md`](generated/s_series_status.md).

This is the root audit document. It consolidates the current state, finished
work, and remaining work across the compiler, runtime/backend, platform
backends, coverage, and domain tracks. Generated dashboards remain the source of
truth for counts; theme audit documents carry the reasoning and work plan.

## Current Truth Snapshot

| Area | Current state | Still open |
|---|---|---|
| Compiler and IR | Canonical compile, IR bundle, named gates, and conformance matrix exist; a single generated-doc registry (`tessera.compiler.generated_docs`) now drives both the CI gate and one `--write` sprint regen, 9 dashboards are CSV-canonical, and the surface (6→1) + test-coverage (2→1) dashboards were consolidated. | Multi-op metadata, fusion groups, and layout contracts are now carried through the compile artifact and authoritative for dispatch (2026-06-22). Remaining: wiring `LayoutAssignmentPass` into the named pipelines + broader Graph-IR folder coverage (per-op effect interfaces landed 2026-06-22), fixture-driven numerical proof for complete cells, and optional dashboard consolidation (target maps, e2e/s_series rollups). |
| Runtime/backend | Runtime execution matrix and C ABI dashboards are generated and drift-gated; the distributed MegaMoE stack (expert-parallel 2× all-to-all, FP8×FP4, async comm/compute overlap) runs with the expert FFN on Apple GPU. | NVIDIA and ROCm have no executable runtime rows yet; MegaMoE multi-rank is mock-collective until a real NCCL/RCCL (or Apple multi-GPU) lane exists. |
| Apple backend | Apple CPU/GPU are runtime-backed; Metal 4, MPSGraph, encode-session, and packaged-kernel lifecycle work exist. | Apple binding specs, feature-limit-guided lowering, production packaged kernels, and canonical one-command-buffer JIT path remain. |
| NVIDIA | CUDA/NVIDIA plans and target maps exist; artifacts/toolchain path is represented. | Real hardware execute-and-compare and runtime launch bridge remain. |
| ROCm | ROCm/gfx target map and execute-and-compare plan exist. | Real HIP/ROCm hardware proof and runtime launch bridge remain. |
| Coverage | Partial-op uplift is closed; direct-test debt is not ordinary missing tests. | Backend-kernel axis is still open across all S-series primitives; batching/transpose/sharding have smaller long-tail gaps. |
| Domain tracks | GA/EBM, attention, CorrDiff/SciML, sharding, and autodiff plans have been reduced to clearer scope locks and implementation history. | Domain claims must stay tied to generated coverage and backend proof, not old roadmap prose. |

Generated dashboards are the **count authority** — this page links them and
never copies their numbers (a copied count silently goes stale; per Decision
#25/#26 the generated docs under `generated/` are the source of truth, drift-
gated by `scripts/check_generated_docs.sh`). For live figures, read:

- [`generated/runtime_abi.md`](generated/runtime_abi.md) — C ABI symbol totals + Apple symbol/family counts.
- [`generated/runtime_execution_matrix.md`](generated/runtime_execution_matrix.md) — executable rows per target (Apple CPU/GPU, native CPU, JIT CPU numpy).
- [`op_target_conformance.md`](op_target_conformance.md) — complete/partial/missing op×target cells.
- [`generated/e2e_op_coverage.md`](generated/e2e_op_coverage.md) — native-complete / runnable-reference split (no partial/planned tail).
- [`generated/s_series_status.md`](generated/s_series_status.md) — per-axis open/complete (lowering closed; backend-kernel universally open).
- [`generated/test_coverage.md`](generated/test_coverage.md) — direct-test-debt classification (actionable / hardware-gated).

## Finished Work

### Compiler And IR

Finished:

- Canonical compiler driver and `CompileResult` are in place.
- `@jit` and `runtime.launch()` carry canonical compile metadata.
- Pipeline gates name first failing axes instead of returning vague unsupported status.
- Op-target conformance matrix is generated and drift-gated.
- Schedule-to-Tile metadata preservation landed.
- C++ `LowerScheduleToTargetPass` now fails honestly instead of silently succeeding as a no-op.
- Tile-to-Apple C++ status aligns with the Python/runtime Apple envelope.
- Dynamic control-flow lowering has explicit diagnostics and fallback behavior.
- Frontend lowering bugs found by audit are fixed, including AugAssign sub/div and ROCm/platform gate issues.
- Compiler correctness tests include pass-order matrices and oracle lanes for several high-risk paths.
- `runtime_abi` and `verifier_coverage` dashboards are CSV-canonical (machine-readable, byte-diffable) with a non-byte-gated Markdown companion, wired into `check_generated_docs.sh --write` for one-command sprint regeneration.

Still needs work:

- ✅ Make compile metadata component-aware for real multi-op programs (2026-06-07; derived per-component and carried to `fn.runtime_artifact().metadata`, locked by 57 tests — see COMPILER_AUDIT Next Work #1).
- Carry fusion groups, layout contracts, shape envelopes, effects, and backend strategy through the compiler artifact.
- Stop rediscovering fusion/program identity separately in Target IR and runtime dispatch. *(Runtime half closed 2026-06-10 — the apple_gpu executor consumes `fusion_groups` known_chain metadata; Target IR C++ fusion passes still re-match. See [compiler/CODE_AUDIT_2026_06_10.md](compiler/CODE_AUDIT_2026_06_10.md).)*
- Tie complete compiler claims to direct compare fixtures or hardware/package validation.
- Generated-doc registry landed (`tessera.compiler.generated_docs`): one source of truth consumed by both `check_generated_docs.sh` and `release_gate.py`, a fleet-wide `--write`/`--check`, an orphan-guard test, and 9 CSV-canonical dashboards. Remaining: optional further consolidation (target maps 3→1, fold e2e/s_series rollups into their primaries).

Primary detail: [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md).

### Runtime And Backend

Finished:

- Runtime execution matrix is the launch source of truth.
- Runtime ABI surface is generated and drift-gated.
- CPU native, CPU JIT numpy, Apple CPU, and Apple GPU executable rows are explicit.
- Non-Apple hardware targets are recognized but honestly non-executable in `runtime.launch()`.
- **C-ABI GPU launch bridge (G7, 2026-06-10):** `tsrLaunchKernel` gained a pluggable launcher hook (`tsrRegisterGpuLauncher` + `tsrGpuLaunchParams`) — the core runtime stays backend-agnostic and a backend registers a name→symbol launcher. Proven end-to-end on Metal (a C-ABI GEMM launch runs through the Apple runtime and equals `A@B`; unregistered kernels still report `UNIMPLEMENTED`). NVIDIA/ROCm plug into the same hook once hardware exists. See [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md).
- Toolchain pins for CUDA, NCCL, and ROCm agree in generated ABI/toolchain dashboards.
- Distributed **MegaMoE** stack landed: single-device MoE layer, fused expert-FFN kernel, expert-parallel 2× all-to-all forward, FP8×FP4 mixed precision, and a real async comm/compute overlap engine (GPU command buffer ∥ CPU comm) with demonstrated wall-clock overlap on Apple. Multi-rank runs over in-process mock collectives (Decision #6); see [`../distributed_megamoe.md`](../distributed_megamoe.md).

Still needs work:

- Add runtime execution rows for NVIDIA/ROCm only after actual launch paths exist.
- Keep artifact-only, compileable, executable, numerical, and hardware-verified statuses separate.
- Use `execute_compare_fixture` consistently for promoted backend claims.

Primary detail: [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md).

### Apple Backend

Finished:

- Apple CPU executes through Accelerate.
- Apple GPU executes through the MPS / MPSGraph / custom MSL runtime envelope.
- Metal 4 lanes and runtime probes exist.
- Encode-session and one-command-buffer chain substrate exist.
- Apple chain planner and auto-batch substrate exist.
- Conv2d encode-session lanes exist across f32/f16/bf16 wrapper surfaces.
- Packaged-kernel lifecycle PK1-PK7 is proven with a real Apple fixture.
- Apple GA/EBM specialized runtime kernels and benchmarks exist.

Still needs work:

- Nothing open on the Apple compiler track. (Real-hardware NVIDIA/ROCm/
  Metalium execution proof remains the cross-backend gate — see those tracks.)

Closed 2026-06-02 → 2026-06-09: binding specs/descriptors for all kernel
families; descriptor-driven dispatch (single-source envelope in
`apple_gpu_envelope.py`, runtime lane-table dispatch, generated C++
`kRuntimeOps`); feature-limit-driven selection (tiled softmax N-cap, bf16
gate, fused-chain/head_dim caps, threads-per-row); canonical one-command-
buffer decode; production packaged-kernel rows; manifest-attached benchmarks
+ perf ratchet (`perf_gate --ratchet` + recorded `apple_gpu_hot_paths.json`);
auto_batch auto-detection + Graph-IR-emission skip.

Primary detail: [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md).

### NVIDIA And ROCm

Finished:

- Target maps exist for NVIDIA SM90 and ROCm.
- CUDA/ROCm toolchain plans and execute-and-compare backlog are documented.
- The repo distinguishes artifact generation from hardware execution.

Still needs work:

- Run real execute-and-compare on NVIDIA and ROCm hardware.
- Add runtime launch bridges and execution-matrix rows only after real execution works.
- Promote backend manifest rows with toolchain, runtime ABI, smoke, and numerical proof.

Primary detail:

- [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md)
- [backend/rocm/ROCM_AUDIT.md](backend/rocm/ROCM_AUDIT.md)

### Coverage And Primitive Contracts

Finished:

- Partial-op uplift closed the legacy partial bucket; the E2E dashboard now
  shows no partial/planned rows ([`generated/e2e_op_coverage.md`](generated/e2e_op_coverage.md)).
- `lowering_rule` is closed project-wide (0 open) ([`generated/s_series_status.md`](generated/s_series_status.md)).
- No actionable direct-test-debt (`needs_direct_test = 0`) ([`generated/test_coverage.md`](generated/test_coverage.md)).
- KV-cache has named diagnostics and explicit target coverage history.
- Advanced examples mostly shifted from missing Python APIs to backend/hardware proof.

Still needs work:

- Close backend-kernel proof across hardware targets.
- Close remaining batching, transpose, and sharding long-tail counts.
- Keep generated dashboards as count truth and avoid copying stale numerical snapshots into prose.

Primary detail: [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md).

### Domain Tracks

Finished:

- GA and EBM scope locks are established.
- GA/EBM Python surfaces and Apple-specialized runtime paths are substantially built.
- Attention/MLA/KV-cache planning has shifted from API invention to backend proof.
- CorrDiff analysis clarified what belongs in compiler primitives vs library/runtime code.
- Sharding audit classified long-tail buckets.
- **DFlash block-diffusion speculative-decoding draft landed (2026-06-12, PR #67).**
  P0 added an additive `attn_bias` operand to `FlashAttnOp` end-to-end — Graph IR
  ODS + verifier, Tile→Apple lowering, MPSGraph `flash_attn_bias_{f32,f16,bf16}`
  runtime symbols (+ stub), eager/CPU/GPU dispatch, VJP (`dbias = dS`), `op_catalog`
  arity, and the `runtime_abi` audit — validated on Metal at 3.3e-7. P1 built the
  DFlash draft on that substrate: `nn.functional.block_diffusion_attention`
  (QK-norm, KV injection, GQA, sliding-window-via-bias), `tessera.dflash` (draft
  model, multi-layer `HiddenStateTap`, `dflash_step`/`dflash_generate`), the
  `apple_gpu_attention_fn` seam, and a `@jit(target="apple_gpu")` flash_attn(bias)
  proof reporting `metal_runtime`. The gold-standard invariant — greedy spec-decode
  output == greedy autoregressive decode — is proven against an independent numpy
  port of the `z-lab/dflash` MLX reference.
  **Integration items 1–9 landed (2026-06-12):** (1) per-layer draft KV cache
  (`DraftKVCache`, cached==non-cached); (2) non-greedy sampling + distribution-
  preserving rejection acceptance (`make_sampler`/`dflash_speculative_verify`,
  marginal==target by Monte Carlo); (3) stateful target KV cache + rollback and
  (4) a real reference target (`dflash_reference.ReferenceDecoderLM`; stateful==
  stateless; full cached+stateful loop == greedy AR); (5) whole-draft attention on
  Metal via the `attention_fn` seam; (6) `DFlashDraft(nn.Module)`; (7) safetensors
  checkpoint I/O + HF state-dict mapping (`dflash_io`); (8) GQA via exact repeat
  (the native kernel doesn't fit the concat-KV+bias layout); (9) position-weighted
  training loss, `RotatingDraftKVCache`, tokenizer-wired `dflash_generate_text`, and
  `DFlashScheduler` (`dflash_serve`). Remaining gates are external: numerical parity
  vs a downloaded `z-lab/*-DFlash` checkpoint (network), and a single fully-jitted
  GPU draft artifact (GPU gather/embedding). Detail: [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md).

Still needs work:

- Keep domain claims tied to coverage/backend proof.
- Avoid treating old domain roadmaps as current status.
- Close domain-specific backend proof through the same generated dashboards and runtime gates.

Primary detail: [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md).

## Priority Work List

### P0

- ✅ **Fixture-driven proof claims for complete conformance cells (2026-06-07).**
  A conformance cell can now only reach `numerical_check = complete` when a
  **manifest-declared `execute_compare_fixture`** exists on disk (the
  legacy filename/keyword heuristic is capped at `partial`). The generator
  enforces it (`conformance_matrix._numerical_proof_source` →
  `"fixture"`/`"heuristic"`/`None`); the gate is locked by
  [`test_conformance_complete_cells_proven.py`](../../tests/unit/test_conformance_complete_cells_proven.py)
  (every complete cell is fixture-proven, the fixture exists and genuinely
  `assert_allclose`-compares a component op, and heuristic-only cells never
  reach complete). Tightening demoted 35 keyword-only `numerical_check` sub-cells
  to `partial` (e.g. `softmax/nvidia`, which has no execution path) with **zero
  `overall` flips**; the 5 published complete cells (Apple matmul/softmax/
  matmul_softmax/flash_attn) are each backed by a real execute-compare and all
  15 declared fixtures pass `conformance_matrix --verify-fixtures`. Two
  mis-declared fixtures (`matmul`/`rope` on apple_gpu pointed at the buffer-pool
  RAII test) were corrected to genuine numerical compares.
- Backend-kernel hardware proof on real NVIDIA/ROCm hardware.
- Runtime execution rows only for genuinely executable backends.

### P1

- ✅ Multi-op compiler metadata and component-aware gates (landed 2026-06-07;
  `component_ops` / `program_executable` / `component_blockers` +
  `effects` / `shape_envelope` / `layout_contracts` / `fusion_groups` /
  `outputs` carried to the `@jit` artifact, fusion dispatch authoritative).
  Forward work moved to COMPILER_AUDIT **Phase 1**: ✅ per-op effect interfaces
  on the 23 non-pure Graph-IR ops (landed 2026-06-22 — `[Pure]` vs
  `MemWrite`/`MemRead`, so generic CSE/DCE is sound); still open — wiring
  `LayoutAssignmentPass` into the named x86/GPU pipelines and broadening folder
  coverage.
- ✅ Apple binding/kernel descriptor unification (2026-06-09 — descriptor-driven dispatch + generated C++ runtime-ops table).
- ✅ Apple feature-limit-guided lowering (2026-06-09 — bf16 gate, fused-chain caps, threads-per-row).
- ✅ Canonical Apple one-command-buffer decode through `tessera.ops` / `@jit` (2026-06-02).
- ✅ Production packaged-kernel rows with reflection, dispatch, and numerical proof (2026-06-02; 7 rows).

### P2

- **Batching/transpose/sharding long-tail — assessed closed for everything
  provable (2026-06-17).** `transpose_rule` and `lowering_rule` are fully closed
  (0 partial); `batching_rule` and `sharding_rule` carry the only residual
  partials — live counts are dashboard-owned in
  [`generated/s_series_status.md`](generated/s_series_status.md) (6 + 47 = 53 as
  of 2026-06-22, up from 4 + 39 = 43 on 2026-06-17 as the EDM/DiffusionBlocks
  primitives in `427f595`/`25111fe` added mesh-gated rows). **All residual
  partials sit in genuinely distributed-mesh-gated categories** — `attention`
  (the reasoning-model fused family: sparse/delta/gated/lightning variants, where
  head-split equivalence isn't trivially true — the *standard* family was already
  proven complete in `test_attention_sharding_mock_mesh.py`), `spectral`
  (distributed FFT = all-to-all transpose), `linalg_decomposition`/`linalg_solver`
  (distributed cholesky/qr/svd), `moe`/`moe_transport` (all-to-all dispatch/
  combine), `ebm`, `state_space`/`state_update`, `sparse`, `loop_nest`. These are
  **Phase-G-gated by design, not bookkeeping debt** — the `_SHARDING_RULE_BY_CATEGORY`
  classifier marks each partial with a documented "known but mesh-aware" reason.
  Flipping them without real mock-mesh proofs would be the audit-inflation
  Decision #25 forbids; closing any further requires a genuine per-variant proof
  (the established `test_*_sharding_mock_mesh.py` pattern) or real Phase-G mesh
  hardware. So the closable closure is **done**; the rest is correctly gated.
- Domain roadmap hygiene and stale-claim cleanup.
- ✅ Benchmark/performance gates tied to backend manifest rows — done for Apple GPU (2026-06-09: `benchmark_json` on hot-path + packaged rows; `perf_gate --ratchet`); other backends follow with Phase G/H/I hardware.
- Unify generated-doc regeneration + drift into one registry/`--write` contract (fold in `release_gate.py`, standardize generator CLIs, extend CSV-canonical to data-shaped dashboards). Detail: [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) Next Work #6.

## Compiler-Completeness & Testing Program

Started 2026-06-07. Reframe: the compiler is **not stub-riddled** — the
software-actionable gap surface is small and specific; most incompleteness is
hardware-gated (expected) or thin-test (better closed by generative differential
testing than hand-written tests). Run
[`scripts/stub_surface_report.py`](../../scripts/stub_surface_report.py) for the
live ranked rollup → [`stub_surface.md`](stub_surface.md).
Stubs are an **oracle/conformance problem, not a fuzz problem** (a stub that
returns a plausible artifact never crashes); fuzzing is layered on top of the
differential oracle to catch the long tail.

The actionable surface, ranked (numbers live in `stub_surface.md`):
- **Verifier stubs — closed (2026-06-10): `trivial_stub` is now 0.** Three
  verifier sprints (V8 norm/softmax, V9 control-flow + stubs + MoR/quant/FFT)
  took `verifier_coverage` from 73 → **100+ `real`**, and the **final trivial
  stub `ArchSTEOneHotOp` was removed 2026-06-10**: `arch.ste_one_hot` maps an
  opaque parameterless `ArchParam → ArchGate`, so the ODS type constraints
  fully constrain it and a `verify(){return success();}` was a false claim —
  dropping `hasVerifier=1` reclassifies it honestly to `no_verifier`
  (`trivial_stub` 1 → **0**). The `no_verifier` tail is mostly pure elementwise
  (legitimately need none); a few structural collective/reshape ops could still
  get one.
- **Software conformance gaps — CPU numerical cells closed (2026-06-10).**
  `conv2d → cpu` and `kv_cache_read → cpu` ("executes but unverified") now carry
  manifest-declared `execute_compare_fixture`s (the existing
  `test_jit_cpu_executes_conv2d_nhwc_reference` and `test_kv_cache_handle`
  read-compares), flipping `numerical_check` `partial → complete`. A
  pipeline-gate fix made `_eval_numerical` honor a declared fixture, so the
  worklist's software-actionable count dropped **6 → 4**. The remaining 4
  (`conv2d`/`kv_cache_read` → nvidia/rocm, stop @ `codegen`) emit no code AND
  have no silicon — effectively hardware-gated.
- **Thin-test tail — differential generator extended (2026-06-10).**
  `_diff_lane.numeric_cases` adds 12 elementwise/reduction ops
  (`exp`/`log`/`sqrt`/`rsqrt`/`abs`/`softplus`/`maximum`/`minimum`/`sum`/`mean`/
  `amax`/`amin`) run on `@jit(apple_gpu)` vs an **independent numpy oracle** (a
  true impl-vs-reference check across the dispatch envelope, beyond the 13-op
  straight-line tracer lane), wired into both the stdlib and hypothesis
  harnesses.

Workstream (chosen 2026-06-07): **(#1) quantify** — `stub_surface_report.py`
(✅ done); **(#4) IR round-trip property + fuzz** (✅ done —
[`test_ir_roundtrip_fuzz.py`](../../tests/unit/test_ir_roundtrip_fuzz.py):
generate→render→parse→compare op-names + malformed-input crash-safety; found &
fixed a real parser EOF crash where `parse_module` asserted instead of raising a
named `FrontendSyntaxError`); **(#2) differential generator** (✅ done —
[`test_differential_generator.py`](../../tests/unit/test_differential_generator.py):
synthesizes random programs over the executable lane (`tessera.ops` +
`tessera.control.fori_loop`/`cond`) and diffs the **eager numpy oracle** against
the real **trace → GraphFn / `execute_traced` Metal path** — a miscompile
detector for straight-line, fused `run_graph_loop`, and fused `run_graph_cond`;
51 cases green on Apple GPU, runtime-free trace/op-name properties run
everywhere. A **hypothesis-backed sibling**
[`test_differential_generator_hypothesis.py`](../../tests/unit/test_differential_generator_hypothesis.py)
drives the same shared grammar/oracle [`_diff_lane.py`](../../tests/unit/_diff_lane.py)
via `@given` for **automatic shrinking** — a Metal miscompile reduces to the
minimal failing program — guarded by `importorskip("hypothesis")` so CI without
it still passes on the stdlib harness). (The "claimed-complete must be proven"
gate is P0 above —
*Fixture-driven proof claims for complete conformance cells*.)

## Where To Go Next

| Need | Read |
|---|---|
| Current all-up status | This document |
| Folder map | [README.md](README.md) |
| Compiler/IR open work | [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) |
| Shared backend proof | [backend/BACKEND_AUDIT.md](backend/BACKEND_AUDIT.md) |
| Apple backend performance/runtime | [backend/apple/APPLE_AUDIT.md](backend/apple/APPLE_AUDIT.md) |
| NVIDIA | [backend/nvidia/NVIDIA_AUDIT.md](backend/nvidia/NVIDIA_AUDIT.md) |
| ROCm | [backend/rocm/ROCM_AUDIT.md](backend/rocm/ROCM_AUDIT.md) |
| Primitive/op coverage | [coverage/COVERAGE_AUDIT.md](coverage/COVERAGE_AUDIT.md) |
| GA/EBM/attention/CorrDiff/sharding | [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md) |
| Planning history | [roadmap/ROADMAP_AUDIT.md](roadmap/ROADMAP_AUDIT.md) |
