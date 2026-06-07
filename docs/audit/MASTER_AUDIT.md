# Tessera Audit Master

**Last updated:** 2026-06-04

This is the root audit document. It consolidates the current state, finished
work, and remaining work across the compiler, runtime/backend, platform
backends, coverage, and domain tracks. Generated dashboards remain the source of
truth for counts; theme audit documents carry the reasoning and work plan.

## Current Truth Snapshot

| Area | Current state | Still open |
|---|---|---|
| Compiler and IR | Canonical compile, IR bundle, named gates, and conformance matrix exist; a single generated-doc registry (`tessera.compiler.generated_docs`) now drives both the CI gate and one `--write` sprint regen, 9 dashboards are CSV-canonical, and the surface (6→1) + test-coverage (2→1) dashboards were consolidated. | Multi-op metadata, fusion groups, layout/effect contracts, and fixture-driven proof need to be first-class; remaining dashboard consolidation (target maps, e2e/s_series rollups) is optional cleanup. |
| Runtime/backend | Runtime execution matrix and C ABI dashboards are generated and drift-gated. | NVIDIA, ROCm, and Metalium have no executable runtime rows yet. |
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

- Make compile metadata component-aware for real multi-op programs.
- Carry fusion groups, layout contracts, shape envelopes, effects, and backend strategy through the compiler artifact.
- Stop rediscovering fusion/program identity separately in Target IR and runtime dispatch.
- Tie complete compiler claims to direct compare fixtures or hardware/package validation.
- Generated-doc registry landed (`tessera.compiler.generated_docs`): one source of truth consumed by both `check_generated_docs.sh` and `release_gate.py`, a fleet-wide `--write`/`--check`, an orphan-guard test, and 9 CSV-canonical dashboards. Remaining: optional further consolidation (target maps 3→1, fold e2e/s_series rollups into their primaries).

Primary detail: [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md).

### Runtime And Backend

Finished:

- Runtime execution matrix is the launch source of truth.
- Runtime ABI surface is generated and drift-gated.
- CPU native, CPU JIT numpy, Apple CPU, and Apple GPU executable rows are explicit.
- Non-Apple hardware targets are recognized but honestly non-executable in `runtime.launch()`.
- Toolchain pins for CUDA, NCCL, and ROCm agree in generated ABI/toolchain dashboards.

Still needs work:

- Add runtime execution rows for NVIDIA/ROCm only after actual launch paths exist.
- Keep artifact-only, compileable, executable, numerical, and hardware-verified statuses separate.
- Use `execute_compare_fixture` consistently for promoted backend claims.
- Decide whether Metalium deserves a dedicated platform folder once there is enough target-specific audit content.

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

- Promote Apple tensor/kernel binding specs beyond packaged kernels.
- Wire Apple feature limits into Schedule/Tile/Target choices.
- Make one-command-buffer decode canonical through `tessera.ops` / `@jit`.
- Populate production packaged-kernel manifest rows.
- Move Apple kernel source/fusion/binding metadata into descriptors rather than broad Target IR/runtime pattern logic.
- Attach stable benchmark/perf gates for Apple backend hot paths.

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

Still needs work:

- Keep domain claims tied to coverage/backend proof.
- Avoid treating old domain roadmaps as current status.
- Close domain-specific backend proof through the same generated dashboards and runtime gates.

Primary detail: [domain/DOMAIN_AUDIT.md](domain/DOMAIN_AUDIT.md).

## Priority Work List

### P0

- Fixture-driven proof claims for complete conformance cells.
- Backend-kernel hardware proof on real NVIDIA/ROCm hardware.
- Runtime execution rows only for genuinely executable backends.

### P1

- Multi-op compiler metadata and component-aware gates.
- Apple binding/kernel descriptor unification.
- Apple feature-limit-guided lowering.
- Canonical Apple one-command-buffer decode through `tessera.ops` / `@jit`.
- Production packaged-kernel rows with reflection, dispatch, and numerical proof.

### P2

- Remaining batching/transpose/sharding long-tail closure.
- Domain roadmap hygiene and stale-claim cleanup.
- Benchmark/performance gates tied to backend manifest rows.
- Unify generated-doc regeneration + drift into one registry/`--write` contract (fold in `release_gate.py`, standardize generator CLIs, extend CSV-canonical to data-shaped dashboards). Detail: [compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) Next Work #6.

## Compiler-Completeness & Testing Program

Started 2026-06-07. Reframe: the compiler is **not stub-riddled** — the
software-actionable gap surface is small and specific; most incompleteness is
hardware-gated (expected) or thin-test (better closed by generative differential
testing than hand-written tests). Run
[`scripts/stub_surface_report.py`](../../scripts/stub_surface_report.py) for the
live ranked rollup → [`generated/stub_surface.md`](generated/stub_surface.md).
Stubs are an **oracle/conformance problem, not a fuzz problem** (a stub that
returns a plausible artifact never crashes); fuzzing is layered on top of the
differential oracle to catch the long tail.

The actionable surface, ranked (numbers live in `generated/stub_surface.md`):
- **Verifier stubs** — the `trivial_stub` verifiers (`Arch*` NAS ops +
  `KVCacheCreateOp` + `RingCreateOp`): `verify()` declared but no-ops. Plus a
  manual-triage pass on the `no_verifier` ops (control/collective/reshape-shaped
  ones should get a real verifier; pure elementwise need none).
- **Software conformance gaps** — the op×target cells that stop at `codegen` /
  `numerical` (vs the hardware-gated `hardware_smoke`/`toolchain`/`link` rows):
  `conv2d`/`kv_cache_read` → cpu (numerical: executes but unverified) and
  conv2d/flash_attn/kv_cache_read → metalium/nvidia/rocm (codegen: lowering emits
  no code).
- **Thin-test tail** — the `needs_direct_test` ops: the target for a generative
  differential harness (tracer vs numpy/CPU oracle).

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

