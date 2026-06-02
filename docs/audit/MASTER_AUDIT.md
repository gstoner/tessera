# Tessera Audit Master

**Last updated:** 2026-06-02

This is the root audit document. It consolidates the current state, finished
work, and remaining work across the compiler, runtime/backend, platform
backends, coverage, and domain tracks. Generated dashboards remain the source of
truth for counts; theme audit documents carry the reasoning and work plan.

## Current Truth Snapshot

| Area | Current state | Still open |
|---|---|---|
| Compiler and IR | Canonical compile, IR bundle, named gates, and conformance matrix exist. | Multi-op metadata, fusion groups, layout/effect contracts, and fixture-driven proof need to be first-class. |
| Runtime/backend | Runtime execution matrix and C ABI dashboards are generated and drift-gated. | NVIDIA, ROCm, and Metalium have no executable runtime rows yet. |
| Apple backend | Apple CPU/GPU are runtime-backed; Metal 4, MPSGraph, encode-session, and packaged-kernel lifecycle work exist. | Apple binding specs, feature-limit-guided lowering, production packaged kernels, and canonical one-command-buffer JIT path remain. |
| NVIDIA | CUDA/NVIDIA plans and target maps exist; artifacts/toolchain path is represented. | Real hardware execute-and-compare and runtime launch bridge remain. |
| ROCm | ROCm/gfx target map and execute-and-compare plan exist. | Real HIP/ROCm hardware proof and runtime launch bridge remain. |
| Coverage | Partial-op uplift is closed; direct-test debt is not ordinary missing tests. | Backend-kernel axis is still open across all S-series primitives; batching/transpose/sharding have smaller long-tail gaps. |
| Domain tracks | GA/EBM, attention, CorrDiff/SciML, sharding, and autodiff plans have been reduced to clearer scope locks and implementation history. | Domain claims must stay tied to generated coverage and backend proof, not old roadmap prose. |

Generated-dashboard facts to anchor this snapshot:

- `generated/runtime_abi.md`: 218 total C ABI symbols, 207 Apple symbols, 84 Apple GPU families.
- `generated/runtime_execution_matrix.md`: executable rows exist for Apple CPU, Apple GPU, native CPU, and JIT CPU numpy.
- `op_target_conformance.md`: 5 complete, 14 partial, 23 missing cells.
- `generated/e2e_op_coverage.md`: 34 native-complete ops, 237 runnable-reference ops, 0 partial, 0 planned.
- `generated/s_series_status.md`: 0 open lowering rules, 432 open backend-kernel axes.
- `generated/test_coverage_classification.md`: 0 actionable direct-test-debt ops, 4 hardware-gated ops.

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

Still needs work:

- Make compile metadata component-aware for real multi-op programs.
- Carry fusion groups, layout contracts, shape envelopes, effects, and backend strategy through the compiler artifact.
- Stop rediscovering fusion/program identity separately in Target IR and runtime dispatch.
- Tie complete compiler claims to direct compare fixtures or hardware/package validation.

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

- Partial-op uplift closed the old 47-partial bucket.
- Current E2E dashboard reports 0 partial and 0 planned rows.
- `lowering_rule` is closed across all 432 S-series primitive entries.
- Direct-test classification reports 0 actionable direct-test-debt ops.
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

