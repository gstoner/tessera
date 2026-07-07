# Tessera

**Pre-alpha. Breaking changes expected. Not production-ready.**

Tessera is a **standalone, tile-centric programming model and compiler** for
deep learning and HPC. It makes tiles, explicit memory spaces, numerical
precision, and distributed parallelism first-class compiler objects rather than
runtime heuristics.

**Standalone means runtime-independent of PyTorch, JAX, and Flax.** Those are
reference vocabularies only — the Tessera runtime never imports them. The
Python frontend, training step, data pipeline, custom-op API, and AOT export
are all in-scope (S-series, see below); file-format compatibility (e.g.
SentencePiece protobufs, SafeTensors, GGUF) is the single permitted concession.

Tessera also hosts compiler-native mathematical IR surfaces for structured
models. The active Clifford / geometric algebra (`tessera.ga`) and
energy-based model (`tessera.ebm`) tracks make algebra signatures,
multivector grades, energy minimization, sampling loops, and manifold-aware
integrators visible to the compiler rather than treating them as plain tensor
conventions. Their flat-array keystone shims now project the tensor-clean
subset onto canonical `tessera.ops.clifford_*` and `tessera.ops.ebm_*` ops, so
autodiff, Graph IR emission, and Apple GPU runtime routing share the same op
vocabulary.

Target work exists for x86 AMX/AVX512, Apple Silicon CPU/GPU, NVIDIA CUDA, and
AMD ROCm. Backend maturity varies by target: x86 CPU now has both the executed
`tessera_jit` lane and AVX-512 `runtime.launch()` lanes; Apple CPU/GPU execute
on capable Darwin hosts; AMD **gfx1151** (Strix Halo, RDNA 3.5) has a broad
compiler-generated HIP runtime family; and NVIDIA **sm_120** (RTX 5070 Ti,
consumer Blackwell) runs a hardware-verified `mma.sync` matmul. Remaining
backend breadth and unproven architectures stay hardware-gated; see
[`docs/audit/backend/BACKEND_AUDIT.md`](docs/audit/backend/BACKEND_AUDIT.md)
and [`docs/README.md`](docs/README.md) for status labels.

The fast Python unit suite currently collects ~13,360 fast tests under
`pytest tests/unit -m "not slow"`; generated audits remain the source of truth
for exact status counts.

---

## What Tessera Is

Tessera replaces thread-level GPU programming with a tile-first abstraction.
Programmers express computation in terms of tiles, groups, domains, and meshes.
The compiler handles thread mapping, memory staging, pipeline scheduling, and
collective insertion where those lowering paths are implemented.

The sketch below shows the canonical Python surface; runnable end-to-end
hardware examples are called out separately in backend-specific docs and tests.

```python
import tessera

dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
A = tessera.array.from_domain(
    tessera.domain.Rect((4, 128, 64)), dtype="fp16", distribution=dist
)
W = tessera.array.from_domain(
    tessera.domain.Rect((4, 64, 256)), dtype="fp16", distribution=dist
)
Y = tessera.array.from_domain(
    tessera.domain.Rect((4, 128, 256)), dtype="fp32", distribution=dist
)

@tessera.jit
def step(W: tessera.Region["read"],
         A: tessera.Region["read"],
         Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(A, W)

@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...],
            B: tessera.f16[..., ...],
            C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"), W.parts("tp"), Y.parts("tp")
)
```

---

## Development Status

Use these status words consistently:

| Status | Meaning |
|--------|---------|
| implemented | Source exists in the active tree and has unit or lit coverage. |
| lit-testable | Compiler/dialect behavior is covered by MLIR lit or target-contract tests; native execution is not implied. |
| mock-runtime | Runtime API works through a deterministic mock or CPU fallback for development and tests. |
| hardware-runtime | Native runtime execution is wired for the backend and has a concrete build/test path. |
| scaffolded | Directory, API shape, or design skeleton exists, but behavior is incomplete or artifact-only. |
| planned | Design direction only. |

Current status snapshot (as of June 29, 2026). Generated dashboards are the
source of truth for exact counts and executable lanes:
[`runtime_execution_matrix.md`](docs/audit/generated/runtime_execution_matrix.md),
[`runtime_abi.md`](docs/audit/generated/runtime_abi.md), and
[`s_series_status.md`](docs/audit/generated/s_series_status.md).

### Core Compiler And Runtime

| Area | Current status | What this means |
|------|----------------|-----------------|
| Frontends and IR stack | implemented / lit-testable | Python API, textual DSL, constraints/effects, Graph IR, Schedule IR, Tile IR, and Target IR object models exist with verifier and lit/unit coverage. |
| Production CPU JIT | implemented / hardware-runtime | `@tessera.jit(target="cpu")` runs a real `tessera_jit` MLIR -> LLVM ORC-JIT path before the numpy reference fallback. Covered ops include matmul, arith, activations, softmax, norm, transpose, and multi-op graph functions. |
| Autodiff | implemented / lit-testable | Python tape and Graph IR adjoint pass are present; VJP/JVP coverage includes core tensor ops, recurrent/state-space ops, collectives, losses, and model-family surfaces. |
| Distributed and training APIs | implemented / mock-runtime | DDP/FSDP, collectives, sharding, MoE/MegaMoE, optimizers, losses, and RL policy losses are usable through Python/reference and mock collectives; real NCCL/RCCL execution remains backend-gated. |
| Standalone S-series contracts | implemented / planned | 474 primitive entries are tracked. `lowering_rule` is closed project-wide, while 467 `backend_kernel` rows remain open by design until backend-specific proof lands. |
| Mathematical and model IR surfaces | implemented / lit-testable | GA/EBM, reasoning-attention families, DFlash, DiffusionGemma, and frontier MoE model-class contracts are compiler-visible. Native execution is claimed only where a backend row below or a generated audit proves it. |
| Runtime ABI and audits | implemented | Runtime C ABI surfaces and generated audit dashboards are drift-gated; exact counts are listed in the support snapshot below. |

The ~13,360-test fast unit suite passes under `-m "not slow"`; the full Python
suite collects ~13,360 tests plus slow/heavy benchmark contracts.

### Current Support Snapshot

Generated audits and executable tests are the status authority when they
disagree with prose. As of the June 29, 2026 review:

| Source | Current facts |
|--------|---------------|
| [`docs/audit/generated/e2e_op_coverage.md`](docs/audit/generated/e2e_op_coverage.md) | 309 ops tracked: 214 native end-to-end, 95 runnable reference, 0 artifact-only/partial/planned. |
| [`docs/audit/generated/runtime_abi.md`](docs/audit/generated/runtime_abi.md) | 381 unique `extern "C" tessera_*` C ABI symbols: 304 Apple, 63 x86, 10 ROCm, 4 NVIDIA; 134 Apple GPU kernel families. |
| [`docs/audit/generated/s_series_status.md`](docs/audit/generated/s_series_status.md) | 474 S-series primitive-contract entries. `lowering_rule` is closed; aggregate `backend_kernel` remains open for 467 entries because backend proof is target-specific. |
| [`docs/audit/generated/docs_freshness.md`](docs/audit/generated/docs_freshness.md) | 92 docs catalogued; 91 dated; 54 updated within 30 days; 0 older than 90 days; 1 undated. |

| Lane | Supported now | Still gated |
|------|---------------|-------------|
| Source validation | `mypy` ratchet, generated-doc drift gates, and focused generated-dashboard checks. | A checkout should not be treated as green until `scripts/validate.sh` or the relevant focused checks pass locally. |
| CPU / x86 | `tessera_jit` MLIR->LLVM CPU execution for the covered op set; native x86 AMX/AVX512 runtime rows for matmul, elementwise, reductions, softmax/norm, losses, FFT/spectral, sparse, linalg, optimizer, RNG, SSM, and related families. | Broader dtype/layout/aliasing and buffer-binding contracts; more Graph IR canonicalizers reaching the executed path. |
| Apple CPU/GPU | Apple CPU uses Accelerate/BNNS lanes. Apple GPU uses MPS, MPSGraph, MSL, packaged kernels, GA/EBM kernels, fused MoE expert FFN, and additive Metal 4 lanes where documented. | Apple hardware proof requires a capable Darwin host; non-Darwin stubs are CI fallbacks only. |
| ROCm | gfx1151 has real `runtime.launch()` lanes, including compiler-generated HIP for matmul-family, attention-family, elementwise, reductions, softmax/norm, losses, quant, FFT/spectral, linalg, sparse, scan, SSM, MoE, RoPE/ALiBi, GA/conformal, and where/argreduce. | CDNA/MI300-class proof, remaining target-map promotions, and performance hardening. |
| NVIDIA | sm_120 (RTX 5070 Ti) runs a compiler-generated CUDA lane over matmul-epilogue, attention, gated-SwiGLU, and pointwise-DAG regions, plus hand-emitted tensor-core `mma.sync` GEMM+epilogue (~6×) and flash-attention (~2.7×) lanes the accuracy-budgeted arbiter selects, through a shipped PTX launch bridge — all F4-oracle-gated. | Hopper sm_90, datacenter sm_100, `wgmma`/`tcgen05`, NVFP4 execution, and broader op coverage. |
| Distributed GPU | APIs, sharding surfaces, and NCCL/RCCL adapter scaffolding exist; mock collectives cover development paths. | Real multi-rank NCCL/RCCL execution and overlap proof. |

---

## Architecture

Tessera compiles through a four-layer IR stack:

```text
Python API + textual DSL frontend
(@tessera.jit, module/func/kernel syntax, Region[...], tessera.domain)
     |
     v
Graph IR    (tessera dialect: math ops, shape/dtype/layout metadata, diagnostics)
     |
     v
Schedule IR (schedule.* dialect: mesh.define/region, pipeline.region, stage, yield)
     |
     v
Tile IR     (tile.* ops, tessera.attn.* FA-4 ops, tessera.queue.* barriers)
     |
     v
Target IR   (backend-specific artifacts: x86, NVIDIA, ROCm, Apple, ...)
```

The Python compiler carries object models and verifier checks for Graph IR,
Schedule IR, Tile IR, and CPU/x86, NVIDIA/CUDA, Apple, and ROCm Target IR.
The JIT artifact spine emits textual MLIR-like inspection strings from those
objects; native hardware execution remains target-specific and is claimed only
where backend docs say so.

For compiler-readiness audits, keep three lanes separate:

- **Reference / contract lane:** Python reference behavior, public APIs,
  Graph IR registration, Schedule/Tile/Target artifacts, lit fixtures, and
  generated audit rows.
- **Executable production lane:** MLIR/LLVM or backend runtime code executes
  the compiled artifact and matches a Python/reference oracle or numerical
  fixture.
- **Hardware proof lane:** target-specific runtime rows, smoke tests,
  benchmark proof bits, packaged-kernel ABI validation, and host capability
  checks. Non-Darwin Apple stubs and NVIDIA/ROCm toolchain artifacts are not
  hardware proof.

Primary named pipelines and target paths are tracked in
[`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md). The
canonical pipelines registered in `tessera-opt` today:

| Pipeline | Status |
|------|--------|
| `tessera-lower-to-x86` | implemented / hardware-runtime through CPU JIT, native CPU ABI, AMX, and AVX-512 compiled lanes |
| `tessera-lower-to-gpu` (NVIDIA SM_90 default) | implemented / lit-testable |
| `tessera-nvidia-pipeline-{sm90,sm100,sm120}` (per-SM aliases) | implemented / lit-testable |
| `tessera-lower-to-rocm` | implemented / lit-testable / hardware-runtime on capable gfx1151 hosts |
| `tessera-lower-to-apple_cpu` (artifact) / `tessera-lower-to-apple_cpu-runtime` (Accelerate) | implemented / hardware-runtime |
| `tessera-lower-to-apple_gpu` (artifact) / `tessera-lower-to-apple_gpu-runtime` (MPS + custom MSL) | implemented / hardware-runtime |
| `tpp-space-time` (Tensor Parallel Primitives) | implemented / lit-testable |
| `ts-spectral-pipeline` (Spectral / FFT) | implemented / lit-testable |

### Front-to-Back Optimizing-Compiler Closure

Tessera is being converged from a library/dispatcher into a true optimizing
compiler whose C++ IR rewrites reach execution. The phased plan and per-item
status live in
[`docs/audit/compiler/COMPILER_AUDIT.md`](docs/audit/compiler/COMPILER_AUDIT.md);
the keystone phases have landed:

- **The executed CPU path is now real codegen.** `@tessera.jit(target="cpu")`
  translates the whole-graph op list into one `GraphFn` and runs it through the
  `tessera_jit` MLIR→LLVM ORC-JIT lane (`tessera-to-linalg → bufferize →
  linalg-to-loops → LLVM`) *before* the numpy reference interpreter — so the
  matmul lowers to a real M×N×K loop nest with a K-reduction, and the C++ Graph-IR
  optimizations finally reach execution. Covered set is f32/f16/bf16/f64 over
  `matmul`, the arith/activation/softmax/norm/transpose families; anything else
  falls back to numpy (correctness-preserving — a fallback handles "couldn't run",
  never "ran wrong"). An opt-in `linalg→vector` tiling+vectorization lane
  (`TESSERA_JIT_VECTORIZE`) runs the GEMM ~13–20× faster than the scalar loops.
- **The Apple GPU fusion seam is closed.** Fusion is now recognized once (by the
  compiler) and carried across to the executor as authoritative `dispatch` roles
  on each fusion group; the four structural re-matchers in the runtime were proven
  equivalent and deleted. The executor no longer re-discovers fusion per invoke.
- **Graph-IR folders/canonicalizers reach the executed path.** `canonicalize` +
  `cse` run before lowering on the CPU JIT lane, with per-op folders shipped
  (`transpose(transpose(x))→x`, identity `cast`, `x+0`/`x*1`/`x/1`, …); CSE/DCE
  fire end-to-end (duplicate QKV-projection matmuls collapse; dead pure ops drop).

### General Fusion Middle-End & Kernel Synthesis

The Apple GPU backend is the proving ground for a **general fusion middle-end**:
instead of a growing catalog of hand-written fused kernels, one *synthesizer*
emits the kernel source for an entire family of fused regions, gated by an
execution-derived oracle. The implementation lives in
[`python/tessera/compiler/fusion.py`](python/tessera/compiler/fusion.py); the
phased design is in
[`docs/audit/compiler/OPTIMIZING_COMPILER_PLAN.md`](docs/audit/compiler/OPTIMIZING_COMPILER_PLAN.md).

- **Region IR** — `FusedRegion` (a matmul root + a pointwise-epilogue chain + an
  optional terminal reduction) and `AttentionRegion` (`softmax(scale·Q·Kᵀ)·V`)
  capture a fusable subgraph as one schedulable unit.
- **Discovery** — `discover_fusable_regions` grows maximal
  `matmul → pointwise(→ reduction)` regions (single-use intermediates only) and
  is wired into the Apple GPU runtime hot path.
- **Synthesis** — `synthesize_matmul_epilogue_msl` emits one MSL kernel for any
  region: a per-thread kernel (`N ≤ 1024`), a threadgroup-tiled kernel
  (`N ≤ 8192`), and a native `half` (f16) variant; bf16 host-converts to f32.
  The accumulator is always fp32, so the math is bit-identical across dtypes.
- **Cost model** — `fusion_cost` / `should_fuse_*` decide profitability
  (stack-fit, dispatch + DRAM-traffic savings); an over-cap region is left to the
  per-op path.
- **Codegen-gated oracle** — `verify_synthesized_region` runs each synthesized
  kernel against the unfused reference before trusting it; a divergent
  synthesizer is rejected (the codegen anti-silent-fallback gate).
- **Autotune** — `autotune_matmul_epilogue` picks the fastest synthesis variant
  that passes the oracle (perf gated behind correctness).

**Catalog retirement:** the synthesizer has *replaced* the entire hand-written
`matmul_{gelu,rmsnorm,softmax}` kernel family across f32/f16/bf16 (12 hand-written
kernels → one synthesized symbol set, `synth_matmul_epilogue{,_tiled,_f16}`), each
retirement proven bit-close to the kernel it replaced on Metal. The catalog
*shrinks* as the general path absorbs it.

### Where the compiler is going (north star)

Apple proved the middle-end; the framework that generalizes it across all four
backends is now largely built. As of 2026-07, the three-tier kernel model +
measured arbiter is implemented in [`python/tessera/compiler/emit/`](python/tessera/compiler/emit/):
the `KernelEmitter`/`KernelRunner` plugin protocol + universal F4 oracle
(`kernel_emitter.py`), the `kernel_cache.py` synth→compile→cache loop, per-arch
plugins (`nvidia_cuda.py`, `rocm_hip.py`, `x86_llvm.py`), and the accuracy-budgeted
arbiter (`candidate.py`) + measured autotune (`autotune.py`) — with NVIDIA sm_120,
ROCm gfx1151, and x86 Zen 5 lanes executing on real hardware. The paired plan +
theory set under [`docs/audit/compiler/`](docs/audit/compiler/) carries the
direction and the landed annotations (status stays in the generated dashboards):

- [`COMPILER_THEORY_OF_OPERATION.md`](docs/audit/compiler/COMPILER_THEORY_OF_OPERATION.md)
  — the **three-tier kernel model** (generic synthesizer / per-arch codegen plugin
  / hand-tuned library) with a **measured, accuracy-budgeted arbiter** that picks
  the fastest in-budget candidate per `(op, shape-bucket, dtype, target)`.
- [`COMPILER_REFACTOR_PLAN.md`](docs/audit/compiler/COMPILER_REFACTOR_PLAN.md)
  — workstreams to share the lowering spine, lift the synthesizer behind a
  per-arch `KernelEmitter` plugin (MSL / PTX / AMDGCN / C-LLVM), and coordinate
  development across the three build systems (Apple dev Mac, Strix Halo / ROCm,
  NR2 Pro / CUDA).
- [`OPTIMIZING_COMPILER_PLAN.md`](docs/audit/compiler/OPTIMIZING_COMPILER_PLAN.md)
  — the F0–F6 middle-end plan (F0–F5 landed on Apple; F6 = the backend-build seam).

Governing rule: **ROCm and CUDA are the lead performance targets; the generic
framework raises the floor and must never cap their ceiling.** Hand-tuned
`wgmma` / `mma.sync` / MFMA / WMMA kernels stay first-class candidates the arbiter
measures — a compiled kernel wins only when it is both faster and in accuracy
budget.

---

## Mathematical IR Surfaces

Tessera's GA + EBM work is the first compiler-native track where mathematical
structure is part of the IR contract:

- **Clifford / geometric algebra (`tessera.ga`)** — v1 supports `Cl(3,0)` and
  `Cl(1,3)`, grade-aware `Multivector` values, geometric products, rotor
  sandwich operations, differential-form primitives, and a parallel geometric
  autodiff registry. The canonical flat-array `tessera.ops.clifford_*` shim
  covers 10 tensor-clean ops and validates closed-form VJP/JVP rules against
  finite differences while preserving the richer `geometric_algebra` coverage
  rows.
- **Energy-based models (`tessera.ebm`)** — energy primitives, inner-loop
  refinement, self-verification, Langevin / MALA / HMC / Gibbs samplers,
  partition-function estimators, EBM losses, and manifold-aware sphere /
  bivector Langevin reference paths. The canonical `tessera.ops.ebm_*` shim
  covers the tensor-clean subset (`energy_quadratic`, `self_verify`,
  `refinement`, `inner_step`); callable/RNG EBM APIs intentionally remain on
  `tessera.ebm`.
- **Reasoning-model attention family + RL** — DeepSeek sparse attention, MiniMax
  Lightning, Kimi-Delta, gated/hybrid/MLA decode (each with VJP+JVP); `tessera.rl`
  ships PPO/GRPO/CISPO policy losses for post-training. Backed by
  `src/transforms/lib/AttentionFamilyPasses.cpp`.

Key documents:

| Document | What it covers |
|----------|----------------|
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford signatures, multivector type contract, GA ops, autodiff, dialect, and lowering |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy primitive contract, inner-loop schedule, training losses, and EBM IR mapping |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | Layered status for Python reference, MLIR/lit, manifests, and native execution |
| [`docs/status/ga_ebm_milestone.md`](docs/status/ga_ebm_milestone.md) | Canonical GA/EBM native milestone status, health check, non-claims, and next targets |
| [`docs/audit/domain/DOMAIN_AUDIT.md`](docs/audit/domain/DOMAIN_AUDIT.md) | Sprint roadmap and acceptance history for the GA + EBM tracks |

The June 8 lane-unification keystone closes the earlier GA/EBM gap between the
specialized math namespaces and the canonical compiler op surface:

- `tessera.ops.clifford_*` exposes 10 flat-coefficient GA ops that flow through
  the autodiff tape, emit canonical Graph IR ops, and route
  `@jit(target="apple_gpu")` to the cl30 MSL kernels with
  `execution_mode="metal_runtime"`.
- `tessera.ops.ebm_*` exposes 4 tensor-clean EBM ops with the same
  autodiff + Graph IR + Apple GPU routing contract.
- The primitive coverage importer skips registry-owned GA/EBM names so the
  authoritative `category="geometric_algebra"` and `category="ebm"` rows keep
  their references, halo/sharding notes, and dialect alignment.

Native execution status is layer-specific:

| Component | Python reference | MLIR / lit | Backend manifest | Native execution |
|-----------|------------------|------------|------------------|------------------|
| GA signature, multivector values, ops, calculus, and autodiff | implemented | implemented / lit-testable for dialect and lowering fixtures | implemented for registered `clifford_*` primitives | 17/17 Apple GPU native primitives benchmarked; x86 and Apple CPU are reference-first; NVIDIA/ROCm planned |
| EBM energy primitives, samplers, partition estimators, losses, and manifold-aware sampling | implemented | implemented / lit-testable for dialect and annotation-pass fixtures | implemented for registered `ebm_*` primitives | **9/9 native Apple GPU ops benchmarked**: `inner_step`, `refinement`, `langevin_step`, `decode_init`, `bivector_langevin`, `sphere_langevin`, `self_verify`, quadratic `energy`, `partition_exact` (stable-logsumexp MSL kernel landed 2026-05-17) |
| GA/EBM composite workloads | implemented as deterministic benchmark workloads | n/a | uses GA/EBM manifest-resolved symbols where native | `ga_feature_pipeline`, `ebt_tiny_refinement`, and opt-in `--ebt-sweep` benchmarked with Apple GPU and Python-reference rows |

---

## Audit-as-Data

Status truth for many areas of the repo is rendered from machine-readable
registries, not prose. The 5 **surface manifests** under
`python/tessera/compiler/` (`examples_manifest.py`, `benchmarks_manifest.py`,
`research_manifest.py`, `tools_manifest.py`, `tests_manifest.py`) are
consolidated into one generated dashboard:

- [`docs/audit/generated/surface_status.md`](docs/audit/generated/surface_status.md) (human) + [`surface_status.csv`](docs/audit/generated/surface_status.csv) (canonical, machine-readable) — examples / benchmarks / research / tools / tests + operator-benchmark coverage.

Plus 13 op-level / compiler-level audit registries covering primitive
coverage (`primitive_coverage.py` / S-series status: 474 entries), backend
kernel manifests, MLIR verifier coverage, dialect registration, named-pipeline
registry, diagnostic codes, docs freshness, effect lattice, runtime C ABI
surface, test coverage by op family, TSOL canonical-op coverage, and
Apple GPU/CPU target maps. Each dashboard is drift-gated by a test in
`tests/unit/`. When you wonder "is X tested / shipped / supported?", the
answer is one of these registries.

CLIs that consume them:

- `tessera-surface-audit --surface=<examples|benchmarks|research|tools|tests>`
- `tessera-claim-lint --surface=<…>`
- `tessera-apple-target-map`, `tessera-gpu-target-map`
- `tessera-e2e-coverage`, `tessera-examples-audit`
- `tessera-operator-benchmarks-coverage`

---

## Documentation

**Start here:** [`examples/getting_started/compile_and_explain.py`](examples/getting_started/compile_and_explain.py)
is the canonical compiler tour — `@tessera.jit` → `fn(...)` →
`fn.explain()` → `tessera.compiler.support(op)` → `tessera.from_text(...)`
in ~80 lines.  Runs on CPU, no accelerator required.

| Document | What it covers |
|----------|----------------|
| [`docs/README.md`](docs/README.md) | Documentation authority tree and status labels |
| [`docs/CANONICAL_API.md`](docs/CANONICAL_API.md) | Public API names and syntax |
| [`docs/spec/PYTHON_API_SPEC.md`](docs/spec/PYTHON_API_SPEC.md) | Public Python symbols and signatures |
| [`docs/spec/COMPILER_REFERENCE.md`](docs/spec/COMPILER_REFERENCE.md) | IR stack, pass registry, pipelines, compiler source map |
| [`docs/spec/AUTODIFF_SPEC.md`](docs/spec/AUTODIFF_SPEC.md) | Tape-based reverse-mode autodiff (Tier 2) + Phase F4 Graph IR adjoint pass (`AdjointInterface` op trait, multi-output rewrite, `tessera-autodiff` MLIR pass) + Phase F5 adjoint collective insertion |
| [`docs/audit/backend/nvidia/archive/nvidia_execution_audit.md`](docs/audit/backend/nvidia/archive/nvidia_execution_audit.md) | Phase G1 — concrete punch list (G1-1 through G1-8) for first SM_90 BF16 GEMM on H100 |
| [`docs/spec/CLIFFORD_SPEC.md`](docs/spec/CLIFFORD_SPEC.md) | Clifford / geometric algebra primitive surface |
| [`docs/spec/EBM_SPEC.md`](docs/spec/EBM_SPEC.md) | Energy-based model primitive surface |
| [`docs/spec/GA_EBM_EXECUTION_STATUS.md`](docs/spec/GA_EBM_EXECUTION_STATUS.md) | GA + EBM execution status by implementation layer |
| [`docs/spec/LOWERING_PIPELINE_SPEC.md`](docs/spec/LOWERING_PIPELINE_SPEC.md) | Pass contracts and invariants |
| [`docs/spec/TARGET_IR_SPEC.md`](docs/spec/TARGET_IR_SPEC.md) | Schedule, Tile, and Target IR dialect details |
| [`docs/spec/RUNTIME_ABI_SPEC.md`](docs/spec/RUNTIME_ABI_SPEC.md) | Runtime C ABI |
| [`docs/reference/tessera_tensor_attributes.md`](docs/reference/tessera_tensor_attributes.md) | Normative tensor attributes + dtype names (six attributes, canonical/alias/planned-gated dtype sets, promotion rules) |
| [`docs/audit/roadmap/ROADMAP_AUDIT.md`](docs/audit/roadmap/ROADMAP_AUDIT.md) | Phases A–I + S-series S0–S15 standalone compiler track with per-task acceptance criteria |
| [`docs/audit/backend/BACKEND_AUDIT.md`](docs/audit/backend/BACKEND_AUDIT.md) | Hardware-gated frontier — what's blocked on real NVIDIA / ROCm |
| [`docs/architecture/README.md`](docs/architecture/README.md) | Architecture guide index |
| [`docs/guides/Tessera_Developer_Frontend_End_To_End.md`](docs/guides/Tessera_Developer_Frontend_End_To_End.md) | First executable frontend path and IR inspection |
| [`docs/distributed_megamoe.md`](docs/distributed_megamoe.md) | **Distributed MegaMoE** — single-device MoE layer, fused expert-FFN kernel, expert-parallel 2× all-to-all forward, FP8×FP4 mixed precision, and real comm/compute overlap |
| [`docs/apple_backend.md`](docs/apple_backend.md) | **Canonical Apple CPU + GPU reference** — architecture, kernel inventory, datatypes, and Metal 4 implementation state (consolidates the former overview / kernel-inventory / datatypes / integration-review docs) |
| [`docs/apple_gpu_metal4_adoption.md`](docs/apple_gpu_metal4_adoption.md) | Current Metal 4 ladder and coexistence model (forward-looking plan) |
| [`docs/dflash.md`](docs/dflash.md) | **DFlash speculative decoding** — the `attn_bias` substrate and the block-diffusion draft (`tessera.dflash` / `dflash_reference` / `dflash_io` / `dflash_serve`): architecture, API, what's proven, and the external gates |

---

## Build & Test

```bash
# Python development install
pip install -e ".[dev]"

# Daily edit-loop sanity check (~13,360 fast tests, < 512 MB RAM)
pytest tests/unit/ -m "not slow" -q

# Full Python suite including heavy benchmarks (~13,866 collected)
pytest tests/unit/ -q

# GA + EBM native Apple GPU health check; skip-recording on non-Darwin
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci

# Optional performance tests
TESSERA_RUN_PERFORMANCE_TESTS=1 scripts/test.sh

# CPU validation spine: versions, unit tests, runtime and benchmark smokes,
# standalone runtime/profiler builds, collectives compile check, audit lane
scripts/validate.sh

# Per-target release gate (the canonical Apple release blocker)
scripts/release_gate.py --target=apple_gpu

# Type check
mypy python/tessera/
```

C++/MLIR builds require LLVM/MLIR 22 or newer:

```bash
# Convenience build wrapper; defaults to CPU-only unless CUDA/HIP is enabled
scripts/build.sh

# Or configure explicitly, for example on macOS with Homebrew LLVM 22
LLVM_PREFIX="$(brew --prefix llvm)"
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$(brew --prefix lit)/bin/lit" \
  -DTESSERA_CPU_ONLY=ON \
  -DTESSERA_BUILD_APPLE_BACKEND=ON

cmake --build build --parallel

# On Ubuntu 24.04 (x86 + AMD ROCm 7.2.4): bootstrap the toolchain once with
#   bash scripts/setup_ubuntu.sh           # LLVM/MLIR 22 from apt.llvm.org + venv
# then configure against the system LLVM and ROCm at /opt/rocm:
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir \
  -DTESSERA_ENABLE_HIP=ON \
  -DTESSERA_BUILD_ROCM_BACKEND=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build tessera-opt

# MLIR lit tests require a built tessera-opt on PATH
lit tests/tessera-ir/ -v

# If the Python lit package is not installed, run a focused Apple contract
# directly with the LLVM FileCheck installed alongside Homebrew LLVM:
build/tools/tessera-opt/tessera-opt tests/tessera-ir/phase8/apple_gpu_lowering.mlir \
  --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu)' --allow-unregistered-dialect \
  | "$LLVM_PREFIX/bin/FileCheck" tests/tessera-ir/phase8/apple_gpu_lowering.mlir

# Phase F4 — Graph IR autodiff. Verify the AutodiffPass + AdjointInterface
# trait build cleanly and the reverse-mode pass produces the expected adjoint IR:
cmake --build build --target tessera-opt --parallel
build/tools/tessera-opt/tessera-opt --tessera-autodiff \
  tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir \
  | "$LLVM_PREFIX/bin/FileCheck" tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir
```

NVIDIA / ROCm toolchain checks (skip cleanly when toolchains absent):

```bash
# Validate CUDA 13.3 PTX patterns against installed nvcc
python scripts/validate_nvcc_compile.py

# Validate ROCm 7.2.4 AMDGCN intrinsics against installed hipcc
python scripts/validate_hipcc_compile.py

# Probe NCCL/RCCL ≥ 2.22 symbols at runtime
python scripts/probe_collective_libs.py
```

Documentation checks:

```bash
scripts/lint_docs.sh
```

---

## Project Layout

For the full canonical layout see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).
A compressed view of the active surface:

```text
python/tessera/
  __init__.py            Public surface — re-exports core, jit, dist, ops, dtype, …
  core/                  Tensor, Module, fundamental abstractions
  compiler/              @jit, textual frontend, IR objects, target maps, audit modules
  distributed/           Region, domain, dist, array, shard, index_launch;
                         MoE router + distributed MegaMoE (expert-parallel
                         2x all-to-all, FP8xFP4, async comm/compute overlap)
  nn/                    Stateful module / layers / functional / utils
  train/                 Agent-native MoE training stack (lazily bound as
                         tessera.train): MoE router/FFN, sparse dispatch,
                         Qwen3-MoE model, GRPO loop — Apple-GPU single-node
  autodiff/              Tape, VJPs, JVPs, mixed-precision, rematerialize (Tier 2)
  cache/                 KVCacheHandle + MemoryStateHandle persistent state ABI
  ebm/                   Energy-based model primitives (M6)
  ga/                    Geometric Algebra / Clifford primitives (M5/M7)
  state/                 Pytree primitives + state-collection taxonomy
  testing/               Mock collectives and Python test helpers
  cli/                   tessera-mlir, tessera-prof, tessera-translate,
                         tessera-runtime-smoke, tessera-surface-audit,
                         tessera-claim-lint, tessera-{apple,gpu}-target-map,
                         tessera-e2e-coverage, tessera-examples-audit,
                         tessera-operator-benchmarks-coverage, tessera-autotune
  runtime.py             ctypes wrapper over the runtime C ABI
  profiler.py            Runtime profiler facade
  autotune.py            Public autotuning facade
  dtype.py               Canonical dtype enforcement + Dtype + result_type
  diagnostics.py         ErrorReporter, ShapeInferenceEngine
  debug.py               DebugTrace, GraphTrace, check_grad, check_determinism
  telemetry.py           Shared telemetry event/report schema
  # S-series reference surface modules:
  aot.py · checkpoint.py · control.py · custom.py · data.py · losses.py ·
  memory.py · optim.py · quantization.py · rl.py · rng.py · sharding.py
  # Domain modules:
  complex.py · conformal_advanced.py · contour.py · distributions.py ·
  energy.py · flow.py · hyperbolic.py · riemann_surface.py · server.py …

src/
  compiler/ir/           Core Tessera Graph IR ODS + C++ dialect sources
  compiler/mlir/         MLIR plugin integration
  compiler/programming_model/  Schedule / programming-model IR
  compiler/tile_opt_fa4/ FA-4 attention and queue dialects/passes
  compiler/tessera_neighbors/  Neighbor / halo / stencil dialect and passes
  compiler/codegen/      x86, NVIDIA, ROCm, Apple
  compiler/diagnostics/  ErrorReporter, ShapeInferencePass
  compiler/autotuning/   Autotuner v1 framework
  collectives/           Collective IR + NCCL/RCCL adapters + chunk planner
  runtime/               Runtime C ABI, CPU/CUDA/HIP backends, scheduler tests
  solvers/               core, linalg, scaling_resilience, spectral, tpp,
                         clifford (M5), ebm (M6)
  transforms/            Canonicalization, lowering, named pipelines;
                         Phase F4 AutodiffPass + AdjointInterface (op trait);
                         Phase F5 AdjointCollectiveInsertionPass

tests/
  unit/                  Python unit and compiler-contract tests
  tessera-ir/            MLIR lit tests by phase / path
  kernel_tests/          System-level kernel and roofline scaffolds
  performance/           Optional performance sweeps
  tessera_numerical_validation/ Numerical reference validation
  integration/, regression/, tessera_tests/

docs/
  spec/                  Normative specs (14 files)
  architecture/          Design documents
  guides/                11 developer guides (~3,400 LOC)
  programming_guide/     11-chapter user manual + Appendix NVL72
  reference/             Tensor attribute + dtype reference, migration guide
  operations/            Canonical operation catalog (TSOL)
  audit/                 Audit reports + generated/ dashboards (drift-gated)
  status/                Per-milestone status docs
  tutorials/             Flash Attention, performance tuning
  context/               Auto-generated context outputs
  benchmarks/            Benchmark-related documentation
```

---

## Key Design Decisions

1. The Python frontend is permanent; MLIR/C++ handles performance-critical
   lowering.
2. Tessera uses static compiler analysis, not a tracing JIT tier.
3. `Region[...]` is a type annotation, not a runtime wrapper.
4. Domains describe shape; distributions describe placement. They remain
   separate objects.
5. Constraint checks run at decoration time when concrete bindings are
   available.
6. **Tessera is a standalone compiler.** PyTorch, JAX, and Flax are reference
   vocabularies only — never imported by the runtime. File-format compatibility
   (SafeTensors, GGUF, SentencePiece protobufs) is the single permitted
   concession.
7. Each backend exposes a **hardware-free Target IR** layer between Tile IR
   and final emission (e.g. `tessera_rocm.mfma`, `tessera_apple.gpu.metal_kernel`).
   This is what makes backends lit-testable and what
   `tests/unit/test_target_ir_contract.py` validates.
8. **`primitive_coverage.py` is the standalone compiler's audit truth.**
   Adding a primitive means updating both the runtime catalog (op_catalog.py)
   and the audit registry; (V)JP / (J)VP registration auto-promotes the
   matching contract axes.
9. Generated dashboards under `docs/audit/generated/` are **not edited by
   hand** — every one is drift-gated by a test that compares the live
   registry to the on-disk snapshot.

For the full design rationale (22 numbered decisions), see
[`CLAUDE.md`](CLAUDE.md).

---

## License

Apache License 2.0 - see [LICENSE](LICENSE).
