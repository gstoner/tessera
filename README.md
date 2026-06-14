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
AMD ROCm. Backend maturity varies by target: x86 AMX and Apple Silicon
(CPU + GPU) execute natively today; NVIDIA and ROCm have toolchain-pinned
Target IR + lit fixtures with native execution gated on real hardware
(Phase G/H — see
[`docs/audit/backend/BACKEND_AUDIT.md`](docs/audit/backend/BACKEND_AUDIT.md));
see
[`docs/README.md`](docs/README.md) for the status labels used across docs.

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

Current high-level status (as of June 14, 2026):

| Area | Status |
|------|--------|
| Python frontend, textual DSL frontend, constraints, effects, Graph IR | implemented |
| Object-backed Graph / Schedule / Tile IR + per-target Target IR artifacts | implemented / lit-testable |
| **x86 AMX BF16 + AVX512 lowering and execution** (Phase 2) | **implemented / hardware-runtime** |
| **Apple Silicon CPU** via Accelerate (cblas_sgemm rank-2/rank-3 + BNNS f16/bf16) (Phase 8.2) | **implemented / hardware-runtime** |
| **Apple Silicon GPU** via MPS, MPSGraph, custom MSL, additive Metal 4 lanes, and packaged `.mtlpackage` loading — 265 Apple C ABI symbols across 113 kernel families (count truth: [`docs/audit/generated/runtime_abi.md`](docs/audit/generated/runtime_abi.md)), GA/EBM/M7 fused kernels, the fused MoE-SwiGLU expert-FFN kernel, MTL4 `matmul2d` bf16 default routing, MTL4 epilogue/session/archive paths, batched linalg MSL kernels, and PK1–PK7 packaged-kernel ABI validation | **implemented / hardware-runtime (Darwin); non-Darwin stubs are CI fallbacks, not hardware proof** |
| **Distributed MoE / MegaMoE** — single-device `nn.functional.moe_layer`, fused `ops.moe_swiglu_block` (Graph-IR op + GPU kernel), expert-parallel `megamoe_forward` (GShard 2× all-to-all dispatch/combine), FP8×FP4 mixed precision, and **real wall-clock comm/compute overlap** (async GPU command buffer ∥ CPU comm) | implemented / hardware-runtime (Apple GPU expert FFN); multi-rank via in-process mock collectives — see [`docs/distributed_megamoe.md`](docs/distributed_megamoe.md) |
| NVIDIA SM_90+ FA-4, WGMMA/TMA, Blackwell TCGEN05/TMEM target artifacts; **CUDA 13.2 Update 1 toolchain pin** | implemented / lit-testable; execution gated on real hardware (Phase G) |
| ROCm MFMA gfx90a / gfx94x / gfx950 / gfx1100; **ROCm 7.2.3 toolchain pin** | implemented / lit-testable; execution gated on real hardware (Phase H) |
| Distributed APIs, cyclic sharding, NCCL/RCCL adapters (≥ 2.22 pin) | implemented / scaffolded |
| Solver, sparse/RNG, linalg, scaling-resilience, **spectral (all 6 passes shipped)**, TPP | implemented / lit-testable |
| **Tier 2 autodiff** — Python tape (`tessera.autodiff.{tape, reverse, custom_rule, rematerialize}`) + 270+ built-in VJPs/JVPs covering matmul, depthwise conv 1d/2d, RNN cells (LSTM), Mamba2 `selective_ssm`, distributed collectives | implemented / hardware-runtime (numpy-reference tape); end-to-end BPTT through multi-step LSTM, depthwise conv chains, and selective SSMs verified vs. central-difference numerical Jacobian at fp64 |
| **Phase F4/F5 — Graph IR adjoint pass** — `tessera-autodiff` MLIR pass + `Tessera_AdjointInterface` op trait + per-op `buildAdjoint` impls (`MatmulOp`, `LayerNormOp`, `SoftmaxOp`, GELU/ReLU/Sigmoid/Sin via `tessera.custom_adjoint_call` placeholder); `tessera-adjoint-collective-insertion` emits real `tessera.collective.{reduce_scatter, all_gather, all_reduce}` on cotangent SSA values from F4's multi-output rewrite; `tessera-autodiff-pipeline` combines F4+F5 | implemented / lit-testable — `tessera-opt` builds clean against MLIR 21 + 22; `tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir` passes FileCheck showing cotangent seed + two transposed matmuls (`dA = seed @ B^T`, `dB = A^T @ seed`) + `tessera.autodiff.arg_cotangents` annotation |
| **Phase I — DDP / FSDP wrappers** — `tessera.distributed.DDP` (replicated weights, `all_reduce` mean-reduction on `Parameter.grad`); `tessera.distributed.FSDP` (ZeRO stage 2/3, leading-dim sharding, `gather_for_forward` / `reshard_after_forward` / `reduce_scatter` to local shard) | implemented / mock-runtime (in-process `MockRankGroup`); real NCCL/RCCL bindings land alongside Phase G |
| **S-series standalone compiler track** (S0–S15): RNG, state/pytrees, control flow, sharding, NN functional, quantization, optimizers, losses, **`tessera.rl` PPO/GRPO/CISPO**, AOT export, custom-primitive API, dataset combinators + tokenizers | implemented (Python reference); 456 entries × 12 contract axes tracked in `primitive_coverage.py` (count truth: [`docs/audit/standalone_primitive_coverage.md`](docs/audit/standalone_primitive_coverage.md)); backend-kernel proof remains the universal open Phase G/H gate |
| **Reasoning-model attention family** — DeepSeek sparse attention, MiniMax Lightning, Kimi-Delta, gated/hybrid/MLA decode + RL post-training losses, all with VJP+JVP | implemented / lit-testable |
| **Speculative decoding — `attn_bias` substrate + DFlash block-diffusion draft** — additive `attn_bias` operand on `FlashAttnOp` end-to-end (Graph IR ODS + verifier, Tile→Apple lowering, MPSGraph `flash_attn_bias_{f32,f16,bf16}` runtime symbols, eager/CPU/GPU dispatch, VJP, ABI audit); `tessera.dflash` block-diffusion draft (KV injection, QK-norm, GQA, sliding-window-via-bias, draft KV cache, sampling + distribution-preserving rejection, `DFlashDraft` `nn.Module`, training loss), `tessera.dflash_reference` (stateful KV-cached target + rollback), `tessera.dflash_io` (safetensors checkpoint loader), `tessera.dflash_serve` (tokenizer-wired generation + scheduler) — see [`docs/dflash.md`](docs/dflash.md) | implemented (Python reference; attention core on Apple GPU `metal_runtime`); greedy spec-decode == greedy AR proven vs the `z-lab/dflash` MLX reference; real-checkpoint numerical parity (network) + a fully-jitted GPU draft (GPU gather) are external gates |
| **DiffusionGemma model graph** (`tessera.models`) — a Gemma-4-calibrated block-diffusion MoE text model as a **shape-only Graph + config-aware verifier** (the contract layer): `DiffusionGemmaConfig`, `build_text_block` / `verify_text_block` / `verify_config`, param-budget estimator; plus reference layers for MoE top-k routing + packing (`route_top_k` / `moe_forward`), the entropy-bound sampler, the block-diffusion step graph + decode loop (`BlockDiffusionDecoder`), and quantization/vision staging manifests. Adds the differentiable `tessera.ops.softcap` (Gemma logit soft-cap) with VJP+JVP. **Block-diffusion region native execution** (`block_diffusion_runtime`): a faithful multi-head GQA canvas denoiser (canvas queries over `[encoder_KV ++ canvas_KV]`, bidirectional) + grouped-SwiGLU MoE FFN, composed through `ops.flash_attn` + `moe_swiglu_block`. **`backend="apple_gpu"`** runs the *whole* step on Metal (attention + MoE + LM-head matmul); `execute_block_diffusion_step` does denoiser → LM head → entropy sampler, and `NativeBlockDiffusionDecoder` is the end-to-end native decode loop (commit/freeze/re-noise + KV promotion). The compiler sees the step as **one structured Graph-IR op** `tessera.diffusion_block_step` (verifier: bidirectional canvas / GQA / head_dim≤256; Tile→Apple tags it `metal_runtime`) | implemented (Python reference — shape-only graph + verifier; **the whole block-diffusion step executes on Apple GPU `metal_runtime`** via `backend="apple_gpu"`, validated against the numpy backend at production head_dim=256; the `tessera.diffusion_block_step` Graph-IR region op + Tile→Apple lowering are lit-proven). Hardware-fused single-kernel region + real-checkpoint weights are future |
| **Frontier MoE model-class compiler track** (`tessera.models.{deepseek_v32, glm5, kimi_k2}` + `tessera.stdlib.{quant, moe, attention}`) — Kimi-K2 / DeepSeek-V3.2 / GLM-5 as shared-pillar model graphs with executable scaled instances. **stdlib pillars:** packed-INT4/FP8 quant + **fused dequantize-into-GEMM** (`dequant_matmul` / `dequant_grouped_gemm`, fp32 accumulate); capacity-aware MoE dispatch + quantized grouped SwiGLU; **MLA** (decoupled-RoPE + weight absorption + paged latent cache); **DSA + LSA** block-sparse attention (offset-aware, decode-loop-consistent). **Model runtime** (`moe_transformer_runtime`): full decoder stack + a KV-cached autoregressive decode loop, oracle-gated **decode ≡ recompute** for all three scaled models with real DSA/LSA sparsity. First-class IR: `tessera.dequant_matmul` / `dequant_grouped_gemm` registered MLIR dialect ops + catalog/coverage rows + VJP/JVP; **fused `tessera_apple_gpu_dequant_matmul_f32` Metal kernel** (in-register dequant). Full-config artifacts verified at production dims via `tests/tessera-ir/model_class/*.mlir`. See [`docs/audit/roadmap/MODEL_CLASS_ROADMAP.md`](docs/audit/roadmap/MODEL_CLASS_ROADMAP.md) | implemented (Python reference + scaled Apple GPU execution; full scale + NVIDIA hardware-gated) |
| Clifford / geometric algebra Python surface, canonical `tessera.ops.clifford_*` shim, autodiff registry, dialect, lowering passes, and Apple GPU fused kernels | implemented / lit-testable; **all 17 canonical `tessera.ops.clifford_*` ops route through autodiff + Apple GPU metal_runtime — the GA autodiff surface is fully closed (16 vjp/jvp complete + 1 not_applicable `clifford_integral`, 0 planned), incl. rotor-sandwich, exp/log, hodge-star, and the ext_deriv/vec_deriv/codiff field operators; 17/17 Apple GPU GA primitives benchmarked** |
| Energy-based model Python surface, canonical `tessera.ops.ebm_*` tensor-clean shim, samplers, losses, partition estimators, dialect, annotation passes, and Apple GPU kernels | implemented / lit-testable; **4 tensor-clean canonical ops (`energy_quadratic`/`self_verify`/`refinement`/`inner_step`) + 5 EBM training losses (CD/PCD/score-matching/ISM/DSM, MPSGraph kernels) route through autodiff + Apple GPU metal_runtime; the callable/RNG ops (energy, langevin, partition_*) are correctly not_applicable on the flat tape; 9/9 native Apple GPU EBM ops benchmarked** (incl. `ebm_partition_exact` via stable-logsumexp MSL kernel) |
| Runtime C ABI and Python wrapper | mock-runtime; hardware-runtime when C runtime is built |
| **Audit-as-data infrastructure** — 5 manifest families + 17 generated audit dashboards (15 under `docs/audit/generated/` + 2 root: `op_target_conformance`, `standalone_primitive_coverage`), incl. the **S-series accelerator-proof map** (`s_series_accelerator_proof.md`), with CSV-canonical artifacts where applicable, drift-gated by `tests/unit/` | implemented |

The ~8,240-test fast unit suite passes under `-m "not slow"`; the full Python
suite collects ~9,685 tests including heavy benchmark contracts.

### Current Source and Documentation Health

This repository is moving quickly; prefer generated audits and executable tests
over phase prose when they disagree. As of the June 9, 2026 source review:

| Surface | Health |
|---------|--------|
| Source and static checks | The `mypy` ratchet and generated-doc drift gates are part of the validation spine; rerun `scripts/validate.sh` or the focused generated-doc checks before treating a local checkout as green. |
| Runtime ABI inventory | Drift-gated and current: `docs/audit/generated/runtime_abi.md` reports 267 unique `extern "C" tessera_*` C ABI symbols, 256 unique Apple symbols, and 109 Apple GPU kernel families. |
| Apple backend | The source has moved beyond the older Phase 8.4.7 overview: MPS/MPSGraph remain the default lanes, while Metal 4 is additive for bf16/f16 `matmul2d`, fused epilogues, resident MLP sessions, pipeline archives, opt-in conv2d, and control-flow experiments. See `docs/apple_backend.md` (canonical CPU+GPU reference) and `docs/apple_gpu_metal4_adoption.md` (forward-looking ladder). |
| Compiler gap focus | The frontend and IR artifact spine are broad, but production promotion still depends on executable codegen plus oracle/fixture comparison. Current open work is runtime consumption of `fusion_groups`, stronger dtype/layout/aliasing/buffer-binding contracts, graph outputs in canonical metadata, and fixture-backed numerical proof before backend cells become complete. |
| S-series backend proof | `docs/audit/generated/s_series_status.md` reports 445 entries with complete tests and lowering rules, but every entry still has an open `backend_kernel` axis (`partial` or `planned`). Treat broad S-series status as Python/reference + contract coverage until a backend-specific proof closes that axis. |
| Known Apple test gaps | Apple runtime claims require a capable Darwin host. Non-Darwin stubs are CI fallbacks, and hardware proof should come from fresh-process runtime checks or the backend-specific benchmark/test lanes before promoting a new Apple claim. |
| Documentation | The freshness dashboard is healthy but date-sensitive (66 docs catalogued; current manifest summary: 62 dated, 25 within 30 days, 0 older than 90 days, 4 undated). Semantic freshness is uneven. Generated dashboards and `docs/README.md` are the most reliable surfaces; the consolidated `docs/apple_backend.md` is now the canonical Apple CPU+GPU reference. |

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
| `tessera-lower-to-x86` | implemented (hardware-runtime via AMX) |
| `tessera-lower-to-gpu` (NVIDIA SM_90 default) | implemented / lit-testable |
| `tessera-nvidia-pipeline-{sm90,sm100,sm120}` (per-SM aliases) | implemented / lit-testable |
| `tessera-lower-to-rocm` | implemented / lit-testable / artifact-only |
| `tessera-lower-to-apple_cpu` (artifact) / `tessera-lower-to-apple_cpu-runtime` (Accelerate) | implemented / hardware-runtime |
| `tessera-lower-to-apple_gpu` (artifact) / `tessera-lower-to-apple_gpu-runtime` (MPS + custom MSL) | implemented / hardware-runtime |
| `tpp-space-time` (Tensor Parallel Primitives) | implemented / lit-testable |
| `ts-spectral-pipeline` (Spectral / FFT) | implemented / lit-testable |

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
coverage (`primitive_coverage.py` — 443 entries × 12 contract axes), backend
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

# Daily edit-loop sanity check (~8,240 fast tests, < 512 MB RAM)
pytest tests/unit/ -m "not slow" -q

# Full Python suite including heavy benchmarks (~9,685 collected)
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
# Validate CUDA 13.2 U1 PTX patterns against installed nvcc
python scripts/validate_nvcc_compile.py

# Validate ROCm 7.2.3 AMDGCN intrinsics against installed hipcc
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
