# CorrDiff-Like Stack: Compiler / Library / Runtime Split Evaluation

> **Purpose:** Evaluate what a CorrDiff-like AI forecasting model for mountainous regions
> actually needs from the Tessera compiler, what belongs in a `tessera.sciml` / `tessera.fields`
> library, and what is purely runtime responsibility. Cross-referenced against the existing
> Phase 7 / S-series roadmap and both the NVL72 (NVIDIA) and Helios/MI455x (AMD) scale-out targets.
>
> Last updated: 2026-05-20 (updated: added Helios/MI455x section)

---

## 1. The Proposed Split Is Correct — With Nuance

The user's partition (compiler knows generic optimization-relevant structure; libraries carry
domain vocabulary) matches Tessera's existing architecture philosophy and the S0 scope decision.
This document validates, sharpens, and extends it.

**Core rule:** put domain vocabulary in libraries; put optimization-relevant structure in the
compiler. The compiler does not need to know "weather," "San Juans," or "CorrDiff." It does
need to know "this op reads a 5-cell halo," "this tile overlaps neighbors," "this reduction
must be deterministic," and "this random stream is per ensemble member."

---

## 2. Compiler Primitive Set — Current Coverage vs. What's Needed

### 2.1 What Tessera Already Has (Can Be Used Today)

| Primitive | Where | Status | Notes |
|-----------|-------|--------|-------|
| **Halo inference** | `tessera_neighbors/HaloInferPass` | ✅ Complete | Auto-computes per-axis halo widths from stencil tap definitions |
| **Stencil op dialect** | `tessera_neighbors/StencilLowerPass` | 🟡 Partial | Graph IR + attributes emitted; loop expansion deferred to target passes |
| **Async halo pipeline** | `PipelineOverlapPass` | 🟡 Partial | Stream IDs assigned; actual DMA/transport not emitted |
| **Boundary conditions (semantic)** | `tessera.neighbors` dialect | ✅ Complete | Periodic, Dirichlet, Neumann, Reflect as dialect attributes |
| **Boundary conditions (lowering)** | — | ❌ Missing | No IR materialization for padding/ghost-region fills |
| **Domain decomposition** | `distributed/`, `DistributionLoweringPass` | ✅ Complete | Block/Cyclic/Replicated with collective insertion |
| **Halo + mesh boundary integration** | — | ❌ Missing | No automated overlap handling at decomposition edges |
| **Sliding window attention** | `AttentionFamilyPasses.cpp` | ✅ Complete | Graph IR op + VJP + JVP; reference path working |
| **2D local window attention** | — | ❌ Missing | Only 1D sliding window; spatial grids need 2D extension |
| **Flash Attention (standard)** | `tile_opt_fa4/` | ✅ Complete | Tile IR → PTX/HIP via WGMMA/TMA; Apple GPU MSL via Phase 8 |
| **Conv2D** (Graph IR) | `TesseraOps.td` | ✅ Complete | NHWC×HWCF op; TilingInterface declared |
| **Conv2D** (backend kernels) | — | ❌ Missing | x86 uses im2col decomposition; GPU kernels pending Phase G |
| **Deterministic RNG effects** | `EffectLattice`, `S4 RNG` | ✅ Complete | Per-rank stream assignment; Philox-backed; effect-annotated |
| **Checkpoint / remat** | `InsertRecomputePass`, `S5 remat` | ✅ Complete | Budget-guided greedy; hooks into autotuner |
| **Mixed precision policy** | `NumericPolicy`, Sprint C2 | ✅ Complete | Storage/accum separation; TF32 via math_mode |
| **Collective insertion (Graph IR)** | `GPUCollectiveInsertionPass` | ✅ Complete | Effect-aware; reduce_scatter/all_gather on cotangents |
| **Layout metadata / transforms** | `dtype.py`, `shape.py` | ✅ Complete | NHWC/NCHW; Sprint A0 canonical dtypes |
| **Autotuning hooks** | `BayesianAutotuner`, `autotune_v2.py` | ✅ Complete | Optuna TPE + Hyperband; SQLite cache |
| **Fusion (producer/consumer)** | `CanonicalizeTesseraIR.cpp` | ✅ Complete | 4 Graph IR fusion patterns; pipeline alias carries fusion order |
| **Distributed sharding + collectives** | `sharding.py`, Phase 4 | ✅ Reference | Python reference layer complete; GPU execution pending Phase G |

### 2.2 Gaps That Must Be Added to the Compiler

These are not library work — they affect correctness, fusion, and memory layout.

#### Gap 1: Stencil Loop Materialization
**What:** `StencilLowerPass` emits structured attributes but does not yet emit actual loop nests
in target IR. For a weather model, stencil.apply needs to lower to real loops (x86 scalar,
NVIDIA TMA-backed async, or CPU/GPU vectorized).

**Priority:** High. Without this, stencil ops are decorative in the IR.

**Effort:** 2–3 weeks. Start with x86 scalar path; hook TMA async for NVIDIA (Phase G dependent).

#### Gap 2: Boundary Condition Lowering
**What:** Semantic BC attributes (`periodic`, `dirichlet(v)`, `neumann(v)`, `reflect`) exist in
the dialect but have no lowering to concrete padding ops or ghost-region fills. For a regional
forecast model, incorrect boundary handling corrupts the solution.

**Priority:** High. Boundary errors are silent and numerically catastrophic.

**Effort:** 1–2 weeks (new pass in `src/transforms/lib/`). Periodic + reflect are the common cases.

#### Gap 3: Halo + Mesh Boundary Integration
**What:** Domain decomposition (`Block/Cyclic`) and halo inference (`HaloInferPass`) are
orthogonal subsystems. When a domain decomposes across ranks, rank boundaries must become
halo exchange boundaries automatically. There is no mechanism today that connects
`ShardSpec` to `tessera.neighbors.halo.region`.

**Priority:** High for distributed runs, medium for single-GPU.

**Effort:** 2–3 weeks. Requires `DistributionLoweringPass` to emit halo exchange around
stencil.apply when partition crosses multiple ranks.

#### Gap 4: 2D Local Window Attention
**What:** `attn_sliding_window` operates over a 1D sequence axis. Spatial weather grids
need 2D window attention (a query patch attends to an H×W neighborhood). This is structurally
a stencil over the KV plane.

**Priority:** Medium. Library code can express this as nested 1D sliding windows as a
workaround, but compiler-visible 2D windows enable better tiling and fusion.

**Effort:** 1 week to register the op + VJP/JVP; 2–3 weeks for efficient tile lowering.

#### Gap 5: Async Halo Transport Kernels
**What:** `PipelineOverlapPass` assigns stream IDs for overlapping halo exchange with
compute, but does not emit the actual pack/exchange/unpack kernels. These are optimization-
relevant (they determine whether the halo stalls compute or runs concurrently).

**Priority:** Medium. Without this, the pipeline overlap pass is a no-op at runtime.

**Effort:** 3–4 weeks (backend-specific: CPU memcpy/barrier, NVIDIA DMA staging, ROCm).

---

## 3. Library Work — What Belongs in `tessera.sciml` / `tessera.fields`

None of the following need compiler awareness. They compile through normal Tessera IR.

### 3.1 Data Model

```python
# tessera.fields — thin wrappers over DistributedArray
class Field2D:    # (H, W, C) tensor with domain metadata
class Field3D:    # (D, H, W, C) tensor
class FieldBatch: # (B, ...) ensemble stack

# Static conditioning
class TerrainLayer:   # elevation, slope, aspect, land mask
class ClimatologyLayer: # seasonal means, variances
```

These are plain Python classes that construct `DistributedArray` instances with appropriate
`ShardSpec` and domain annotations. Zero compiler change needed.

### 3.2 Neural Network Blocks

All of the following are standard conv/attention/norm compositions that
compile through existing Graph IR ops:

- Conv blocks: standard, depthwise, separable, dilated
- Norm/activation/residual (LayerNorm, RMSNorm, GroupNorm already in `nn/`)
- U-Net encoder/decoder blocks
- Upsample/downsample (bilinear, transposed conv, pixel shuffle)
- Window attention over spatial grids (using `attn_sliding_window` or 2D variant once added)
- Axial attention (row-wise + column-wise — expressible as two 1D attentions)

The one exception: if the library wants depthwise conv to fuse with norm/activation,
the compiler needs to recognize that pattern. Add a `DepthwiseConvFusion` Graph IR
canonicalization rather than baking depthwise into the compiler as a first-class concept.

### 3.3 Diffusion Sampler

```python
# tessera.sciml.diffusion
class NoiseSched:    # DDPM / DDIM / EDM noise schedules
def denoise_step(x_t, t, model_fn, noise_sched): ...
def classifier_free_guidance(uncond, cond, w): ...
```

The iterative structure of diffusion (loop over T denoising steps) compiles through
`tessera.control.fori_loop` or `while_loop`. The autoregressive/iterative scheduling
aspect is compiler-visible via the loop's effect annotation — `memory` effect if each
step reads the previous output. Checkpointing across denoising steps hooks into the
existing `remat` infrastructure.

**Compiler interaction points (not new compiler work):**
- `RNG` effect: each denoising step draws noise → compiler assigns a per-step stream
- `remat`: checkpoint intermediate activations rather than materializing T full-resolution fields
- Loop-level scheduling: `fori_loop` emits structured `scf.for` in Graph IR, which the
  `PipelineStageInsertionPass` can pipeline across ensemble members

### 3.4 Domain-Specific Losses and Metrics

```python
# tessera.sciml.losses
def weighted_mse(pred, target, weights): ...
def crps(pred_ensemble, target): ...          # via sort + rank operations
def spectral_loss(pred, target, wavenumbers): ... # calls tessera.ops.rfft
def gradient_loss(pred, target): ...          # calls tessera.ops.conv2d with finite-diff kernel
def masked_skill(pred, target, mask): ...     # regional evaluation

# tessera.sciml.verification
def by_elevation_bin(skill_fn, pred, target, dem, bins): ...
def by_region(skill_fn, pred, target, regions): ...
```

These are ordinary Tessera ops. `crps` decomposes into sort + rank-based reductions
(existing primitives). `spectral_loss` uses `tessera.ops.rfft` (already complete with VJP/JVP).
`gradient_loss` uses a conv2d with finite-difference weights.

### 3.5 Dataset Adapters

```python
# tessera.sciml.data — wraps tessera.data (S15)
class HRRRDataset(tessera.data.Dataset): ...
class ERA5Dataset(tessera.data.Dataset): ...
class WRFDataset(tessera.data.Dataset): ...
class NetCDFAdapter: ...
class ZarrAdapter: ...
```

These should live outside core Tessera (`tessera.sciml.data`) and possibly in a separate
`tessera-geo` package. The S15 `Dataset` / `ShardedDataset` combinators are the correct
foundation. NetCDF/Zarr I/O is handled by external libraries; the adapter presents data as
`DistributedArray` batches.

---

## 4. Runtime Responsibilities — Already Scoped Correctly

The runtime responsibilities the user listed map directly to existing Phase 4–8 runtime work:

| Responsibility | Current Status |
|----------------|----------------|
| Device memory allocation + reuse | ✅ `tsrMalloc`, RAII Metal buffer pool (Phase 8) |
| Static field caching on device | 🟡 Planned (static conditioning arrays) |
| Async streams/events | ✅ CUDA/HIP streams; Apple GPU CommandQueue |
| Prefetch + compute/data overlap | 🟡 `PipelineOverlapPass` assigns streams; transport unimplemented |
| Multi-GPU domain decomposition | ✅ Reference layer; real NCCL pending Phase G |
| Halo exchange | 🟡 Dialect-level; runtime transport missing (Gap 5) |
| Collective communication | ✅ Reference; NCCL/RCCL wiring post-Phase G |
| Deterministic RNG stream state | ✅ Philox per-rank; `to_state`/`from_state` (S4) |
| Checkpoint/restart | ✅ `tessera.checkpoint` (S12); save/load/sharded |
| Profiling/tracing | ✅ `tprof`; Perfetto export; roofline tools |
| Schedule cache / autotune results | ✅ SQLite cache v2 (BayesianAutotuner) |

Static field caching (terrain, climatology) is the one meaningful gap: the runtime
should be able to pin a `Field2D` to device and reuse it across inference steps without
re-uploading. This is a runtime metadata annotation (`persistent=True` buffer, already
sketched in `Module.register_buffer`) — no compiler change needed.

---

## 5. NVL72 Scale-Out — Current State vs. What the Stack Needs

### 5.1 What's Already First-Class

The distributed training architecture for NVL72-scale runs is fully designed and
the reference layer is complete:

| Component | Status |
|-----------|--------|
| Named mesh abstraction (`NamedMesh`) | ✅ Complete |
| Partition specs (DP × TP × PP) | ✅ Complete |
| DDP / FSDP wrappers | ✅ Complete against mock collectives |
| Graph IR collective insertion | ✅ Complete (F5) |
| Collective VJP/JVP rules | ✅ Complete (psum/pmean/pmax/pmin/collective_permute) |
| Pipeline parallelism (1F1B) | ✅ Complete (`PipelinePlan`, `PipelineStageInsertionPass`) |
| ZeRO stage 2 | ✅ Complete (`OptimizerShardPass`) |

### 5.2 Critical Blocker: Phase G (NVIDIA GPU Execution)

**The entire distributed story is wire-ready but cannot execute** until Phase G lands.
DDP/FSDP are defined and correct; they call mock collectives. The moment you need real
NVIDIA hardware, every distributed operation stalls because there is no GPU kernel launch
path in `runtime.py` for NVIDIA targets.

**Phase G critical path** (from `docs/audit/backend/nvidia/NVIDIA_AUDIT.md`):
1. Python GPU dispatcher in `jit.py` (`_execute_nvidia_gpu_artifact`) — 2–3 days
2. `nvidia_sm90` branch in `runtime.py` — 2–3 days
3. Real NCCL wiring (post-G1–G5) — 1–2 weeks
4. First H100 BF16 GEMM verified — ~1 week hardware time

### 5.3 NVL72-Specific Gaps for First-Class Support

The NVL72 appendix (see `docs/programming_guide/Tessera_Programming_Guide_Appendix_NVL72.md`) describes a
72-GPU system with 9 NVSwitch domains × 8 GPUs/domain, 1.8 TB/s bisection bandwidth.
To make NVL72 a first-class target:

| Capability | Current | Gap |
|-----------|---------|-----|
| Multi-axis shard_map (DP+TP+PP simultaneously) | Single-axis only | Needs extension — 2–3 weeks |
| Topology-aware collective routing | Designed, not implemented | No cost model for NVSwitch vs. NVLink latency |
| Hierarchical collectives (ring-allgather + SHARP) | Sketched in docs | No Tessera-native orchestration |
| Dynamic mesh reshaping | Not implemented | Post-Phase G |
| Intra-domain vs. inter-domain scheduling | Not implemented | Post-Phase G |

### 5.4 What This Stack Specifically Needs at NVL72 Scale

For a CorrDiff-like ensemble model at NVL72 scale, the relevant parallelism axes are:

```
DP axis: ensemble members (B ensemble × N GPU domains)
TP axis: spatial tile decomposition (8 GPUs per domain share a regional tile)
No PP: diffusion models don't have clean pipeline stages (each step is full-model)
```

This is primarily **data-parallel across ensemble members** with **tensor-parallel
within a domain** for spatial tiling. The current `shard_map` can express this
today with two `NamedMesh` axes. The gap is execution, not design.

**Halo exchange at NVL72 scale** is intra-domain (NVLink, 900 GB/s, ~2µs). The
`PipelineOverlapPass` stream-ID scheme already models this correctly — it just needs
the actual transport code (Gap 5 above).

### 5.5 Recommended: Make These Two Things First-Class Now

Rather than treating NVL72 as a future milestone, two specific changes would make the
distributed story first-class for this workload before Phase G even lands:

**1. Multi-axis `shard_map`:** Extend the current single-axis reference `shard_map` to
handle a 2D mesh (e.g., `NamedMesh(dp=9, tp=8)`) where the DP axis shards ensemble
members and the TP axis shards spatial tiles. This is pure Python reference work —
no compiler changes needed — and it unblocks correct distributed simulation even on CPU.

**2. Ensemble-aware RNG stream assignment:** The current `stream_id = global_seed × num_ranks + rank`
formula works for DP-only parallelism. For an ensemble diffusion model, each ensemble
member needs an independent but deterministic noise trajectory. The library API would be:

```python
ensemble_key = tessera.rng.from_seed(seed).split(ensemble_size)  # already works
# but the compiler needs to route each member's key to its DP-assigned shard
```

This is a `fold_in(key, member_id)` call at the library level — already supported —
but the runtime needs to know which shard holds which member for stream assignment.
This is a metadata annotation, not a new primitive.

---

## 6. Final Prioritized Gap List

### Compiler Gaps (in order of impact)

1. **Stencil loop materialization** — High priority. Unblocks the entire halo/stencil path.
2. **Boundary condition lowering** — High priority. Silent correctness risk without it.
3. **Halo + mesh boundary integration** — High priority for distributed; medium for single-GPU.
4. **Multi-axis `shard_map`** — High priority for NVL72 ensemble parallelism.
5. **2D local window attention** — Medium. Workaround exists (nested 1D); native is better.
6. **Async halo transport kernels** — Medium. Pipeline overlap is no-op without this.
7. **`DepthwiseConvFusion` Graph IR pattern** — Low. Only needed if depthwise+norm is hot.

### Runtime Gaps

1. **Phase G: NVIDIA GPU execution** — Critical blocker for everything GPU-related.
2. **NCCL/RCCL wiring** — Follows directly from Phase G; 1–2 weeks post-G.
3. **Static field pinning** — Small but meaningful for inference: terrain/climatology on device.

### Not Needed in the Compiler

The following were correctly identified as library work and require no compiler changes:

- CorrDiff / regional super-resolution model definitions
- Diffusion noise schedules and sampler steps
- Conv/attention/norm/residual blocks (they use existing Graph IR ops)
- All weather-domain losses (CRPS, spectral loss, gradient loss)
- Terrain, land mask, climatology data models
- Dataset adapters (HRRR, ERA5, WRF, NetCDF, Zarr)
- Meteorological formulas
- Evaluation, visualization, forecast product generation

---

## 7. Suggested Package Structure

```
tessera.sciml/                   # New package (library, not compiler)
  fields/
    __init__.py                  # Field2D, Field3D, FieldBatch
    terrain.py                   # TerrainLayer, ClimatologyLayer, elevation bins
    conditioning.py              # static_field_cache, patch_extract, tile_stitch
  models/
    unet.py                      # U-Net encoder/decoder blocks
    corrdiff.py                  # CorrDiff-like architecture definition
    super_resolution.py          # Regional SR model wrappers
  diffusion/
    noise_sched.py               # DDPM, DDIM, EDM schedules
    sampler.py                   # denoise_step, cfg_guidance, ensemble_sample
  losses/
    weather.py                   # weighted_mse, crps, spectral_loss, gradient_loss
    verification.py              # masked_skill, by_elevation, by_region
  data/                          # Thin wrappers over tessera.data (S15)
    hrrr.py
    era5.py
    wrf.py
    adapters.py                  # NetCDF, Zarr → DistributedArray

tessera/                         # Core compiler — additions only
  compiler/
    transforms/
      StencilMaterializePass.cpp  # Gap 1: loop materialization
      BoundaryConditionLowerPass.cpp  # Gap 2: BC lowering
      HaloMeshIntegrationPass.cpp  # Gap 3: halo + decomposition join
  sharding.py                    # Gap 4: extend shard_map to multi-axis
  nn/functional.py               # Gap 5: add attn_2d_local_window op
```

---

---

## 8. Helios / MI455x — AMD Target Additions Required

> This section covers what needs to happen in Tessera to support the Helios cluster
> running AMD MI455x accelerators alongside the NVL72 / NVIDIA path.

### 8.1 Current AMD/ROCm State

Tessera's ROCm story is **compiler-complete but runtime-incomplete** — the same structural
position as NVIDIA before Phase G. Specifically:

| Layer | Status |
|-------|--------|
| AMDArch enum + feature matrix | ✅ GFX_90A → GFX_1200; MFMA shapes for all six archs |
| ROCm toolchain pin | ✅ ROCm 7.2.3 / HIP 7.2.3 |
| HIP memory API (malloc/memcpy/streams/events) | ✅ `hip_backend.cpp` |
| MFMA table generation | ✅ `generate_mfma_table.py` regenerates from `rocm_target.py` |
| H-series pre-work (H-1 through H-8) | ✅ Complete (toolchain pins, lit fixtures, RCCL pin) |
| **HIP kernel launch** | ❌ `launchHostKernel()` explicitly returns `TSR_STATUS_UNIMPLEMENTED` |
| **Runtime dispatch path** | ❌ No `_execute_rocm_artifact()` in `runtime.py` |
| **JIT fast-call path** | ❌ No `_execute_rocm_metadata()` in `jit.py` |
| **Roadmap priority** | 🔲 Deferred post-Phase G; no Phase H-ROCm track exists yet |

There is one specific design debt: `launchHostKernel()` in `hip_backend.cpp` returns
UNIMPLEMENTED because the current `tsrLaunchParams` contract cannot be satisfied via
`hipLaunchHostFunc` (which carries only one `void*` payload). This is not a conceptual
blocker — it requires extending the launch params ABI to pass a serialized args bundle
via a `hipModuleLaunchKernel` / `hipExtModuleLaunchKernel` call instead of the host-func
shortcut — but it must be explicitly resolved before any ROCm kernel runs.

### 8.2 GFX1250 Architectural Profile — What LLVM Tells Us

GFX1250 is confirmed in LLVM (initial stub: commit
[llvm/llvm-project@6997465](https://github.com/llvm/llvm-project/commit/69974658f079cec82a9fc13dd4993ab1e072c811),
Sept 2025 follow-on adding `cluster_load_async_to_lds`). The architecture has several
properties that matter for Tessera:

| Property | GFX1250 | Existing CDNA (gfx90a–gfx950) |
|----------|---------|-------------------------------|
| Wavefront size | **Wave32** | Wave64 |
| Matrix instruction | **WMMA** (not MFMA) | MFMA |
| Async memory | `cluster_load_async_to_lds` | `lds_async_copy` |
| LDS atomic barriers | 29-bit count + 3-bit phase (new) | Standard |
| Feature set base | `+gfx12-insts`, `+gfx1250-insts` | `+gfx9-insts` |
| Chip type | Chiplet (2nm compute + 3nm I/O) | Monolithic/multi-die |

**The key implication:** GFX1250 is structurally more like GFX_1100 (RDNA3, Wave32, WMMA)
and GFX_1200 (RDNA4, Wave32, WMMA) than like the CDNA line. Tessera already has the
WMMA-capable path for GFX_1100/GFX_1200 — GFX_1250 extends that, not the MFMA path.

The `_MFMA_VARIANTS` table is **not the right home** for GFX1250 shapes. The entry will
be empty (like GFX_1100/GFX_1200 today), and the WMMA instruction shapes belong in a
`_WMMA_VARIANTS` table that doesn't yet exist in `rocm_target.py`.

### 8.3 Adding GFX1250 to Tessera

**Code changes — `rocm_target.py` (enum registration, ~half day):**

```python
class AMDArch(IntEnum):
    GFX_90A  = 90    # MI250, CDNA 2, Wave64, MFMA
    GFX_940  = 940   # MI300A, CDNA 3, Wave64, MFMA
    GFX_942  = 942   # MI300X, CDNA 3, Wave64, MFMA
    GFX_950  = 950   # MI325X, CDNA 4, Wave64, MFMA+F4/F6
    GFX_1100 = 1100  # RDNA 3, Wave32, WMMA
    GFX_1200 = 1200  # RDNA 4, Wave32, WMMA+FP8
    GFX_1250 = 1250  # MI455x (Helios), Wave32, WMMA — NEW

_WAVEFRONT_SIZE = {
    ...,
    AMDArch.GFX_1250: 32,  # same as RDNA path
}

_ROCM_ARCH_STRINGS = {
    ...,
    AMDArch.GFX_1250: "gfx1250",
}

# MFMA table: empty for GFX_1250 (uses WMMA, not MFMA)
_MFMA_VARIANTS = {
    ...,
    AMDArch.GFX_1250: [],  # WMMA-only arch
}

# NEW table needed — parallel to _MFMA_VARIANTS
_WMMA_VARIANTS = {
    AMDArch.GFX_1100: [
        {"M": 16, "N": 16, "K": 16, "dtype": "fp16"},
        {"M": 16, "N": 16, "K": 16, "dtype": "bf16"},
        {"M": 16, "N": 16, "K": 16, "dtype": "int8"},
    ],
    AMDArch.GFX_1200: [
        # GFX12 adds FP8
        {"M": 16, "N": 16, "K": 16, "dtype": "fp16"},
        {"M": 16, "N": 16, "K": 16, "dtype": "bf16"},
        {"M": 16, "N": 16, "K": 32, "dtype": "fp8_e4m3"},
        {"M": 16, "N": 16, "K": 32, "dtype": "fp8_e5m2"},
    ],
    AMDArch.GFX_1250: [
        # Confirm exact shapes from AMD ISA guide for MI455x
        # Placeholder based on GFX12 baseline — update when spec confirmed
        {"M": 16, "N": 16, "K": 16, "dtype": "fp16"},
        {"M": 16, "N": 16, "K": 16, "dtype": "bf16"},
        {"M": 16, "N": 16, "K": 32, "dtype": "fp8_e4m3"},
        {"M": 16, "N": 16, "K": 32, "dtype": "fp8_e5m2"},
        # MI455x may also support scaled_wmma or block-scaled variants
    ],
}

_ROCM_7_2_FEATURES = {
    ...,
    AMDArch.GFX_1250: {
        "mfma": False,        # WMMA arch, not MFMA
        "wmma": True,
        "wmma_f8": True,      # confirm
        "lds_async_copy": False,
        "cluster_load_async_to_lds": True,  # confirmed in LLVM
        "fp8_conversion": True,
        "gfx12_insts": True,
        "gfx1250_insts": True,
        # confirm: atomic barrier instructions (29-bit count + 3-bit phase)
    }
}
```

**Schema extension — `generate_mfma_table.py` and `mfma_table.inc`:**

The existing X-macro `TESSERA_MFMA_VARIANT(arch_id, arch_name, M, N, K, K_blocks)` only
covers MFMA. Need a parallel macro for WMMA:

```c
// mfma_table.inc addition
TESSERA_WMMA_VARIANT(1250, gfx1250, 16, 16, 16, fp16)
TESSERA_WMMA_VARIANT(1250, gfx1250, 16, 16, 16, bf16)
TESSERA_WMMA_VARIANT(1250, gfx1250, 16, 16, 32, fp8_e4m3)
TESSERA_WMMA_VARIANT(1250, gfx1250, 16, 16, 32, fp8_e5m2)
```

The C++ lowering pass (`MFMAFullCoveragePass`) reads `mfma_table.inc` — it needs a
parallel `WMMALoweringPass` that reads the WMMA table for GFX_1100/GFX_1200/GFX_1250.

**Capabilities entry:**

```python
TARGET_CAPABILITIES["rocm_gfx1250"] = TargetCapability(
    family="rocm", arch="gfx1250", runtime_backend="hip",
    wavefront_size=32,
    available=False,  # flip to True when hardware is in CI
    matrix_instruction="wmma",  # not "mfma"
    ...
)
```

**What to confirm from hardware / AMD ISA doc before finalizing:**
- Exact WMMA shapes for MI455x (M×N×K per dtype) — GFX12 shapes are a reasonable baseline
- Whether `scaled_wmma` (block-scaled variant) is supported, and its shape table
- ROCm version required (7.2.3 may cover initial support; confirm)
- LDS bytes and max waves per CU for MI455x specifically
- Interconnect: Pensando Vulcano NIC bandwidth (for `ChunkPlanner.cpp` chunk size tuning)

### 8.3 Phase H-ROCm: A Formal Track Is Needed

The NVL72 path has Phase G as its execution unblock. Helios/MI455x needs an equivalent
**Phase H-ROCm** track. Suggested tasks, mirroring the Phase G audit structure:

| Task | Description | Effort |
|------|-------------|--------|
| H-ROCm-1 | Fix `launchHostKernel()` — extend `tsrLaunchParams` ABI for `hipModuleLaunchKernel` | 2–3 days |
| H-ROCm-2 | Add `_execute_rocm_artifact()` to `runtime.py` (mirror Apple GPU path) | 2–3 days |
| H-ROCm-3 | Add `_rocm_fast_call()` to `jit.py` (mirror `_apple_gpu_fast_call`) | 1–2 days |
| H-ROCm-4 | Wire `hipModule` loading + function lookup from compiled HSA/HSACO artifact | 3–4 days |
| H-ROCm-5 | First BF16 GEMM on real MI3xx/MI455x hardware (validation) | 1 day hardware |
| H-ROCm-6 | RCCL wiring (parallel to NCCL post-Phase-G; adapters are in `src/collectives/`) | 1–2 weeks |
| H-ROCm-7 | ROCm CI spine (`validate.sh --rocm`) | 1 week |

Estimated timeline: **3–4 weeks to first kernel + 2 weeks RCCL** = ~6 weeks to distributed
training on Helios, assuming hardware access from day one.

### 8.4 Helios Interconnect and Collective Strategy

For ensemble diffusion at Helios scale, the parallelism axes are the same as NVL72:

```
DP axis: ensemble members (1 member per AMD node/partition)
TP axis: spatial tile decomposition (within a node, multiple MI455x share a regional tile)
```

Key questions to confirm with the hardware team:
- Interconnect: Infinity Fabric (same-node) + Ethernet/InfiniBand (inter-node)?
  RCCL handles both; its collective strategy differs from NCCL in flat-allreduce topology.
- NPS (NUMA per socket) configuration on MI455x nodes — affects `ShardSpec` alignment
- Available HBM bandwidth (determines whether halo exchange saturates memory bus before
  NIC; affects the `PipelineOverlapPass` stream-overlap value)

The `ChunkPlanner.cpp` chunk sizes (`NVLink=512KiB`, `PCIe=128KiB`, `RDMA=256KiB`) will
need a new `"infinity_fabric"` entry once the MI455x interconnect bandwidth is confirmed.

### 8.5 Dual-Target CI Strategy

With both NVL72 (NVIDIA) and Helios (AMD) as real execution targets, CI should be structured as:

```
validate.sh --cpu       # always, fast, no hardware required (current)
validate.sh --gpu       # Phase G gate: NVIDIA H100
validate.sh --rocm      # Phase H-ROCm gate: AMD MI455x / Helios
```

The lit fixture suites are already partitioned:
- `tests/tessera-ir/phase3/cuda13/` — NVIDIA WGMMA/TMA (8 fixtures)
- `tests/tessera-ir/phase8/rocm_7_2/` — AMD MFMA (6 fixtures)
- `tests/tessera-ir/phase8/` — Apple (4 fixtures; not relevant for cluster CI)

Add a `tests/tessera-ir/phase8/rocm_mi455x/` directory for MI455x-specific MFMA patterns
once the ISA is confirmed.

---

*See also:*
- `docs/audit/roadmap/ROADMAP_AUDIT.md` — Phase G punch list and acceptance criteria
- `docs/audit/backend/nvidia/NVIDIA_AUDIT.md` — Phase G1–G8 per-task breakdown
- `python/tessera/compiler/rocm_target.py` — AMDArch enum + MFMA variants
- `src/backend/hip_backend.cpp` — HIP memory/stream APIs; kernel launch gap
- `scripts/generate_mfma_table.py` — MFMA table codegen
- `src/compiler/tessera_neighbors/` — Halo/stencil dialect (Phase 7)
- `python/tessera/sharding.py` — NamedMesh, shard_map reference
- `docs/programming_guide/Tessera_Programming_Guide_Appendix_NVL72.md` — 72-GPU topology design
