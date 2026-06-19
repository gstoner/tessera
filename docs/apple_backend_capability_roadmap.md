# Apple Backend — Capability Roadmap (engine · kernels · optimization)

> Status: **planning (2026-06-19).** A consolidated forward plan for the Apple
> Silicon backend (CPU via Accelerate/BNNS, GPU via MPS / MPSGraph / MSL),
> derived from the [MLX ecosystem survey](apple_gpu_mlx_ecosystem_survey.md) and
> reconciled against what is *already in-tree*. The serving / app / dflash
> production track is a **separate plan** —
> [`tessera_inference_serving_plan.md`](tessera_inference_serving_plan.md).
>
> **Scope boundary.** This doc covers compiler/runtime/kernel capability for the
> Apple backend. It does NOT cover the OpenAI/Anthropic server, EnginePool,
> tiered prefix cache, model registry, or the Swift app — those are the serving
> plan. Items here make the *engine* faster, broader, and more general.

---

## 1. Grounding — what already exists (verified in-tree, do not re-plan)

Before reading the roadmap, the bar is high. Already landed on Apple GPU:

- **Fused op surface** parity-or-ahead of `mx.fast`: rmsnorm / layer_norm / rope
  / softmax / log_softmax / silu_mul / gelu + flash_attn with the **`attn_bias`
  substrate** (MLX's SDPA has no bias), flash_attn_gqa (native KV-group, no
  repeat-KV), fused `bsmm` (`softmax(QKᵀ·scale)V` one dispatch).
- **MLA decode, fully built out:** compressed-KV, decoupled-RoPE, **weight
  absorption** (per-head K/V never materialized, ~8.9× smaller cache),
  `{f32,f16,bf16}` matrix, paged (`MLAPagedDecoder`) + vLLM-style block-table
  (`MLABlockPagedCache`) caches with same-length `B>1` batching.
- **Tier-2/3 ops:** `bmm` (broadcast B), `qkv_projection`, `linear_general`,
  batched-MHA (unbounded head_dim), reductions/scan/argreduce, conv2d/conv3d.
- **MPSGraph lane** (Tier-1 activations/norms, no N≤256 limit) + graph caching
  (`mpsgraph_cache_size()`), **GPU dequant-matmul** (`dequant_matmul_f32`),
  Gumbel sampler.
- **Infra:** RAII Metal buffer pool, MLIR Tile→Apple lowering aligned to the
  runtime envelope (G3 enforcer test), 264 C ABI symbols, MLIR 22.1.6 build.

The survey's "gaps" are narrowed accordingly. What is genuinely missing is below.

---

## 2. The five capability pillars

```
P1 Runtime scheduler        — kill the sync-bound, host-orchestrated decode loop
P2 General fusion compiler  — replace hardcoded longest-chain pass matching
P3 Quantization execution   — fused packed-weight quant matmul (not dequant-then-GEMM)
P4 Kernel-gen automation    — [[host_name]] template instantiation; precompiled metallib
P5 Memory & introspection   — public budget ABI; residency; in-place shared-buffer wrap
```

Each pillar below: **what / why / concrete work / acceptance / effort / risk**.
Sequencing rationale in §3.

---

## P1 — Runtime scheduler (lazy tape + command-buffer batching)

**What.** Replace the per-op, host-orchestrated dispatch (each op:
`newBufferWithBytes` upload → run → `readBytes` download into numpy) with a lazy
tape + GPU-resident activations + multi-op command-buffer batching.

**Why.** The [resident-activations plan](apple_gpu_resident_activations_plan.md)
already proved the per-token decode loop is **sync-bound, not compute-bound**.
MLX's `status`-enum tape + `Fence` + auto-RAW-barrier + multi-dispatch-per-buffer
is the canonical fix, and unified memory means the "transfer" is only a redundant
memcpy — wrap the numpy memory as a shared `MTLBuffer` in place.

**Concrete work.**
1. **In-place shared-buffer wrap** (prerequisite, also P5): adopt host numpy as
   `MTLResourceStorageModeShared` `MTLBuffer` without copy; eliminate redundant
   `newBufferWithBytes`/`readBytes`. (This is the single highest-leverage change.)
2. **GPU-resident value handles:** a `values` dict entry can be a device buffer,
   not numpy; ops consume/produce handles; download only at boundaries.
3. **Lazy tape + batched command buffer:** accumulate a recognized op-chain into
   one `MTLCommandBuffer`, insert RAW barriers from input/output tracking, commit
   at op-count/MB threshold or at a forced eval (download / control flow).
4. **Fence-based cross-encoder deps** mirroring MLX's `stream.outputs` map.

**Acceptance.** Decode-loop tokens/sec rises materially on a fully-fused decode
(the Gumbel-sampler finding becomes compute-bound); a microbench shows N chained
ops dispatch in 1 command buffer with 0 intermediate host round-trips; existing
numerics unchanged (bit-exact vs current per-op path).

**Effort:** High · **Risk:** Med (correctness of barrier insertion). **This is
the backbone item** — P2 fusion and the serving decode loop both compound on it.

---

## P2 — General element-wise fusion compiler

**What.** A graph-driven fuser (MLX `Compiled` + `compile_simplify`) that fuses
arbitrary element-wise chains into one synthesized MSL kernel, replacing
Tessera's hardcoded longest-chain matching in pass ordering.

**Why.** Today fusion is enumerated by hand (matmul→softmax→matmul, matmul→gelu,
…). It doesn't generalize to new chains, and every chain is bespoke. A real fuser
turns "add a fused kernel" into "it just fuses."

**Concrete work.**
1. **Fusion region detection** over Graph IR: same-stream, compatible-shape,
   element-wise (+ broadcast-duplication via split) maximal subgraphs; depth/arity
   caps (`max_compile_depth`, `max_compile_arrays`).
2. **CSE + DCE** over the fused region (`compile_simplify`).
3. **MSL emission** from the region using the P4 template generator; cache keyed
   by region signature + dtype (`shapeless` → key on rank).
4. Wire as a `tessera-lower-to-apple_gpu-runtime` pass ahead of the existing
   hand-fusion passes; hand-fusions become a fast-path the fuser can still match.

**Acceptance.** A novel element-wise chain (e.g. `silu(x)*sigmoid(y)+z`) with no
hand-written kernel fuses to one dispatch and matches numpy; the existing fused
chains still hit their fast paths; evaluator horizontal oracle (fused ≡ unfused)
passes on generated kernels.

**Effort:** High · **Risk:** Med. **Depends on P4** (emission) and benefits from
P1 (resident handles between fused regions).

---

## P3 — Fused packed-weight quantization execution

**What.** A real on-GPU `quantized_matmul` over **packed** weights (uint8, 2/4/8
vals/byte, per-group scale/bias), not the current dequant-to-fp32-then-GEMM.

**Why.** Tessera already *names* the dtypes (`nvfp4`, `fp4_e2m1`, `fp6_*`,
`fp8_*`, planned `mxfp*`) and ships `dequant_matmul_f32`, but dequant-then-GEMM
loses the bandwidth win that is the entire point of quantization on
memory-bound Apple decode. MLX's affine + MXFP4(g32) + NVFP4(g16) algorithm and
QMV/QVM/QMM dispatch are turnkey and map onto the already-named dtypes.

**Concrete work.**
1. **Packing/storage** in `quantization.py` (S9): uint8 bit-packing + per-group
   `(scale, bias)` layout; affine quant kernel (SIMD min/max → scale → bias →
   round). Reuse canonical dtype names.
2. **Dimension-specialized MSL kernels** — QMV (decode: small M, transposed),
   QVM (split-K for K≥1024), QMM (prefill: large M, 32×32×32 blocks). Native
   variant gated on Metal 4 / M3+ (`qmm_t_nax` analog) — cross-ref
   [`apple_gpu_metal4_adoption.md`](apple_gpu_metal4_adoption.md) and
   [`ldt_primitives_metal4_mapping.md`](ldt_primitives_metal4_mapping.md).
3. **C ABI symbols** `tessera_apple_gpu_quantized_matmul_{affine,mxfp4,nvfp4}_*`
   + `runtime.py` dispatch + envelope + driver gating.
4. **VJP** (affine: input/scale/bias grads; MXFP/NVFP: scale-grad only) — Tessera
   already has int4/int8 STE in `quantization.py`.

**Acceptance.** 4-bit affine GEMM on GPU matches a dequant-reference at quant
tolerance, reads ~1/4 the weight bytes (measured), and beats `dequant_matmul`
latency at decode shapes; MXFP4/NVFP4 produce correct packed layouts. KV-cache
quant (already present) stays orthogonal.

**Effort:** Med–High · **Risk:** Low–Med (algorithm is well-specified). **Largely
independent** of P1/P2 — can proceed in parallel. Highest direct decode-perf ROI.

---

## P4 — Kernel-generation automation

**What.** (a) An `[[host_name]]`-style **template-instantiation generator** that
emits the `{f32,f16,bf16}` (and quant) MSL variants from one functor + body, and
(b) an optional **precompiled `.metallib`** build mode.

**Why.** Tessera hand-writes per-dtype kernel variants (26+ symbols × dtype).
MLX generates them from string templates with systematic names. (a) removes the
hand-maintenance and is a prerequisite for P2/P3 emitting variants; (b) cuts the
JIT cold-start (100–500 ms first-use today, per-`(source,entry)` sha256).

**Concrete work.**
1. **MSL template generator:** functor library (unary/binary ops) + body
   templates + `get_template_definition`-style instantiation → systematic kernel
   names; integrate with the existing `(msl_source, entry)` sha256 cache.
2. **Precompiled-metallib CMake mode** (`TESSERA_METAL_JIT=OFF`): bake known
   instantiations into one `.metallib`, load at runtime; JIT stays the default
   for novel/fused kernels.
3. Migrate existing hand-written dtype variants onto the generator incrementally
   (start with the activation/norm family).

**Acceptance.** Adding a new element-wise op yields all 3 dtype kernels with no
hand-written MSL; cold-start of a precompiled kernel is measurably lower than
JIT; generated kernels bit-match the hand-written ones they replace.

**Effort:** Med · **Risk:** Low. **Enabler** for P2 and P3.

---

## P5 — Memory & introspection

**What.** Public memory-budget ABI + residency + the in-place shared-buffer wrap
(shared with P1).

**Why.** MLX exposes `set_memory_limit` / `set_cache_limit` / `get_active_memory`
/ `get_peak_memory` / `ResidencySet`. Tessera's pool is RAII-solid but exposes
only `mpsgraph_cache_size()`. Serving (the other plan) needs a process memory
ceiling and active/peak introspection; P1 needs the in-place wrap.

**Concrete work.**
1. **Budget ABI:** `tessera_apple_gpu_{get_active,get_peak,set_memory_limit,
   set_cache_limit,clear_cache}` over the existing pool.
2. **Two-tier alloc** (<256 B heap + size-keyed cache) if profiling shows
   small-alloc churn.
3. **`ResidencySet`** integration for hot weights/caches.
4. **In-place shared-buffer wrap** (the P1 prerequisite) — list here too since it
   is a memory-subsystem change.

**Acceptance.** `get_active/peak_memory` track real allocation; a memory limit is
enforced (alloc fails / evicts predictably); the in-place wrap removes the
redundant upload/download memcpys (verified by instrument or counter).

**Effort:** Low–Med · **Risk:** Low. **Quick wins**; the budget ABI is a
dependency the serving plan's `ProcessMemoryEnforcer` consumes.

---

## 3. Sequencing

```
        ┌─────────────────────────────────────────────────────────┐
Wave A  │ P5 budget ABI + in-place shared-buffer wrap   (quick)    │
(found-  │ P4 template generator                          (enabler) │
 ation)  └───────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐   ┌──────────────────────┐
Wave B  │ P1 runtime scheduler / lazy tape │   │ P3 packed quant matmul│ (parallel,
(core)  │   (consumes the in-place wrap)   │   │   (independent)       │  high ROI)
        └────────────────┬─────────────────┘   └──────────────────────┘
                         │
        ┌────────────────▼─────────────────┐
Wave C  │ P2 general fusion compiler        │  (needs P4 emission + P1 handles)
        └───────────────────────────────────┘
```

- **Wave A (foundation, low risk):** P5 budget ABI + in-place wrap, and the P4
  template generator. Cheap, unblock everything, immediately useful (the wrap
  removes memcpys; the budget ABI feeds serving).
- **Wave B (core, parallelizable):** P1 runtime scheduler (the backbone) and P3
  packed-quant matmul (independent, highest direct decode ROI) can run
  concurrently.
- **Wave C:** P2 fusion compiler, once P4 emits variants and P1 provides resident
  handles between fused regions.

**Recommended first move:** the **in-place shared-buffer wrap** — it is the
prerequisite for P1, a P5 deliverable, removes real memcpy cost today, and is
low-risk. Then the P4 generator in parallel.

---

## 4. Smaller / opportunistic borrows (not pillars)

| Item | Source | Note |
|---|---|---|
| `fast.*` naming convention | mlx core | Cosmetic: separate fused-restricted-grad ops from `ops.*`. |
| Function-transform composition for higher-order grad | mlx core | Study vs tape-based `autodiff`; only if higher-order grad demand appears. |
| `fast.metal_kernel`-style user escape hatch | mlx core | A Metal-specific public custom-kernel surface (Tessera's `custom_call` is the generic analog). |
| Embedding head (`mean-pool → L2-norm → typed output`) | mlx-embeddings | Small `nn` surface over existing encoders. |
| VLM tower/projector/LM + `VisionFeatureCache` + `pixel_shuffle`/`interpolate` | mlx-vlm | Multimodal; overlaps the JEPA/DiffusionGemma graphs — coordinate with model-class roadmap. |
| `dynamic_batch` token-budget packing + Buffer/Stream split | mlx-data | S15 data-pipeline ergonomics; independent of GPU work. |

---

## 5. Out of scope (explicitly)

- The OpenAI/Anthropic server, EnginePool, tiered prefix cache, continuous
  batching scheduler, model download/registry, Swift app, dflash production
  envelope → [`tessera_inference_serving_plan.md`](tessera_inference_serving_plan.md).
- AIR bitcode codegen (Apple ships no public path — MLX confirms; revisit only on
  a perf wall).
- Training-side on-GPU RNG / dropout (would break Decision #18 bit-exactness;
  inference Gumbel sampler already shipped #18-safe).

---

## 6. Acceptance for "the roadmap is done"

Per-pillar acceptance above. The backend reaches a new tier when: (1) a fully-
fused decode loop is compute-bound, not sync-bound (P1); (2) novel element-wise
chains fuse automatically (P2); (3) 4-bit decode reads ~1/4 the weight bandwidth
on GPU (P3); (4) dtype variants are generated, not hand-written (P4); (5) the
runtime exposes memory budget + enforces a ceiling (P5). Track status inline in
this doc as items land (mirror the Tier-2/3 plan's ✅ convention) and surface the
all-up picture in `docs/audit/MASTER_AUDIT.md`.
