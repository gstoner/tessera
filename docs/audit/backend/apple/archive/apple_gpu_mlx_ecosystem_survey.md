# MLX Ecosystem Survey — ideas for the Tessera Apple backend

> Status: **reference / research snapshot (2026-06-19).** A survey of Apple's
> MLX ecosystem and adjacent projects, read with one lens: *what is extractable
> for Tessera's Apple Silicon CPU + GPU backend and its inference story?* This
> doc is the source material; the action plans derived from it live in
> [`apple_backend_capability_roadmap.md`](apple_backend_capability_roadmap.md)
> (engine / kernels / optimization) and
> [`docs/architecture/inference/serving.md`](../../../../architecture/inference/serving.md)
> (serving tier + embeddable app + dflash production envelope).
>
> **Repos surveyed:** `ml-explore/mlx`, `ml-explore/mlx-c`, `ml-explore/mlx-lm`,
> `ml-explore/mlx-data`, `Blaizzy/mlx-embeddings`, `Blaizzy/mlx-vlm`,
> `bstnxbt/dflash-mlx`, `jundot/omlx`.

---

## 0. The headline finding

**MLX independently confirms Tessera's core Apple-GPU thesis.** Apple's own
framework reaches the same conclusion as the
[`apple_gpu_codegen_path_constraint`] memory: there is no public LLVM→AIR path,
so **MSL-source synthesis + MPS where Apple ships kernels** is the only viable
Apple-GPU codegen path. MLX does not codegen via bitcode either. Tessera is on
the right road; the gaps are *orchestration* and a few *engine mechanisms*, not
architecture.

**Where Tessera stands vs. MLX:**

- **At / ahead of parity:** fused `fast.*`-equivalent ops (rmsnorm / layernorm /
  rope / softmax / silu_mul / flash_attn — plus the `attn_bias` substrate MLX's
  SDPA lacks, batched-MHA with unbounded head_dim, MLA weight-absorption + paged
  / block-table caches, the Llama-decoder E2E proof); the RAII Metal buffer pool;
  the autodiff contract surface (12-axis registry vs MLX's ~4 per-primitive
  transform hooks); proven dflash correctness.
- **Behind:** general element-wise fusion compiler; a *fused packed-weight*
  group-quant matmul (Tessera has dequant-then-matmul, not packed QMV/QVM/QMM);
  the lazy runtime scheduler / command-buffer batcher (the
  [`apple_gpu_resident_activations_plan`](apple_gpu_resident_activations_plan.md)
  sync-bound finding); and the **serving orchestration tier**.

---

## 1. MLX core (`ml-explore/mlx`) — engine internals

### 1.1 Lazy / deferred evaluation & unified memory
Every op returns an `array` wrapping an `ArrayDesc` (shape / strides / dtype /
producing `primitive` / inputs / siblings / shared buffer / `status` enum:
`unscheduled → evaluated → available`). The array *is* a graph node. `mx.eval()`
runs a 4-stage pipeline — DFS to discover deps, BFS to build a topological
"tape," execute in tape order dispatching `eval_cpu` / `eval_gpu` per primitive,
then detach evaluated arrays. Arrays live in unified memory; the *operation*
picks device/stream at eval time, no host↔device copy. Dynamic graph
construction → shape changes don't recompile.

> **Tessera angle.** Different model (Tessera = eager Python + decoration-time
> `@jit` to MLIR; MLX defers everything to `eval()`). MLX's `status` enum + tape
> + automatic detach + fence-based per-stream scheduling is a clean *runtime
> scheduler* design Tessera lacks — it dispatches each op discretely through
> `runtime.py`. This is the substrate the resident-activations plan needs.

### 1.2 Metal kernel strategy
- **Two build modes** (`MLX_METAL_JIT`): precompiled (all template
  instantiations baked into one `mlx.metallib`) vs JIT-from-embedded-strings
  (100–500 ms first-use).
- **Source generation:** `make_jit_source` preprocesses headers into
  string-returning C++ funcs; runtime concatenates `metal::utils()` +
  `metal::unary_ops()` functors + templated body, then `get_template_definition`
  emits explicit MSL instantiations via `[[host_name]]` / `[[kernel]]` with
  systematic names (`v_abs`, `gn4large_abs`, `ss_add`); GEMM uses hash-of-params
  names.
- **Caching / dispatch:** `metal::Device` holds the `MTLLibrary` cache (metal-cpp,
  not Obj-C); `CommandEncoder` tracks `all_inputs_` / `all_outputs_` for
  automatic RAW-barrier insertion and batches multiple dispatches per command
  buffer (committed at op-count / MB threshold); cross-encoder deps via `Fence`.
- **Fusion (`mx.compile`):** `Compiled` primitive fuses element-wise ops into one
  kernel (`max_compile_depth=11`, `max_compile_arrays=24`); `compile_simplify`
  does CSE + DCE; `CompilerCache` keys traced tapes by signature + constant
  hashes (`shapeless=True` keys on rank). **No LLVM→AIR** — MSL synthesis + MPS.

> **Tessera angle.** Validates the codegen-path thesis. Borrow: (a) a **general
> element-wise fusion compiler** (replace hardcoded longest-chain pass-order
> matching); (b) the **`[[host_name]]` template-instantiation generator** to kill
> per-dtype hand-written MSL variants; (c) the **precompiled `.metallib` build
> option** to cut cold-start; (d) **auto RAW-barrier + multi-dispatch-per-command-
> buffer** batching.

### 1.3 Primitive / transform design
Each `Primitive` implements `eval_cpu` / `eval_gpu` + `jvp` / `vjp` / `vmap` /
`output_shapes`. Transforms are **function-level (JAX-style)**: `mx.grad`,
`value_and_grad`, `jvp`, `vjp`, `vmap` take a function, return a function →
compose to arbitrary depth (`grad(grad(sin))`). Custom kernels via
`mx.fast.metal_kernel(name, input_names, output_names, source, template=[("T",
dtype)], …)` — user supplies the MSL *body*, MLX generates the full signature.

> **Tessera angle.** Tessera's contract surface is richer (12 axes vs ~4). MLX's
> **function-transform composition** is cleaner than the tape-based
> `autodiff.tape()` for higher-order grads. MLX's `fast.metal_kernel`
> auto-signature is a good model for a user-facing Tessera escape-hatch MSL
> kernel surface (`custom.py` is the analog but not Metal-specific).

### 1.4 Quantization
Affine (`w = scale·w_q + bias`, bits ∈ {2,3,4,5,6,8}, group ∈ {32,64,128}),
MXFP4 (g32, shared exponent, no bias), MXFP8, NVFP4 (g16). API: `mx.quantize`
→ `(w_q, scales, biases)`, `mx.dequantize`, `mx.quantized_matmul`, `GatherQMM`
(batched/indexed). Packed into `uint8` (2/4/8 vals/byte) + per-group scale/bias.
**Dimension-specialized kernels:** QMV (small M, transposed), QVM (small M,
split-K for K≥1024), QMM (large M, 32×32×32 blocks, `qmm_t_nax` for M3+).

> **Tessera angle.** Tessera *names* `nvfp4` / `fp4_e2m1` / `mxfp*` and has GPU
> **dequant-matmul** (`tessera_apple_gpu_dequant_matmul_f32`), but **no fused
> packed-weight quant matmul**. MLX's uint8 packing + per-group scale/bias +
> QMV/QVM/QMM dispatch is a turnkey design for a real `quantized_matmul` lane.

### 1.5 Allocator & memory budget
`MetalAllocator`: `malloc` / `free` / `set_memory_limit` / `set_cache_limit` /
`get_active_memory` / `get_peak_memory` / `clear_cache`. Two-tier: <256 B heap
(1 MB, capped buffers) + size-keyed `BufferCache<MTL::Buffer>`; `ResidencySet`
tracks residency; encoder-scoped temporaries freed in completion handlers.

> **Tessera angle.** Tessera already has the RAII pool (`TS_METAL_BUF_ACQUIRE`).
> Cheap win: add **public budget ABI** (`get_active/peak_memory`, cache limit) —
> Tessera only exposes `mpsgraph_cache_size()` today. Consider the <256 B-heap
> split + `ResidencySet`.

### 1.6 `mx.fast` namespace
`rms_norm`, `layer_norm`, `rope`, `scaled_dot_product_attention` — hand-fused
single-kernel primitives with per-op VJP/JVP and vector/full/fallback dispatch.

> **Tessera angle.** Near-exact overlap; Tessera is *ahead* in breadth
> (attn_bias, unbounded head_dim, Llama-decoder E2E). Borrow only the **naming
> convention**: a stable `fast.*` namespace signalling "fused, restricted grads,"
> separate from general `ops.*`.

---

## 2. mlx-c — the C API binding layer

- **Object model:** every object is `typedef struct { void* ctx; } mlx_array;`
  (handle-by-value, heap payload) — ABI-stable as the C++ class evolves. Same for
  `mlx_stream`, `mlx_vector_array`, `mlx_map_string_to_array`, `mlx_closure`.
- **Ownership:** `mlx_array_new_data(...)`, `mlx_array_new_data_managed(data, …,
  dtor)` (**adopt a foreign buffer + its destructor → zero-copy**),
  `mlx_array_free`, `mlx_array_set(dst*, src)` (rebind via refcount).
- **Errors:** global settable handler `mlx_set_error_handler(fn, void* data,
  dtor)`; funcs return `int`, rich errors flow through the handler.
- **Closures:** `mlx_closure_new_func_payload(fn, void* payload, dtor)` — the
  `(fn, payload, dtor)` triple passes **callbacks with captured state across the
  C boundary**; specialized `value_and_grad` / `custom_jvp` / `custom_vmap`
  variants make transforms expressible entirely from C.

> **Tessera angle (keystone for embeddability).** Borrow, in priority order:
> (1) the **closure triple** — the mechanism to expose custom ops / samplers /
> logit processors across the FFI without touching Python; (2) **typed
> handle-by-value** `tessera_array`; (3) **`_new_data_managed(dtor)`** zero-copy
> adopt (especially valuable on the unified-memory lane); (4) **settable error
> handler** to surface Tessera's source-attributed diagnostics across FFI
> (currently absent in the C ABI).

---

## 3. mlx-lm — LLM inference/training stack

- **Generation core:** three-tier `generate_step` / `stream_generate` /
  `generate`, with **`sampler` (`logits→token`) + `logits_processors`
  (`[tokens, logits]→logits`) callable injection**. Prefill chunked
  (`prefill_step_size=2048`) with progress callback.
- **KV cache taxonomy** (`models/cache.py`): `KVCache`, `RotatingKVCache`
  (`keep` sink tokens), `QuantizedKVCache`, `ChunkedKVCache`,
  `ConcatenateKVCache`, `MambaCache`. Lifecycle: `make_prompt_cache`,
  **`save/load_prompt_cache` to safetensors + class-name metadata**,
  `maybe_quantize_kv_cache` (deferred quant via `quantized_kv_start`),
  **`can_trim_prompt_cache` / `trim_prompt_cache(n)` via per-cache
  `is_trimmable()` / `trim()`**.
- **Sampling:** composable `make_sampler(temp, top_p, min_p, top_k, xtc_*)` +
  `make_logits_processors(logit_bias, repetition/presence/frequency penalties
  with separate context sizes)`.
- **Model registry:** `MODEL_REMAPPING` aliases + `importlib.import_module(
  f"...models.{model_type}")` dynamic dispatch (new arch = drop a module).
  Quantize-on-load, AWQ/GPTQ packed-weight conversion, `make_shards` (5 GB).
- **Server:** OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, `/health`), SSE streaming, `ModelProvider` with an
  **`LRUPromptCache` reusing KV across requests** (reports `cached_tokens`).
- **Speculative decoding:** `speculative_generate_step` — draft proposes
  `num_draft_tokens`, target verifies in one pass, accept-prefix, **rewind both
  caches via `trim`**.
- **Continuous batching:** `batch_generate` / `BatchGenerator` — right-pad,
  per-seq length tracking, completion-draining queue, new prompts join mid-flight.

> **Tessera angle.** Tessera has paged/quantized/sliding-window handles + dflash.
> Borrow: the **`trim(n)` / `is_trimmable()` protocol** (serving + speculative
> both depend on it); **save/load prompt cache to safetensors**; **deferred KV
> quant**; the **composable sampler / logits-processor factories**; the
> **`importlib`-by-`model_type` registry**; **continuous batching scheduler**;
> the OpenAI server shape + **`LRUPromptCache`**.

---

## 4. mlx-data — data loading

- **Sample = keyed dict-of-arrays** (`{"image": b"path", "label": 42}`);
  transforms target keys, not positions.
- **Two containers:** `Buffer` (indexable / finite / random-access — `shuffle`,
  `perm`, `partition`, `ordered_prefetch`) vs `Stream` (sequential / infinite /
  nestable — `dynamic_batch`, `sliding_window`, `buffered`, `repeat`,
  `prefetch`, file readers).
- **Lazy pull-based DAG;** per-op conditional variant `<op>_if(cond, …)`;
  native-C++ threaded `prefetch(prefetch_size, num_threads)` (sidesteps GIL);
  `dynamic_batch` packs a *variable* count under a token/element budget.

> **Tessera angle (S15).** Adopt the **Buffer (finite) vs Stream (infinite)**
> type split so `shuffle`/`shard` are type-checked to finite sources; add
> **`dynamic_batch` token-budget packing** feeding varlen attention; adopt the
> **`_if` conditional-op** pattern for branch-free augmentation. The C++-core /
> thin-Python-binding split matches Tessera's stance.

---

## 5. mlx-embeddings & mlx-vlm

### mlx-embeddings (`Blaizzy/mlx-embeddings`)
Families: BERT / XLM-RoBERTa / ModernBERT, LLM-as-embedder (Qwen3,
Llama-Bidirectional), late-interaction (ColPali / ColQwen), vision (SigLIP,
Qwen3-VL). Tail = mean-pool → L2-norm → typed `outputs.text_embeds` dataclass.
`load()` → `(model, tokenizer|processor)`; batch via `batch_encode_plus` +
`attention_mask`. Convert with affine/mxfp4/nvfp4/mxfp8.

> **Tessera angle.** Small bounded `nn` "embedding head" (mean-pool → L2-norm →
> typed output) over existing encoder graphs; structured named-field outputs.

### mlx-vlm (`Blaizzy/mlx-vlm`)
**vision_tower + multi_modal_projector + language_model** triple (text reused
from mlx-lm). `encode_image()` → projector → features scattered into
`inputs_embeds` at `image_token_index`. Shared `base.py`: `BaseImageProcessor`
(`preprocess` / `rescale` / `normalize`), typed `LanguageModelOutput` /
`InputEmbeddingsFeatures`, helpers `expand2square` / `pixel_shuffle` /
`interpolate` / `chunked_attention`. **`VisionFeatureCache`** — LRU over
*projected* features keyed by image (separate from text K/V; ~11× multi-turn).
APC (Automatic Prefix Caching) blocks + disk spill. 16+ model families.

> **Tessera angle.** Standardize multimodal-JEPA / DiffusionGemma on the
> tower/projector/LM triple + a Graph-IR "vision-embed-merge" op; model **caches
> as distinct primitives** — text K/V (`KVCacheHandle`), projected-vision LRU
> (`VisionFeatureCache`), prefix blocks (APC). Port `pixel_shuffle` /
> `interpolate` / `expand2square` as native `nn.functional` (no HF dep, Decision
> #23).

---

## 6. dflash-mlx & omlx — the production envelope

### dflash-mlx (`bstnxbt/dflash-mlx`)
Productionized sibling of the `z-lab/dflash` reference Tessera already matched.
- **Drafting:** block size 16 — draft (~1B) emits 16 tokens in parallel via
  block diffusion; target verifies all 16 in one pass. Matched
  `<model>-DFlash` draft checkpoints (83–91% acceptance). **Adaptive block size**
  keyed on acceptance + long-context pressure.
- **Rejection / rollback:** greedy lossless accept-prefix; **tape-replay
  recurrent-state rollback** — record an "innovation tape" during verify, replay
  only accepted steps through a Metal kernel; `RecurrentRollbackCache` keeps
  GatedDeltaNet state coherent (KV truncates easily, recurrent state does not).
- **Serving:** L1/L2 prefix cache snapshotting *KV + GDN recurrent state + hidden
  + logits*, SSD spill, eviction; `verify_qmm` (verify-shape-specialized quant
  matmul); CLI/server `dflash generate|serve|benchmark|models|doctor` +
  OpenAI `/v1/chat/completions`.
- **Perf (M5 Max, 1024 tok):** Qwen3.5-4B 3.40× (86.4% accept), 9B 4.37%×
  (89.6%), 27B-4bit 2.95×, 35B-A3B-4bit 2.20×.

> **Tessera angle.** Tessera proved *correctness*; borrow the *envelope*:
> tape-replay recurrent rollback (directly expressible over existing
> `gated_deltanet` / `lightning_attention` — and a strong evaluator metamorphic
> target); L1/L2 prefix cache over `KVCacheHandle` / `MemoryStateHandle`;
> adaptive block size in `dflash_step`.

### omlx (`jundot/omlx`) — the embeddable-app blueprint
Thin **SwiftUI menubar app (12.5%) → FastAPI engine (Python 75%) over localhost
HTTP** — *the engine is the product.* Architecture: FastAPI (OpenAI + Anthropic
compat) → **EnginePool** (multi-model LRU / TTL / pin) → **ProcessMemoryEnforcer**
(RAM−8 GB ceiling) → **Scheduler** (continuous batching) → tiered cache
(**PagedCacheManager** GPU CoW + prefix-share → Hot RAM → **PagedSSDCacheManager**
safetensors, survives restart). Specialized `BatchedEngine` / `VLMEngine` /
`EmbeddingEngine` / `RerankerEngine`. Built-in HF downloader, profiles
(`<model>:<profile>`, no extra memory), `/admin/chat` web UI, venvstacks
portable Python bundle.

> **Tessera angle.** The outer-shell blueprint for "Tessera embedded for
> inference." Minimum surface to slot in: OpenAI/Anthropic streaming server +
> EnginePool + Scheduler + tiered KV cache + HF-style download/register + portable
> bundle. Tessera's `server.py` scaffolding (`Scheduler`, `KVCacheManager`,
> `ModelManifest`, `TesseraPackage`) is the seed; the gap is wiring it to a live
> streaming engine. See [`docs/architecture/inference/serving.md`](../../../../architecture/inference/serving.md).

---

## 7. Consolidated borrow list (pointer)

Ranked and sequenced in the two derived plans:
- **Engine / kernels / optimization →**
  [`apple_backend_capability_roadmap.md`](apple_backend_capability_roadmap.md)
- **Serving tier + embeddable app + dflash envelope →**
  [`docs/architecture/inference/serving.md`](../../../../architecture/inference/serving.md)

### Sources
MLX docs (ml-explore.github.io/mlx): lazy_evaluation, compile, function_transforms,
custom_metal_kernels, python/fast; DeepWiki MLX architecture / quantization /
JIT-kernels / device-encoder / compilation pages; GitHub `mlx`, `mlx-c`,
`mlx-lm`, `mlx-data`, `mlx-embeddings`, `mlx-vlm`, `dflash-mlx`, `omlx`.
