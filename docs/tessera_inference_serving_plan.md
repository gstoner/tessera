# Tessera Inference Serving + Embeddable App — Plan (omlx + dflash envelope)

> Status: **planning (2026-06-19).** The serving-tier and embeddable-app track,
> separated from the Apple *engine* roadmap
> ([`docs/audit/backend/apple/archive/apple_backend_capability_roadmap.md`](audit/backend/apple/archive/apple_backend_capability_roadmap.md)).
> Source material: the [MLX ecosystem survey](audit/backend/apple/archive/apple_gpu_mlx_ecosystem_survey.md)
> §3 (mlx-lm), §6 (dflash-mlx + omlx).
>
> **Thesis (validated by omlx).** The app is *thin*; the **engine + server is the
> product**. omlx ships Python 75% / Swift 12.5% — a SwiftUI menubar shell over a
> FastAPI engine on localhost HTTP. Tessera already owns the *hard correctness
> cores* (proven dflash, `attn_bias` substrate, paged/quantized/block-table KV +
> memory-state handles, Apple GPU `metal_runtime`, MLA weight-absorption). The
> recurring gap across mlx-lm / dflash-mlx / omlx is the **serving orchestration
> tier**: a live streaming server + EnginePool + Scheduler + tiered prefix cache +
> model registry — plus the dflash production envelope.

---

## 1. Grounding — what exists vs. what's missing (verified in-tree)

**Exists (the seed):**
- `python/tessera/server.py` (365 LOC) — an **in-process route registry**:
  `App` (`@app.route(stream=)`, `@app.model`, `healthz`, `metrics`, `run`→mapping),
  `Route`, `Scheduler` (`continuous_batch`/`sequence_batch`/`priority` — config
  only), `SchedulerSession.generate`, `KVCacheManager`, `KVCacheConfig`,
  `ModelManifest`, `TesseraPackage`, `RuntimeCapabilities`, `load_package`,
  `capabilities`.
- `python/tessera/dflash_serve.py` (76 LOC) — `DFlashScheduler.generate` /
  `generate_text`, `dflash_generate_text`.
- KV/state primitives: `KVCacheHandle` (paged, int4/8 quant, sliding-window),
  `MLAPagedDecoder`, `MLABlockPagedCache` (vLLM-style block tables, ragged
  `decode_batch`), `MemoryStateHandle`, `RotatingDraftKVCache`.
- Proven `tessera.dflash` (greedy spec-decode == greedy AR vs MLX reference);
  `dflash_io` (safetensors ↔ weights), HF state-dict import.

**Missing (this plan):**
- A **live HTTP transport** (server.py `run()` returns a mapping; no ASGI/SSE).
- **OpenAI/Anthropic-compatible endpoints** with real streaming.
- **EnginePool** — multi-model load/unload, LRU, TTL, pin.
- **Scheduler execution** — the continuous-batching *loop* (config exists; the
  `BatchGenerator`-style drain queue does not).
- **Tiered prefix cache** — RAM hot + SSD cold, cross-request KV reuse.
- **KV `trim(n)` / `is_trimmable()` protocol** + save/load prompt cache.
- **Composable sampler / logits-processor** factories.
- **Model registry + downloader** (`importlib`-by-`model_type`, HF fetch).
- **dflash production envelope** — tape-replay recurrent rollback, L1/L2 prefix
  cache, adaptive block size, OpenAI server wiring.
- **Packaging** — portable bundle for a Swift shell.

---

## 2. Two layers

```
┌──────────────────────────────────────────────────────────────────┐
│ LAYER B — Embeddable app (omlx blueprint)                          │
│   Thin SwiftUI menubar shell ──HTTP localhost──▶ Layer A           │
│   + HF-style model download UI, profiles, /admin chat, venvstacks  │
└───────────────────────────────────┬──────────────────────────────┘
                                     │ consumes
┌───────────────────────────────────▼──────────────────────────────┐
│ LAYER A — Serving engine (Python, the product)                     │
│   FastAPI (OpenAI + Anthropic compat, SSE)                         │
│   ├─ EnginePool (multi-model LRU/TTL/pin)                          │
│   ├─ ProcessMemoryEnforcer (← Apple P5 budget ABI)                 │
│   ├─ Scheduler (continuous batching loop)                          │
│   ├─ Generation core (step/stream/generate + sampler/processors)   │
│   ├─ Model registry (importlib-by-type + HF download)              │
│   └─ Tiered prefix cache (RAM hot → SSD cold, KV+state)            │
│        ▲ dflash production envelope plugs in here                  │
└────────────────────────────────────────────────────────────────────┘
```

**Layer A is the priority** (it is the product; the Swift shell is thin and can
come last or be community/optional). Layer A reuses the Apple engine via
`@jit(target="apple_gpu")` `metal_runtime` and the existing cache handles.

---

## 3. Layer A — workstreams

### A1. Generation core (foundation)
**What.** Three-tier `generate_step` / `stream_generate` / `generate` over a
Tessera-compiled model graph, with **`sampler` + `logits_processors` callable
injection** and chunked prefill (`prefill_step_size` + progress callback).

**Work.**
1. `generate_step(prompt, model, *, max_tokens, sampler, logits_processors,
   prompt_cache, prefill_step_size, kv_bits, ...) -> generator[(tokens, logprobs)]`
   over the existing decode loop (`MLAPagedDecoder` / `MLABlockPagedCache`).
2. `make_sampler(temp, top_p, top_k, min_p, xtc_*)` +
   `make_logits_processors(logit_bias, repetition/presence/frequency penalties
   with separate context sizes)` — port the mlx-lm factories.
3. `stream_generate` yields a `GenerationResponse` (text, token, logprobs,
   timing, `finish_reason`, `from_draft`).

**Acceptance.** Greedy `generate` matches the existing dflash/AR path; sampler +
processors are pluggable callables; long prompts prefill in chunks with progress.
**Effort:** Med · depends on nothing new.

### A2. KV cache protocol + prompt cache
**What.** The `trim` protocol + serializable prompt cache that serving and
speculative decode both require.

**Work.**
1. Add **`is_trimmable()` / `trim(n)`** to `KVCacheHandle` /
   `MLABlockPagedCache` / draft caches (rewind-on-rejection + multi-turn).
2. **`save_prompt_cache` / `load_prompt_cache`** to safetensors + class-name
   metadata (reuse `dflash_io`).
3. **Deferred KV quant** (`quantized_kv_start`) — keep recent tokens full
   precision, quantize the tail (KV quant already exists; add the start offset).

**Acceptance.** A draft-rejection trims both caches correctly (cross-checked vs
recompute); a prompt cache round-trips to disk and resumes bit-exact.
**Effort:** Med · **A1 dependency** for the rewind path.

### A3. Tiered prefix cache + EnginePool
**What.** Cross-request KV reuse (RAM hot → SSD cold) + multi-model pool.

**Work.**
1. **`LRUPromptCache`** keyed by prompt-prefix hash → KV/state snapshot; report
   `cached_tokens`; prefix-hit skips prefill.
2. **SSD spill** (safetensors, survives restart) over a block store — leverage
   `MLABlockPagedCache`'s block pool + free-list for the paged GPU tier; RAM
   write-back; SSD cold tier.
3. **EnginePool** — load/unload, LRU eviction, per-model TTL, pin; backed by
   **`ProcessMemoryEnforcer`** consuming the Apple **P5 budget ABI**
   (`get_active/peak_memory`, memory limit).

**Acceptance.** A repeated prompt prefix hits the cache and skips prefill
(measured `cached_tokens`); a cold prompt survives a process restart from SSD;
the pool evicts under memory pressure without OOM.
**Effort:** High · **depends on A2** + Apple **P5**.

### A4. Continuous-batching scheduler
**What.** Turn the existing `Scheduler` *config* into an execution loop —
`BatchGenerator` over right-padded prompts, per-seq length tracking, a
completion-draining queue, new requests joining mid-flight.

**Work.** Implement the batch loop using the Apple batched lane (`bmm`,
`absorb_decode_batch` same-length grouping in `MLABlockPagedCache`); FCFS +
configurable concurrency; per-seq `lengths`/`right_padding` through caches.

**Acceptance.** Throughput (agg tokens/sec) rises with concurrent requests vs
sequential; a finishing sequence drains and a queued prompt joins without
stalling the batch. **Effort:** High · depends on A1/A2; benefits from Apple **P1**.

### A5. HTTP server (OpenAI + Anthropic compat)
**What.** A live ASGI server wrapping Layer A.

**Work.** `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`,
Anthropic `/v1/messages`; **SSE** (`text/event-stream`, `data: {json}\n\n`,
`data: [DONE]`), `stream_options.include_usage`; request → sampling dataclass
(temp/top_p/.../seed/stop/logit_bias). Wire `server.py`'s `App`/`Route`/`Scheduler`
to an ASGI transport (FastAPI/Starlette — a *serving-time* dep, not a runtime
dep; Decision #23 governs the compiler runtime, not the optional server).

**Acceptance.** An OpenAI client library streams a completion end-to-end against
a Tessera-served model; `/v1/models` lists models + profiles; usage stats report
`cached_tokens`. **Effort:** Med · depends on A1; A3/A4 enrich it.

### A6. Model registry + downloader
**What.** `MODEL_REMAPPING` + `importlib.import_module(f"...models.{model_type}")`
dynamic dispatch; HF-MLX-style search + download + register through existing
importers.

**Work.** Formalize `tessera.models` under by-`model_type` dispatch (the
`minimax_m3_importer` / DiffusionGemma importers become registered modules);
quantize-on-load + AWQ/GPTQ packed-weight conversion (pairs with Apple **P3**);
auto-detect local model dirs; HF fetch with size preview + mirror.

**Acceptance.** A new architecture is added by dropping a module (no central
switch); a HF MLX/safetensors model downloads, converts, and serves.
**Effort:** Med · independent; pairs with Apple P3 for quant-on-load.

---

## 4. dflash production envelope (plugs into Layer A)

Tessera proved dflash *correctness*; this is the *production envelope* from
dflash-mlx. It rides A1–A5.

| Item | What | Tessera mapping | Effort |
|---|---|---|---|
| **D1 Tape-replay recurrent rollback** | Record an "innovation tape" during verify, replay only accepted steps through a Metal kernel; keep GatedDeltaNet state coherent across cycles | New `tessera.ops` rollback over existing `gated_deltanet`/`lightning_attention`; strong **evaluator metamorphic** target (replayed-state ≡ recomputed-state) | High |
| **D2 L1/L2 prefix cache** | Snapshot *KV + recurrent state + hidden + logits*; SSD spill; eviction; skip prefill on revisit | A3 tiered cache, extended to recurrent + hidden + logits snapshots | Med (on A3) |
| **D3 Adaptive block size** | Auto-tune draft block size on observed acceptance + long-context pressure | Knob in `dflash_step` / `DFlashScheduler` | Low |
| **D4 Verify-shape quant matmul** | `verify_qmm` specialized for the many-token single-pass verify shape | Apple **P3** QMM variant tuned for verify shape | Med (on P3) |
| **D5 Server wiring** | dflash as a draft/target pair behind the OpenAI server | `DFlashScheduler` → A5 endpoints; `--draft-model` in EnginePool | Low (on A5) |

**D1 is the headline** — it is the one capability neither Tessera nor mlx-lm's
KV-only speculative path has, and it generalizes spec-decoding to
linear/recurrent-attention models. **D3 is a cheap high-value win** independent of
the rest.

---

## 5. Layer B — embeddable app (omlx blueprint)

Thin and last. Once Layer A serves OpenAI/Anthropic over localhost:
1. **SwiftUI menubar shell** → HTTP localhost (read-only/streaming chat UI).
2. **HF model download UI**, profiles (`<model>:<profile>`, no extra memory),
   `/admin/chat` web UI (history, model switch, image upload for VLM, reasoning
   display).
3. **Packaging:** venvstacks-style portable Python bundle shipping Tessera's
   runtime + Apple `.dylib` (264 C ABI symbols) so the shell needs no managed env;
   Homebrew tap (`tessera serve`) as the CLI surface.
4. **C ABI for native embedding** (alternative to HTTP): the
   [survey §2](audit/backend/apple/archive/apple_gpu_mlx_ecosystem_survey.md) borrows — closure triple,
   typed handles, `_new_data_managed`, settable error handler — let a Swift/Rust
   host drive the engine in-process without the HTTP hop. Decide HTTP-only vs
   HTTP+C-ABI embedding once Layer A is real.

---

## 6. Sequencing

```
Wave 1 (engine usable as a server)
  A1 generation core ─▶ A2 KV trim/prompt-cache ─▶ A5 HTTP/OpenAI+Anthropic SSE
                                                   A6 model registry/downloader (parallel)
Wave 2 (throughput + reuse)
  A3 tiered prefix cache + EnginePool (needs A2 + Apple P5)
  A4 continuous-batching scheduler  (needs A1/A2; better with Apple P1)
Wave 3 (dflash envelope)
  D3 adaptive block (cheap) ; D1 tape-replay rollback ; D2 L1/L2 (on A3) ;
  D4 verify-qmm (on Apple P3) ; D5 server wiring (on A5)
Wave 4 (app)
  Layer B Swift shell + packaging + (optional) C-ABI embedding
```

**Cross-plan dependencies on the Apple engine roadmap:**
`ProcessMemoryEnforcer`←**P5**; continuous-batch throughput compounds on **P1**;
`verify_qmm`/quant-on-load←**P3**.

**Recommended first move:** **A1 generation core + A2 trim protocol** — they make
the existing engine a real generator, unblock A5 (a demoable OpenAI server) and
the whole dflash envelope, and are Med-effort with no new engine dependency.

---

## 7. Boundaries

- **Decision #23.** The *compiler runtime* imports no PyTorch/JAX/Flax. The
  *optional server* (FastAPI/Starlette/uvicorn) and the Swift app are
  serving-time/app-time deps, not runtime deps — allowed, but isolated behind the
  server package so `import tessera` stays clean.
- File-format compat only for weights (safetensors / GGUF / HF state dicts via
  existing importers); the consuming runtime is Tessera's own.
- Engine capability (scheduler internals, kernels, fusion, quant execution) lives
  in the [Apple backend roadmap](audit/backend/apple/archive/apple_backend_capability_roadmap.md), not here.
