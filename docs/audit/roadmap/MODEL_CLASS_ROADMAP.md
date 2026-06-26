---
last_updated: 2026-06-26
audit_role: plan
plan_state: landing
---

# Model-Class Roadmap — compiling Kimi-K2 / DeepSeek-V3.2 / GLM-5.2 / MiniMax-M3

> Status: **M0–M5 landed** (scaffolding + quant + MoE + MLA + DSA + full stack &
> autoregressive decode), plus the M3.1/M4.1 composed Apple GPU attention lanes
> and the MiniMax-M3 text-tower graph contract/runtime decode/artifact KV-outer
> lowering path plus local HF tokenizer/safetensors import gates and projected
> media embedding splice/reference vision tower. MiniMax-M3 text checkpoint,
> tokenizer, processor, vision/projector alias, production-shape multimodal
> build, and scaled image+video decode-after-prefill gates are now closed;
> JEPA training/selective-decode graph contracts are closed as compiler-visible
> artifact lowerings.
> Remaining work is aggregated in **Open Closure Backlog** below.
> Last updated: 2026-06-19.

## Goal & definition of done

Stand up the frontier MoE models as Tessera model graphs. On this Apple
Silicon machine "compile model X" is two provable claims:

1. **Artifact at full config** — the full-scale architecture lowers to a valid,
   verified op graph (lit + verifier where native IR fixtures exist; full-layer
   graph verifier for shape/config-only contracts). Provable here; no execution.
2. **Scaled execution where runtime-wired** — a structurally-faithful, shrunk
   instance runs end-to-end on Apple GPU, gated against a numpy reference. For
   newly staged graph contracts, the scaled gate starts as build/verify and is
   promoted to decode/recompute parity once the runtime path is wired.

Full-scale execution + NVIDIA FP8/sparse kernels + distributed MoE stay
**hardware-gated** (Phase G/H) but lit-provable. HF tokenizer/safetensors
metadata import is now staged for MiniMax-M3 with typed runtime-weight
materialization covered by realistic fixtures. Serving and native full-scale
vision/video tower execution remain later hardware/backend workstreams. This
roadmap is **compiler-core**: graph + kernels + decode math, weights/tokens
assumed provided.

The models are one architecture family (MoE-FFN transformer with latent or
grouped attention, optional sparse attention, low-precision weights), so the work
is **shared stdlib pillars** built once + thin per-model configs:

| Model | Attention | Sparse | Quant |
|-------|-----------|--------|-------|
| DeepSeek-V3.2 (north-star) | MLA | DSA | FP8 |
| GLM-5.2 | MLA | DSA + IndexShare | BF16 / rollout KV FP8 |
| Kimi-K2 | MLA | — | INT4 |
| MiniMax-M3 | GQA | MSA | BF16 + staged multimodal metadata |

## Milestone ladder

| M | Title | Pillar | Status |
|---|-------|--------|--------|
| M0 | Scaffolding + failing north-star gate | `tessera.models.*`, `tessera.stdlib.*` | ✅ landed |
| M1 | Fused dequant-into-GEMM (packed INT4/FP8 + scales, fp32 accum) | `stdlib.quant` | ✅ landed |
| M2 | Capacity-aware MoE dispatch + quantized grouped SwiGLU | `stdlib.moe` | ✅ landed |
| M3 | MLA as production decode primitive (absorption + prefill + decoupled-RoPE + paged latent cache) | `stdlib.attention` | ✅ landed |
| M4 | DSA native block-sparse lowering (indexer + selection + exact block-sparse attention) | `stdlib.attention` | ✅ landed |
| M3.1/M4.1 | Composed Apple GPU attention lanes (MLA absorb + DSA per-head matmuls on Metal) | `stdlib.attention` | ✅ landed (composed; fused MSL still open) |
| M5 | Full layer stack + autoregressive KV-cached decode loop; per-model scaled-exec gates | `tessera.models.moe_transformer_runtime` | ✅ landed (decode ≡ recompute for the runtime-gated families, including MiniMax-M3 MSA) |

Sequencing: **M1 unblocks M2** (and dense projections); M3 and M4 are
independent; M5 integrates. Every kernel lands a perf ratchet; a contract cell
flips to `complete` only when an oracle independently re-derives it.

## Open Closure Backlog

This is the single aggregate list of known opens from the model-class track.
Milestone sections below may mention context, but closure should be tracked here.

| ID | Status | Area | Open item / closure evidence | Close condition / acceptance |
|----|--------|------|------------------------------|------------------------------|
| O1 | Open | Attention kernels | Fused MSL attention kernels for MLA absorb / DSA block-sparse paths remain open; current Apple GPU support is composed over existing matmul lanes. | Single fused Apple GPU kernel path lands with parity vs numpy/reference and a perf ratchet that beats the composed path on representative scaled configs. |
| O2 | Open | MiniMax-M3 MSA backend | Native NVIDIA CUDA/H800/Blackwell KV-outer sparse MSA kernel is still artifact-only. Current Target IR intentionally emits `status = "artifact_only"`. | `tessera.attn.msa_kv_outer_sparse` lowers to a native executable backend kernel with dense-equivalence oracle, decode/prefill coverage, and target-specific lit/runtime gates. |
| O3 | Closed | Full text checkpoint materialization | `load_text_runtime_weights(_from_safetensors)` maps full scaled text-tower HF tensor names into typed `ModelWeights`; `test_full_scaled_text_safetensors_materialize_runtime_weights` covers realistic sharded HF-style fixtures and selected runtime layout parity. | Full text-tower tensor-name map loads at least one realistic sharded HF-style fixture into typed runtime weights; selected layer outputs match the synthetic/reference loader layout expectations. |
| O4 | Closed | Tokenizer parity | `HFTokenizerAdapter` wraps `tokenizers.Tokenizer`, preserves image/video specials and chat templates, and `test_hf_tokenizer_roundtrips_specials_and_chat_template` checks token IDs against upstream tokenizer output. | HF tokenizer path can round-trip fixture prompts, special image/video tokens, and chat templates with token IDs matching upstream tokenizer output. |
| O5 | Closed | HF processor fixture parity | `test_processor_fixture_reproduces_pixels_and_video_frame_sampling` imports an HF-style processor fixture and verifies resize/rescale/normalize plus deterministic frame sampling against exact expected arrays. | `processor_config.json` / image processor fixtures reproduce resize/rescale/normalize/frame-sampling outputs within a defined numeric tolerance. |
| O6 | Closed | Full vision/projector checkpoint aliases | `load_vision_runtime_weights` resolves observed vision/projector aliases and `test_vision_projector_aliases_load_with_precise_diagnostics` verifies alias acceptance plus precise missing-key diagnostics. | Loader accepts observed upstream MiniMax-M3 vision/projector key aliases and rejects ambiguous/missing aliases with precise diagnostics. |
| O7 | Closed | Native/fused vision tower lowering | Media graph ops now cover preprocess/frame sample/patch embed/patch merge/project/splice and lower as named Schedule/Tile/Target contract kernels; reference tower parity remains covered by raw media execution tests. | Full multimodal graph lowers through native/composed backend kernels for patch/project/splice surfaces with parity against the reference tower. |
| O8 | Closed | Full MiniMax multimodal execution at production geometry | `build_multimodal_graph` builds full production image+video shape contracts; `test_raw_image_and_video_prefill_can_continue_decode` covers scaled raw image+video prompts end-to-end with cached decode after multimodal prefill. | Full MiniMax-M3 image+text and video+text graph builds at production dimensions, and a scaled execution gate covers image and video prompts end-to-end with decode after multimodal prefill. |
| O9 | Closed | JEPA native training lowering | `tessera.jepa.train_step` lowers through Schedule/Tile/Target as a compiler-visible stateful training artifact; reference tests prove deterministic mask RNG, EMA update semantics, and latent-loss parity. | JEPA training step lowers as a compiler-visible stateful training graph with mask RNG determinism, EMA update semantics, and latent-loss parity against the reference. |
| O10 | Closed | VL-JEPA selective decoder integration | `tessera.jepa.selective_decode` lowers as an optional conditional decode artifact with latent-score gating and retrieval/classification/decode branch attrs; reference tests prove the decoder branch can be skipped. | Conditional/selective decoder graph compiles with latent-score gating, retrieval/classification/decode branches, and tests proving decoder invocation is data-dependent and optional. |
| O11 | Open | Quantized model-weight runtime bridge | INT4/FP8 packed dequant kernels are landed, but some model-family full-weight runtime paths still use synthetic/reference weights. | Frontier model configs can load quantized weight fixtures into runtime-compatible typed weights and hit the fused dequant path in model-level tests. |
| O12 | Open | Quantized packed-byte fused-kernel memory path | The fused dequant-GEMM kernel exists, but native packed-byte INT4 nibble / FP8 operands still need the full no-materialization memory path. | Fused dequant kernels consume packed-byte operands directly, preserve separate scale operands, avoid full-weight materialization, and pass parity/perf ratchets. |
| O13 | Open | Distributed/full-scale execution | Full-scale execution, distributed MoE, and NVIDIA FP8/sparse performance remain hardware-gated. | Hardware-backed CI or reproducible artifact gate covers full-scale launch metadata, distributed routing, and target-specific performance/regression thresholds. |

## What M0–M2 landed (2026-06-14)

### M0 — scaffolding
- `python/tessera/stdlib/{__init__,quant,moe,attention}.py` — the shared
  production-primitive namespaces. `attention` reserves the M3/M4 API as
  milestone stubs that raise a clear pointer (never a silent dense fallback).
- `python/tessera/models/moe_transformer.py` — one shape-only
  `MoETransformerConfig` + verifier + `build_block` graph builder + param-budget
  estimator, covering MLA/GQA × DSA/LSA/MSA/dense × INT4/FP8/BF16/none.
- `python/tessera/models/{deepseek_v32,glm5,kimi_k2,minimax_m3}.py` — thin
  `config()` + `scaled_config()` factories. GLM-5.2 and MiniMax-M3 use
  released HF-shape contracts.
- North-star integration test (`test_model_class_frontier.py`) — full configs
  build+verify; scaled MoE pillar executes vs reference; the full scaled forward
  is a `strict xfail` until M3/M4.

### M1 — quant keystone (`stdlib.quant`)
- `PackedQuantTensor` — packed codes (genuine int4 two-per-byte nibble packing,
  int8, or fp8 grid) + a **separate per-group scale tensor** + `QuantScheme`
  (per-channel or per-group along the K axis). The scale is a load-bearing
  operand, not a declared contract.
- `quantize_weight` (per-channel / group-wise INT4·INT8·FP8), `pack_int4` /
  `unpack_int4`, `dequantize`.
- `dequant_matmul` / `dequant_grouped_gemm` — the fused contract: packed weights
  + scales in, **fp32-accumulated** result out, evaluated group-wise (the real
  kernel's accumulator policy). Oracle proves it equals dequant-then-matmul
  (DESIL cross-path); runs on the Apple GPU Metal matmul lane (`backend="apple_gpu"`).
- Tests: `tests/unit/test_stdlib_quant.py` (parity, group-beats-per-channel,
  int4-packs-smaller, oracle equivalence, Apple GPU exec).

### M2 — MoE production lowering (`stdlib.moe`)
- `compute_capacity` + `plan_dispatch` (capacity/bucketing, deterministic
  overflow drop) + `dispatch` / `combine` (token permutation, weighted scatter).
- `shared_expert_swiglu` + residual combine in `moe_forward` (returns a
  `MoEResult` carrying the dispatch plan / drop stats).
- `moe_swiglu_quantized` — grouped SwiGLU on packed INT4/FP8 experts via the M1
  path (the M1↔M2 join).
- Tests: `tests/unit/test_stdlib_moe.py` (no-capacity ≡ `models.moe_routing`
  reference, capacity drop semantics, quant ≈ dense).

### M3 — MLA decode primitive (`stdlib.attention`)
- `MLAWeights` + `mla_attention` — DeepSeek-style MLA with a **decoupled
  (partial) RoPE** split and **weight absorption**: `W_uk` folded into the query
  and `W_uv` into the output, so per-head K/V are never materialized — only the
  compressed latent `c` (+ a tiny shared RoPE key) is read.
- `mla_prefill` (fill latent cache from prompt) + `mla_decode_step` (advance one
  chunk via `cache.LatentKVCacheHandle` — paged latent + rope).
- Oracles (`tests/unit/test_stdlib_attention.py`): `absorb ≡ no-absorb`
  (production latent-only path == explicit-K/V reference); prefill-then-decode ≡
  full-prefill (autoregressive consistency over the paged cache); chunk ≡
  token-by-token.

### M4 — DSA block-sparse lowering (`stdlib.attention`)
- `dsa_block_index` (lightning-indexer-style per-GQA-group block scores) →
  `dsa_select_blocks` (top-k + causal + forced-local keep-mask) →
  `dsa_block_sparse_attention` (exact attention over only the selected blocks).
- Oracles: `select-all ≡ dense causal` (DESIL cross-path); explicit top-1 hand
  check; restricted top-k ≠ dense; selection determinism.

### M3.1 / M4.1 — composed Apple GPU attention lanes
- `mla_attention(..., backend="apple_gpu")` runs the absorbed per-head score +
  output matmuls on the Metal matmul lane; `dsa_block_sparse_attention(...,
  backend="apple_gpu")` runs the per-head QKᵀ / weights·V matmuls on Metal.
  Both numpy-fallback on a Metal miss; parity-gated vs the reference
  (`test_stdlib_attention.py`, skipped when Metal absent).
- Open closure item O1 tracks the single fused MLA-absorb / block-sparse MSL
  kernel promotion; these lanes currently compose existing Metal matmuls.

### M5 — full stack + autoregressive decode (`models.moe_transformer_runtime`)
- `synthetic_weights` + `forward` (full causal forward, all layers: RMSNorm →
  attention → residual → RMSNorm → FFN → residual → final norm → LM head) +
  `prefill` / `decode_step` / `greedy_generate` (KV-cached: MLA latent cache for
  MLA layers, K/V cache for GQA).
- Oracle (`test_moe_transformer_runtime.py`): **KV-cached greedy decode ≡ full
  recompute** for the runtime-gated scaled models, with per-step logit parity
  (not just argmax) so the equality isn't a tie coincidence. MiniMax-M3 now
  joins this gate through the offset-aware MSA path.
- DSA block-sparsity in the decode loop is closed in M5.1 below via the
  offset-aware indexer; MiniMax-M3 MSA runtime wiring is closed in M5.3.

### Honesty notes
- **Composed vs fused on Apple GPU**: `dequant_matmul` runs the heavy matmul on
  the Metal lane after a per-group host dequant. A single MSL kernel folding the
  dequant into the GEMM tile is the M1.1 follow-up — mirrors the existing
  `moe_swiglu_block` composed-default / fused-behind-env pattern. What is real
  and load-bearing today: packed storage, separate per-group scales, fp32 accum.
- **Not yet first-class IR ops**: the quant/MoE pillars are `stdlib`
  compositions over existing catalog ops (`grouped_gemm`, `matmul`,
  `quantize_fp8`). Promoting `dequant_matmul` to a `tessera.*` Graph IR op +
  `primitive_coverage` row (so it is compiler-visible and lit-lowerable) is
  tracked for M1.1 / M2.1.
- **MiniMax-M3** now has a text-tower compiler contract with scaled MSA runtime
  decode parity, artifact KV-outer lowering, HF tokenizer/safetensors import
  gates, full text-weight fixture materialization, reference vision execution,
  production media-shape build gates, and scaled raw image+video
  decode-after-prefill coverage. Open closure item O2 tracks the remaining
  native NVIDIA MSA backend.

## Open work

The authoritative closure list is **Open Closure Backlog** above. Short map:
fused/composed kernel promotion is O1 and O12; MiniMax-M3 native MSA is O2; the
MiniMax-M3 checkpoint/tokenizer/processor/multimodal closures O3-O8 and JEPA
closures O9-O10 are landed; model-weight and hardware-gated execution closure is
O11-O13.

## Perf follow-ups (landed)

### Dedicated dequant `.td` ops
`tessera.dequant_matmul` and `tessera.dequant_grouped_gemm` are now first-class
**MLIR dialect ops** (`src/compiler/ir/TesseraOps.td` + verifiers in
`TesseraOps.cpp`): packed low-precision weight codes + a separate per-group
`w_scale` operand, fp32 accum, `weight_dtype ∈ {int4,int8,fp8_e4m3,fp8_e5m2}`
validated. `tessera-opt` rebuilds clean; lit fixture
`tests/tessera-ir/model_class/dequant_ops.mlir` (+ integrated into
`deepseek_v32_block.mlir`) verifies them at the IR level; the full tessera-ir
sweep is 0 FAIL. The coverage `graph_ir_lowering` metadata is now genuinely
`registered` (a real MLIR op, not just a catalog name).

### Fused dequant-into-GEMM MSL kernel
`tessera_apple_gpu_dequant_matmul_f32` (`apple_gpu_runtime.mm` + non-Darwin
stub) is a real Metal kernel computing
`O[m,n] = Σ_k X[m,k]·(codes[k,n]·scale[k/GS,n])` — the dequant is done
**in-register inside the contraction** with an fp32 accumulator, so the full
fp32 weight is never materialized and it is one dispatch (vs. per-group GEMMs).
`stdlib.quant.dequant_matmul(backend="apple_gpu")` now uses it (composed path is
the fallback), and `_apple_gpu_backend.gpu_dequant_matmul` exposes it via ctypes.
Validated on Metal vs. dequant-then-matmul (~5e-7) for INT4/INT8/FP8 in
`tests/unit/test_stdlib_quant.py`.

## M5.1 — quant pillars promoted to first-class IR ops + registry rows (landed)

The fused dequant-into-GEMM keystone is now compiler-visible, not just a
`stdlib` composition:

- **Catalog**: `tessera.dequant_matmul` (`x, w_codes[, w_scales]`) and
  `tessera.dequant_grouped_gemm` (`x, codes, group_sizes[, scales]`) added to
  `op_catalog.OP_SPECS` (lowering `loop_nest`); callable via `tessera.ops.*` and
  listed in `docs/spec/PYTHON_API_SPEC.md` (spec-sync gate).
- **Coverage**: drift-gated `primitive_coverage` rows in the `quantization`
  category — math/shape/dtype/lowering/tests **complete**, `numeric_policy`
  (fp32 accumulate) attached, `dequant_grouped_gemm` carries the contiguous
  `grouped_layout`. `backend_kernel` is honestly **partial** (composed Apple GPU
  matmul lane; the fused dequant MSL kernel is M1.1).
- **Autodiff**: `dequant_matmul` VJP + JVP registered (STE GEMM rule), so those
  axes auto-flip to complete; `dequant_grouped_gemm` is structural (N/A).
- **Guards**: `tests/unit/test_model_class_registry.py`; generated dashboards
  regenerated and drift-gated (`scripts/check_generated_docs.sh`).

The attention pillars already have catalog anchors (`mla_decode_fused`,
`deepseek_sparse_attention`, `msa_*`).

## M5.1 — DSA-in-decode-loop + full-config artifact lit fixtures (landed)

- **Offset-aware DSA in the decode loop**: `dsa_select_blocks` /
  `dsa_block_sparse_attention` gained a `q_positions` argument and rank
  strictly-past blocks only (own block force-added), so block selection is
  identical for a full forward and an incremental decode step (the property that
  makes KV-cached decode ≡ recompute under sparsity). `Sk` need not be a
  multiple of `block_size` — keys are zero-padded internally and causally
  masked. `models.moe_transformer_runtime` now wires DSA into both prefill and
  decode (materialized K/V cache; MLA expands the latent, GQA projects).
  Guards: `tests/unit/test_moe_transformer_runtime.py` — decode ≡ recompute with
  real multi-block sparsity, plus a "DSA genuinely engaged ≠ dense" check.
- **Full-config artifact lit fixtures**: `tests/tessera-ir/model_class/{deepseek_v32,glm5,kimi_k2,minimax_m3}_block.mlir`
  — each model's decoder-block core at production dims (DeepSeek H=7168/256
  experts FP8, GLM-5.2 H=6144/256 experts BF16, Kimi H=7168/384 experts INT4,
  MiniMax-M3 H=6144/128 experts BF16 + MSA), run
  through `tessera-opt` so the registered `moe_swiglu_block` /
  `deepseek_sparse_attention` / `msa_sparse_attention` / `latent_kv_compress`
  ops are **verified at full scale**. A runnable Python companion
  (`test_full_config_artifact_all_layers_anchored`) builds every layer at full
  config and checks each compute-core op is catalog-anchored.

## M5.2 — MiniMax-M3 text-tower contract (landed)

- **Model family config**: `python/tessera/models/minimax_m3.py` adds the
  released text contract: 60 layers, hidden size 6144, 64 Q heads / 4 KV heads,
  1M context, first 3 dense layers, 128 routed experts, top-4 routing, one
  shared expert, BF16 weights, and MSA block size 128 / top-16.
- **Shared graph emission**: `MoETransformerConfig` now accepts `sparse="msa"`
  with per-layer sparse frequency; dense early layers emit plain `attention`,
  later layers emit `msa_sparse_attention` with `top_k_blocks`, `block_size`,
  `index_dim`, `num_index_heads`, `score_type`, `force_local_block=True`, and
  `causal=True`.
- **Staged multimodal metadata**: MiniMax image/video token ids, image sequence
  shape, patch/merge sizes, projector hidden size, and max-frame metadata are
  exposed for future import work with `vision_execution_supported=False`.
- **Guards**: `tests/unit/test_minimax_m3_contract.py` verifies the full config,
  the exact dense/MSA frequency over all 60 production layers, MSA attrs and
  verifier failures, scaled config, and parameter-budget estimates. The
  model-class lit fixture `tests/tessera-ir/model_class/minimax_m3_block.mlir`
  verifies the production-dim MSA + BF16 MoE artifact.

## M5.3 — MiniMax-M3 MSA runtime decode + KV-outer artifact lowering (landed)

- **Runtime decode**: `models.moe_transformer_runtime` now honors
  `msa_sparse_layer_freq`, emitting dense GQA cache entries for MiniMax's warmup
  layers and materialized MSA K/V cache entries for sparse layers. The shared
  MSA reference path accepts global `q_positions`, ranks strictly-past blocks,
  force-adds the local block, and pads non-divisible K/V lengths internally, so
  KV-cached decode and full recompute choose consistent selected blocks.
- **MSA primitive reuse**: the public `tessera.ops.msa_*` wrappers delegate to
  `stdlib.attention`, keeping model runtime, examples, and op tests on one
  reference implementation. `selected_block_ids` is accepted as an explicit
  backend worklist contract and bounds-checked.
- **Backend artifact path**: Graph IR `tessera.msa_sparse_attention` lowers to
  `schedule.attn.kv_outer_sparse`, then Tile IR
  `tessera.attn.msa_kv_outer_sparse`, then NVIDIA Target IR
  `kernel = "msa_kv_outer_sparse"` with `status = "artifact_only"`,
  `block_ids_layout = "B,Hkv,Sq,top_k"`, `gqa_group_size`, `tile_q`, `tile_kv`,
  `mode`, and `kv_traversal = "kv_outer"`.
- **Guards**: `tests/unit/test_moe_transformer_runtime.py` covers MiniMax-M3
  decode ≡ recompute, MSA genuinely engaged ≠ dense, and dense-warmup-vs-MSA
  cache kinds. `tests/unit/test_msa_kv_outer_schedule.py` covers the
  Graph→Schedule→Tile→Target artifact path.
- **Open closure item**: O2 tracks native NVIDIA CUDA/H800/Blackwell kernel
  implementation and hardware benchmark proof; this slice intentionally stops
  at a verified compiler artifact contract.

## M5.4 — MiniMax-M3 tokenizer/safetensors importer + multimodal execution gate (landed)

- **HF metadata importer**: `python/tessera/models/minimax_m3_importer.py`
  reads local HF-style `config.json`, `tokenizer_config.json`, `tokenizer.json`,
  `.safetensors`, and `.safetensors.index.json` files. It validates present
  config fields against Tessera's MiniMax-M3 contract and exposes a
  `MiniMaxM3ImportManifest`.
- **Tokenizer contract**: tokenizer import records special tokens and chat
  templates. When `tokenizer.json` is loadable through Hugging Face
  `tokenizers`, `HFTokenizerAdapter` preserves upstream token IDs; deterministic
  `VocabTokenizer` fallback remains available for minimal fixtures. Missing
  tokenizer files are rejected when requested.
- **Safetensors manifest + text materialization**: dependency-free safetensors
  header parsing produces `TensorSpec` entries without loading the full
  checkpoint payload. Shape validators cover the HF-layout text-tower tensors,
  and full scaled text-tower fixtures materialize through
  `load_text_runtime_weights(_from_safetensors)` into typed runtime
  `ModelWeights`.
- **Multimodal execution gate**: prompt preparation expands text, image, and
  video segments into token ids plus explicit media spans. Text-only prepared
  prompts execute through `moe_transformer_runtime.forward`; image/video spans
  raise `MiniMaxM3VisionExecutionError` unless projected media embeddings are
  supplied through the M5.5 splice path.
- **Guards**: `tests/unit/test_minimax_m3_importer.py` covers config mismatch
  rejection, HF tokenizer/chat-template parity, safetensors manifest/shape
  validation, full scaled text `ModelWeights` materialization from sharded
  fixtures, multimodal span construction, text-only execution, and image/video
  execution rejection.
- **Closure**: O3 and O4 are closed by the typed text-weight fixture and HF
  tokenizer parity tests. O5-O8 are closed in the M5.5/M6 media graph and
  execution gates below.

## M5.5 / M6 — projected-media runtime, reference vision tower, and JEPA contracts (landed)

- **Embedding-first decoder substrate**:
  `python/tessera/models/moe_transformer_runtime.py` now exposes
  `forward_embeds` and `prefill_embeds` alongside the token-id APIs. The old
  `forward` / `prefill` path is now a thin embedding lookup wrapper, and tests
  prove token IDs and precomputed embeddings produce identical logits/cache
  states.
- **MiniMax-M3 projected media splice**:
  `python/tessera/models/minimax_m3_importer.py` adds
  `splice_media_embeddings` plus projected-media and raw-media execution/prefill
  paths. Media placeholder token IDs do not need to be inside the scaled
  synthetic vocab: text positions are embedded from token IDs, media spans are
  replaced with caller-supplied `(span_tokens, hidden_size)` projector outputs
  or reference tower outputs, and missing or wrong-shaped media embeddings fail
  before decoder execution.
- **Decode after multimodal prefill**: `prefill_multimodal_prompt(...,
  media_embeddings=...)` feeds projected media through the same per-layer KV
  cache path as text. The guard compares the next decoded logits against full
  recompute over the spliced embedding prefix plus generated token embedding.
  A scaled raw image+video prompt now runs through the reference vision tower,
  multimodal prefill, and one cached decode step.
- **Reusable multimodal contracts**:
  `python/tessera/models/multimodal.py` adds `MediaSegment`, `MediaSpan`,
  `MediaBatch`, `PatchGrid`, `ProjectedMediaEmbeddings`, and
  `MediaProcessorConfig`, plus HF processor metadata import for
  `processor_config.json` / image-processor configs.
- **Reference vision tower**:
  `python/tessera/models/vision_transformer.py` implements numpy image/video
  preprocessing, patch embedding, patch merge/resampling, tiny ViT blocks, and
  projection into text hidden size. MiniMax-M3 exposes scaled executable vision
  metadata for local tests while full metadata remains a shape/import contract.
- **HF vision/projector mapping**:
  `minimax_m3_importer.expected_hf_vision_tensor_shapes` and
  `load_vision_runtime_weights` map HF-style `(out, in)` linear/patch/projector
  tensors into typed `VisionRuntimeWeights`. Tiny safetensors fixtures round-trip
  into the typed runtime and reproduce identical reference outputs; observed
  upstream-style vision/projector aliases are accepted, while missing aliases
  produce precise diagnostics.
- **MiniMax-M3 production media graph**:
  `minimax_m3.build_multimodal_graph` builds full production image+video shape
  contracts for preprocess/frame sampling, patch embed, patch merge, projector,
  and splice surfaces while scaled configs remain executable through the numpy
  reference tower.
- **Compiler-visible multimodal contracts**:
  Graph IR ops for `tessera.image_preprocess`, `tessera.video_frame_sample`,
  `tessera.patch_embed`, `tessera.patch_merge`, `tessera.media_project`, and
  `tessera.splice_embeddings` now lower as named `schedule.media.*` /
  `tile.media.*` artifacts and reach target-level backend contract kernels.
- **JEPA-ready latent contracts**:
  `tessera.jepa.{mask_blocks_2d,mask_tubes_3d,gather_context,gather_targets,
  stop_gradient,ema_update,latent_predict,l2_loss,train_step,selective_decode}`
  lower through
  `schedule.jepa.*` / `tile.jepa.*` and target artifact kernels. This gives
  multimodal JEPA work a shared mask/target/predictor/training/conditional
  decoder vocabulary without claiming a hardware-native training backend.
- **First-class JEPA model contract**:
  `python/tessera/models/jepa.py` adds an executable reference for deterministic
  2-D/3-D masks, context/target gathers, stop-gradient target latents, EMA
  target encoder updates, latent prediction/loss, multimodal shared-latent
  encoding, and optional selective decoding as a downstream consumer.
- **Guards**: `tests/unit/test_moe_transformer_runtime.py` covers
  token-vs-embedding parity; `tests/unit/test_minimax_m3_contract.py` covers
  production and scaled MiniMax-M3 media graph builds;
  `tests/unit/test_minimax_m3_importer.py` covers projected media splice, raw
  image/video tower execution, image+video prefill+decode, processor fixture
  parity, processor metadata import, text and vision safetensors weight mapping,
  alias diagnostics, and shape rejection; `tests/unit/test_jepa_model_contract.py`
  covers JEPA masks, deterministic training-step loss parity, EMA, multimodal
  latent contracts, and optional selective decoding;
  `tests/unit/test_multimodal_jepa_contracts.py` covers
  Graph→Schedule→Tile→NVIDIA Target contract preservation.
- **Closure**: O5-O10 are closed at the compiler/reference-contract layer. The
  remaining backend-native gaps are O1, O2, and O11-O13 in the aggregate table.

---

## Appendix — 2026-H1 Frontier Model Survey

> Consolidated from the former standalone `FRONTIER_MODEL_SURVEY_2026.md` (Decision #26 — fewer audit entry points; merged 2026-06-26).

> Status: **survey + grounded re-ranked roadmap**. No code landed yet.
> Companion to [`MODEL_CLASS_ROADMAP.md`](MODEL_CLASS_ROADMAP.md) (which stood up
> DeepSeek-V3.2 / GLM-5.2 / Kimi-K2 / MiniMax-M3, M0–M5 plus the MiniMax-M3
> text-tower graph/runtime/importer contracts). This doc covers the *next* wave — hybrid
> linear-attention models, dual-stream world models, native MTP, agent serving —
> and defines **Track L** (the linear-mixer keystone) as the M6-equivalent.
> Last updated: 2026-06-15.

This is a survey of ~12 recent model/system papers read at primary-source depth,
each distilled to the **specific execution contract** a compiler must model, then
**grounded against Tessera's actual code** (not the CLAUDE.md prose — see the
doc-drift note below). The output is a re-ranked roadmap whose ordering falls out
of a primitive dependency graph, not a wish list.

### Meta-finding: the contracts mostly *exist*; they don't *execute*

Tessera already has Graph IR ops + Python reference + "complete" contract-axis
status for nearly the whole surveyed frontier (`gated_deltanet`,
`deepseek_sparse_attention`, the reasoning-attention family, `moe_swiglu_block`,
`NativeSparseAttention`, `MixtureOfRecursions`). **Almost none execute on a
backend.** So the dominant lever is **promote reference → executable lowering**,
plus a smaller set of genuinely-absent *stateful/temporal* contracts. Tessera
already shipped the right validation tool for promotion: the evaluator's DESIL
cross-path + metamorphic oracles (`python/tessera/compiler/evaluator.py`) — a
fused kernel is auto-provable against its reference.

#### Grounding corrections (verified at source)

Two claims that floated through the audit, both checked against the tree:

1. **The false "Mamba2 op landed" claim was real — in `primitive_coverage.py`,
   not CLAUDE.md.** CLAUDE.md proper says the op is "*pending*"; but the coverage
   registry's comment asserted *"dedicated Mamba2 Graph IR op landed (2026-05-18)
   as `tessera.selective_ssm`"* and set `graph_ir_lowering="registered"` while **no
   ODS op existed** in `TesseraOps.td` — registry intent that outran the compiler
   surface (Decision #25/#26). **L4 closed this for real** (see below): the op is
   now materialized + lit-proven, so `registered` is finally honest. (`gated_deltanet`
   always did have its ODS op at `TesseraOps.td:1109`.)

2. **The real correctness finding (now test-proven):** the shipped
   `gated_deltanet` / `kimi_delta_attention` / `modified_delta_attention`
   reference (`__init__.py::_delta_attention_impl`, lines 1215-1225) computes
   `Ŝ_t = α_t·Ŝ_{t-1} + β_t·k_t v_tᵀ` — **gated linear attention, missing the
   DeltaNet `(I − β_t k_t k_tᵀ)` erase term**. The ODS summary ("Gated DeltaNet
   recurrent attention") and the runtime comment ("the delta recurrence is
   algebraically the quadratic form `(QKᵀ⊙mask)@V`", `runtime.py:6428`) describe
   the delta rule, but the math is linear attention — and every parity test passes
   because the GPU path faithfully matches the mislabeled reference.
   `tessera.stdlib.delta_rule` (Track L L1/L2, landed below) adds the genuine rule
   and the oracle that locks the distinction
   (`tests/unit/test_stdlib_delta_rule.py`).

### Per-model contract → Tessera status (grounded)

| Model | Verified compiler contract | Tessera today (file) |
|---|---|---|
| **Cosmos 3** (NVIDIA) [2606.02800] | **Two-launch varlen attention** with *independent q/k `cu_seqlens`* (causal launch + rectangular-block launch) — explicitly *not* a mask/bias (they beat FlexAttention 22% by moving structure into packing metadata). Plus **positional dual-tower weight binding** (segment→weightset, not MoE routing). | ❌ zero varlen surface; `attn_bias` is the wrong substrate. |
| **Nemotron-3 Super** (NVIDIA) [2604.12374] | Mamba2+**LatentMoE** paired blocks + sparse GQA anchors. LatentMoE = down-proj `ℓ=1024` → **route in latent** → `ℓ→ℓ` experts → up-proj (d/ℓ=4; 512 experts/top-22). **2 shared-weight recursive MTP** heads. NVFP4 per-layer dtype map (16-elem micro-blocks + BF16/MXFP8 islands). | SSM ref-only; no LatentMoE; MTP = Python orchestration; nvfp4 planned-gated dtype only. |
| **Qwen3.6-35B-A3B** (Alibaba) | `layer_types=[linear_attention×3, full_attention]×10`. **Gated DeltaNet dual-form**: chunked UT-transform GEMM (C=64, S∈[128,128], only non-GEMM step = C×C `(I−A)⁻¹`) for prefill / rank-1 gated state update for decode. 256e/top-8+1shared. 1 MTP layer. | `gated_deltanet` ODS exists (`TesseraOps.td:1071`); recurrent `linear_attn_f32` MSL kernel exists but **decay/state/β unwired** (`LinearAttnToAppleGPU.cpp:130`); **no chunked path anywhere**. |
| **Mellum2** (JetBrains) [2605.31268] | 3:1 SWA (window 1024) : full; **layer-selective YaRN** (only global layers); MTP self-speculation. | `attn_sliding_window` composes via bias; no per-layer-type RoPE table. |
| **LFM2.5-8B-A1B** (Liquid) | **LIV** double-gated short conv (depthwise causal k=3) in 18/24 layers; 32e/top-4. | `depthwise_conv1d` exists — closest existing primitive; not wired as a mixer. |
| **GLM-5.1** (Z.ai) [2602.15763] | DSA: **deterministic** top-k=2048 token gather, **all layers**, over MLA latent. Consistent-hash rollout→rank affinity; keep-recent-5 + discard-all@32k KV folding. (Non-deterministic top-k "caused drastic performance degradation" in RL.) | sparse-attn ref-only; no deterministic-topk primitive; no KV folding. |
| **MiniMax-M2** [2605.26494] | Dense GQA + **prefix-tree "compute-prefix-once, fork-branches" attention** + DFS radix global KV cache + windowed-FIFO (W=0.3N). MoE-under-branch dispatch undocumented (open Q). | scheduler single-request, resets cache each call (`dflash_serve.py:780`). |
| **DeepSeek-V4** [official report — REAL] | CSA: block-compress (m=4) → **FP4 lightning-indexer** → det top-k=1024 block gather → MQA, **interleaved with dense HCA** (m′=128). **Two-tier paged + SSM-state heterogeneous cache** keyed on compression-block boundaries. Per-*dim* FP8/BF16 KV split. | M4 `dsa_*` stdlib exists (block-index/select/sparse); paging metadata-only; no FP4 indexer; no heterogeneous cache. |
| **DiffusionGemma** (Google) | Block-AR + within-block parallel diffusion; **non-monotonic** accept/re-noise; `entropy_bound=0.1`; KV promote at block boundary only. | sampler already emits `accepted_mask`+`renoise_mask`, computes entropy then **discards it** (`models/sampler.py:77`). |
| **MemTrace** [2605.28732] | Bipartite **provenance DAG** (Variables w/ timestamps + Operations w/ In/Out sets) → "decisive error set" = earliest minimal faulty cut. | `memory_read/write/evict` + effect tracking exist; no In/Out provenance edges. |
| **YOLO26** (Ultralytics) [2606.03748] | Static-K=300, NMS-free (dual-head topk=7→1), DFL-free (`reg_max=1`) **fused decode → `f32[N,300,6]`**. | only `conv2d`/depthwise; zero detection tail. |
| **Holo3.1** (H Co.) | **Unpinnable** from primary sources (140ms / action-schema are third-party and contradict vendor's own 3.3s figure). | n/a — needs `config.json` inspection before any claim. |

### The cross-model convergence (where to bet)

**Architectural insight:** the SSM/linear-mixer recurrent state and the KV cache
are *the same heterogeneous-cache problem*. Nemotron's "constant Mamba state, KV
only on anchors," Qwen3.6's "dual recurrent-S + KV," and DeepSeek-V4's "two-tier
paged + SSM-state pool" are three statements of one contract. **Designing the
dual-form mixer's decode state correctly *is* designing the heterogeneous cache.**

Primitive dependency graph (build order falls out of this):

```
deterministic top-k ──────────► DSA/CSA sparse attention  +  RL bit-parity
varlen / packing attention ───► Cosmos dual-stream  +  CSA gather  +  prefix-fork
dual-form linear mixer ───────► Nemotron · Qwen3.6 · Mellum2 · LFM2.5   (4 flagships)
   └─ decode state == cache ──► heterogeneous cache (SSM-state ∪ windowed ∪ compressed)
MTP draft-head graph object ──► near-universal (every model here ships one)
```

Two tracks the original 8-item backlog **missed**: **deterministic top-k** (load-
bearing for all sparse attention + RL parity) and **sub-tensor mixed precision**
(NVFP4 micro-blocks + per-layer BF16/MXFP8 islands; per-dim KV split) — a concrete
`numeric_policy` extension, not a new dtype-by-fiat.

### Re-ranked roadmap

Validation rule (inherited from `MODEL_CLASS_ROADMAP.md`): a contract cell flips
to `complete` only when an oracle independently re-derives it. The natural oracle
for each promotion is named.

**Tier 0 — cheap, well-specified, Tessera already ~80% there:**
- **Diffusion commit-trace.** Sampler already produces `accepted_mask` +
  `renoise_mask` + per-position entropy — just retain a per-step
  `(step, position, entropy, accepted, renoised)` buffer. Non-monotonic semantics
  already modeled. Oracle: replay determinism. *Days.*
- **Memory provenance DAG.** MemTrace gives the exact schema; map In/Out sets +
  variable timestamps onto existing effect/dependency tracking + `MemoryStateHandle`.
  Hard rule: evict/overwrite must be an op with *both* In and Out edges or
  information-loss is un-attributable.
- **Deterministic top-k** (NEW, foundational). Unblocks all sparse attention + RL
  bit-parity. Oracle: metamorphic permutation-invariance of the selected set.

**Tier 1 — the keystone (4-flagship coverage): Track L, see below.**

**Tier 2 — new primitive classes:**
- **Varlen / two-launch attention** (Cosmos) — independent q/k `cu_seqlens`,
  causal + rectangular block. Same primitive is the substrate for CSA gather and
  prefix-fork → pays for itself 3×. Plus positional dual-tower weight binding.
- **LatentMoE** (precise spec) + close the `moe_swiglu_block` VJP/JVP gap.
- **MTP draft-head as a graph contract** — promote DFlash Python orchestration to
  an internal shared-weight, recursively-applied graph object. Near-universal.

**Tier 3 — serving + sparse + numerics (after their deps):**
- **Heterogeneous cache + prefix-shared/branch-fork attention** — common
  denominator of GLM sticky-hash / MiniMax radix-tree / DeepSeek two-tier. Couples
  with Track L's decode state.
- **DSA/CSA sparse attention executable** (block-compress → indexer → det-topk →
  varlen-gather → MQA). Depends on Tier-0 det-topk + Tier-2 varlen. Extends M4.
- **Sub-tensor mixed-precision `numeric_policy`** (NVFP4 micro-blocks, MXFP8
  islands, per-dim KV split).

**Scope decisions (explicit, not silent backlog):** **YOLO26** is a clean static-
shape NMS-free decode but the weakest strategic fit for an LLM compiler — decide
yes/no, don't drift. **Holo3.1 / LFM2.5-offload** contracts are unpinnable from
primary sources — don't plan against them until `config.json`/inference code is
inspected.

---

### Track L — dual-form linear mixer + hybrid schedule (the keystone)

Milestone ladder, mirroring `MODEL_CLASS_ROADMAP.md`. Unblocks Nemotron-3,
Qwen3.6, Mellum2 (schedule), LFM2.5 (LIV variant). "Definition of done" =
the two provable claims: full-config artifact (lit + verifier + conformance) and
scaled execution on Apple GPU gated vs numpy.

#### Accurate starting state (grounded)

| Piece | Status | Anchor |
|---|---|---|
| `gated_deltanet` ODS op (q/k/v/gate/β/decay/state, return_state, state_dtype) | ✅ exists | `TesseraOps.td:1071` (base `Tessera_DeltaAttentionOp`) |
| Python emit path | ✅ | `runtime.py:6408` (`_gated_deltanet`) |
| Recurrent Apple GPU kernel `linear_attn_f32` | 🟡 partial | `apple_gpu_runtime.mm:8224`; pass `LinearAttnToAppleGPU.cpp` |
| — but **β/decay/state unwired** (→ it's plain linear attn, not the gated delta rule) | ❌ gap | `LinearAttnToAppleGPU.cpp:130` |
| — f32 / rank-4 / `D_qk·D_v ≤ 256` / causal only | constraint | `LinearAttnToAppleGPU.cpp:84-125` |
| **Chunked UT-transform prefill path** | ❌ absent | none in tree (prefill is O(S) sequential) |
| `selective_ssm` ODS op (Mamba2) | ❌ absent | python ref + JVP only (`autodiff/jvp.py`) |
| Hybrid `layer_types` as first-class schedule attr | ❌ absent | none |

#### Ladder

| L | Title | Definition of done | Oracle |
|---|---|---|---|
| **L0** ✅ | Grounding correction + contract lock | Correct the propagated misquote; document the *real* finding (delta family = linear attn, no erase); lock it with an oracle | `test_existing_gated_deltanet_is_linear_attention_not_delta` |
| **L1** ✅ | **Genuine gated delta recurrence** (decode form) | `gated_delta_rule_recurrent` adds the `(v_t − α_t v̂_t)` erase, fp32 state, return_state; **not** "wire β/decay" (those were already wired) — the existing reference was missing the erase entirely | vs independent brute-force `(I−βkkᵀ)` recurrence; `erase=False` ≡ existing ref; state-carry |
| **L2** ✅ | **Chunked UT-transform prefill (the keystone)** | `gated_delta_rule_chunked`: chunk C, `Ã=tril(β·γ-ratio·KKᵀ,−1)`, `(I+Ã)⁻¹` via explicit forward substitution (`_forward_substitution`), WY/output as GEMM, γ-decay folding, cross-chunk state carry | **chunk ≡ recurrent** across ungated/β/fully-gated/output-gated + chunk-size-invariant (the make-or-break proof) |
| **L1.1** ✅ | Genuine delta rule on Metal (decode form) | `tessera_apple_gpu_gated_delta_rule_f32` — per-(b,h) sequential MSL scan with the erase; `backend="apple_gpu"` on the recurrent reference | **Metal ≡ numpy** (DESIL) + Metal ≡ L2 chunked (independent routes) |
| **L2.1** ✅ | Chunked UT-transform on Metal (prefill form) | `tessera_apple_gpu_gated_delta_rule_chunked_f32` — one threadgroup per (b,h), the within-chunk `(I+Ã)⁻¹` solve on-device; `backend="apple_gpu"` on the chunked reference | **Metal chunked ≡ numpy** (all chunk sizes incl. partial) + **Metal chunked ≡ Metal recurrent** |
| **L2.2** ✅ | Cooperative-parallel chunk kernel | **Measured headroom found** at high occupancy (256+ threadgroups), where L2.1's lane-0 advantage shrinks to ~1.3× over recurrent. Key insight: the within-chunk solve's **d_v columns are independent chains** → each thread owns columns and solves **barrier-free**; state carry parallelizes over cells. `coop=True` (default). | **L2.2 ≡ L2.1 ≡ numpy** (correctness) + **2.3× over L2.1 lane-0, 2.9–3.1× over recurrent** (measured, high occupancy) |
| **L3** ✅ | **Hybrid layer schedule** as a first-class attribute | `HybridSchedule` lowers `layer_types` literally; reference stack threads the **dual cache** (recurrent Ŝ for linear layers, KV for full layers) | **streaming dual-cache decode ≡ full recompute** + Qwen3.6 full-config schedule check |
| **L3.1** ✅ | `gated_deltanet` shipped-numerics decision | **Decision: opt-in, don't flip.** Added `erase=False` (default = current linear attn, backward-compatible) to `gated_deltanet`/`kimi_delta`/`modified_delta`; `erase=True` is the genuine rule. Flipping the default would break every caller's numerics — a future major version may. ODS `erase` attr deferred until graph→kernel honors it (no-op attr = drift) | `erase=True` ≡ `stdlib.delta_rule`; default ≡ existing; no regression |
| **L4** ✅ | `selective_ssm` (Mamba2) ODS op | Materialize the op the registry falsely claimed: `Tessera_SelectiveSsmOp` + verifier; close the drift. Chunk-scan (`_mamba_ssd.py`) + chunk≡sequential oracle already existed | lit roundtrip/verifier + chunk-scan ≡ sequential-scan |
| **L4.1** ✅ | Hybrid SSM mixer (Nemotron) | `linear_mixer="ssm"` adds a Mamba SSM mixer to the L3 hybrid stack; SSM state `h[D,N]` carried in the dual cache alongside attention-anchor KV | dual-cache decode ≡ recompute with SSM layers + `_ssm_scan` ≡ shipped `selective_ssm` |
| **L5** ✅ | LFM2.5 LIV mixer variant | `linear_mixer="liv"` — `Linear→(B⊙x̃)→depthwise-causal-conv(k=3)→(C⊙z)→Linear_out`; constant conv state (last k-1) in the dual cache | dual-cache decode ≡ recompute + conv causality |
| **Full blocks** ✅ | MoE FFN + MTP draft head | `ffn="moe"` (routed experts + shared, exact per-token routing); `HybridLM` + graph-level MTP head + lossless self-speculation | decode ≡ recompute (MoE); spec == AR (MTP, lossless) |

#### Landed 2026-06-15 — L0–L2 (reference tier, host-free)

`python/tessera/stdlib/delta_rule.py` + `tests/unit/test_stdlib_delta_rule.py`
(18 oracles, all green). Mirrors the M-series house pattern: numpy reference +
oracle first, fused MSL kernel (L1.1/L2.1) as the hardware-gated follow-up. What
is proven host-free: the genuine gated delta rule (recurrent ≡ independent
brute-force in the paper's `(I−βkkᵀ)` layout), the chunk-parallel UT-transform
(`chunk ≡ recurrent` across all gating modes, chunk-size-invariant), the
triangular-solve primitive, cross-chunk state carry, and the L0 lock that the
shipped `gated_deltanet` is linear attention (`erase=False`), materially distinct
from the true rule when keys correlate.

#### Landed 2026-06-15 — L1.1 (genuine delta rule on Metal)

`tessera_apple_gpu_gated_delta_rule_f32` (`apple_gpu_runtime.mm` + non-Darwin
stub parity) — a per-(b,h) sequential MSL scan carrying the `(v_t − α·v̂_t)`
erase, registered in `_apple_gpu_backend` and reachable via
`gated_delta_rule_recurrent(..., backend="apple_gpu")`.
`tests/unit/test_apple_gpu_gated_delta_rule.py` (7 oracles, all run on Metal):
**Metal ≡ numpy** for true delta / β+decay / output-gate / non-square head dims,
`erase=False` ≡ the shipped linear reference, and **Metal recurrent ≡ the L2
chunked UT-transform** (the genuine rule reached by two fully independent routes).

**Numerics finding (realism, → `numeric_policy`):** the delta rule is only
well-conditioned with **L2-normalized keys** — then `β·‖k‖²=β<1` makes
`(I−βkkᵀ)` a contraction and f32≡f64. With unnormalized keys (`‖k‖²≫1`) the
recurrence *expands* (eigenvalue `1−β‖k‖²<0`) and f32 legitimately diverges from
f64 (~10% here) — genuine ill-conditioning, not a kernel defect. A production
`gated_delta_rule` op should carry key-normalization in its contract (real models
do), with fp32 state accumulation.

#### Landed 2026-06-15 — L2.1 (chunked UT-transform on Metal)

`tessera_apple_gpu_gated_delta_rule_chunked_f32` — one threadgroup per (b,h), C
threads (one per token-in-chunk), state Ŝ in threadgroup memory across the chunk
loop.  The GEMM-shaped rows (A, W̃, output) parallelize across threads; the
`(I+Ã)⁻¹` forward substitution + rank-1 state carry run on lane 0 (cooperative
parallelization of those is L2.2, a perf follow-up).  Reachable via
`gated_delta_rule_chunked(..., backend="apple_gpu")`.  Oracles (in the same test
file): **Metal chunked ≡ numpy** across chunk sizes 1/4/8/16/32 (S=20 exercises a
partial last chunk), **Metal chunked ≡ Metal recurrent** (two independent
on-device kernels), output-gate, and `erase=False` ≡ the shipped linear ref.

#### Landed 2026-06-15 — L3 (hybrid layer schedule + dual cache)

`tessera.stdlib.hybrid` — `HybridSchedule` makes `layer_types` first-class
(`qwen3_6_schedule` = `[lin,lin,lin,full]·N`; `nemotron_schedule` = sparse
anchors) + a reference stack that threads the **dual cache**: constant-size
recurrent Ŝ for linear (genuine gated-delta) layers, growing KV for full-attention
layers.  Linear layers L2-normalize keys (the L1.1 conditioning finding).
`tests/unit/test_stdlib_hybrid.py` (9 oracles): the headline **streaming
dual-cache decode ≡ full recompute** across prefill points and schedules
(all-linear, every-other-anchor), schedule validation, and the Qwen3.6 full-config
check (30 linear / 10 full at 40 layers).

**Not yet done:** L2.2 (cooperative-parallel chunk kernel — perf), L3.1 (promote
the `gated_deltanet` ODS op to the true rule — a shipped-numerics decision), and
MoE/MTP composition into the hybrid stack (`stdlib.moe` exists; wiring is additive).

#### Landed 2026-06-15 — L4 (`selective_ssm` Mamba2 ODS op)

`Tessera_SelectiveSsmOp` (`src/compiler/ir/TesseraOps.td`) + `SelectiveSsmOp::verify`
(`TesseraOps.cpp`) — `tessera.selective_ssm` is now a genuine Graph IR op
(rank-checked: rank-3 x / shape-equal delta / matching b,c / A rank-1|2 / optional
`gate` shape-equal x / `init` state rank-3). `tessera-opt` rebuilds clean (MLIR
22.1.6); lit fixture `tests/tessera-ir/model_class/selective_ssm.mlir` passes
(model_class sweep 5/5). The **drift is closed**: the coverage registry's
`graph_ir_lowering="registered"` for `selective_ssm` is now backed by a real op
(comment corrected in `primitive_coverage.py`). The chunked-parallel SSD lowering
(`_mamba_ssd.py::selective_ssm_parallel`) and its **chunk ≡ sequential** oracle
(`test_mamba_ssd_gpu.py`, 12 tests) already existed and stay green.

#### Landed 2026-06-15 — L4.1 (hybrid SSM mixer — Nemotron expressible)

`tessera.stdlib.hybrid` now takes `linear_mixer = "delta" | "ssm"`. The stack was
refactored to per-mixer **span functions** so `hybrid_forward` (one span) and
`hybrid_decode` (streamed spans) run identical per-layer code — the dual-cache
oracle is meaningful for all three mixer types (delta Ŝ, SSM h[D,N], attention
KV).  `_ssm_scan` reproduces the shipped `tessera.ops.selective_ssm` (the L4 op's
reference) **and returns the carried state** (which the public reference does
not), so streaming SSM decode is exact.  Nemotron is now the second flagship
(after Qwen3.6) expressible end-to-end: `nemotron_schedule` + `linear_mixer="ssm"`
= Mamba layers + sparse attention anchors.  `tests/unit/test_stdlib_hybrid.py`
(+5 L4.1 oracles): `_ssm_scan ≡ selective_ssm`, Nemotron-shaped dual-cache decode
≡ recompute (across prefill points), and an all-SSM stack.

#### Landed 2026-06-15 — L5 + MoE/MTP (full model blocks) + L3.1

**L5 (LIV):** `linear_mixer="liv"` — the LFM2.5 double-gated causal short-conv
(`Linear→B⊙x̃→depthwise-conv k=3→C⊙z→Linear_out`), constant conv state (last k-1
inputs) in the dual cache. Third hybrid family expressible. **MoE FFN:** `ffn="moe"`
wires `stdlib.moe` (routed experts + optional shared) with **exact, no-capacity-drop
routing** so it stays per-token → decode≡recompute holds; `top_k` proven
load-bearing; a Nemotron SSM+MoE full block runs. **MTP:** `HybridLM` (tied-embed
LM head + a shared-weight MTP draft head predicting t+2 from `h_t` + `embed(t+1)`)
+ `mtp_speculative_generate` — greedy self-speculation that is **lossless (== AR)**
by construction, with the accept-path exercised on a constructed predictable model.
27 oracles in `test_stdlib_hybrid.py`.

**L3.1 — the `gated_deltanet` shipped-numerics decision (made):** added
`erase=False` (default = current gated *linear* attention, **backward-compatible**)
to `gated_deltanet`/`kimi_delta`/`modified_delta`; `erase=True` opts into the
genuine DeltaNet rule (≡ `stdlib.delta_rule`). The default is **deliberately not
flipped** — that would silently change every caller's numerics; a future major
version may. The ODS `erase` attr is deferred until the graph→kernel path honors
it (a no-op attr would just be new drift). Guards in `test_stdlib_delta_rule.py`;
no regression in the existing delta/attention suites.

#### Landed 2026-06-15 — L2.2 + erase routing + named models

**L2.2 (cooperative chunk kernel):** the L2.1 chunked kernel gained a `coop` mode
(default on) — the within-chunk `(I+Ã)⁻¹` solve parallelizes its independent d_v
column-chains across threads **barrier-free**, and the state carry parallelizes
over cells. Measured (`benchmark_delta_rule.py`, high occupancy): **2.3× over L2.1
lane-0, 2.9–3.1× over recurrent**, both modes bit-equal to numpy. The earlier
"deferred" call was overturned by sharper data (the lane-0 cost only shows at
high occupancy). **graph→Metal `erase` routing:** `gated_deltanet(erase=True)` on
`@jit(target="apple_gpu")` now runs the genuine rule on Metal (the L1.1 kernel),
not the composed linear form — verified e2e (`test_apple_gpu_delta_erase_routing.py`).
**Named models:** `models/{qwen3_6,nemotron3,lfm2_5}.py` wire the full-block stack
into named `config()`/`scaled_config()` factories (shapes match published configs;
scaled instances run decode≡recompute).

**Still open:** the ODS `erase` attr (deferred until lowering honors it), LatentMoE
(distinct from standard MoE) for a weight-faithful Nemotron, and per-layer-type
head dims / short-conv for a weight-faithful Qwen3.6.

Sequencing: **L1 unblocks L2** (decode state is the chunk carry); **L2 is the
keystone** (only the chunked GEMM form is tensor-core-viable for prefill — the
papers are unanimous); L3 is independent of L1/L2; L4 reuses L2; L5 is parallel.

#### The one hard kernel detail

The only non-GEMM step in the chunked form is the within-chunk `T=(I−A)⁻¹` where
`A` is strictly-lower-triangular C×C — solved by forward substitution, **not** a
GEMM, and it is the throughput bottleneck on accelerators. It wants a dedicated
tile primitive (C=64 → a 64×64 unit-lower-triangular solve). Everything else
(WY factors `W=TβK`, `U=TβV`; cross-chunk `S_new=S_prev+(U−W S_prevᵀ)ᵀK`; intra-
chunk `O=QS_prevᵀ+(QKᵀ⊙M)(U−W S_prevᵀ)`) is tensor-core GEMM. **State accumulates
in fp32** regardless of bf16 storage — the `(I−A)⁻¹` and rank-updates are
numerically sensitive.

### Sources

Primary sources (verified at depth) live inline in the per-model table above:
Cosmos 3 [arXiv:2606.02800 + NVIDIA tech report], Nemotron-3 Super
[arXiv:2604.12374], Qwen3.6 [HF config.json] + Gated Delta Networks
[arXiv:2412.06464], Mellum2 [arXiv:2605.31268], LFM2.5 [Liquid blog + LFM2
report arXiv:2511.23404], GLM-5 [arXiv:2602.15763], MiniMax-M2 [arXiv:2605.26494],
DeepSeek-V4 [official report, HF deepseek-ai/DeepSeek-V4-Pro], DiffusionGemma
[Google ai.google.dev docs], MemTrace [arXiv:2605.28732], YOLO26
[arXiv:2606.03748 + Ultralytics docs], Holo3.1 [hcompany.ai — existence only].
