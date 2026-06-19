# Model-Class Roadmap — compiling Kimi-K2 / DeepSeek-V3.2 / GLM-5.2 / MiniMax-M3

> Status: **M0–M5 landed** (scaffolding + quant + MoE + MLA + DSA + full stack &
> autoregressive decode), plus the M3.1/M4.1 composed Apple GPU attention lanes
> and the MiniMax-M3 text-tower graph contract/runtime decode/artifact KV-outer
> lowering path plus local HF tokenizer/safetensors import gates and projected
> media embedding splice/reference vision tower.
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
metadata import is now staged for MiniMax-M3; full weight materialization,
serving, and real vision/video tower execution remain later workstreams. This
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

| ID | Area | Open item | Close condition / acceptance |
|----|------|-----------|------------------------------|
| O1 | Attention kernels | Fused MSL attention kernels for MLA absorb / DSA block-sparse paths remain open; current Apple GPU support is composed over existing matmul lanes. | Single fused Apple GPU kernel path lands with parity vs numpy/reference and a perf ratchet that beats the composed path on representative scaled configs. |
| O2 | MiniMax-M3 MSA backend | Native NVIDIA CUDA/H800/Blackwell KV-outer sparse MSA kernel is still artifact-only. | `tessera.attn.msa_kv_outer_sparse` lowers to a native executable backend kernel with dense-equivalence oracle, decode/prefill coverage, and target-specific lit/runtime gates. |
| O3 | Full text checkpoint materialization | MiniMax-M3 text safetensors are parsed and shape-validated, but full HF checkpoint materialization into `ModelWeights` is not complete. | Full text-tower tensor-name map loads at least one realistic sharded HF-style fixture into typed runtime weights; selected layer outputs match the synthetic/reference loader layout expectations. |
| O4 | Tokenizer parity | Importer has deterministic vocab/byte fallback but not exact HF tokenizer/BPE/chat-template parity. | HF tokenizer path can round-trip fixture prompts, special image/video tokens, and chat templates with token IDs matching upstream tokenizer output. |
| O5 | HF processor pixel parity | Reference image/video preprocessing is executable, but exact MiniMax/HF processor pixel parity is not claimed. | `processor_config.json` / image processor fixtures reproduce upstream resize/rescale/normalize/frame-sampling outputs within a defined numeric tolerance. |
| O6 | Full vision/projector checkpoint aliases | Vision/projector safetensors mapping has a stable test contract, but broad HF tensor-name alias coverage is incomplete. | Loader accepts observed upstream MiniMax-M3 vision/projector key aliases and rejects ambiguous/missing aliases with precise diagnostics. |
| O7 | Native/fused vision tower lowering | Reference `vision_transformer` execution is numpy-level; backend-native/fused patch embed, ViT block, patch merge, and projector lowering are not complete. | Full multimodal graph lowers through native/composed backend kernels for patch/project/splice surfaces with parity against the reference tower. |
| O8 | Full MiniMax multimodal execution at production geometry | Scaled raw image/video prompts run through the reference tower; full production geometry remains shape/import contract. | Full MiniMax-M3 image+text and video+text graph builds at production dimensions, and a scaled execution gate covers image and video prompts end-to-end with decode after multimodal prefill. |
| O9 | JEPA native training lowering | JEPA model contract and artifact ops exist, but native training lowering for masks, stop-grad/EMA state updates, latent prediction, and losses is not complete. | JEPA training step lowers as a compiler-visible stateful training graph with mask RNG determinism, EMA update semantics, and latent-loss parity against the reference. |
| O10 | VL-JEPA selective decoder integration | Selective decode exists as a reference downstream consumer, not an integrated compile-time conditional decoder flow. | Conditional/selective decoder graph compiles with latent-score gating, retrieval/classification/decode branches, and tests proving decoder invocation is data-dependent and optional. |
| O11 | Quantized model-weight runtime bridge | INT4/FP8 packed dequant kernels are landed, but some model-family full-weight runtime paths still use synthetic/reference weights. | Frontier model configs can load quantized weight fixtures into runtime-compatible typed weights and hit the fused dequant path in model-level tests. |
| O12 | Quantized packed-byte fused-kernel memory path | The fused dequant-GEMM kernel exists, but native packed-byte INT4 nibble / FP8 operands still need the full no-materialization memory path. | Fused dequant kernels consume packed-byte operands directly, preserve separate scale operands, avoid full-weight materialization, and pass parity/perf ratchets. |
| O13 | Distributed/full-scale execution | Full-scale execution, distributed MoE, and NVIDIA FP8/sparse performance remain hardware-gated. | Hardware-backed CI or reproducible artifact gate covers full-scale launch metadata, distributed routing, and target-specific performance/regression thresholds. |

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
- **MiniMax-M3** in this slice is a text-tower compiler contract with scaled
  MSA runtime decode parity, artifact KV-outer lowering, and local
  tokenizer/safetensors import gates. Open closure items O2-O8 track the
  remaining MiniMax-M3 backend, checkpoint, tokenizer, processor, and
  multimodal execution work.

## Open work

The authoritative closure list is **Open Closure Backlog** above. Short map:
fused/composed kernel promotion is O1 and O12; MiniMax-M3 native MSA is O2;
MiniMax-M3 checkpoint/tokenizer/processor/multimodal closure is O3-O8; JEPA
native training and selective decode are O9-O10; model-weight and hardware-gated
execution closure is O11-O13.

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
- **Tokenizer contract**: tokenizer import records special tokens, chat-template
  presence, and a deterministic `VocabTokenizer` surface when `tokenizer.json`
  carries a vocab map. Full BPE parity remains a frontend/importer integration
  concern; missing tokenizer files are rejected when requested.
- **Safetensors manifest**: dependency-free safetensors header parsing produces
  `TensorSpec` entries without loading the full checkpoint payload. Shape
  validators cover the HF-layout text-tower tensors used by the Tessera runtime
  mapping work, and selected named tensors can be materialized from a
  file/index/directory without loading every shard.
- **Multimodal execution gate**: prompt preparation expands text, image, and
  video segments into token ids plus explicit media spans. Text-only prepared
  prompts execute through `moe_transformer_runtime.forward`; image/video spans
  raise `MiniMaxM3VisionExecutionError` unless projected media embeddings are
  supplied through the M5.5 splice path.
- **Guards**: `tests/unit/test_minimax_m3_importer.py` covers config mismatch
  rejection, tokenizer import, safetensors manifest/shape validation, multimodal
  span construction, text-only execution, and image/video execution rejection.
- **Open closure items**: O3-O8 track full HF checkpoint mapping into
  `ModelWeights`, exact BPE tokenizer parity, exact HF processor pixel parity,
  and native/fused multimodal tower execution.

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
  into the typed runtime and reproduce identical reference outputs.
- **Compiler-visible multimodal contracts**:
  Graph IR ops for `tessera.image_preprocess`, `tessera.video_frame_sample`,
  `tessera.patch_embed`, `tessera.patch_merge`, `tessera.media_project`, and
  `tessera.splice_embeddings` now lower as named `schedule.media.*` /
  `tile.media.*` artifacts and reach target-level backend contract kernels.
- **JEPA-ready latent contracts**:
  `tessera.jepa.{mask_blocks_2d,mask_tubes_3d,gather_context,gather_targets,
  stop_gradient,ema_update,latent_predict,l2_loss}` lower through
  `schedule.jepa.*` / `tile.jepa.*` and target artifact kernels. This gives
  multimodal JEPA work a shared mask/target/predictor vocabulary without
  claiming a full training backend.
- **First-class JEPA model contract**:
  `python/tessera/models/jepa.py` adds an executable reference for deterministic
  2-D/3-D masks, context/target gathers, stop-gradient target latents, EMA
  target encoder updates, latent prediction/loss, multimodal shared-latent
  encoding, and optional selective decoding as a downstream consumer.
- **Guards**: `tests/unit/test_moe_transformer_runtime.py` covers
  token-vs-embedding parity; `tests/unit/test_minimax_m3_importer.py` covers
  projected media splice, raw image/video tower execution, prefill+decode,
  processor metadata import, vision safetensors shape/weight mapping, and shape
  rejection; `tests/unit/test_jepa_model_contract.py` covers JEPA masks,
  latent prediction/loss, EMA, multimodal latent contracts, and selective
  decoding;
  `tests/unit/test_multimodal_jepa_contracts.py` covers
  Graph→Schedule→Tile→NVIDIA Target contract preservation.
- **Open closure items**: O5-O10 track exact MiniMax/HF processor parity, full
  checkpoint tensor-name aliases beyond the stable test contract,
  native/fused vision-tower lowering, native JEPA training lowering, and
  selective decoder integration.
