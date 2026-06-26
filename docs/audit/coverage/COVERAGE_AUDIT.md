---
last_updated: 2026-06-21
audit_role: theme
---

# Coverage Audit

This document consolidates primitive, op, example, KV-cache, and coverage audit
material.

> **Counts live in `docs/audit/generated/`, never in this prose.** Per
> Decision #25/#26, every numeric coverage claim is owned by a drift-gated
> generated dashboard. This page states *qualitative* status and **links** to
> the dashboard that holds the live number — it does not copy counts (a copied
> number silently goes stale). When you need a figure, read the linked
> dashboard.

## Finished

- **Partial-op uplift closed the legacy partial rows.** E2E op coverage now
  shows no `partial` / `planned` rows — see
  [`generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md).
- **E2E op pipeline is native-complete or runnable-reference end to end** (no
  partial/planned tail). Live native-complete / runnable-reference split:
  [`generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md).
- **`lowering_rule` is closed project-wide** (0 open across all S-series
  categories) — see [`generated/s_series_status.md`](../generated/s_series_status.md).
- **No actionable direct-test-debt** (`needs_direct_test = 0`) — see
  [`generated/test_coverage.md`](../generated/test_coverage.md)
  for the full classification (covered-by-family / structural-only /
  hardware-gated breakdown).
- Advanced examples largely moved from missing APIs to backend/hardware proof.
- KV-cache coverage has explicit target diagnostics and historical matrices.
- **Manifold Langevin backend coverage (2026-06-02):** the EBM/manifold
  Langevin ops moved off `backend_kernel=planned` — `ebm_sphere_langevin_step`
  + `ebm_bivector_langevin_step` are now `partial` with a **real fused Apple
  GPU kernel** (sphere: dedicated MSL; bivector: the affine `ebm_langevin_step`
  kernel on grade-2 coeffs), and the two chain wrappers
  (`*_langevin_sample`) are `partial` via their numpy reference. See
  [`generated/s_series_status.md`](../generated/s_series_status.md) /
  [`generated/apple_target_map.md`](../generated/apple_target_map.md). (Their
  *distributed-mesh* axis stays Phase-G-gated — see the `hardware_gated` row in
  [`generated/test_coverage.md`](../generated/test_coverage.md);
  single-device kernel ≠ multi-GPU mesh.)

## Still Open

- **Backend-kernel proof is open on every S-series primitive** — a universal
  Phase-G/H/I gate (each entry needs *all* declared targets to ship real
  kernels; gated on NVIDIA/ROCm hardware). Live open/complete counts:
  [`generated/s_series_status.md`](../generated/s_series_status.md).
- **Long-tail transform axes** — `batching_rule` and `transpose_rule` are now
  closed; `sharding_rule` is the remaining increment (2026-06-02).
  `batching_rule` closed for the textbook-batchable families (collective /
  recurrent / state_space / linalg decomposition+solver / sparse /
  segment_reduce) — only the genuinely mesh-aware ones (moe / moe_transport /
  kv-cache state) stay partial. `transpose_rule` closed to **zero open** on the
  linear-vs-nonlinear principle: linear maps (sparse spmm/sddmm/bsmm,
  moe_transport gather/scatter adjoints, segment_reduce, tri_solve, avg_pool)
  are `complete`; nonlinear families (optimizers, recurrent cells, linalg
  decomposition, ebm energy/sampling, moe routing, max/min/adaptive pool) are
  `not_applicable` (their backward is the registered VJP, not a linear
  transpose). `sharding_rule` remains largely Phase-G-mesh-pending. Live open
  counts + per-category breakdown:
  [`generated/s_series_status.md`](../generated/s_series_status.md).
- **Hardware-gated tests** remain for a small set of EBM/manifold Langevin ops
  — see the `hardware_gated` row in
  [`generated/test_coverage.md`](../generated/test_coverage.md).

### VLM coverage gap (2026-06-21)

Audit of the registry against the broader VLM landscape (LLaVA, Qwen2-VL /
Qwen2.5-VL, Flamingo / BLIP-2, Idefics3 / SmolVLM, Fuyu / encoder-free Gemma-4 —
the last surfaced by HF's *Train Your Own Encoder-Free VLM in $100*). Headline:
**the heavy vision compute already ships; the VLM-specific connector /
preprocessing / fusion layer was entirely untracked.** A VLM forward pass is
already expressible through existing ops — `conv2d`/`conv3d` (ViT/SigLIP patch
stem, apple_gpu `hardware_verified`), `flash_attn` / `multi_head_attention` /
`gqa_attention`, `varlen_sdpa` (the knapsack / `cu_seqlens` sequence packing the
encoder-free post relies on), `attn_local_window_2d` (Qwen2-VL window attn),
`gated_attention` (Flamingo-style gated x-attn), `layer_norm` / `rmsnorm`,
`linear_general` / `lora_linear`, and `cross_entropy_loss` with `ignore_index`
masking for image/pad tokens. The gap is the **glue**, not the math.

**P0 — landed as `partial` (Python reference + autodiff; 2026-06-21).** Each
ships a numpy reference on `tessera.ops.*`, a registered VJP **and** JVP
(`vjp`/`jvp` axes `complete`), and tape integration; Graph IR lowering +
backend kernels remain the open axes (`lowering_rule` / `backend_kernel`). A new
`vlm` model-family groups them — filter `render_markdown()` from
`tessera.compiler.primitive_coverage` to the `vlm` family, or see the drift-gated
[`generated/test_coverage.md`](../generated/test_coverage.md) /
[`.csv`](../generated/test_coverage.csv). (Note: `generated/support_table.md` is
an *existing-op* surface whose "Family" column is the primitive *category*, not
the model-family.) Tests: `tests/unit/test_vlm_primitives.py` (forward numerics
+ finite-difference VJP/JVP + tape end-to-end). Shared resample/layout helpers:
`python/tessera/_image_ops.py`.

- `masked_scatter` (`indexing`) — modality fusion: overwrite the embeddings at a
  boolean image-placeholder mask with the projected patch embeddings
  (`combined[mask] = image_embd`). The single most VLM-defining op; previously
  only `masked_fill` (scalar) and index `scatter` existed, not the tensor-source
  variant. Supports the VLM `(B,S)`-mask-over-`(B,S,D)` partial-indexing pattern
  and `torch.masked_scatter` flatten semantics.
- `image_resize`, `center_crop`, `image_normalize` (`vision`) — the three-step
  standard preprocessing transform (resize shorter side → center crop →
  per-channel affine). No pixel ops existed; the `data` category is
  dataset-plumbing only.
- `interpolate` (`vision`) — bilinear/nearest resample for variable-resolution
  inputs (NaViT / dynamic-res Qwen2-VL) and positional-embedding-table
  interpolation; shares the (exact, transpose-of-resample) VJP with
  `image_resize`.

**P1 — patch embedder, landed as `partial` (2026-06-21).** Two atomic
primitives with numpy reference + registered VJP/JVP:

- `patchify` (`layout_transform`) — `(B,C,H,W) → (B,nh*nw,C*P*P)` reshape/permute
  (the article's recipe); standalone + differentiable (VJP = inverse permute),
  layout-aware (nchw/nhwc/chw/hwc).
- `factorized_pos_emb` (`position_encoding`) — `pos[i,j] = row[i] + col[j]` over
  the patch grid, the exact Gemma-4 embedder trick (8× fewer params than a full
  table); differentiable through both tables, unused table rows get zero grad.

  *Pre-existing-op note:* the registry tour missed that
  `tessera.ops.patch_embed` (NHWC patchify + optional projection) and
  `tessera.ops.patch_merge` (Qwen-style token merge) **already ship** in
  `__init__.py` — but were absent from the coverage registry (untracked). The
  new ops sit alongside `patch_embed` rather than replacing it; the Gemma
  embedder composes `patchify → matmul → factorized_pos_emb` and is verified
  differentiable end-to-end through the tape
  (`test_gemma_embedder_composition_tape`). Registering the existing
  `patch_embed` / `patch_merge` / `video_frame_sample` ops in the coverage
  registry is follow-up tracked under P2/next-work.

**P1 — `mrope_2d` landed as `partial` (2026-06-21).** Multimodal M-RoPE
(`rotary_embedding`): rotary split into temporal/height/width sections by a
per-axis `positions` tensor and `sections` partition (Qwen2-VL); reduces exactly
to `rope` for a single section (test-pinned), norm-preserving, VJP+JVP
registered. Only the backend-kernel axis remains.

**P2 — landed as `partial` (2026-06-21).**

- `pixel_unshuffle` / `pixel_shuffle` (`layout_transform`) — space-to-depth /
  depth-to-space token reduction (Idefics3 / InternVL); exact inverses
  (roundtrip test-pinned), VJP = the inverse rearrange, JVP registered.
- `cross_attention` (`attention`) — scaled-dot-product attention where the query
  attends to a separate K/V source (Flamingo / BLIP-2 / Q-Former). Real
  reference SDPA with a hand-written analytic VJP **and** JVP (forward-mode
  through the softmax Jacobian), both finite-difference-pinned.
- `perceiver_resampler` (`attention`) — learned latents cross-attend to a
  variable-length feature sequence, compressing it to `len(latents)` tokens. A
  composite over `cross_attention`, differentiable through the tape
  (verified by `test_perceiver_resampler_compresses_and_is_differentiable`).

The P0/P1 rows are `partial` (Python reference + autodiff complete; Graph IR
lowering + backend kernel are the open axes). Live status:
[`generated/test_coverage.md`](../generated/test_coverage.md).

## Next Work

1. Treat generated dashboards as the only count authority.
2. Close backend-kernel proof through platform backend work.
3. Prioritize remaining batching/transpose/sharding gaps by model impact.
4. Keep KV-cache status tied to runtime/conformance proof, not only Graph IR
   lowering.
5. Update example status only when generated support/e2e dashboards agree.
6. Promote the VLM P0 rows off `planned` (Python reference → Graph IR lowering →
   backend kernel), then register the P1/P2 VLM connector ops (`patch_embed`,
   factorized 2D pos-emb, `mrope_2d`, `pixel_shuffle`/`pixel_unshuffle`,
   `cross_attention` / `perceiver_resampler` contracts).

## Source Material Consolidated

- `archive/advanced_examples_capability_gap.md`
- `archive/kv_cache_coverage_matrix.md`
- `archive/partial_ops_uplift_plan.md`
- `archive/primitive_coverage_state.md`
