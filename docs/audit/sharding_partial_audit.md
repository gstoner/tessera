---
status: Informative
classification: Sharding-axis audit plus Sprint #19 Bucket A closure
authority: Companion to `docs/audit/standalone_primitive_coverage.md` and `docs/audit/generated/s_series_status.md`
last_updated: 2026-05-22
---

# Sharding-rule partial audit (Sprint #18 / #19, 2026-05-22)

The `sharding_rule` contract axis stood at **104 partial** entries after the
2026-05-22 sharding-rule promotion sprint closed 8 categories
(`stencil`, `pooling`, `tensor_algebra`, `layout_transform`, `indexing`,
`sort`, `grad_transform`, `control_flow` — 189 → 104 open).

This document originally classified **every remaining partial** into one of
three buckets so the next promotion sprints could be sized honestly:

* **Bucket A — promote-now (host/reference semantics fully documented):** the
  partition-spec rule is decidable at host level; no mesh runtime needed.
* **Bucket B — mock-mesh (needs `MockRankGroup` round-trip):** the rule is
  documented but its correctness depends on a multi-rank collective trace
  that today's `tests/testing/mock_collective.py` thread-mocks can exercise.
* **Bucket C — real-hardware (Phase G/H/I gate):** validation requires NCCL,
  RCCL, real distributed FFT all-to-all, or hardware-distributed execution.
  Promotion is honestly blocked until the corresponding Phase ships.

Sprint #19 has now closed Bucket A. Bucket B/C remain classification-only
until their mock-mesh or real-hardware proof lands.

## Sprint #19 Bucket A closure

Sprint #19 closed the promote-now set with host/reference partition-spec
rules only; no mock mesh or hardware claim was required.

| Change | Count | Result |
|--------|------:|--------|
| RL losses | 3 | `ppo_policy_loss`, `grpo_policy_loss`, and `cispo_policy_loss` promote to `sharding_rule=complete`. |
| State updates | 4 | `kv_cache_append`, `kv_cache_prune`, `kv_cache_read`, and `online_softmax_state` promote to `complete`. |
| Recurrent batch-shardable ops | 3 | `bidirectional_scan`, `gru_cell`, and `simple_rnn_cell` promote to `complete` for the documented non-state-axis rule. |
| EBM pointwise ops | 3 | `ebm_energy`, `ebm_self_verify`, and `ebm_decode_init` promote to `complete`. |
| GA pointwise ops | 13 | Non-differential Clifford ops promote to `complete`; the four differential/halo-bound ops remain partial. |
| LoRA adapter | 1 | `lora_linear` follows `linear_general` partitioning and promotes to `complete`. |
| `complex_jit` decorator | 1 | Removed from primitive-contract counting; it is a frontend decorator, not a primitive. |

Current dashboard state after regeneration:

| Axis | Open | Complete |
|------|-----:|---------:|
| `sharding_rule` | 76 | 356 |
| `backend_kernel` | 432 | 0 |

The initial audit called the Bucket A size "24/25" in prose, but the explicit
name list closes **28 dashboard entries**: 27 sharding promotions plus the
`complex_jit` registry-classification fix.

## Summary

| Bucket | Count | Categories |
|--------|------:|-----------|
| **A** | **closed: 28 dashboard entries** | rl_loss (3), state_update (4 — KV cache by head), recurrent (3 — batch-sharded), ebm pointwise (3), GA pointwise (13), lora_linear, complex_jit decorator cleanup |
| **B** | **47** | attention (14 standard + family), contraction (1), fused_epilogue (1), GA differential (4), loop_nest (7), model_layer (3), normalization (7), projection (1), segment_reduce (1), sparse-CSR (3), ebm sampling (8) |
| **C** | **33** | linalg_decomposition (3), linalg_solver (1), moe (1), moe_transport (2), spectral (9), state_space (1), reasoning-model attention with fused kernels (7), sparse-COO (1), recurrent state-axis (0 — folded into B), ebm bivector/sphere Langevin (4), GA codiff field ops (would belong here if not already in B), normalization with feature-axis collectives on real fabric (already in B) |
| **Total** | **104** | |

(Some entries appear in both attention-B and attention-C buckets when the
op has both a CPU-reference path and a fused real-kernel path; the audit
walker chooses the higher bucket — see per-category notes below.)

## Bucket A — promote-now

These entries have closed-form host-level partition-spec rules. The
rule is decidable without a mesh runtime, and CLAUDE.md Decision #25's
"is the contract documented?" question is yes. Sprint #19 has applied
the per-name overrides and regenerated the dashboard.

### rl_loss (3)
`ppo_policy_loss`, `grpo_policy_loss`, `cispo_policy_loss` — pure
reductions to a scalar loss; the canonical `psum` over data-parallel
rank is the same pattern `_SHARDING_RULE_BY_CATEGORY["loss"] = "complete"`
uses. These now promote through `_RL_LOSS_HARDENED`.

**Closure:** `_RL_LOSS_HARDENED["sharding_rule"] = "complete"`.

### state_update (4)
`kv_cache_append`, `kv_cache_prune`, `kv_cache_read`, `online_softmax_state` —
KV cache shards by head axis (the canonical transformer-TP partition for
KV); `online_softmax_state` carries a streaming softmax row state that
broadcasts across data-parallel ranks. Per-name overrides in
`_EXISTING_CONTRACT_OVERRIDES` now sets `kv_cache_*["sharding_rule"] =
"complete"`.

**Closure:** the contract IS the head-axis partition — host-level
KVCacheHandle layout proves it. Mock-mesh proof is desirable but not
required for the contract claim. A future hardening sprint can add a
parallel `tests/unit/test_kv_cache_sharding_mock_mesh.py` that walks the
handle layout under a mocked 2-rank head split.

### recurrent (3)
`bidirectional_scan`, `gru_cell`, `simple_rnn_cell` — sharding along the
batch axis (the canonical case) is trivial; state-axis sharding is C.
Currently `partial` covers both.

**Closure:** the existing partial-shape claim was conservative.
Family-level promotion is honest if we declare the contract as
"shardable along non-state axes; state axis requires real distributed
recurrence semantics."

### ebm pointwise (3)
`ebm_energy`, `ebm_self_verify`, `ebm_decode_init` — pointwise on
(B, K) with no cross-shard reduction. The Apple GPU backend manifest
already ships these as fused kernels.

**Closure:** category `ebm` currently sits at `partial` because
of the sampling subfamily; splitting via per-name override is the path.

### GA pointwise (13)
The 13 non-differential Clifford ops — `clifford_geometric_product`,
`clifford_wedge`, `clifford_inner`, `clifford_left_contraction`,
`clifford_rotor_sandwich`, `clifford_grade_projection`,
`clifford_reverse`, `clifford_norm`, `clifford_conjugate`,
`clifford_grade_involution`, `clifford_hodge_star`, `clifford_exp`,
`clifford_log` — are per-element on the batch axis. Decision #25
already documents this; the category-level `partial` is set conservatively
to cover the differential 4 (`clifford_codiff`, `clifford_vec_deriv`,
`clifford_ext_deriv`, `clifford_integral`) which are halo-bound.

**Closure:** per-name overrides for the 13 pointwise ops.

### Other (2)
- `lora_linear` — two small matmuls per layer; sharding follows the base
  `linear_general` pattern. Could ride a `model_layer` per-name override.
- `complex_jit` — a **decorator**, not a primitive. The registry entry was
  removed so the primitive dashboard counts only primitive contracts.

## Bucket B — mock-mesh (needs `MockRankGroup`)

These 47 entries have documented rules but the cross-shard collective
behavior should be exercised under the `MockRankGroup` thread-based
fake-rank harness (see `tests/testing/mock_collective.py`) before
flipping the axis. Mock-mesh proof is cheap and CPU-only.

### attention — standard family (14)
`flash_attn`, `multi_head_attention`, `gqa_attention`, `mqa_attention`,
`mla_decode`, `mla_decode_fused`, `attn_sliding_window`,
`attn_top_k_blocks`, `attn_compressed_blocks`, `attn_local_window_2d`,
`linear_attn`, `linear_attn_state`, `power_attn`, `retention`

The TP-by-head pattern is textbook; sequence-parallel (S-axis split with
all-gather of K/V) is also documented. A mock-mesh test walking a 2-rank
head split + a 2-rank sequence split would prove the contract.

**Sprint shape:** add `tests/unit/test_attention_sharding_mock_mesh.py`
covering head-split and seq-split for `flash_attn` + `mha`. Other 12
variants follow by category default; promote `_ATTN_HARDENED`
`"sharding_rule"` to `"complete"` once the mock-mesh oracle is green.

### loop_nest (7)
`matmul`, `gemm`, `batched_gemm`, `factorized_matmul`,
`latent_kv_compress`, `latent_kv_expand_k`, `latent_kv_expand_v`

Megatron-style TP (contraction-axis split with all-reduce, or
output-axis split with all-gather) is the textbook ML sharding pattern.
The `tessera.distributed_planner` already produces deterministic plans
for these.

**Sprint shape:** mock-mesh test that runs a 2-rank contraction-axis
matmul under `MockRankGroup` and asserts the all-reduce sum matches the
single-rank reference. Then category promotion.

### normalization (7)
`layer_norm`, `rmsnorm`, `rmsnorm_safe`, `group_norm`, `instance_norm`,
`spectral_norm`, `weight_norm` — feature-axis all-reduce of mean/var
when sharded on the feature axis; identity when sharded on batch.

### model_layer (3)
`linear_general`, `conv1d`, `conv_transpose` — same matmul/halo pattern.

### projection / contraction / fused_epilogue (3)
`qkv_projection`, `einsum`, `fused_epilogue` — matmul family.

### GA differential (4)
`clifford_codiff`, `clifford_vec_deriv`, `clifford_ext_deriv`,
`clifford_integral` — these are halo-bound stencil-style ops; the
stencil halo machinery (already promoted to `complete` in Sprint #14)
applies here. Mock-mesh proof = halo exchange + apply.

### segment_reduce (1)
`segment_reduce` — scatter+reduce by segment id; with replicated
segment IDs and sharded values, it's a standard reduction.

### sparse-CSR (3)
`spmm_csr`, `sddmm`, `bsmm` — row-block partition is the canonical
distributed sparse pattern.

### ebm sampling (8)
`ebm_inner_step`, `ebm_langevin_step`, `ebm_partition_exact`,
`ebm_partition_ais`, `ebm_partition_monte_carlo`, `ebm_decode_init` (B
classified above as A — keep A), plus the 2 Langevin-step samplers.
Partition functions need a cross-shard sum (canonical reduction);
inner-step Langevin is pointwise.

## Bucket C — real-hardware (Phase G/H/I gate)

These 33 entries genuinely need real distributed execution to validate.
Mock-mesh cannot exercise their target-specific routing or fabric-bound
collectives faithfully.

### attention — reasoning-model fused family (7)
`deepseek_sparse_attention`, `lightning_attention`, `gated_attention`,
`hybrid_attention`, `gated_deltanet`, `kimi_delta_attention`,
`modified_delta_attention`

These ship fused Apple GPU MSL kernels and have NVIDIA/ROCm planned
kernels in the manifest. Their sharding is the **fused kernel's**
sharding, which is target-specific and needs the real backend to verify.

### moe / moe_transport (3)
`moe`, `moe_dispatch`, `moe_combine` — token routing IS the all-to-all
collective; expert-parallel sharding requires real distributed execution
to validate the dispatch trace.

### spectral (9)
`fft`, `ifft`, `rfft`, `irfft`, `dct`, `stft`, `istft`, `spectral_conv`,
`spectral_filter` — distributed FFT uses the butterfly/ring all-to-all
pattern. The `DistributedFFTPass` (Spectral solver, 2026-05-10) exists
but its correctness against the single-rank reference is itself a Phase
G validation step.

### linalg_decomposition / linalg_solver (4)
`cholesky`, `qr`, `svd`, `tri_solve` — ScaLAPACK-style 2-D process-grid
distribution. Wavefront communication patterns; mock-mesh can prove
shape conformance but not numerical correctness on real grids.

### state_space (1)
`selective_ssm` — Mamba-style recurrence; sharding the time axis breaks
the recurrence. Sharding the batch axis is A but currently the partial
covers both; promotion path needs a Mamba-specific sharding pass.

### ebm Langevin family — vector / sphere (4)
`ebm_bivector_langevin_sample`, `ebm_bivector_langevin_step`,
`ebm_sphere_langevin_sample`, `ebm_sphere_langevin_step` — bivector +
sphere Langevin steps live on manifolds whose sharding requires GA-aware
halo exchange; not yet implemented.

### sparse-COO (1)
`spmm_coo` — hash-partition for COO requires real distributed execution
to verify the hash-sharding doesn't collide.

## How this audit maps to follow-up sprints

| Sprint candidate | Bucket | Expected close | Effort |
|------------------|--------|---------------:|--------|
| #19a — `_RL_LOSS_HARDENED["sharding_rule"] = "complete"` | A | 3 | done |
| #19b — KV cache + online softmax state per-name `"sharding_rule"` flip | A | 4 | done |
| #19c — recurrent batch-axis + GA pointwise per-name | A | 16 | done |
| #19d — `ebm` per-name pointwise (3 ops) + `lora_linear` + `complex_jit` removal | A | 5 | done |
| **Sprint #19 cumulative** | A | **28** | **done** |
| #20a — attention mock-mesh test + `_ATTN_HARDENED["sharding_rule"] = "complete"` | B | 14 | medium |
| #20b — loop_nest mock-mesh test + category promotion | B | 7 | medium |
| #20c — normalization + projection + contraction + fused_epilogue + model_layer (subset) mock-mesh | B | 14 | medium |
| #20d — GA differential mock-mesh (rides Sprint #14 halo) + segment_reduce + sparse-CSR | B | 8 | small |
| #20e — ebm sampling mock-mesh | B | 4 | small |
| **Sprint #20 cumulative** | B | **47** | **multi-PR sub-sprint** |
| Sprint #21 onwards | C | **33** | **Phase G/H/I gated** |

After Sprint #20 lands, the `sharding_rule` axis should stand near
**~29-33 partial entries**, depending on which Bucket B names require
real-hardware reclassification during mock-mesh proof.

## Notes / corrigenda

- `complex_jit` is a decorator factory, not a primitive op; Sprint #19
  removed it from primitive-contract counting. It remains documented as a
  frontend lane in the Visual Complex status docs.
- Several attention names appear in `_EXISTING_CONTRACT_OVERRIDES` via
  `_ATTN_HARDENED` (24 names total — `flash_attn`, the MLA family, the
  sparse/linear/recurrent family, AND the 7 reasoning-model fused
  variants). The reasoning-model 7 should arguably split off into a
  separate `_REASONING_FUSED_ATTN_HARDENED` override with a distinct
  `sharding_rule = "partial"` justification, so the standard family can
  promote to `complete` independently. Captured as a recommendation,
  not an action.
