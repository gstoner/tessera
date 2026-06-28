# S-series gap-closure plan + planned fused-kernel inventory

> **Scope:** drive the remaining S-series primitives to **real, on-device-validated
> execution on the two devices this repo can prove on today — x86 AVX-512 and
> AMD ROCm gfx1151 (RDNA 3.5 WMMA)**. NVIDIA SM_90/100/120 and Apple are gated to
> their own hardware (Phase G/H/I) and tracked separately.
>
> **Counts are dashboard-owned** (Decision #25/#26). The numbers below are
> planning approximations; the truth is in `docs/audit/generated/` —
> `support_table.csv` (per-op kernel status), `runtime_execution_matrix.csv`
> (per-lane), and the per-target maps (`rocm_target_map`, `nvidia_sm90_target_map`,
> `apple_target_map`). Regenerate, never hand-edit.
>
> Companion docs: the **hardware-gated** WGMMA/MFMA fused-kernel inventories live
> in [`docs/nvidia_cuda13_kernel_inventory.md`](../../nvidia_cuda13_kernel_inventory.md)
> and [`docs/rocm_mfma_kernel_inventory.md`](../../rocm_mfma_kernel_inventory.md),
> with their schema + lit fixtures guarded by
> `tests/unit/test_kernel_inventory_and_lit_fixtures.py`. **This doc is the
> *native-execution* companion** — the kernels we can build and numerically prove
> on x86 + gfx1151 now, not the CDNA-MFMA / Blackwell-tcgen05 artifact path.

---

## 1. Where we are (per-device truth, not the universal gate)

The registry's `backend_kernel = complete` axis is a **universal multi-target
gate** — it only flips when *every* documented target ships (x86 **and** ROCm
**and** NVIDIA **and** Apple), so `s_series_status` reads `0/474 complete` **by
design**. That is not the per-device picture. Measured per device (a target
"executes" iff its manifest status ∈ {`fused`, `compiled`, `hardware_verified`}):

| Bucket | ≈count | Meaning |
|---|---:|---|
| **execute on both x86 + ROCm** | ~127 | closed (incl. the #180–#186 sweep) |
| **ROCm-only** (x86 gap) | ~10 | mostly the WMMA attention lanes — need an x86 partner |
| **x86-only** (ROCm gap) | ~1 | |
| **neither** (both-device gap) | ~171 | the actionable surface, triaged below |

The recently-closed sweep (PRs #180–#186): sparse (spmm/sddmm/bsmm), state_space
(selective_ssm), linalg (cholesky/tri_solve/cholesky_solve/lu/qr/svd), moe-compute.

---

## 2. Triage of the both-device gap (~171 ops)

Not all gaps are compute. Four dispositions:

### Tier 0 — Host / structural → **classify `not_applicable`, do NOT build a kernel**
Pure view/shape/data-movement ops; AVX-512 / WMMA buy nothing. The honest action
is to mark `backend_kernel = not_applicable` per op (Decision #25 — only after
verifying zero FLOP content), which closes the open count without inflating it.

- **layout_transform (~31):** reshape, view, squeeze, unsqueeze, expand, broadcast,
  permute, transpose, flatten, cat, stack, split, chunk, pack/unpack, tile_view,
  rope_split/merge, rearrange, repeat, roll, flip, pad, masked_fill, arange, cast,
  mor_partition/router/scatter.
- **indexing (~15):** slice, select, take, index_select, gather, scatter,
  dynamic_slice/update_slice, nonzero, masked_categorical, memory_index_select(_ste),
  msa_select_blocks. *Caveat:* `scatter_add` / `scatter_reduce` have real
  reduction content → Tier 2 (atomic-add kernel), not Tier 0.

> **Sub-task 0a:** a single audit PR that walks these, confirms zero-FLOP, and sets
> `not_applicable` with a one-line reason each (the canonical KV-cache→target
> pattern, Decision #21). ~40 rows close with no kernel work.

### Tier 1 — Mesh-gated transport → **stays gated (needs multi-accelerator HW)**
- **collective (4):** all_gather, all_reduce, all_to_all, reduce_scatter.
- **moe_transport (2):** moe_dispatch, moe_combine.

No action on this box; they prove on a real multi-GPU mesh (Phase H). Leave
`partial`/`planned`, documented reason.

### Tier 2 — Easy elementwise / predicate compute → **extend existing lanes**
Flat per-element or simple-reduce ops that drop straight into the proven
unary/binary/reduce lanes (`x86_unary_compiled` / `rocm_unary_compiled`, etc.).

- **elementwise (~13):** add, mul, sin, atan2, clip, floor_div, mod, softcap,
  score_combine, digamma, lgamma, popcount.
- **numeric_helper (~5):** abs, clamp, isfinite, isinf, isnan.
- **position_encoding (1):** ntk_rope (variant of the existing rope lane).
- **stable_reduction (1):** reduce. **segment_reduce (1):** segment_reduce (scatter-add).

> Lowest effort / highest row-count. Mostly new op-name → existing-kernel wiring
> + a couple of new transcendental/predicate cases.

### Tier 3 — Composable medium compute → **compose on reduce / GEMM / elementwise**
- **functional_optimizer_step (~6):** sgd, momentum, adam, **adamw**, lion,
  adafactor — fused per-parameter update kernels (the user-called-out "fused
  AdamW"). See §3.
- **normalization (~4):** group_norm, instance_norm, rmsnorm_safe, weight_norm.
- **loss (~8):** EBM/diffusion losses — score_matching / denoising_score_matching
  / implicit_score_matching / contrastive_divergence / persistent_cd / ddpm_noise_pred
  / vlb / load_balance. Compose on reduce + elementwise.
- **energy_based_models (~10):** EBM energy/grad compute.
- **geometric_algebra (~18):** Clifford/GA products — mostly structured
  multiply-add (the S2 GA family already has a CPU reference; many are
  vectorizable bilinear forms).
- **visual_complex (~20):** complex arithmetic (complex_abs/arg/conj/div/log/pow/sqrt)
  + conformal geometry (mobius, cross_ratio, dbar/dz, laplacian_2d, conformal_*).
  Complex arith = interleaved-f32 elementwise (reuse the FFT complex substrate).

### Tier 4 — New kernel families (high value, real engineering) → **the marquee work**
The fused kernels worth a dedicated kernel class. See §3 for the inventory.

- **attention (~13 both-gap + ~10 ROCm-only needing x86):** flash_attn (x86),
  MLA decode, NSA (deepseek_sparse_attention), lightning, kimi_delta, gated /
  delta / linear / retention / power / sliding-window / local-2d variants.
- **stencil (2):** conv2d, conv3d.
- **sort (3):** argsort, sort, top_k (bitonic; top_k feeds NSA/MoE routing).
- **loop_nest (~8):** quantized_matmul, dequant_grouped_gemm, latent_kv_compress,
  latent_kv_expand_k/v (the MLA building blocks).
- **random_source (2) + random_mask (1):** device Philox RNG → rng_uniform /
  rng_normal / dropout (the accel-proof "special" class).
- **state_update (~3):** kv_cache_append / prune / read (paged scatter-copy).

---

## 3. Planned fused-kernel inventory (native x86 AVX-512 + gfx1151 WMMA)

Marquee fused kernels, with the **native** approach for each device. Tile shapes
for the hardware-gated WGMMA/MFMA path are in the companion inventory docs; here
we record the **executable-now** plan. ROCm uses RDNA 3.5 WMMA `16×16×16`
(f16/bf16; **no FP8/FP4 WMMA on gfx1151** — those are CDNA4/RDNA4, see the
[RDNA ISA archive](../../reference/isa/rdna/)). x86 uses AVX-512 (+ AMX BF16 for
the GEMM tile).

| Kernel | x86 approach | gfx1151 approach | ROCm status today | Validation |
|---|---|---|---|---|
| **matmul / gemm** | AMX BF16 + AVX-512 GEMM (shipped) | WMMA `16×16×16` (shipped, `hardware_verified`) | ✅ executes | vs numpy |
| **flash_attn** (FA-style) | AVX-512 tiled QKᵀ→softmax→·V (online softmax) | WMMA flash kernel (shipped `hardware_verified`) | ✅ ROCm; **x86 = gap** | vs reference attn |
| **multi_head_attention** | x86 `fused` (shipped) | WMMA `compiled` (shipped) | ✅ both | vs reference |
| **mla_decode_fused** (DeepSeek MLA) | tiled latent-KV decode | WMMA `compiled` (shipped); native decode kernel | ✅ ROCm; x86 = gap | vs reference |
| **deepseek_sparse_attention** (NSA) | top-k block gather → tiled attn over selected blocks | block-sparse WMMA; needs top_k kernel (Tier 4 sort) | artifact; **gap both** | vs dense-masked ref |
| **lightning_attention** (MiniMax) | linear-attn state recurrence (cumulative) | WMMA `compiled` (shipped); native state scan | ✅ ROCm; x86 = gap | vs reference |
| **kimi_delta_attention** | delta-rule state update + readout | WMMA + state scan (compose on deltanet lane) | gap both | vs reference |
| **gated_deltanet / retention / power_attn / linear_attn** | state-recurrence scans (reuse the SSM scan substrate) | one-thread-per-channel scan (the selective_ssm pattern) | gap both | vs reference |
| **swiglu_mlp** | fused gate·up GEMM → SiLU → down GEMM (reuse AMX GEMM + silu_mul lane) | WMMA GEMM + the shipped `rocm_silu_mul` lane | gap both (compose) | vs reference |
| **matmul_softmax_matmul** | fused score→softmax→context (the FA inner core) | WMMA + softmax lane | gap both (compose) | vs reference |
| **adamw_step / adam / lion / sgd / momentum / adafactor** | AVX-512 per-parameter update kernel (elementwise; bias-correction scalar) | flat elementwise update kernel (one thread per param) | gap both | vs reference optimizer |
| **conv2d / conv3d** | im2col + AMX GEMM, or direct AVX-512 | im2col + WMMA, or direct | gap both | vs reference (apple_gpu has it) |
| **quantized_matmul / dequant_grouped_gemm** | AVX-512 VNNI int8 GEMM (shipped int8 path) + dequant epilogue | WMMA int8 + dequant | gap both | vs reference |
| **rng (Philox) → rng_uniform/normal/dropout** | AVX-512 Philox counter-based | per-thread Philox (counter = global stream id, Decision #18) | gap both | vs reference RNG stream |
| **sort / argsort / top_k** | AVX-512 bitonic / partial-sort | bitonic network in LDS | gap both | vs numpy sort |

These tie to the existing inventory's canonical family names (so
`test_kernel_inventory_and_lit_fixtures.py::TestG2/H3` and the cross-target
parity test keep both surfaces aligned): `matmul`, `flash_attn`, `mla_decode`,
`deepseek_sparse_attention`, `lightning_attention`, `kimi_delta_attention`,
`swiglu_mlp`, `matmul_softmax_matmul`, `adamw_step`.

---

## 4. Sequenced execution (proven per-family cadence)

Each PR follows the cadence proven across #180–#186: x86 C-ABI kernel
(`avx512_<fam>_f32.cpp`, into both CMake lists + the `.so`) **and** ROCm
`GenerateROCM<Fam>Kernel.cpp` (PassWrapper + Passes.h/.cpp/CMakeLists + ODS op)
+ runtime executor + manifest/matrix rows + fixtures + `test_{x86,rocm}_<fam>`,
both numerically validated on real hardware, then regen dashboards + drift gate.

**Phase A — bookkeeping honesty (1 PR, no kernels).** Tier 0/1 dispositions:
mark layout_transform + pure-indexing `not_applicable`; document collective +
moe_transport mesh-gated reasons. Closes ~46 open rows truthfully. *Biggest
count-per-effort.*

**Phase B — elementwise/predicate sweep (1–2 PRs).** Tier 2: add/mul/sin/atan2/
clip/floor_div/mod/abs/clamp/isfinite/isinf/isnan/popcount/softcap + ntk_rope +
segment_reduce. Extend the existing unary/binary/reduce lanes. ~21 ops.

**Phase C — optimizer steps (1 PR).** Tier 3: sgd/momentum/adam/**adamw**/lion/
adafactor as fused per-parameter update kernels. User-prioritized. ~6 ops.

**Phase D — normalization + complex/GA (1–2 PRs).** group_norm/instance_norm/
weight_norm/rmsnorm_safe; complex arithmetic (reuse FFT complex substrate);
the vectorizable GA bilinear forms. ~30+ ops.

**Phase E — EBM + diffusion losses (1 PR).** energy_based_models + the 8 score/
diffusion losses, composing on reduce + elementwise. ~18 ops.

**Phase F — attention x86 partners (2–3 PRs).** Give the ROCm-only WMMA attention
lanes their x86 AVX-512 counterparts: flash_attn, mla_decode, lightning, then the
state-recurrence family (deltanet/retention/power/linear) on the SSM scan
substrate. Closes the ~10 ROCm-only gaps + the attention both-gaps.

**Phase G — new kernel families (sequenced, highest effort).** sort/top_k →
unlocks NSA; conv2d/3d; quantized_matmul + MLA latent-KV (loop_nest); device
Philox RNG → dropout; swiglu_mlp + matmul_softmax_matmul fusions; kv_cache
append/prune.

Ordering rationale: A→B→C front-loads the cheap, high-count closes (honest gate
movement) before the marquee Tier-4 engineering; F depends on B (softmax/scan
substrate) and G's sort precedes NSA.

---

## 5. Honesty guardrails (carry from the sweep)

- **Validate factorization/iterative kernels by invariants**, not exact arrays
  (sign/order ambiguity) — as svd/lu/qr did.
- **`not_applicable` only after verifying zero backend execution on every target**
  (grep trace.py + the per-target dispatch); a category being *allowed*
  not_applicable in the universal-gate test is a ceiling, not a directive
  (Decision #25, the PR #132 lesson).
- **Generated dashboards carry the numbers**; this doc carries the plan. When a
  phase lands, regenerate and let `support_table` / `runtime_execution_matrix`
  report the new per-device truth.
- **gfx1151 = RDNA 3.5:** WMMA f16/bf16/iu8/iu4 only — **no FP8/BF8 WMMA, no
  sparse SWMMAC** (RDNA4/CDNA4). Don't plan FP8 WMMA lanes for this box.
