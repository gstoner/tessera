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

### Tier 0 — Structural ops → **a compiler-foundation workstream, not "just not_applicable"**
This is **not** a single not_applicable audit. `backend_kernel = not_applicable`
is the right *disposition for the kernel axis* on the pure-view subset, but the
ops still need real **IR + lowering + runtime-movement + contract** support to be
*compiled* rather than numpy-interpreted. Today **28 of 33 structural ops have no
Graph IR op at all** — they exist only at the Python frontend + numpy runtime, so
they cannot enter a compiled pipeline, be fused, or lower to device memory
movement. The foundation work is specified in **§6**. Three kernel-axis sub-classes:

- **0-view — pure metadata (`backend_kernel = not_applicable`).** reshape, view,
  squeeze, unsqueeze, expand, broadcast, permute, transpose, flatten + the
  rearrange/rope_split/merge/tile_view family. Zero FLOP, zero data movement when
  layouts are stride-compatible. The kernel axis is genuinely not_applicable; the
  *foundation* axis is view analysis + stride-aware lowering (§6.B).
- **0-move — memory movement (`backend_kernel` = a strided-copy/gather lane).**
  cat, stack, split, chunk, pad, roll, flip, slice, select, take, index_select,
  dynamic_slice/update_slice, gather, scatter, masked_fill, repeat, tile. These
  **move data** — the honest backend lane is a memory-bound strided-copy /
  gather-scatter primitive (§6.C), NOT not_applicable and NOT a FLOP kernel.
- **0-reduce — genuine compute.** scatter_add / scatter_reduce (indexed
  reduction / atomics), nonzero / masked_categorical (data-dependent output
  shape). Real kernels (atomic scatter; stream-compaction). → folds into §6.C.

> **Reframe:** Tier 0 is the compiler's structural backbone. The deliverable is
> the §6 foundation (Graph IR dialect + view/movement lowering + runtime ABI +
> vmap/shard contracts), after which the kernel axis resolves per sub-class
> (not_applicable for 0-view, a movement lane for 0-move/0-reduce).

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

---

## 6. Tier 0 compiler foundation (the structural substrate)

Tier 0 is where the compiler-foundation question bites: these ops are the glue of
every model (reshape/gather/slice/pad/concat), yet most of them live only in the
Python frontend today. This section is the **foundation plan** the gap-closure
ask requires — what the **compiler itself, the runtime, and the Tessera Standard
Library** each need so structural ops are *compiled, optimized, lowered, batched,
and sharded* — not numpy-interpreted.

### 6.0 Current state (verified, file-grounded)

| Concern | State today | Evidence |
|---|---|---|
| **Stdlib surface** | ✅ all 33 ops registered with `lowering="layout_transform"/"indexing"`; numpy reference complete; exposed via `tessera.ops.*` | `python/tessera/compiler/op_catalog.py` (47 layout/indexing OpSpecs) |
| **Autodiff (VJP)** | ✅ largely complete — incl. gather (`vjp.py:535`), scatter (`:444`), scatter_add/reduce (`:458/:467`), masked_fill (`:688`), reshape/transpose/pad/cat/slice/take/… | `python/tessera/autodiff/vjp.py` |
| **Autodiff (JVP)** | ❌ only scatter_add/scatter_reduce; rest deferred | `python/tessera/autodiff/jvp.py` |
| **Graph IR (ODS)** | ❌ **only 5/33**: transpose (`TesseraOps.td:1693`), reshape (`:1724`), cast (`:1743`), select (`:1314`), masked_fill (`:1320`). 28 ops have **no IR op**. | `src/compiler/ir/TesseraOps.td` |
| **Lowering** | ❌ no structural→Tile/Target path; transpose only folds into matmul; reshape has a folder | `src/transforms/lib/CanonicalizeTesseraIR.cpp` |
| **View vs copy** | ❌ no IR concept; zero-copy view exists ONLY on Apple Metal (`DeviceTensor.reshape_view`), not backend-agnostic | runtime Apple GPU path |
| **Runtime movement** | ❌ numpy advanced-indexing only; no strided-copy / gather-scatter device primitive | `python/tessera/__init__.py` op refs |
| **Contracts** | ⚠️ `batching_rule` vmap semantics **UNSPECIFIED** (registry comment); `sharding_rule`/`transpose_rule` `partial` | `python/tessera/compiler/primitive_coverage.py` |

**Conclusion:** autodiff and the numpy reference are in good shape; the foundation
gap is **Graph IR representation → lowering → runtime movement → contracts**.

### 6.A Compiler — a structural / layout dialect

The 28 missing ops need IR representation. **Decision to make first:** native
Tessera ops vs. reuse of upstream MLIR `tensor`/`linalg`.

- **Reuse where upstream already models it** (lower onto, don't reinvent):
  `tensor.expand_shape`/`collapse_shape` (reshape/flatten/squeeze/unsqueeze),
  `tensor.extract_slice`/`insert_slice` (slice/dynamic_slice/update_slice),
  `tensor.pad`, `tensor.concat` (cat), `tensor.gather`/`scatter` (gather/scatter).
- **Define native Tessera ops** only where (a) upstream has no op (roll, flip,
  repeat, tile, index_select with Tessera index semantics, pack/unpack), or (b) we
  need to carry a **Tessera layout/distribution attribute** the tensor dialect
  can't (the whole point of Decision #2/#3 — layout + distribution are
  first-class). These become a small `tessera.layout` / `tessera.index` op set.
- **Per op:** ODS def + verifier + **shape inference** (TilingInterface where
  tileable) + **folders/canonicalizers**: no-op reshape elision, reshape∘reshape,
  transpose∘transpose (exists), broadcast propagation, `gather∘scatter`/`split∘cat`
  identities, dead-view removal.
- **Frontend wiring:** `@jit` Graph-IR emission must emit these ops (today it
  drops to numpy for the 28). This is the gate that turns them *compileable*.

> **Deliverable A:** `tessera.layout`/`tessera.index` ODS ops (or tensor-dialect
> lowering shims) for the 28, with shape verification + canonicalizers + frontend
> emission. Drift-gated by `op_catalog` ↔ ODS parity.

### 6.B Compiler — view/layout passes (the optimization that makes structural ops free)

- **View-vs-copy analysis pass.** Classify each structural op, given operand
  layout, as **zero-copy view** (stride/offset rewrite) or **materializing copy**.
  This is the single highest-leverage pass: reshape/slice/transpose/expand on a
  stride-compatible layout must NOT copy. Emits a `tessera.view` (aliasing) vs a
  movement op.
- **Layout propagation / canonicalization.** Push transposes through elementwise,
  fuse layout chains, sink/hoist reshapes, eliminate round-trips — so a chain of
  20 structural ops collapses to a handful of real movements.
- **Bufferization.** Lower views to `memref` with **strided layout attributes**
  (offset + strides), copies to explicit `memref.copy` / a movement op. Integrate
  with one-shot bufferize; alias analysis to keep views safe.
- **Movement lowering.** The 0-move ops that survive as copies lower to a
  strided-copy / gather-scatter loop nest (linalg.generic or a Tessera movement
  op) → Tile IR → the per-backend runtime primitive in §6.C.

> **Deliverable B:** ViewAnalysis + LayoutCanonicalize + structural bufferization
> passes, lit-tested (view elision FileCheck fixtures). This is what makes
> "not_applicable for a kernel" *true in practice* — the op compiles to a stride
> rewrite, not a copy.

### 6.C Runtime — a movement ABI (the backend-agnostic view + gather/scatter primitive)

- **View descriptor in the runtime ABI.** Generalize the Apple-only
  `reshape_view` into a backend-agnostic **(base_ptr, offset, shape, strides)**
  tensor view, so reshape/slice/transpose/expand are zero-copy on x86 + ROCm too
  (today they copy via numpy). One descriptor struct across the C ABI
  (`tessera_runtime.h`) + per-backend honoring (CPU pointer math; HIP device
  pointer + strided access).
- **Strided-copy movement kernel** (the 0-move backend lane): a generic
  gather/scatter/strided-copy primitive — pad/cat/stack/split/chunk/roll/flip and
  contiguous-incompatible reshape lower here. x86 = AVX-512 strided copy; ROCm =
  one-thread-per-output-element copy (the same lane pattern as the #180–#186
  kernels). This is the honest `backend_kernel` for 0-move (memory-bound, not FLOP).
- **Gather / scatter kernel** (0-reduce): indexed load (gather) and indexed store
  with **atomics for scatter_add/scatter_reduce**. x86 = gather loop + scalar
  atomics; ROCm = `generate-rocm-gather-kernel` / `-scatter-kernel` with
  `llvm.amdgcn` atomic add. This is a *real* kernel (the one Tier-0 sub-family that
  earns a `fused` row), reusing the proven ROCm Generate-pass + HIP-launch cadence.
- **Stream compaction** (nonzero / masked_categorical): prefix-sum + compact
  (data-dependent output shape) — a dedicated kernel, lowest priority.

> **Deliverable C:** the view descriptor ABI + a strided-copy movement lane +
> gather/scatter(+atomic) kernels on x86 + ROCm, with the same dual on-hardware
> validation as the sweep. Resolves the 0-move/0-reduce kernel axis honestly.

### 6.D Contracts — formalize vmap (batching) + sharding for structural ops

The registry flags **batching_rule UNSPECIFIED** for layout ops — the single
biggest correctness gap (vmap/`shard_map` can't route structural ops without it).

- **Batching rule per op family:** permute/transpose/flip/roll → axis indices
  shift +1 under the mapped axis; broadcast/expand/repeat/tile → the mapped axis
  is broadcast-preserved; reshape/view/flatten/squeeze/unsqueeze → the mapped axis
  must stay separable (specify + verify); slice/select/pad → leading mapped axis
  preserved, spec applies to trailing. Encode as `batching_rule` registry entries
  → axis auto-flips to `complete`.
- **Sharding rule:** partition-spec propagation through each op (how a
  `ShardSpec` transforms across reshape/transpose/gather; when a `gather` across a
  sharded axis needs an `all_to_all` — ties to `distributed_planner.py`).
- **Transpose (autodiff) rule:** VJPs already exist → register the `transpose_rule`
  axis so it flips `partial → complete` (free win, mirrors how matmul-family did).

> **Deliverable D:** batching + sharding rule registry entries for the 33 ops +
> `test_*_batching`/`_sharding` (the mock-mesh pattern). Flips three contract axes
> off `partial` honestly.

### 6.E Stdlib — hardening

- **Implement pack/unpack** (in `op_catalog`, no runtime body) or remove from the
  surface if subsumed by quantization.
- **Fix the gather `lowering=` inconsistency** (catalogued `layout_transform` but
  semantically `indexing`).
- **JVP rules** for the structural ops (forward-mode) — currently only
  scatter_add/reduce; the rest are mechanical given the VJPs (Phase F item).

### 6.F Sequencing of the Tier-0 foundation

1. **F1 — Graph IR ops + frontend emission (§6.A)** for the 28 missing ops
   (reuse tensor-dialect lowering where possible). Unblocks everything else.
2. **F2 — View analysis + layout canonicalize + bufferization (§6.B).** Makes
   0-view ops compile to stride rewrites (the real meaning of not_applicable).
3. **F3 — Runtime view ABI + strided-copy lane (§6.C, 0-move).** Backend-agnostic
   zero-copy views + the movement primitive on x86 + ROCm.
4. **F4 — gather/scatter(+atomic) kernels (§6.C, 0-reduce).** The one real-kernel
   sub-family; dual on-hardware validated.
5. **F5 — vmap + sharding contracts (§6.D)** + transpose_rule flip + stdlib
   hardening (§6.E).

F1→F2 are pure compiler; F3→F4 reuse the proven kernel cadence; F5 is contracts.
Only after F1 do the 28 ops become *compileable*; only after F3 is "not_applicable"
on the 0-view set **true in the compiled pipeline**, not just on paper.
