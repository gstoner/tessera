# Apple Backend — Capability Roadmap (engine · kernels · optimization)

> Status: **planning (2026-06-19).** A consolidated forward plan for the Apple
> Silicon backend (CPU via Accelerate/BNNS, GPU via MPS / MPSGraph / MSL),
> derived from the [MLX ecosystem survey](apple_gpu_mlx_ecosystem_survey.md) and
> reconciled against what is *already in-tree*. The serving / app / dflash
> production track is a **separate plan** —
> [`docs/architecture/inference/serving.md`](../../../../architecture/inference/serving.md).
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

## 1a. Audit reconciliation (2026-06-19) — the roadmap overstated the gaps

A pillar-by-pillar grounding against actual in-tree code found that **three of the
five pillars were already substantially built**; only two were genuine gaps, and
both are now landed:

| Pillar | Audited reality | Verdict |
|---|---|---|
| **P1** Runtime scheduler | R0 resident `DeviceTensor` (zero-copy unified-memory wrap) ✅; R1 producer→consumer ✅; **R2 `EncodeSession` command-buffer batching** with `_dev_f32_enc` encoded elementwise ✅. `mtl4_residency` set in use. | **Largely built.** Remaining = measure the R0/R1/R2 decode-latency gain (per `apple_gpu_resident_activations_plan.md` §9–10), then extend only if numbers justify. NOT a reimplementation. |
| **P2** Fusion compiler | `python/tessera/compiler/fusion.py` (**3122 LOC**) is a general fusion middle-end: F0 `FusedRegion`, F1a `discover_fusable_regions` (maximal single-use chain growth), F2a `synthesize_matmul_epilogue_msl` (runtime MSL synthesis) + tiled/prologue/reduction variants; **already replaced the 168-kernel catalog** for matmul→pointwise. ~25 fusion test files. Gated by the Evaluator horizontal oracle. | **Substantially done** — this IS the "replace hardcoded chain matching" goal. Remaining = broaden the fusable vocabulary (non-matmul roots, attention-epilogue) incrementally; audit, not greenfield. |
| **P3** Packed quant matmul | `dequant_matmul_f32` reads full-width f32 codes (no bandwidth win); **no packed lane existed.** | **Genuine gap → ✅ LANDED** (packed-int4 QMV/QMM, this session). |
| **P4** Kernel-gen automation | `[[host_name]]` generator usage = **0** (absent). But per-dtype MSL duplication is mostly **2×** (f32 + one), not 3×; kernels are inline MSL literals in the 23k-line `.mm`; `fusion.py` already does runtime MSL *synthesis* (a more general mechanism than static template instantiation) for the epilogue family. | **Generator absent, but ROI lower than assumed.** A big-bang template refactor is high-risk/low-marginal-value; recommend incremental seed-helper adoption + a separable precompiled-`.metallib` build mode. Audit recorded; defer the refactor. |
| **P5** Memory & introspection | Budget ABI = **0 symbols** (absent); in-place shared-buffer wrap already done via R0. | **Genuine gap → ✅ LANDED** (budget ABI, this session). |

**Net:** the two real gaps (P3, P5) are landed + tested on real Metal. P1/P2 are
mature subsystems needing *measurement / incremental breadth*, not rebuilds. P4's
generator is genuinely absent but lower-value than the survey implied. The pillar
descriptions below are retained as the original analysis; read them through this
reconciliation.

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

## P2 — General element-wise fusion compiler — 🟢 built; vocabulary mature (audited 2026-06-19)

> **Status.** Done as a subsystem. `fusion.py` (3122 LOC) is the general fuser:
> `discover_fusable_regions` (maximal single-use matmul→pointwise chains) +
> `synthesize_matmul_epilogue_msl` (runtime MSL synthesis, oracle-gated) — the
> "replace hardcoded chain matching" goal. **Fusable vocabulary is broad:**
> `POINTWISE_OPS` has **22 ops** (add/sub/mul/div, relu/sigmoid/tanh/silu/gelu/
> neg/abs/exp, Phase-C long tail sqrt/rsqrt/log/log1p/expm1/reciprocal/softplus,
> maximum/minimum/sign) via a single-source `_POINTWISE_NAMES` shared by runtime +
> compile-time routing; `EPILOGUE_OPS` has the 6 hot in-matmul activations.
>
> **Finding: growth is deliberately demand-driven, not a present gap.** The C2
> close-out (2026-06-17) explicitly caps `EPILOGUE_OPS` at hot activations
> ("growing further would be speculative") and routes rarer ops through the
> pointwise-DAG path. Each `POINTWISE_OPS` addition is `verify_synthesized_
> pointwise`-gated and must correspond to an op a frontend actually emits — adding
> speculative vocabulary would be dead entries. **Recommendation:** broaden only
> when a model needs a specific op; the extension *mechanism* is proven (one
> ~3-file consistent change: catalog + vocab entry + drift gate). No speculative
> additions made.


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

## P3 — Fused packed-weight quantization execution — 🟢 int4 lane landed (2026-06-19)

> **Status.** The **packed-int4 QMV/QMM lane landed** and is validated on real
> Metal. Grounding correction: the pre-existing `dequant_matmul_f32` passes
> `Wcodes` as **`float*`** (4 bytes/weight — *no* bandwidth win); it dequants
> in-register from full-width codes. The new lane stores weights as **packed
> 4-bit codes (uint8, 2 nibbles/byte = 0.5 B/weight, ~8× less weight traffic)** —
> the actual decode bandwidth win.
>
> **Landed:** MSL kernel `quantized_matmul_i4_f32` + C ABI
> `tessera_apple_gpu_quantized_matmul_i4_f32(X, Wq, scales, biases, O, M, K, N,
> group_size)` (affine `w = scale·code + bias`, W[N,K] nn.Linear layout, M=1 QMV
> / M>1 QMM, partial + odd-K groups) + non-Darwin stub parity;
> `tessera.quantization.quantize_int4_packed` / `dequantize_int4_packed` (per-group
> affine packing, 2-nibble/byte); runtime `apple_gpu_quantized_matmul_i4`. Guard:
> `tests/unit/test_apple_gpu_quantized_matmul.py` (8 — matches dequant reference
> across QMV/QMM/partial/odd-K/single-group at rtol 1e-4, 8× bandwidth assert,
> pack roundtrip). mypy clean; 91 existing quant tests green.
>
> **Extended (2026-06-19):** **f16 X** (`quantized_matmul_i4_f16` — half activation
> upload, f32 accum), **tiled QMM** (`quantized_matmul_i4_tiled_f32` —
> threadgroup-cached X CHUNK=512, multi-chunk K; matches untiled), and the MSL
> source is now a **dtype-parameterized builder** (`quant_i4_msl_source(fname,
> xtype)` — one body, f32+f16; seeds P4). **op-catalog wiring:**
> `OpSpec("quantized_matmul", "tessera.quantized_matmul", 4, 4)` + `PYTHON_API_SPEC`
> row (spec-sync gate green). Guard: 11 tests (f16 at rtol 2e-3, tiled≡untiled).
> `runtime.apple_gpu_quantized_matmul_i4(..., variant="f32"|"f16"|"tiled")`.
>
> **Full `@jit(target="apple_gpu")` artifact dispatch (2026-06-19):** end-to-end
> landed. A jitted function calling `ops.quantized_matmul(x, w_packed, scales,
> biases, group_size=…)` lowers AST → `tessera.quantized_matmul` graph op (generic
> via the op_catalog `graph_name_for`, no bespoke AST code) → metadata artifact →
> driver gating (`_is_apple_gpu_mps_executable`) → executor → `quant_matmul` lane
> (`_apple_gpu_dispatch_quantized_matmul`) → packed-int4 Metal kernel. Wiring:
> `_APPLE_GPU_QUANT_OPS` in `apple_gpu_envelope.py` → `quant_matmul` lane;
> regenerated `apple_runtime_ops.inc` + **tessera-opt rebuilt** so the C++ Tile→
> Apple pass tags it `metal_runtime` (G3 enforcer green). Guard: `test_full_jit_
> apple_gpu_quantized_matmul` + metadata-artifact + driver-gating + lane tests (16
> total in `test_apple_gpu_quantized_matmul.py`; envelope/table/G3 drift gates green).
>
> **Follow-ups landed (2026-06-19):**
> - **VJP** — `vjp_quantized_matmul` (straight-through, frozen quantized weight:
>   `dx = dout @ dequant(w)`; no grad to packed codes/scales/biases). Verified
>   analytic + finite-difference.
> - **QVM split-K** — `quantized_matmul_i4_splitk_f32`: parallel K-reduction
>   (M·N·S threads writing `[S,M,N]` partials + host S-way sum; heuristic
>   `S = clamp(K/512, 1, 16)`). Avoids device float atomics (Apple7/M1-safe).
>   `variant="splitk"`; matches untiled.
> - **MXFP4 / NVFP4 packed layouts** — `quantize_fp4_packed(W, group_size,
>   scale_mode="mx"|"nv")` (FP4 e2m1 codes 2/byte + per-group symmetric scale, no
>   bias; MX rounds scale to pow-2/g32, NV g16) + `quant_matmul_fp4_f32` MSL kernel
>   (e2m1 LUT decode, f32 — native FP4 matrix ops gated on macOS-27.0 SDK) +
>   `runtime.apple_gpu_quantized_matmul_fp4`. Both layouts match the dequant
>   reference. Guard: 21 tests total in `test_apple_gpu_quantized_matmul.py`.
>
> **Remaining (optional):** FP4 lane `@jit` op-catalog wiring (int4 is wired; FP4
> is a runtime helper today); native FP4 matrix-unit path when the 27.0 SDK lands.

**What (original scope).** A real on-GPU `quantized_matmul` over **packed**
weights (uint8, 2/4/8 vals/byte, per-group scale/bias), not the current
dequant-to-fp32-then-GEMM.

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

## P4 — Kernel-generation automation — 🟡 seed-helper demonstrated; metallib low-ROI (audited 2026-06-19)

> **Status.** (a) **Template seed-helper: first instance landed.** The P3 f16 work
> introduced `quant_i4_msl_source(fname, xtype)` in `apple_gpu_runtime.mm` — one
> MSL body emitting both the f32 and f16 kernels via `stringWithFormat`, instead
> of two hand-copied kernels. This is the concrete seed of the `[[host_name]]`
> idea; further kernels can adopt the same pattern incrementally (the audited
> ~2× per-dtype duplication means the marginal value per migration is modest, so
> a big-bang refactor stays unwarranted).
>
> (b) **Precompiled `.metallib` mode: genuine but low-ROI.** The from-source
> loader (`_apple_gpu_dispatch.py`) already caches the compiled dylib keyed on
> `.mm` mtime, so cold-start is amortized across processes (paid once per source
> edit, not per run). A build-time `.metallib` would shave first-use JIT only;
> given the existing dylib cache, ROI is low. Recommend deferring unless a
> cold-start profile demands it.


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

## P5 — Memory & introspection — ✅ budget ABI landed (2026-06-19)

> **Status.** The **budget ABI + accounting landed** and is validated on real
> Metal. The **in-place shared-buffer wrap is already done** via the R0 resident
> `DeviceTensor` (`ts_dev_*` + unified-memory shared buffers — see P1). Residency
> sets are already used on the MTL4 lane (`mtl4_residency`). Remaining optional:
> the <256 B heap split (only if small-alloc churn shows in a profile).
>
> **Landed:** byte accounting on the buffer pool (`metal_buffer_acquire/release`)
> + device-tensor handles (`ts_dev_alloc/free`) with relaxed atomics; 7 C ABI
> symbols `tessera_apple_gpu_{get_active,get_cache,get_peak,reset_peak,
> set_memory_limit,get_memory_limit,clear_cache}_memory`; non-Darwin stub parity;
> Python surface `runtime.apple_gpu_memory_stats()` / `apple_gpu_set_memory_limit`
> / `apple_gpu_reset_peak_memory` / `apple_gpu_clear_cache`. Semantics mirror MLX
> (active = checked-out + resident; cache = pooled-for-reuse; peak = high-water;
> limit advisory for the serving ProcessMemoryEnforcer). Guard:
> `tests/unit/test_apple_gpu_memory_budget.py` (6 — exact device-tensor delta,
> peak tracking, limit round-trip, clear_cache=0-cache invariant; 53 pool/dispatch
> regression tests green). mypy clean.

**What (original scope).** Public memory-budget ABI + residency + the in-place
shared-buffer wrap (shared with P1).

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
  envelope → [`docs/architecture/inference/serving.md`](../../../../architecture/inference/serving.md).
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
