---
status: Normative (development roadmap)
classification: Audit / Plan
authority: Sequences every open capability gap into executable phases
last_updated: 2026-05-09
---

# Tessera Development Roadmap

This document is the **authoritative sequencing plan** for every issue raised
in the May 2026 capability review. Tasks are grouped into phases by dependency
and impact. Each task has explicit acceptance criteria so Claude (or any
contributor) can pick it up and execute without further design conversation.

If a task lists "Open question:", that is a decision the implementer must
surface back to the user before writing code. Otherwise, the task is
self-contained.

## How to use this doc

1. Scan **Phase ordering rationale** to understand dependencies.
2. Pick the lowest-numbered task whose dependencies are all ✅.
3. Read its **Acceptance criteria**; those are the test(s) you must make pass.
4. After landing, mark the task ✅ and update any cross-referencing audits
   (e.g., `advanced_examples_capability_gap.md`).

Status legend: 📋 planned · 🚧 in progress · ✅ done · 🔲 deferred (out of
scope this cycle, with a tracked reason).

## Phase ordering rationale

```
A (quick wins)   ───┐
                    ├──► C (Theme 1 cleanup)
B (protocols)    ───┘                ┐
                                     ├──► D (streaming)
                                     ├──► E (KV-cache)
F (autodiff follow-ups, parallelizable)
G (NVIDIA execution — biggest lift, parallelizable with F)
H (Conv2d + remaining nn cleanup, after G picks the layout)
I (DDP/FSDP — depends on F4+F5 + G)
```

**Critical paths:**
- **G is the highest-leverage phase.** Without NVIDIA execution, the
  autotuner, FA-4 verification, GPU-only tier, and GPU CI are all dark.
- **B1 (buffer protocol)** unblocks `BatchNorm1d` (C1) + streaming kernels
  with state (D).
- **B2 (`KVCacheHandle` value type)** unblocks Theme 4 entirely (E1–E3) +
  the `KVCache` Module wrapper (C2).
- **F4 (Graph IR adjoints) → F5 (effect-aware adjoint collectives) → I
  (DDP/FSDP).** This chain is the long pole for distributed training.

Estimated runway end-to-end: **12–20 weeks** of focused work, parallelizable
across F + G.

---

## Phase A — Quick wins (independent, parallelizable, ~1–2 weeks)

### [A1] Debugging story — env-var IR dumps + per-pass diff + how-to doc 📋

**Scope:** S (~250 LOC code + ~300 LOC docs).

**Files (new):**
- `python/tessera/debug_env.py` — env-var parser (`TESSERA_DEBUG_IR=graph,schedule,tile,target` and `TESSERA_DEBUG_DUMP_DIR=/tmp/tessera`)
- `tests/unit/test_debug_env.py`

**Files (modify):**
- `python/tessera/compiler/jit.py` — emit IR snapshots after each lowering stage when env var is set
- `docs/guides/Tessera_Debugging_Tools_Guide.md` — add "Dumping IR mid-pipeline" section
- `CLAUDE.md` — list the env vars in the Testing/Debug section

**Acceptance:**
- `TESSERA_DEBUG_IR=graph,schedule TESSERA_DEBUG_DUMP_DIR=/tmp/d ./run.py` writes `graph.mlir` and `schedule.mlir` for every JIT artifact emitted in that run.
- Per-pass diff helper: `tessera-mlir diff /tmp/d/graph.mlir /tmp/d/schedule.mlir` prints a textual line-diff (use Python `difflib`).
- Test: env var off → no files written; env var on → files exist + are non-empty MLIR.
- Doc: a recipe titled "kernel ran, results wrong, what now?" walks through `TESSERA_DEBUG_IR`, the diff tool, and `tessera.debug.replay_manifest`.

### [A2] Dynamic shapes — audit + doc + test 📋

**Scope:** S (mostly investigation + ~150 LOC test + doc).

**Files (modify):**
- `docs/spec/SHAPE_SYSTEM.md` — add "Dynamic shape support matrix per backend" section
- `tests/unit/test_dynamic_shapes.py` (new) — every supported backend × symbolic-dim case

**Acceptance:**
- Documented matrix of which `Dim("S")`/symbolic dims actually flow through to which backend (x86 / Apple_cpu / Apple_gpu).
- For each "supported" cell, a test that builds a `@tessera.jit` with a symbolic dim, calls it with two different concrete shapes, and validates correct output.
- For each "unsupported" cell, a test that asserts a clear error message at decoration or first-call time (no silent fallback).
- Update `CANONICAL_API.md` with a one-paragraph dynamic-shape semantics block.

**Audit result (2026-05-09):** dynamic shapes work on CPU reference, Apple
CPU, and Apple GPU — symbolic dims flow through to actual execution. The
real gap is **call-time constraint enforcement**: `@jit(bindings={K:7})` +
`require(Divisible(K, 8))` raises at decoration time, but the same function
called *without* `bindings=` lets a violating shape through silently. See
`tests/unit/test_dynamic_shapes.py::TestConstraintEnforcement::test_call_time_check_xfail`
(currently `xfail`, becomes `xpass` when fixed) and `docs/spec/SHAPE_SYSTEM.md` §10.

### [A2-followup] Call-time constraint enforcement 📋

**Scope:** S (~150 LOC). Lift the constraint check from `@jit` decoration
into `JitFn.__call__` so that constraint violations on real argument shapes
raise `TesseraConstraintError` even when no `bindings=` was supplied.
Acceptance: the `xfail` test in `test_dynamic_shapes.py` flips to `xpass` and
the `xfail` mark is removed.

### [A3] KV-cache lowering coverage matrix 📋

**Scope:** XS (doc-only, ~80 LOC).

**Files (new):**
- `docs/audit/kv_cache_coverage_matrix.md`

**Files (modify):**
- `CLAUDE.md` Architecture Decision #21 — link to the matrix

**Acceptance:**
- Per-target table: rows = `kv_cache_append`, `kv_cache_prune`, FA-4 with cache; columns = x86, Apple_cpu, Apple_gpu, NVIDIA, ROCm, TPU, Cerebras, Metalium, RubinCPX. Cells: ✅ executes / 🟡 lowers but no execution / 🔲 emits diagnostic / ❌ silent no-op (bug).
- Audit method: grep each backend's lowering passes for `kv_cache_*`, verify behavior, document.
- For any 🔲 cells found that turn out to be ❌, file a follow-up task in this roadmap.

### [A4] Theme 1 cleanup — small phantoms that don't need new infrastructure 📋

**Scope:** S (~300 LOC code + ~200 LOC tests).

**Files (modify):**
- `python/tessera/nn/__init__.py` — replace 8 phantoms with real classes/aliases
- `python/tessera/nn/layers.py` — add the new Module wrappers
- `tests/unit/test_nn_module.py` — add coverage

**Per-phantom resolution:**
| Phantom | Resolution |
|---------|-----------|
| `SiLU`, `Sigmoid`, `GELU`, `ReLU`, `Tanh`, `Identity` | Stateless `Module` wrappers — `def forward(self, x): return ts.ops.silu(x)` etc. (`Identity` returns input unchanged) |
| `MultiHeadCrossAttention` | Subclass of `MultiHeadAttention` whose `forward(q, k, v)` requires explicit K/V (no self-attention shortcut) |
| `RotaryEmbedding` | Module owning `theta` (precomputed); `forward(x)` calls `ops.rope(x, self.theta)` |
| `CastedLinear`, `CastedEmbedding` | Subclass `Linear` / `Embedding` with extra `cast_dtype` arg; `forward` does `ops.cast(super().forward(x), self.cast_dtype)` |
| `CrossEntropyLoss` | Functional + Module form: `-mean(log_softmax(logits)[target])`. Composes through `ops.softmax` + `ops.reduce` for autodiff |
| `nn.utils.clip_grad_norm_(params, max_norm)` | Real impl: compute total `||grad||₂`, scale `.grad` in place if above threshold |

**Acceptance:**
- Each phantom in the list above replaced with a real implementation that passes `forward` shape tests + composition tests.
- `CrossEntropyLoss` tested end-to-end through a tape — gradients to logits match numerical Jacobian.
- `clip_grad_norm_` tested: above-threshold case scales correctly; below-threshold leaves grads untouched.
- Update `advanced_examples_capability_gap.md` to mark these ✅.

### [A5] `flash_attn` VJP via `custom_rule` 📋

**Scope:** XS (~50 LOC code + ~60 LOC test).

Today, calling `flash_attn` inside a tape raises `TesseraAutodiffError`. This is correct (no kernel-level adjoint exists), but the standard reference VJP is well-known and worth shipping for the numpy reference path so `MultiHeadAttention` modules can train end-to-end.

**Files (new):**
- `python/tessera/autodiff/_flash_attn_vjp.py` — registered via `custom_rule("flash_attn")`
- Tests: numerical-Jacobian against `flash_attn` reference impl

**Acceptance:**
- VJP computes `dQ`, `dK`, `dV` by recomputing scores + softmax during backward (memory-efficient is out of scope; correctness first).
- Numerical-Jacobian test passes at fp64 with `rtol=1e-5`.
- `MultiHeadAttention` module training step tested end-to-end.

---

## Phase B — Foundational protocols (sequential, ~1 week) — **all three landed 2026-05-09**

### [B1] Module buffer protocol — `register_buffer` + state_dict integration ✅

**Scope:** M (~250 LOC code + ~150 LOC tests).

Buffers are non-trainable named tensors that ride alongside parameters in
`state_dict()` (BatchNorm running stats, RoPE precomputed `theta`, attention
masks, etc.). They differ from parameters in two ways: no `.grad`,
`requires_grad` is meaningless. They're persisted by `state_dict` like
parameters.

**Files (modify):**
- `python/tessera/nn/module.py` — add `_buffers: OrderedDict`, `register_buffer(name, value, persistent=True)`, attribute routing for `Buffer`-tagged tensors, `buffers()` / `named_buffers()` iterators, `state_dict()` includes persistent buffers
- `python/tessera/nn/__init__.py` — export `Buffer` (a thin tagged-ndarray class so `__setattr__` can route)
- `tests/unit/test_nn_module.py` — buffer round-trip, `persistent=False` excluded from state_dict, `to(dtype)` migrates buffers

**Acceptance:**
- `module.register_buffer("running_mean", np.zeros(64))` → `module.running_mean` returns the buffer; `module.named_buffers()` yields it; `module.state_dict()` includes it under the right name.
- `module.register_buffer("temp", arr, persistent=False)` excluded from `state_dict()` but still accessible.
- `module.parameters()` / `named_parameters()` does **not** yield buffers (regression test).
- `Module.zero_grad()` does not touch buffers (no `.grad` slot to reset).

**Decision (locked 2026-05-09):** `Buffer` is a wrapper class analogous to
`Parameter`, with `_data: DistributedArray`, no `.grad` slot, no `requires_grad`,
and `persistent: bool` for state-dict participation.

### [B2] `KVCacheHandle` opaque value type ✅

**Scope:** M (~200 LOC code + ~150 LOC tests).

A handle that flows through Tile IR / Graph IR as a first-class value
representing the state of a paged KV cache. Today, `ops.kv_cache_append`
returns the cache, but there's no formal handle type — passing it through
ops works only by Python convention. Theme 4 needs a real handle.

**Files (new):**
- `python/tessera/cache/__init__.py` — `KVCacheHandle` class (Python-side; opaque to ops)
- `python/tessera/cache/handle.py` — internal storage (paged numpy buffers), `pages`, `current_seq`, `max_seq` attributes

**Files (modify):**
- `python/tessera/__init__.py` — export `cache` namespace
- `python/tessera/ops` callsites — `kv_cache_append`/`kv_cache_prune` accept and return `KVCacheHandle` instances (with backward-compat for the existing `ReferenceKVCache`)

**Acceptance:**
- `cache = ts.cache.KVCacheHandle(num_heads=4, head_dim=64, max_seq=128, dtype="fp32")` constructs.
- `cache = ts.ops.kv_cache_append(cache, k, v)` returns a new handle (functional style).
- `ts.ops.kv_cache_read(cache, slice(0, 64))` returns `(k, v)` tensors. (Adds a new op + VJP-stub; mark `flash_attn` consumers TODO.)
- Round-trip: append then read returns the appended values.
- Test: appending past `max_seq` raises a clear `TesseraAutodiffError`-style error.

### [B3] `Module.to(dtype)` — dtype migration ✅

**Scope:** S (~100 LOC + ~80 LOC tests).

Migrate every `Parameter` and persistent `Buffer` in a module tree to a new
dtype. `to(device)` deferred until a real device handle exists post-Phase G.

**Files (modify):**
- `python/tessera/nn/module.py` — `Module.to(dtype: str) -> Module` (mutates in place + returns self for chaining)
- `tests/unit/test_nn_module.py` — coverage

**Acceptance:**
- `mlp.to("fp16")` migrates every Parameter and persistent Buffer; non-persistent buffers untouched.
- Subsequent `forward(x)` on `fp16` parameters works (uses `_as_array` extraction; numpy handles the cast).
- Round-trip: `to("fp16").to("fp32")` returns to the original dtype shape; values within fp16 quantization noise of the original.
- `to("invalid")` raises `ValueError` with the list of valid dtypes.

---

## Phase C — Theme 1 cleanup that depends on Phase B (parallelizable, ~1 week) — **landed 2026-05-09**

### [C1] `BatchNorm1d` (real Module) ✅

**Scope:** S (~80 LOC + ~80 LOC tests). Depends on **B1**.

**Files (modify):**
- `python/tessera/nn/layers.py` — `BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)`
- `tests/unit/test_nn_module.py`

**Acceptance:**
- `register_buffer("running_mean", zeros)`, `register_buffer("running_var", ones)`, `register_buffer("num_batches_tracked", 0)`.
- Train mode: uses batch stats; updates running stats with `momentum`.
- Eval mode: uses running stats (no update).
- `state_dict()` includes the buffers; `load_state_dict()` restores them.
- Replace the phantom in `nn/__init__.py`.

### [C2] `KVCache` Module wrapper ✅

**Scope:** XS (~60 LOC + ~50 LOC tests). Depends on **B2**.

Module form of the KV cache for layered transformer use:

```python
class KVCache(Module):
    def __init__(self, num_heads, head_dim, max_seq, dtype="fp32"): ...
    def forward(self, k, v):  # appends and returns full (K, V) so far
```

**Acceptance:** transformer block test uses `ts.nn.KVCache` to maintain decoding state across calls; second forward returns concatenated K/V.

---

## Phase D — Theme 3 streaming kernels (~2–3 weeks) — **D1/D2/D4 landed 2026-05-09; D3 deferred**

### [D1] `ops.depthwise_conv1d` ✅

**Scope:** M (~250 LOC + ~150 LOC tests). Depends on **B1** for streaming state.

**Acceptance:**
- `ops.depthwise_conv1d(x, w, *, kernel_size, padding, groups, causal=False, state=None) -> (y, state_out)`.
- Numpy reference matches torch's `F.conv1d(..., groups=in_channels)`.
- Causal version produces no future leakage (test with one-hot inputs).
- VJP for autodiff (Tier 2 v1 op set extended).
- Streaming variant: passing `state=` into a sequence of length-1 calls produces the same output as one length-N call.

### [D2] `ops.online_softmax` (streaming, numerically stable) ✅

**Scope:** M (~150 LOC + ~100 LOC tests).

**Acceptance:** matches naive `softmax` to fp32 precision while accepting one chunk at a time + carry state. Required for FA-4 reference path; useful standalone.

### [D3] `ops.selective_ssm` (Mamba2 selective state-space op) 📋

**Scope:** L (~400 LOC + ~200 LOC tests). Depends on **D1** + **B1**.

**Acceptance:**
- Mamba2 algorithm: A/B/C/Δ projections, chunked scan (size 128), output gate.
- Replaces the placeholder reference inside `examples/advanced/Nemotron_Nano_12B_v2/`.
- VJP shipped (chunked scan adjoint is the interesting part).

### [D4] `nn.DynamicDepthwiseConv1d` Module ✅

**Scope:** XS (~50 LOC). Depends on **D1** + **B1**.

Replaces the phantom; wraps `ops.depthwise_conv1d` with state buffer.

---

## Phase E — Theme 4 KV-cache + block quantization (~1–2 weeks) — **landed 2026-05-09**

### [E1] `ops.quantize_kv` / `ops.dequantize_kv` ✅

**Scope:** S (~150 LOC + ~80 LOC tests).

**Acceptance:**
- `quantize_kv(k, v, bits=4) -> (k_q, v_q, scale, residual_bits)` — matches the algorithm sketched in `examples/advanced/kv_cache_serving/`.
- `dequantize_kv(...)` round-trips with bounded error.
- Numerical: max relative error ≤ `2^(-bits)` on N(0, 1) inputs.

### [E2] `ops.kv_cache_update` / `ops.kv_cache_read` (with `KVCacheHandle`) ✅

**Scope:** M (~200 LOC + ~150 LOC tests). Depends on **B2** + **E1**.

Functional API on `KVCacheHandle`. Replaces the legacy `kv_cache_append`/`kv_cache_prune` (which become thin shims).

### [E3] Rolling-window KV-cache state machine ✅

**Scope:** S (~150 LOC + ~100 LOC tests). Depends on **E2**.

**Acceptance:** Cache supports `evict_oldest(n)`; auto-eviction when `current_seq == max_seq`; tracked entries ≤ window size.

---

## Phase F — Tier 2 autodiff follow-ups (parallelizable, ~2–4 weeks) — **F1/F2/F3 landed 2026-05-09; F4 ODS scaffolded; F5/F6/F7 deferred**

### [F1] Mixed-precision: autocast context + GradScaler ✅

**Scope:** M (~300 LOC + ~150 LOC tests). Independent.

**Acceptance:**
- `with ts.autodiff.autocast("fp16"): y = model(x)` casts forward inputs to fp16, accumulates in fp32 for matmul, casts back.
- `GradScaler.scale(loss); scaler.step(optimizer)` follows the standard fp32 master-copy pattern.

### [F2] Activation checkpointing (`rematerialize`) ✅

**Scope:** M (~250 LOC + ~150 LOC tests). Independent.

**Acceptance:**
- `with ts.autodiff.rematerialize(): y = expensive_block(x)` — forward stores only the recipe, recomputes during backward.
- Memory test: peak resident activations during backward < forward-only peak by a measurable margin on a 4-layer MLP.

### [F3] Custom kernel adjoints — `flash_attn` ✅, `fft`/`ifft`/`rfft`/`irfft` ✅, `moe` 🔲

**Scope:** S each (~80 LOC + ~80 LOC tests per op). Independent.

Sized as A5 — derive standard analytical VJP, register via `custom_rule`.

### [F4] Graph IR adjoint ops 🟡 (ODS + pass stub landed; MLIR build integration is follow-up)

**Scope:** L (~600 LOC code + ~300 LOC tests). Foundational.

Move autodiff from numpy-reference (Tier 2 v1) to IR-level. Adds adjoint ops
to `TesseraOps.td`; teaches the lowering pipeline to materialize backward
computations as Graph IR rather than tape-walked numpy.

**Files (modify):**
- `src/compiler/ir/TesseraOps.td` — adjoint ops
- `src/transforms/lib/AutodiffPass.cpp` (new) — Graph IR adjoint generation
- `tools/tessera-opt` — register the pass + a `tessera-autodiff` pipeline

**Acceptance:**
- A `@jit` function with `@autodiff.reverse` lowers to Graph IR with adjoint ops.
- `tessera-opt --tessera-autodiff` on a forward IR produces a forward+backward IR.
- Numerical equivalence with Tier 2 v1 numpy reference.

**Decision (locked 2026-05-09):** `AdjointInterface` op trait. Each
differentiable op declares an `adjoint` method on its ODS interface; the
`AutodiffPass` walks the IR and inserts adjoint ops by interface dispatch.
Avoids doubling the op count and keeps lowering tables small. Custom adjoints
register via the same `tessera.autodiff.custom_rule(name)` Python API used in
the v1 numpy reference, with the registration also visible to the IR pass.

### [F5] Effect-aware adjoint collective insertion 📋

**Scope:** M (~250 LOC). Depends on **F4**.

Extends `GPUCollectiveInsertionPass` to insert `reduce_scatter` / `all_gather`
on adjoint paths for distributed parameters.

**Acceptance:** A 2-rank mock-collective test of MLP training shows correct
gradient aggregation across ranks.

### [F6] JAX-style transforms — `jacrev`, `jacfwd`, `vmap` 📋

**Scope:** L. Independent. **Mark 🔲 deferred** unless explicit user demand —
high cost, unclear immediate payoff for ML training. Tracked here so it isn't
forgotten.

### [F7] Higher-order derivatives 🔲

Out of scope for the foreseeable cycle. Tracked.

---

## Phase G — NVIDIA execution path (THE BIG ONE, ~4–8 weeks)

The single highest-leverage block. Until this lands, the autotuner is dark,
FA-4 is unverified, the GPU-only tier is theoretical, and GPU CI is impossible.

### [G1] Audit current state — what's actually missing? 📋

**Scope:** S (1–2 days investigation).

**Files (new):**
- `docs/audit/nvidia_execution_audit.md` — concrete punch list

**Acceptance:** list every IR pass + runtime call needed for one shape (SM_90 BF16 GEMM 128×128×128) to actually launch on a real H100. Items become G2–G7.

### [G2] CUDA runtime backend wiring verification 📋

**Scope:** M. Likely audit shows `cuda_backend.cpp` is partially wired; finish what's missing.

### [G3] WGMMA SM_90 BF16 GEMM end-to-end 📋

**Scope:** L (~500 LOC + tests). The first real GPU execution. Pick one shape, drive it through the whole stack: Graph IR → Schedule IR → Tile IR → NVIDIA Target IR → PTX → cuBin → launch.

**Acceptance:** `@jit(target=GPUTargetProfile(isa=ISA.SM_90))` on a fixed-shape BF16 GEMM produces correct output (within fp32 tolerance) when run on a real H100.

### [G4] TMA descriptor wiring 📋

**Scope:** M. `NVTMADescriptorPass` exists but its descriptors must reach the runtime launch. Depends on G3.

### [G5] FA-4 forward verification on H100 📋

**Scope:** M. With G3+G4, run FA-4 forward on real hardware; compare against the numpy reference; iterate on tile-size tuning.

### [G6] Autotuner sweep against cuBLAS baseline 📋

**Scope:** S. With G3 done, the Bayesian autotuner finally has something to time. Run a sweep over `tile_q`/`tile_kv`/`pipeline_stages`; produce a JSON of best configs per shape.

### [G7] CI: GPU-spine equivalent of `validate.sh` 📋

**Scope:** M. CUDA-required tests gated; `scripts/validate.sh --gpu` runs the GPU subset.

---

## Phase H — Conv2d Module + remaining nn cleanup (~1 week)

### [H1] Conv2d Module — layout NHWC (decision locked) 📋

**Scope:** S.

**Decision (locked 2026-05-09):** **NHWC default** — matches existing
`tessera.ops.conv2d`. Ship a `tessera.nn.Conv2dNCHW` shim that does
`x.transpose(0, 2, 3, 1)` → `Conv2d(...)` → `out.transpose(0, 3, 1, 2)` for
torch-port code. Both forms share weight storage in HWIO (`(kH, kW, in, out)`).

**Acceptance:** `Linear`-shaped Module wrapper; `register_buffer("bias", ...)` if `bias=True`; tested forward shape.

### [H2] `LSTM` Module 🔲

**Mark deferred.** RNN cells need state-propagation primitives that are a
separate build-out. Defer until concrete user demand.

---

## Phase I — DDP / FSDP wrappers (post-F4 + F5 + G)

### [I1] `tessera.distributed.DDP(module, mesh_axis="dp")` 📋

**Depends on F4 + F5 + G.** All-reduce on adjoint path; backward triggers gradient sync.

### [I2] `tessera.distributed.FSDP(module, mesh_axis="dp")` 📋

**Depends on I1.** Sharded parameters + gather-on-forward + reduce-scatter-on-backward. `OptimizerShardPass` (Phase 5) is the underlying machinery.

---

## Out-of-scope / consciously deferred 🔲

| Item | Reason |
|------|--------|
| `LSTM` Module (H2) | RNN cell state-propagation needs its own primitive design. No active demand. |
| Higher-order derivatives (F7) | Niche; high implementation cost. |
| JAX-style `vmap`/`jacrev`/`jacfwd` (F6) | High cost, unclear payoff for ML training workloads. Tracked. |
| Module device migration (`to("cuda")`) | Requires a real device handle; tied to Phase G. |
| AIR bitcode codegen on Apple GPU | MPS+MSL covers everything we need; revisit only if a perf wall demands it. |

---

## Cross-references

- `docs/audit/advanced_examples_capability_gap.md` — per-example status tied
  to these phases (Theme 3 = Phase D, Theme 4 = Phase E, etc.)
- `docs/spec/AUTODIFF_SPEC.md` — Tier 2 v1 spec; Phase F lands the follow-ups
- `docs/CANONICAL_API.md` — public surface; update as each task lands
- `CLAUDE.md` Architecture Decisions #19, #21, #22 — relevant invariants
- `examples/advanced/README.md` — honest per-example status; refresh after C/D/E
- `tests/unit/test_nn_module.py`, `tests/unit/test_autodiff.py` — current test
  surface that future phases extend

---

## Phase summary (for at-a-glance scope)

| Phase | LOC est | Wks | Independent? | Unblocks |
|-------|---------|-----|--------------|----------|
| A — quick wins | ~1,000 + 600 docs | 1–2 | ✅ all 5 tasks | Theme 1 90% closed; debugging story; autograd-via-flash_attn |
| B — protocols | ~700 | 1 | sequential within | C, D, E |
| C — Theme 1 cleanup | ~250 | 0.5 | ✅ within phase | BatchNorm1d, KVCache module |
| D — streaming | ~1,000 | 2–3 | partly sequential | Jet_nemotron, Nemotron_Nano forward |
| E — KV-cache | ~700 | 1–2 | sequential within | kv_cache_serving, Fast_dLLM_v2, paged-MLA |
| F — autodiff follow-ups | ~1,500 | 2–4 | F1+F2+F3 ‖, F4→F5 | distributed training, mixed precision, checkpointing |
| G — NVIDIA execution | ~2,000 | 4–8 | sequential within | autotuner, FA-4 verification, GPU CI |
| H — Conv2d Module | ~150 | 0.5 | indep | torch-port examples |
| I — DDP/FSDP | ~600 | 2 | post-F+G | distributed training at scale |

**Total ~7,900 LOC code + ~3,000 LOC tests + ~1,000 LOC docs over 12–20 weeks.**
