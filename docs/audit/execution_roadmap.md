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

**Critical paths (all three chains are now closed as of 2026-05-09 — G is the only remaining frontier):**
- ✅ **B1 (buffer protocol)** unblocked `BatchNorm1d` (C1) + streaming kernels
  with state (D). Closed.
- ✅ **B2 (`KVCacheHandle` value type)** unblocked Theme 4 entirely (E1–E3) +
  the `KVCache` Module wrapper (C2). Closed.
- ✅ **F4 (Graph IR adjoints) → F5 (effect-aware adjoint collectives) → I
  (DDP/FSDP).** This chain — the long pole for distributed training — is
  closed at v1. F4+F5 verified end-to-end on MLIR 21; I1/I2 ship against
  mock_collective.
- 🚧 **G is the highest-leverage remaining phase.** Without NVIDIA execution, the
  autotuner, FA-4 verification, GPU-only tier, and GPU CI are all dark. G1
  audit is done; G2–G7 sequenced in `docs/audit/nvidia_execution_audit.md`.

Estimated remaining runway: **4–8 weeks of focused work on Phase G** (4–6 days
to first H100 BF16 GEMM per the G1 audit; remainder is sweep + verification +
CI). All other phases complete.

---

## Phase A — Quick wins (independent, parallelizable, ~1–2 weeks)

### [A1] Debugging story — env-var IR dumps + per-pass diff + how-to doc ✅

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

### [A2] Dynamic shapes — audit + doc + test ✅

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
CPU, and Apple GPU — symbolic dims flow through to actual execution.
Call-time constraint enforcement also landed: `JitFn.__call__` resolves
symbolic dims from concrete argument shapes and raises `TesseraConstraintError`
for `Divisible`, `Range`, and `Equal` violations.

### [A2-followup] Call-time constraint enforcement ✅

**Scope:** S (~150 LOC). Lift the constraint check from `@jit` decoration
into `JitFn.__call__` so that constraint violations on real argument shapes
raise `TesseraConstraintError` even when no `bindings=` was supplied.
Acceptance: the `xfail` test in `test_dynamic_shapes.py` flips to `xpass` and
the `xfail` mark is removed.

### [A3] KV-cache lowering coverage matrix ✅

**Scope:** XS (doc-only, ~80 LOC).

**Files (new):**
- `docs/audit/kv_cache_coverage_matrix.md`

**Files (modify):**
- `CLAUDE.md` Architecture Decision #21 — link to the matrix

**Acceptance:**
- Per-target table: rows = `kv_cache_append`, `kv_cache_prune`, FA-4 with cache; columns = x86, Apple_cpu, Apple_gpu, NVIDIA, ROCm, TPU, Cerebras, Metalium, RubinCPX. Cells: ✅ executes / 🟡 lowers but no execution / 🔲 emits diagnostic / ❌ silent no-op (bug).
- Audit method: grep each backend's lowering passes for `kv_cache_*`, verify behavior, document.
- For any 🔲 cells found that turn out to be ❌, file a follow-up task in this roadmap.

### [A4] Theme 1 cleanup — small phantoms that don't need new infrastructure ✅

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

### [A5] `flash_attn` VJP via `custom_rule` ✅

**Scope:** XS (~50 LOC code + ~60 LOC test).

The numpy-reference VJP is shipped via the built-in `custom_rule` registry so
`MultiHeadAttention`-style code can train end-to-end on the reference path.

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

### [D3] `ops.selective_ssm` (Mamba2 selective state-space op) ✅ (forward; VJP is follow-up)

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

## Phase F — Tier 2 autodiff follow-ups (parallelizable, ~2–4 weeks) — **F1–F7 landed; +Phase F-MoR shipped 2026-05-10**

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

### [F3] Custom kernel adjoints — `flash_attn` ✅, `fft`/`ifft`/`rfft`/`irfft` ✅, `moe` ✅, `selective_ssm` (D3 VJP) ✅, `linear_attn` ✅, `silu_mul` ✅

**Scope:** S each (~80 LOC + ~80 LOC tests per op). Independent.

Sized as A5 — derive standard analytical VJP, register via `custom_rule`.

**Status (verified 2026-05-10):** all originally-listed F3 ops + the
post-F3 follow-ups have analytical VJPs registered in
`python/tessera/autodiff/vjp.py`:

- `flash_attn` — line 590 (recompute-scores adjoint)
- `fft` / `ifft` / `rfft` / `irfft` — lines 637–667 (spectral adjoints)
- **`moe`** — line 219 (per-token routed-matmul adjoint, accumulating
  per-expert weight gradient when multiple tokens share an expert)
- **`selective_ssm`** — line 680 (Mamba2 chunked-scan adjoint that
  recomputes the forward trajectory and walks ``t = S-1 → 0``)
- `linear_attn` / `linear_attn_state` — lines 407 / 524
- `silu_mul` / `gather` / `clip` / `masked_fill` — extras shipped
  with Theme 9 / SwiGLU work

The original `moe 🔲` and "D3 VJP" follow-ups are both ✅ closed —
both predate this 2026-05-10 update; the doc is being refreshed to
reflect that.

### [F4] Graph IR adjoint ops ✅ — verified end-to-end on MLIR 21
ODS + pass body + per-op `buildAdjoint` impls + CMake tablegen target +
multi-output rewrite that exposes argument cotangents as additional
function outputs. `tessera-opt --tessera-autodiff` builds clean against
`/opt/homebrew/opt/llvm@21` and the lit fixture
`tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir` passes FileCheck
showing: cotangent seed (constant tensor of 1.0), two transposed matmuls
(dA = seed @ B^T, dB = A^T @ seed), multi-result return signature, and
`tessera.autodiff.arg_cotangents` annotation. Build recipe:
```bash
cmake .. -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
         -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir
make -j tessera-opt
./tools/tessera-opt/tessera-opt --tessera-autodiff <input.mlir> | FileCheck <input.mlir>
```

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

### [F5] Effect-aware adjoint collective insertion ✅ — full rewrite landed
Real `tessera.collective.{reduce_scatter, all_gather, all_reduce}` ops
emitted on cotangent SSA values from F4's multi-output rewrite. Per-arg
`tessera.adjoint_collective_plan` attribute records the choice. Pipeline
alias `tessera-autodiff-pipeline` runs F4+F5 together. Compiles clean.

**Scope:** M (~250 LOC). Depends on **F4**.

Extends `GPUCollectiveInsertionPass` to insert `reduce_scatter` / `all_gather`
on adjoint paths for distributed parameters.

**Acceptance:** A 2-rank mock-collective test of MLP training shows correct
gradient aggregation across ranks.

### [F6] JAX-style transforms — `vmap`, `jacrev`, `jacfwd` ✅

**Scope:** L (~600 LOC code + ~400 LOC tests). Independent.

**Status (landed 2026-05-09 via deferred-items plan Item 5):**
- `tessera.autodiff.vmap(fn, in_axes=0, out_axes=0)` — naive scan-then-stack;
  per-arg `in_axes` (int / sequence / None); `out_axes=None` returns the
  per-element list as-is.
- `tessera.autodiff.jacrev(fn, argnums=0)` — reverse-mode Jacobian; one
  backward pass per output dim. Uses Item 4's `retain_graph=True`
  re-runnable tape.
- `tessera.autodiff.jacfwd(fn, argnums=0)` — forward-mode Jacobian via the
  JVP engine in `python/tessera/autodiff/jvp.py`. v1 implementation uses
  central FD (`eps=1e-6`) for the `jvp(fn, primals, tangents)` entry
  point; the parallel JVP-rule registry is in place for 12 core ops.
  A true tape-based dual-number propagation is a Phase G perf
  follow-up that won't change the API contract.
- 15 unit tests in `tests/unit/test_jax_transforms.py` (vmap × 6,
  jacrev × 3, jacfwd × 3, JVP engine × 2, composability × 1).

The canonical `vmap(grad(fn))` per-sample-gradient pattern works as
written.

### [F7] Higher-order derivatives ✅

**Scope:** M (~400 LOC code + ~250 LOC tests).

**Status (landed 2026-05-09 via deferred-items plan Item 4):**
- `tessera.autodiff.grad(fn, argnums=0)` — JAX-style gradient
  transform. Returns ndarray for int argnums, tuple for sequence
  argnums; uses `accumulate_param_grad=False` so it doesn't leak into
  caller `Parameter.grad` slots.
- `tessera.autodiff.hvp(fn, primals, tangents, eps=1e-4)` —
  Hessian-vector product via central finite difference of `grad`.
  ~1e-6 accuracy at fp64.
- `tessera.autodiff.elementwise_grad(fn)` — per-element derivative
  for vector → vector elementwise ops; convenient for inspecting
  activation derivatives.
- `tape.backward(target, *, retain_graph=False, accumulate_param_grad=True)`
  — re-runnable tape with explicit opt-in via `retain_graph=True`.
  Required for `jacrev` (F6).
- 15 unit tests in `tests/unit/test_higher_order_autodiff.py`
  (`grad` × 6, `hvp` × 3, `elementwise_grad` × 3, `retain_graph` × 3).

True forward-over-reverse HVP (`jvp(grad(fn), x, v)`) lands when the
F6 forward-mode tape's analytical-rule path matures; the FD path is
correct for L-BFGS / natural-gradient / GAN-penalty workloads today.

### [F-MoR] Mixture of Recursions ✅ — landed 2026-05-10

**Scope:** M (~400 LOC code + ~280 LOC tests). Independent.

Bae et al. 2025 "Mixture-of-Recursions" — adaptive computation by
routing tokens through different numbers of recursive layer
applications. A learned per-token router assigns each token to a
target depth d ∈ [1, max_depth]; the layer is applied recursively to
each token until it hits its target depth, then the token's hidden
state freezes for the rest of the loop. Computational savings follow
from "easy" tokens routing to lower depths.

**Shipped 2026-05-10:**

- **Three primitive ops** in `python/tessera/__init__.py`:
  - `ops.mor_router(x, w_router, *, max_depth)` — argmax-based
    token-choice depth router; returns ``(B, S)`` int64 in
    ``[1, max_depth]``.
  - `ops.mor_partition(x, depth, *, step)` — bool mask of tokens
    whose target depth ≥ step (1-indexed).
  - `ops.mor_scatter(full, updated, mask)` — write `updated` values
    into `full` at masked positions (frozen-token semantics for the
    unselected rows).
- **VJPs** in `python/tessera/autodiff/vjp.py`:
  - `mor_router` returns zero gradients (argmax is non-differentiable;
    real router-training uses auxiliary load-balance / utilization
    losses the user adds explicitly).
  - `mor_partition` zero-grad on the int-valued depth and the
    real-valued x.
  - `mor_scatter` is linear in `updated`; gradients flow through
    `full` on the False positions and through `updated` on the True
    positions.
- **`nn.MixtureOfRecursions(layer, *, embed_dim, max_depth)`** Module —
  composes the router + recursion loop. The wrapped `layer`'s
  parameters are shared across all recursion steps (the canonical
  MoR contract).
- **ODS ops** `tessera.mor_router` / `tessera.mor_partition` /
  `tessera.mor_scatter` in `src/compiler/ir/TesseraOps.td`.
- **Lit fixture** `tests/tessera-ir/phase8/mor_primitives.mlir` —
  ODS verifier + assembly-format roundtrip.
- **17 unit tests** in `tests/unit/test_mor.py` covering forward
  correctness of all three ops + the Module + per-token recursion
  depth verified end-to-end + VJP shape contracts.

**Acceptance:**
- Per-token recursion depth is honored: with a layer that adds a
  constant, output for token i increases by exactly `depth_i * Δ`.
- `mor_partition(depth=[1,2,3,2,1], step=2)` returns
  `[F, T, T, T, F]`.
- `mor_scatter(full, updated, mask)` writes `updated` only where
  `mask` is True; the rest of `full` is preserved bit-equivalent.
- `nn.MixtureOfRecursions` rejects rank-2 inputs and `max_depth=0`.

**Phase G follow-ups** (not gating the v1 surface):
- Token-active-only kernel: gather active tokens before the layer
  call and scatter back after, instead of the v1 reference's
  apply-to-full-then-mask approach. Saves compute proportional to
  token-depth utilization.
- KV-cache-share-first / KV-cache-recursion policies for attention
  inside the inner layer (the example folder at
  `examples/archive/advanced/Tessera_MoR/` sketches the design).

---

## Phase G — NVIDIA execution path (THE BIG ONE, ~4–8 weeks) — **G1 audit landed 2026-05-09; G2–G8 sized**

The single highest-leverage block. Until this lands, the autotuner is dark,
FA-4 is unverified, the GPU-only tier is theoretical, and GPU CI is impossible.

### [G1] Audit current state — what's actually missing? ✅ (delivered at `docs/audit/nvidia_execution_audit.md`)

Per-component audit + 8-task punch list (G1-1 through G1-8). Critical path
to first H100 BF16 GEMM 128×128×128: **4–6 days** of focused work, of which
only G1-5/G1-6/G1-8 require real H100 hardware. G1-2/G1-3/G1-4/G1-7 can
land on a CUDA-only-no-H100 dev box.

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

### [H1] Conv2d Module — layout NHWC (decision locked) ✅

**Scope:** S.

**Decision (locked 2026-05-09):** **NHWC default** — matches existing
`tessera.ops.conv2d`. Ship a `tessera.nn.Conv2dNCHW` shim that does
`x.transpose(0, 2, 3, 1)` → `Conv2d(...)` → `out.transpose(0, 3, 1, 2)` for
torch-port code. Both forms share weight storage in HWIO (`(kH, kW, in, out)`).

**Acceptance:** `Linear`-shaped Module wrapper; `register_buffer("bias", ...)` if `bias=True`; tested forward shape.

### [H2] `LSTM` Module ✅ (state-propagation primitive shipped)
`ops.lstm_cell` returns packed `[h_t, c_t]` (single-output for v1 tape
compatibility); `ops.lstm_state_h`/`lstm_state_c` extract parts under
autodiff. VJPs registered for all three; BPTT through 2+ steps verified
against numerical Jacobian to 1e-11 at fp64. `nn.LSTMCell` (single-step
Module wrapping the primitive) and `nn.LSTM` (multi-step unroll) both
ship.

---

## Phase I — DDP / FSDP wrappers (post-F4 + F5 + G)

### [I1] `tessera.distributed.DDP(module, mesh_axis="dp")` ✅

**Depends on F4 + F5 + G.** All-reduce on adjoint path; backward triggers gradient sync.

### [I2] `tessera.distributed.FSDP(module, mesh_axis="dp")` ✅ (v1 — per-rank Module instances, sharded leading-dim, mock_collective tested)

**Depends on I1.** Sharded parameters + gather-on-forward + reduce-scatter-on-backward. `OptimizerShardPass` (Phase 5) is the underlying machinery.

---

## Out-of-scope / consciously deferred 🔲

| Item | Reason |
|------|--------|
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

| Phase | Status | LOC est | Wks | Independent? | Unblocks |
|-------|--------|---------|-----|--------------|----------|
| A — quick wins | ✅ complete | ~1,000 + 600 docs | 1–2 | ✅ all 5 tasks | Theme 1 90% closed; debugging story; autograd-via-flash_attn |
| B — protocols | ✅ complete | ~700 | 1 | sequential within | C, D, E |
| C — Theme 1 cleanup | ✅ complete | ~250 | 0.5 | ✅ within phase | BatchNorm1d, KVCache module |
| D — streaming | ✅ (D3 VJP open) | ~1,000 | 2–3 | partly sequential | Jet_nemotron, Nemotron_Nano forward |
| E — KV-cache | ✅ complete | ~700 | 1–2 | sequential within | kv_cache_serving, Fast_dLLM_v2, paged-MLA |
| F — autodiff follow-ups | ✅ all closed (F1–F7 + F3-moe + D3-VJP + Phase F-MoR) | ~2,500 | 2–4 | F1+F2+F3 ‖, F4→F5 | distributed training, mixed precision, checkpointing, JAX-style transforms, Mixture of Recursions |
| G — NVIDIA execution | 🚧 G1 audit only; G2–G7 open | ~2,000 | 4–8 | sequential within | autotuner, FA-4 verification, GPU CI |
| H — Conv2d Module + LSTM | ✅ complete | ~150 | 0.5 | indep | torch-port examples |
| I — DDP/FSDP | ✅ v1 complete | ~600 | 2 | post-F+G | distributed training at scale |

**Total ~7,900 LOC code + ~3,000 LOC tests + ~1,000 LOC docs over 12–20 weeks.**

---

## Status snapshot — 2026-05-10

**Done:** A, B, C, D (forward + D3 VJP), E, F (F1–F7 + F3-moe + Phase F-MoR), H, I.

**Remaining frontier:** **Phase G (NVIDIA execution)** — the only long pole. G1 audit is complete (`docs/audit/nvidia_execution_audit.md`); G2–G7 are open. Per the audit: 4–6 days of focused work to first H100 BF16 GEMM, of which only G1-5/G1-6/G1-8 need real H100 hardware.

**Sequenced next steps for G:**
1. **G2** — finish wiring `cuda_backend.cpp` per the audit (CUDA-only-no-H100 dev box is sufficient).
2. **G3** — first WGMMA SM_90 BF16 GEMM end-to-end (Graph IR → Schedule IR → Tile IR → NVIDIA Target IR → PTX → cuBin → launch). This is the unlock; everything downstream is sweep-on-top.
3. **G4** — TMA descriptors reach runtime launch (depends on G3).
4. **G5** — FA-4 forward verification on real H100 (needs hardware).
5. **G6** — autotuner sweep vs cuBLAS baseline (needs hardware).
6. **G7** — `validate.sh --gpu` CI spine.

**Cleanup items closed 2026-05-10:**
- ✅ **D3 VJP** — `selective_ssm` chunked-scan adjoint shipped; registered at `python/tessera/autodiff/vjp.py:680` (`vjp_selective_ssm` recomputes the forward trajectory and walks ``t = S-1 → 0`` accumulating gradients).
- ✅ **F3-moe** — MoE router `custom_rule` VJP shipped at `python/tessera/autodiff/vjp.py:219` (per-token routed-matmul adjoint with per-expert gradient accumulation).
- ✅ **F6** — `vmap` / `jacrev` / `jacfwd` shipped (deferred-items plan Item 5).
- ✅ **F7** — `grad` / `hvp` / `elementwise_grad` + re-runnable tape shipped (deferred-items plan Item 4).
- ✅ **Phase F-MoR** — Mixture of Recursions primitives + `nn.MixtureOfRecursions` Module + ODS ops + lit fixture + 17 unit tests (2026-05-10).

**Three critical chains called out in the rationale at the top of this doc — all closed:**
- B1 → C1 + D: ✅ complete (D3 VJP shipped)
- B2 → E1–E3 + C2: ✅ complete
- F4 → F5 → I (DDP/FSDP): ✅ v1 complete
