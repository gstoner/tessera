# Single-command-buffer decode chain — audit Action 6 / Pattern table row #6

**Created**: 2026-06-01 (Phase G follow-on, audit Action 6 / table row #6)
**Status**: scaffold landed; per-op encode-session variants are the
incremental work.

## What the audit asked for

From the Apple Metal 4 doc-deep review (`skills.md` lines 110-158):

> **ML passes on the GPU timeline** — keep prefill/decode/attn/MLP/projection
> on one command buffer; avoid CPU turnarounds.

A typical decoder layer has ~7 ops in flight per token:

```
  pre_norm → qkv_projection → rope_q → rope_k → flash_attn →
  output_projection → post_norm → mlp_up → silu → mlp_down → residual
```

Dispatching each op as its own `MTLCommandBuffer` (current pattern in
`apple_gpu_runtime.mm` for nearly every dispatcher) means N command-
buffer commits + N GPU↔CPU wait pairs per token. At per-token latency
budgets <1 ms the per-op commit overhead dominates the actual compute.

Apple's `MTL4MachineLearningCommandEncoder` (used in PK4 for packaged
ML) demonstrates the right shape: *one* command buffer holds the
entire neural network, the encoder appends ops into it, and the GPU
sees a contiguous timeline.

## Current state (post-bmm)

Tessera already has the encode-session ABI for the BMM case:

* `ts_enc_begin()` — opens a session, returns a `TsEncodeSession*`
  wrapping an `MPSCommandBuffer` + the underlying `MTLCommandBuffer`.
* `ts_enc_commit_wait(s)` — commits + waits (since batch 5, with
  Pattern-4 timeout protection via a shared event).
* `tessera_apple_gpu_bmm_dev_f32_enc(s, A, B, O, …)` — encodes a BMM
  into the session's command buffer. Operates on device-resident
  `TsDeviceTensor*` (no host↔device roundtrips).
* `ts_dev_alloc` / `ts_dev_upload` / `ts_dev_download` / `ts_dev_free` —
  resident-tensor lifecycle.

Only **one** op (`bmm_dev_f32`) supports the encode-session path today.

## Plan

### Stage 1 — scaffold (this PR)

The minimum-viable proof that the architecture extends past BMM:

* **Add `tessera_apple_gpu_layer_norm_dev_f32_enc`**: encode-session
  variant of `layer_norm_f32` operating on `TsDeviceTensor` inputs.
  Picked because (a) it appears in every decoder layer (pre-norm /
  post-norm), (b) the f32 kernel is already migrated to Pattern-4,
  (c) it's structurally simple (one in / one out, no extra state).
* **Add a Python wrapper** at `python/tessera/apple_gpu_batched.py`
  exposing a context-manager session API:
  ```python
  with apple_gpu.batched_session() as s:
      y_dev = apple_gpu.layer_norm_enc(s, x_dev, gamma, beta, eps)
      z_dev = apple_gpu.bmm_enc(s, y_dev, w_dev, …)
      # Both ops encode into ONE command buffer; commit fires on
      # context-manager exit.
  ```
* **Test**: `tests/unit/test_apple_gpu_single_command_buffer.py`
  proves that a `layer_norm + bmm` chain runs in *one* session and
  produces the right numerical answer; the test also probes the
  command-queue's submission count to verify exactly 1 cb committed
  (not 2). End-to-end proof that the scaffold composes.

### Stage 2 — per-op encode-session variants (follow-on PRs)

For each op in the decode envelope, add a `_dev_f32_enc(s, …)` variant:

* `rope_dev_f32_enc` / `rope_dev_f16_enc`
* `flash_attn_dev_f32_enc` / `flash_attn_dev_f16_enc`
* `softmax_dev_f32_enc` / `softmax_dev_f16_enc`
* `silu_dev_f32_enc` (already supported via MPSGraph lane — needs an
  encode-session bridge)
* `gelu_dev_f32_enc`
* `rmsnorm_dev_f32_enc`
* `matmul2d_dev_f32_enc` (the M8 resident path likely already has this
  under a different name — audit at Stage 2 start)

Estimated work: ~6 ops × ~50 LOC each = ~300 LOC across 1-3 PRs.

### Stage 3 — Python ergonomics + jit integration

**Phase 1 (landed 2026-06-01)** — `@decode_chain` decorator. Wraps a
function whose first arg is the session handle; opens + commits the
session automatically. Caller still writes explicit `_enc(s, ...)`
calls but doesn't manage the session lifecycle. 90% of the ergonomic
win for ~30 LOC. Lives in `python/tessera/apple_gpu_batched.py`.

**Phase 2 substrate (landed 2026-06-01)** —
`python/tessera/apple_gpu_chain.py` ships the three pieces
`compiler/jit.py` will consume:

1. `ENCODE_OP_REGISTRY` — canonical `(op_name, dtype) → EncodeOpSpec`
   metadata for every encode-session-aware op (16 entries: 8 ops ×
   2 dtypes f32/f16).
2. `plan_chain(trace)` — walks an `OpRecord` list and groups
   consecutive encode-eligible same-dtype ops into `ChainSegment`s;
   non-eligible ops form singleton segments.
3. `execute_chain(segments)` / `run_trace(trace)` — opens a session
   per encode segment, dispatches each op through its encode helper,
   commits + waits.

15 tests in `tests/unit/test_apple_gpu_chain_planner.py` pin
registry consistency, planner correctness (incl. dtype-boundary +
non-eligible-op break), and executor numerical correctness +
single-cb invariant per segment.

**Phase 2.1a (landed 2026-06-01) — `@auto_batch` decorator +
`apple_gpu_ops` user surface.** Bridges the substrate to a true
"no decorator-per-session" experience for end users.

Pieces:

* `python/tessera/apple_gpu_ops.py` — user-facing op surface
  (`rmsnorm`, `bmm`, `rope`, `softmax`, `silu`, `gelu`, `flash_attn`,
  `layer_norm`) with dual-mode dispatch via a `contextvars`
  `_TRACE_CTX`. Eager mode → one-shot session + run + download.
  Trace mode → append `OpRecord` + return `TraceRef`.
* `TraceRef` substrate addition to `apple_gpu_chain.py` — handle to
  the output of a prior op in the same trace; executor resolves at
  run time. Supports forward refs only (`op_index >= 0`).
* `@apple_gpu_ops.auto_batch` decorator — captures the trace inside
  the wrapped function via `_trace_scope`, calls `run_trace`, then
  walks the function's return value (TraceRef / tuple / dict /
  list / plain) to resolve TraceRefs to DeviceTensors.
* Nested `@auto_batch` flattens into the outer trace (one cb total).
* Top-level import: `tessera.apple_gpu_ops.auto_batch` reachable
  from the standard `import tessera` path.

12 tests in `tests/unit/test_apple_gpu_auto_batch.py` pin:
dual-mode dispatch consistency, single-cb invariant for chains,
full Llama attention block on 1 cb, exception propagation, nested
flattening, TraceRef resolution.

**Phase 2.1b (open) — `tessera.ops.*` interception.** The remaining
work to fold auto-batching into the canonical Tessera op surface
(so users don't have to switch from `tessera.ops.rmsnorm` →
`tessera.apple_gpu_ops.rmsnorm`). Requires either: (a) a
context-aware shim layer that routes `tessera.ops.*` to
`apple_gpu_ops.*` when an `auto_batch` trace is active, or (b)
adding `@jit(target='apple_gpu', auto_batch=True)` as a kwarg that
wraps the user function via `@auto_batch` and exposes
`apple_gpu_ops` as `tessera.ops` inside the wrapped scope.

Implementation surface:

* `compiler/jit.py::jit` — add `auto_batch: bool = False` kwarg.
* When `target='apple_gpu' and auto_batch=True`, wrap the JitFn's
  `__call__` with `apple_gpu_ops.auto_batch`. Existing users who
  call `tessera.ops.*` get auto-routing through a `_TRACE_CTX`-aware
  proxy in `tessera.ops` (small dispatch layer).
* Non-encode-eligible `tessera.ops.*` calls inside the trace fall
  back to eager dispatch (they segment the trace correctly via
  the chain planner's `kind="single"` path).

**Phase 3 (open)** — `pre-decode warmup`: upload weights to device
once and reuse across decode steps (the M8 resident MLP path
established this; needs generalization).

### Stage 4 — full decoder benchmark — LANDED (2026-06-01)

Benchmark at ``benchmarks/apple_gpu/benchmark_decoder_layer_one_cb.py``
compares a Llama-style 8-op attention block in **four modes**:

* ``per_op_cold`` — upload everything (weights + activation) each
  iter, one cb per op (the naive newcomer pattern)
* ``per_op_baseline`` — weights device-resident (pre-uploaded), one
  cb per op
* ``one_cb`` — weights device-resident, all 8 ops in one cb (the
  encode-session architectural win)
* ``warmed_one_cb`` — weights via ``ResidentWeights`` + chain via
  ``@auto_batch`` (the full-stack ergonomic mode — same perf as
  ``one_cb`` plus the natural Python code shape)

Measured on M-series hardware (30 reps, median latency):

| Shape (BxSxD) | cold ms | per_op ms | one_cb ms | warmed ms | resident-weight win | single-cb win | full stack |
|---------------|---------|-----------|-----------|-----------|---------------------|----------------|-------------|
| 1x8x16        | 2.59    | 2.50      | 0.77      | 0.80      | 1.04×               | 3.27×          | **3.24×**   |
| 1x32x64       | 5.11    | 5.31      | 1.98      | 2.23      | 0.96×               | 2.69×          | **2.29×**   |
| 1x64x128      | 8.60    | 9.36      | 4.86      | 4.95      | 0.92×               | 1.93×          | **1.74×**   |

Pattern:

* **Single-cb dominates the win** at every shape (1.9-3.3×). The
  per-op command-buffer commit overhead is the main thing the
  encode-session architecture eliminates.
* **ResidentWeights wins are in the noise at these sizes** —
  weights here are D×D (256 to 16384 fp32 floats). At benchmark
  scale, per-call upload of these tiny tensors costs <0.1 ms.
  ResidentWeights matters when weights are large — a real LLM with
  D=4096 has 16M-float (64 MB) per-projection weights, where
  per-iter upload would cost milliseconds and dominate the
  speedup story.
* **warmed_one_cb ≈ one_cb** at these sizes — same kernels, same
  device-resident state, just different ergonomic shape (decorator
  vs. explicit session). Tracking the architectural promise: no
  perf cost for the cleaner API.

Above 64×128 we'd expect the single-cb speedup to keep tapering
toward 1.0× as compute fully dominates; the architecture stays
relevant for small-batch decode where commit overhead is the
critical bottleneck.

### Stage 5 — Multi-layer transformer benchmark + ops-per-cb cliff — LANDED (2026-06-01)

**Architectural finding (2026-06-01, refined)**: the single-cb
encoded chain hits an empirical cliff that's **shape × op-count
dependent**, not pure op count.

* At 1×33×65 with N=80 layers (160 ops), the chain completes in
  12.5 ms — no cliff observed.
* At 1×32×64 with N=12 (144 ops) full attention+MLP block, the
  chain completes in 15.6 ms in a fresh process — no cliff.
* At 1×64×256 with N=6 layers (54 ops), the chain reliably hangs
  at the 30-second commit_and_wait timeout — cliff fires.

Root cause hypothesis (not definitively isolated): **MPSGraph
compile time scaling with shape × distinct graphs per cb**. Large
shapes amplify per-op compile cost; at some product of (shape,
op count), the first-encounter graph build exceeds 30 s on this
hardware. The cliff is not pure encoder count or buffer count.

An earlier session reported the cliff at 1×32×64 / N=5 (60 ops),
which was real at that moment but no longer reproduces in a fresh
process — suggesting MPSGraph compile state is system-dependent
in ways outside our control. Chunking infrastructure (next section)
remains a valuable defensive default.

Practical mitigations:

* **Smaller per-layer op count** — flash_attn instead of separate
  softmax+matmul keeps the chain compact. Already the design.
* **Multiple cb commits per decode step** — chunk N layers into
  K cb's, each within the budget. Removes the "1 cb per step"
  promise but unblocks deep models. Phase 5b follow-on.
* **Larger per-shape compute** — at larger shapes per op spends more
  time computing, may shift the cliff. Untested.

The honest framing: **single-cb decode is real and substantial for
sub-budget chains** (4.41× at 3 layers / small shape; 1.9-3.3× at the
single attention block). Beyond the budget, the architecture
gracefully degrades to multi-cb dispatch — same encode-session ABI,
just N commits per step instead of 1.



Stacks N attention + MLP sub-blocks under ``ResidentWeights`` +
``@auto_batch`` and measures the full-stack speedup. The benchmark
at ``benchmarks/apple_gpu/benchmark_multi_layer_transformer.py``
runs two modes:

* ``per_op_cold`` — re-upload every weight + activation each step,
  one cb per op (the naive newcomer pattern at multi-layer scale)
* ``warmed_one_cb`` — weights via ``ResidentWeights``, full
  N-layer chain via ``@auto_batch`` (one cb per step)

Measured on M-series, 8 reps median.

Pre-chunking (sub-cliff configurations, single-cb):

| Shape (BxSxD,N)   | cold_ms | warmed_ms | speedup | tok/s (warmed) |
|-------------------|---------|-----------|---------|----------------|
| 1×8×16, N=4       | 24.95   | 3.73      | **6.69×** | 2146         |
| 1×32×64, N=3      | 25.01   | 5.46      | **4.58×** | 5865         |

Post-chunking (Phase 5b — arbitrary depth via cb chunking at 30
ops/cb default):

| Shape (BxSxD,N)   | cold_ms | warmed_ms | speedup | tok/s (warmed) | cbs/step |
|-------------------|---------|-----------|---------|----------------|----------|
| 1×8×16, N=6       | 26.23   | 5.86      | **4.47×** | 1364         | 3 |
| 1×32×64, N=6      | 39.17   | 17.42     | **2.25×** | 1837         | 3 |
| 1×32×64, N=12     | 83.90   | 28.63     | **2.93×** | 1118         | 5 |

Per-layer cost is consistent: ~0.98 ms/layer warmed at small shape,
~2.40-2.90 ms/layer at medium shape (modulo the small-N rounding
overhead). N=12 vs N=6 at the same shape scales nearly linearly in
both modes, confirming chunking has minimal per-cb overhead beyond
the necessary commit + barrier. The combined
(ResidentWeights + single-cb + decorator) lift is greater than
single-cb alone because the multi-layer case amplifies the
per-step weight-upload overhead: 9 weight tensors × 3 layers = 27
host→device transfers per cold step.

Architectural validation: **a 4-layer transformer step (~48 user
ops) commits exactly 1 command buffer per decode step** when run
through the full encode-session stack. Structural correctness
tests in ``tests/unit/test_apple_gpu_multi_layer_transformer.py``
pin (a) one-cb-per-step invariant, (b) `ResidentWeights` handle
stability across multiple decode steps, (c) steady-state
determinism (same X → same output).

## Test architecture

The drift gate for stages 2-4 is the **`submission_count_probe`** —
a new C ABI symbol exposing `MTLCommandQueue` total commits over its
lifetime. A test that opens a session, encodes N ops, commits once,
and reads (`final_count - initial_count == 1`) is the structural
proof the chain stays on one cb.

## Non-goals

* **Cross-device pipelines** — staying on one Apple GPU; multi-device
  is its own line item.
* **Dynamic graphs / control flow in the encoded chain** — out of
  scope. `with batched_session()` accepts only straight-line code;
  conditional / loop control stays on the CPU side.
* **Speculative decoding / multi-token in one cb** — interesting
  follow-up after stage 4 lands.

## Files

* **Stage 1 (this PR)**
  * `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`
    — adds `tessera_apple_gpu_layer_norm_dev_f32_enc`
  * `python/tessera/apple_gpu_batched.py` — new module
  * `tests/unit/test_apple_gpu_single_command_buffer.py` — new
  * `docs/audit/backend/apple/APPLE_AUDIT.md` — this doc
