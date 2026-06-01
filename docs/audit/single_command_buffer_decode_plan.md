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
compares a Llama-style 8-op attention block on N command buffers
versus 1. Measured speedups on M-series hardware (30 reps,
median latency):

| Shape (BxSxD) | per-op (8 cb) | one cb | speedup |
|---------------|---------------|--------|---------|
| 1x8x16        | 2.49 ms       | 0.76 ms | **3.29×** |
| 1x32x64       | 4.82 ms       | 2.41 ms | **2.00×** |
| 1x64x128      | 8.63 ms       | 4.53 ms | **1.90×** |

Pattern: smaller shapes benefit more from single-cb because the
per-op command-buffer overhead dominates the actual GPU compute.
At 1x8x16 the 7 saved cb commits are worth ~1.7 ms; the actual
attention math takes ~0.7 ms either way. As shapes grow, compute
dominates and the per-cb overhead becomes a smaller fraction —
but the 2× win at 1x64x128 is still very significant for the
typical decode-step latency budget.

Above 64×128 we'd expect the speedup to keep tapering toward 1.0×
as compute fully dominates; this is the expected behavior and not
a regression of the architecture.

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
  * `docs/audit/single_command_buffer_decode_plan.md` — this doc
