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

**Phase 2 (open)** — true `@jit(target="apple_gpu")` auto-detection.
The compiler/runtime would walk the function body, identify maximal
straight-line subgraphs of encode-session-aware ops, and rewrite them
to use `batched_session()` transparently. Implementation surface:

* `compiler/jit.py::_execute_apple_gpu_*` paths need to track which
  ops are encode-session-compatible (driven by a registry —
  `_APPLE_GPU_ENCODE_SESSION_OPS`).
* The metadata layer (`runtime.py::_execute_apple_gpu_metadata`)
  needs to materialize device-resident tensors between ops instead of
  download/upload roundtrips.
* A new `BatchedExecutionPlan` IR node represents "encode N ops into
  one session" — the lowering pass identifies adjacent encode-
  compatible ops and groups them.

Estimated work: ~4-8 hours of careful changes spanning jit.py +
runtime.py + the apple_gpu dispatch wiring. Stage-2 ops are now in
place so the encode-session ABI surface that the JIT would route
through is complete.

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
