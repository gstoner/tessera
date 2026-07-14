---
status: Normative (Tier 2 first-slice)
classification: Spec
authority: Programming Guide Ch.7 supersedes; this doc specifies the v1 implementation
last_updated: 2026-07-14
---

# Tessera Autodiff — v1 Spec

> **Scope:** This document specifies the **first slice** of Tessera autodiff —
> tape-based reverse-mode at the numpy-reference op layer, integrated with the
> Tier 1 stateful `Module`/`Parameter` surface. The full Phase 5 epic (Graph/Tile
> IR adjoints, effect-aware adjoint collectives, rematerialization, JAX-style
> transforms) is *not* in scope here; cross-references appear inline.

## Goals

1. **Train a small model end-to-end on CPU.** Forward through `tessera.nn.*`,
   compute a scalar loss, backprop, populate `Parameter.grad`, manual SGD update.
2. **Be honest about what's missing.** The first slice is numpy-reference,
   not Graph/Tile IR. Distributed grad collectives are deferred.
3. **Surface that survives the upgrade.** When the Graph/Tile IR adjoint
   work lands, `tessera.autodiff.tape()` and `tessera.autodiff.reverse(fn)`
   keep the same shape; only the implementation underneath swaps.

## Non-goals (deferred follow-ups, refreshed 2026-05-10)

| Item | Status |
|------|--------|
| Graph/Tile IR adjoint ops | **✅ Phase F4 landed at the IR level** — `AdjointInterface` ODS + `AutodiffPass.cpp` produce valid backward IR, lit-verified on MLIR 22. *Not* an execution claim: no target executes backward natively yet — see [`autodiff_connection_ledger.md`](../audit/generated/autodiff_connection_ledger.md) + [`AUTODIFF_UNIFICATION_PLAN.md`](../audit/compiler/AUTODIFF_UNIFICATION_PLAN.md). |
| Effect-aware adjoint collective insertion | **✅ Phase F5 landed** — Python: `tessera.distributed.DDP` / `FSDP` validate against `mock_collective`. IR: `AdjointCollectiveInsertionPass` (`--tessera-adjoint-collective-insertion`) gates on per-arg `tessera.effect` (memory-class) and inserts `reduce_scatter`/`all_gather`/`all_reduce` by sharding kind; lit `tests/tessera-ir/phase_f5/`. |
| Activation checkpointing / rematerialization | **✅ Phase F2 landed** — Python: `tessera.autodiff.rematerialize` (alias `checkpoint`). IR: `ActivationRematerializationPass` (`--tessera-activation-rematerialization`) clones `tessera.recompute`-tagged pure ops to their backward consumers; lit `tests/tessera-ir/phase_f2/`. |
| Mixed-precision autocast + loss scaling | **✅ Phase F1 landed** — `tessera.autodiff.autocast(dtype)` + `GradScaler`. fp8 backend lowering still pending Phase G. |
| Higher-order derivatives (HVP, jacrev, jacfwd) | **✅ Phase F7 landed** — `tessera.autodiff.{grad, hvp, jacrev, jacfwd, elementwise_grad}`. |
| `jax.vmap`-style batched transforms | **✅ Phase F6 landed** — `tessera.autodiff.vmap` + `tessera.control.{vmap, pmap}`. |
| Backward through `flash_attn`, attention family | **✅ Phase F3 landed** — `custom_rule`-registered VJP+JVP; reasoning-model variants (`deepseek_sparse_attention`, `lightning_attention`, `kimi_delta_attention`, etc.) also shipped. |
| Backward through spectral, MoE, selective_ssm, sparse, linalg | **Mixed** — `fft`/`ifft`/`rfft`/`irfft`/`moe`/`selective_ssm` shipped (Phase F3); long-tail (`stft`/`istft`/`dct`/`spectral_*`, `spmm_*`/`sddmm`/`bsmm`, `cholesky`/`qr`/`svd`/`tri_solve`) tracked at `docs/audit/coverage/COVERAGE_AUDIT.md`. |
| Custom CUDA / Metal adjoint kernels | **Pending Phase G/H** — autodiff contract is numpy-reference complete; hardware kernels arrive with each backend. |
| Custom-primitive registration API | **✅ Sprint S13 landed** — `tessera.custom.custom_vjp`/`custom_jvp`/`custom_batching`/`custom_primitive`. |

## Surface

### `tessera.autodiff.tape()` (context manager)

```python
import tessera as ts

with ts.autodiff.tape() as t:
    y = model(x)
    loss = ((y - target) ** 2).mean()
    t.backward(loss)

# .grad now populated on every parameter touched by the forward pass
for p in model.parameters():
    p._data._data -= 0.01 * p.grad.numpy()
    p.zero_grad()
```

While a tape is active, every `tessera.ops.<name>` call that has a registered
VJP is intercepted: forward is computed normally, and a tape entry is appended
recording the op, inputs (with `Parameter` provenance), kwargs, output, and
a VJP function pointer. Calling `t.backward(scalar)` walks the tape in reverse,
seeding the cotangent at the scalar with `1.0` and propagating gradients
through each VJP. When a tape entry's input came from a `Parameter`, the
gradient is accumulated into that `Parameter.grad`.

### `tessera.autodiff.reverse(fn)` (function transform)

```python
@ts.autodiff.reverse
def loss_fn(model, x, target):
    return ((model(x) - target) ** 2).mean()

loss, param_grads = loss_fn(model, x, target)
# param_grads: dict[str, np.ndarray] keyed by named_parameters
```

`reverse(fn)` is a thin convenience wrapper that opens a tape, calls `fn`,
runs `tape.backward(loss)`, and returns `(loss, {name: grad})`. The Module(s)
referenced inside `fn` have their `.grad` populated as a side effect (same as
`with tape()`).

### `tessera.autodiff.custom_rule(op_name)` (decorator)

```python
@ts.autodiff.custom_rule("flash_attn")
def _vjp_flash_attn(dout, Q, K, V, **kwargs):
    # Hand-written VJP. Returns a tuple of cotangents matching the input order.
    dQ, dK, dV = ...
    return (dQ, dK, dV)
```

`custom_rule` registers (or overrides) the VJP for a registered op name. The
existing `tessera.ops.<name>` callable is automatically tape-wrapped if it
wasn't already. Use this to opt op families *into* autodiff incrementally.

### `Parameter.grad` semantics

- `.grad` is `None` until autodiff populates it.
- Multiple backward passes accumulate into `.grad` (matches PyTorch).
- `Module.zero_grad()` (already shipped Tier 1) resets every parameter's `.grad` to `None`.

## VJP / JVP coverage

The spec ships a registry-driven coverage model rather than a static table.
Per-primitive contract status is the single source of truth at
`python/tessera/compiler/primitive_coverage.py`, rendered as a dashboard
at `docs/audit/standalone_primitive_coverage.md`, and audited at
`docs/audit/coverage/COVERAGE_AUDIT.md`.

**Live counts are dashboard-owned, not copied here.** The number of
registered VJPs/JVPs and the per-axis `complete`/`not_applicable`/`planned`
split move every sprint as primitives land; the **count authority** is the
generated S-series status dashboard
([`docs/audit/generated/s_series_status.md`](../audit/generated/s_series_status.md))
and the registry itself (`tessera.autodiff.vjp._VJPS` /
`tessera.autodiff.jvp._JVPS`, surfaced through
`python/tessera/compiler/primitive_coverage.py`). Per Decision #26, prose
that copies these numbers silently goes stale — read the dashboard for the
current figures. As of this writing the long-tail `vjp`/`jvp` `planned`
buckets are closed project-wide (the open contract axis is `backend_kernel`,
which is hardware-gated). The qualitative family coverage below is stable
regardless of the exact counts.

**Coverage by family:**

| Family | VJP+JVP coverage |
|---|---|
| Elementwise pointwise (`add`/`mul`/`exp`/`log`/`sqrt`/`pow`/`cos`/`tan`/`sinh`/`cosh`/`asin`/`acos`/`atan`/`atan2`/`erf`/`erfc`/`log1p`/`expm1`/`softplus`/`sigmoid_safe`/`reciprocal`/`absolute`/`sign`/...) | ✅ complete |
| Activations (`gelu`/`silu`/`relu`/`sigmoid`/`tanh`/`softmax`/`log_softmax`) | ✅ complete |
| Reductions (`sum`/`mean`/`prod`/`amax`/`amin`/`var`/`std`/`cumsum`/`logsumexp`) | ✅ complete |
| Normalization (`layer_norm`/`rmsnorm`/`group_norm`/`instance_norm`) | ✅ complete |
| Linear (`matmul`/`gemm`/`linear_general`/`einsum`/`conv1d`) | ✅ complete |
| Attention family — `flash_attn`/`multi_head_attention`/`gqa_attention`/`mqa_attention`/`mla_decode`/`linear_attn`/`linear_attn_state`/`power_attn`/`retention`/`attn_sliding_window`/`attn_top_k_blocks`/`attn_compressed_blocks` | ✅ complete |
| Reasoning-model attention — `deepseek_sparse_attention`/`lightning_attention`/`gated_attention`/`hybrid_attention`/`gated_deltanet`/`kimi_delta_attention`/`modified_delta_attention` | ✅ complete |
| Position encodings (`rope`/`rope_split`/`rope_merge`/`alibi`/`ntk_rope`) | ✅ complete |
| Losses (`mse_loss`/`mae_loss`/`huber_loss`/`smooth_l1_loss`/`log_cosh_loss`/`cross_entropy_loss`/`binary_cross_entropy_loss`/`focal_loss`/`label_smoothed_cross_entropy`/`kl_divergence`/`js_divergence`/`wasserstein_distance`/`triplet_loss`/`contrastive_loss`/`cosine_embedding_loss`/`info_nce_loss`/`nt_xent_loss`/`ddpm_noise_pred_loss`/`score_matching_loss`/`vlb_loss`/`ctc_loss`/`seq2seq_loss`) | ✅ complete |
| RL policy losses (`ppo_policy_loss`/`grpo_policy_loss`/`cispo_policy_loss`) | ✅ complete |
| MoE routing (`moe`/`moe_dispatch`/`moe_combine`/`mor_router`/`mor_partition`/`mor_scatter`) | ✅ complete |
| MLA family (`latent_kv_compress`/`latent_kv_expand_k`/`latent_kv_expand_v`/`mla_decode`/`mla_decode_fused`) | ✅ complete |
| Collectives — `psum`/`pmean`/`pmax`/`pmin`/`collective_permute`/`broadcast_to_axis` | ✅ complete |
| Optimizers — `sgd`/`momentum`/`nesterov`/`adamw` (state-aware single-step) | ✅ complete |
| Spectral — `fft`/`ifft`/`rfft`/`irfft` | ✅ complete |
| Memory (`memory_read` differentiable; `memory_write`/`memory_evict` are state-effect) | ✅ complete |
| Cumulative extrema (`cummax`/`cummin`) | ✅ complete |
| Pooling (`max_pool`/`avg_pool`) | ✅ complete |

The VJPs live in `python/tessera/autodiff/vjp.py`, JVPs in
`python/tessera/autodiff/jvp.py`. Adding a new op is one rule function +
one decorator (`@_vjp("name")` / `@_jvp("name")`). The primitive coverage
registry consults both dicts automatically — registering a (V/J)VP for a
catalog op auto-flips its dashboard status from `planned` to `complete`
without manual intervention.

**Categories that are intentionally non-differentiable** (covered by the
`_NONDIFFERENTIABLE_CATEGORIES` set in `primitive_coverage.py`): RNG
samplers, control-flow transforms (`scan`/`cond`/`while_loop`), LR
schedules, comparisons (`eq`/`ne`/`lt`/`...`), logical ops, sharding
primitives, gradient transforms, sort/argsort/top_k, state trees, data
pipelines, AOT/serialization, conformance suites, and the
`extension` (custom-primitive) declarations themselves. These resolve to
`vjp = not_applicable` / `jvp = not_applicable` in the registry.

Calling an unregistered op inside an active tape raises
`TesseraAutodiffError` with a clear message naming the op and pointing at
`custom_rule` for registration.

## Mechanism

### Tape data structure

```python
@dataclass(frozen=True)
class TapeEntry:
    op: str
    inputs: tuple[InputDesc, ...]    # one per forward-arg
    kwargs: dict
    output_id: int                   # id() of the output numpy array
    output: np.ndarray               # held to keep it alive
    vjp: Callable                    # (dout, *forward_inputs, **kwargs) -> tuple[dinput...]


@dataclass(frozen=True)
class InputDesc:
    param: Parameter | None          # source Parameter, if any
    array_id: int                    # id() of the underlying numpy buffer
    array: np.ndarray                # held to keep it alive (and used for VJP forward inputs)
```

The tape is a flat list — no graph, no topological sort. Reverse iteration
suffices because forward order is already a valid reverse-topological order
of the computation DAG.

### Tape-active-state

A `contextvars.ContextVar` named `_ACTIVE_TAPE` holds the current tape (or
`None`). Op wrappers read this on every call. Async-safe and thread-safe.

### Op interception

On import of `tessera.autodiff`, every op in `_VJPS` is wrapped:

```python
def _make_wrapper(name, original, vjp_fn):
    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)        # always compute forward
        tape = _ACTIVE_TAPE.get()
        if tape is not None:
            inputs = tuple(_describe(a) for a in args)
            tape.record(name, inputs, kwargs, out, vjp_fn)
        return out
    wrapped.__wrapped__ = original
    return wrapped
```

The wrapper is installed on the `tessera.ops` namespace via `setattr`. Calls
not made through `tessera.ops.<name>` (e.g., raw `np.matmul`) are *not*
intercepted — this is intentional: we only differentiate Tessera ops.

### Backward pass

```python
def backward(self, scalar):
    arr = np.asarray(scalar, dtype=np.float64)
    if arr.size != 1:
        raise TesseraAutodiffError("backward expects a scalar loss")
    cotan = {id(scalar): np.array(1.0, dtype=arr.dtype)}
    for entry in reversed(self.entries):
        dout = cotan.get(entry.output_id)
        if dout is None:
            continue
        forward_args = tuple(d.array for d in entry.inputs)
        d_in = entry.vjp(dout, *forward_args, **entry.kwargs)
        for desc, g in zip(entry.inputs, d_in):
            if g is None:
                continue
            cotan[desc.array_id] = cotan.get(desc.array_id, 0.0) + g
            if desc.param is not None:
                _accumulate_param_grad(desc.param, g)
```

`_accumulate_param_grad` writes into `param.grad` (creating it if `None`,
adding into the existing buffer otherwise). Stays in numpy land.

## Errors

| Class | Raised when |
|-------|-------------|
| `TesseraAutodiffError` | scalar `backward()` argument not 0-d; op without VJP encountered inside a tape; multiple `backward()` calls on a stale tape; ... |

## Testing strategy

1. **Per-op numerical-Jacobian checks.** For each VJP, construct a small input,
   compute the analytical Jacobian via the VJP, compare to a finite-difference
   numerical Jacobian. Tolerance: `rtol=1e-4, atol=1e-5` (fp32-friendly).
2. **End-to-end MLP train step.** Build `ts.nn.Sequential(Linear, RMSNorm, MLP, Linear)`,
   forward, MSE loss, backward, manual SGD step, assert loss decreased after
   one step on a fixed seed.
3. **`custom_rule` regression test.** Register a custom VJP for an op, verify
   it overrides the built-in.
4. **Phantom-VJP test.** Calling an unsupported op (e.g., `flash_attn`) inside
   a tape raises `TesseraAutodiffError` with a clear message.
5. **Multiple backward accumulation.** Two `backward()` calls on the same
   parameters double the grad; matches PyTorch's accumulation default.

## File layout

```
python/tessera/autodiff/
├── __init__.py    — public surface (tape, reverse, custom_rule, errors)
├── tape.py        — Tape, TapeEntry, InputDesc, _ACTIVE_TAPE, op wrapping
└── vjp.py         — built-in VJPs + register_vjp helper
```

Estimated v1 LOC: ~600 (code) + ~350 (tests) = **~950 LOC**. Well under the
audit's 1,200–1,800 estimate because we explicitly defer Graph/Tile IR work.

## Phase F4 — Graph IR adjoint via `AdjointInterface` (landed at the IR level)

The numpy-tape implementation in this spec is the v1 surface that user code
binds to. The Phase F4 follow-up replaces the *internals* of that surface
with Graph-IR-level adjoints, while keeping `tape() / reverse(fn) /
custom_rule(name)` unchanged at the Python boundary.

> **Scope of "landed" (reconciled 2026-07-11).** F4 is landed at the **IR
> level**: the tablegen is wired, `AutodiffPass` runs, and the lit smoke test
> passes on MLIR 22 — the pass produces valid *backward IR*. It is **not** an
> end-to-end *execution* claim: no op family binds a compiled backward entry
> point or is oracle-proven for gradients on any target yet. The per-op-family ×
> per-target truth (which ops have a native vs. placeholder IR adjoint, and
> which — none, today — execute backward natively) lives in the generated
> [`autodiff_connection_ledger.md`](../audit/generated/autodiff_connection_ledger.md);
> the promotion path from "IR adjoint" to "native backward execution" is
> [`AUTODIFF_UNIFICATION_PLAN.md`](../audit/compiler/AUTODIFF_UNIFICATION_PLAN.md).

Key pieces (landed 2026-05-09):

* **ODS interface** — `src/compiler/ir/include/Tessera/AdjointInterface.td`
  defines `Tessera_AdjointInterface` (`cppNamespace = ::tessera`) with three
  methods: `buildAdjoint(builder, outputCotangents) -> SmallVector<Value>`,
  `isDifferentiable() -> bool`, and `customAdjointName() -> StringRef`.

* **Pass body** — `src/transforms/lib/AutodiffPass.cpp` documents and codes
  the full four-step reverse walk (collect forward ops, identify scalar
  return as the loss seed, build cotangent map keyed by forward `Value`,
  reverse-walk dispatching `buildAdjoint` per op, accumulate via
  `arith.addf`). The tablegen for `AdjointInterface` is wired
  ([`src/compiler/ir/CMakeLists.txt`](../../src/compiler/ir/CMakeLists.txt)
  emits `AdjointInterface.{h,cpp}.inc`), so the interface dispatch is live —
  the pass is no longer a no-op.

* **Build wiring** — `AutodiffPass.cpp` is in
  `src/transforms/lib/CMakeLists.txt`'s `TesseraPasses` library, registered
  via `createAutodiffPass()` declared in
  `src/transforms/include/Tessera/Transforms/Passes.h` and exposed to
  `tessera-opt` via `Passes.cpp`'s `registerTesseraPasses()`.

* **Lit smoke test** — `tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir`
  **passes** (no longer `XFAIL`): it runs `tessera-opt --tessera-autodiff` and
  FileCheck-verifies the reverse walk rewrites `@train_step` to expose argument
  cotangents and emits the transposed `tessera.matmul` adjoints.

* **Custom-rule bridge** — `tessera.autodiff.custom_rule(name)` continues to
  register Python VJPs at runtime. Ops that opt into a custom rule report the
  registry key via `customAdjointName()`; the AutodiffPass consults the
  bridge during its reverse walk via the `tessera.custom_adjoint_call`
  placeholder op (now present).

* **What has an IR adjoint today** — `matmul`, `tanh`, and `sigmoid` have
  **native** `buildAdjoint` bodies (real backward Graph IR; the pointwise W5 ops
  fall back to the placeholder only for dynamic shapes). `layernorm`, `softmax`,
  `gelu`, `relu`, `sin`, `silu`, `softplus`, `rmsnorm`, `log_softmax` carry a
  **placeholder** `buildAdjoint` that emits `tessera.custom_adjoint_call` and
  round-trips to the Python VJP — i.e. an IR adjoint that is *not* native
  (`layernorm`/`softmax` have hand-written bodies but they still construct the
  placeholder, so they are round-trips, not native — the ledger classifies by
  what the body emits). The live native-vs-placeholder split and the
  `bwd_cpu_ir_oracle` column (backward IR interpreted on CPU and oracle-matched)
  are the generated
  [`autodiff_connection_ledger.md`](../audit/generated/autodiff_connection_ledger.md)
  (the count authority — don't trust this enumeration if it drifts). Native
  *backward execution* is no longer empty: Phase 4 (A2) sources the backward
  rungs from the runtime execution matrix, so `flash_attn` (ROCm) and
  `selective_ssm` (ROCm + x86) now register `bwd_hardware_proven` — read the
  ledger for the live set.

* **Paired forward/backward ABI (Phase 2, landed first cut)** — the in-place
  return expansion above is the bootstrap; `--tessera-autodiff-paired`
  ([`AutodiffPairedPass.cpp`](../../src/transforms/lib/AutodiffPairedPass.cpp))
  emits a **separate** backward function
  `@f__bwd(inputs, out_cotangents) -> input_cotangents` (recompute-all residual
  policy), the deterministic ABI runtime binding + hand-emitted backward kernels
  (e.g. ROCm WMMA) both target. See
  [`AUTODIFF_UNIFICATION_PLAN.md`](../audit/compiler/AUTODIFF_UNIFICATION_PLAN.md).

* **Remaining work** — promote IR adjoints to native backward *execution*
  (runtime binding, per-target oracle proof); broaden native `buildAdjoint`
  coverage beyond `matmul`/`tanh`/`sigmoid` so fewer ops depend on the
  `custom_adjoint_call` Python round-trip; add the SAVE residual policy.

Landed alongside F4 (previously out of scope for the F4 first cut):

* **Effect-aware adjoint collective insertion (Phase F5).**
  `AdjointCollectiveInsertionPass` (`--tessera-adjoint-collective-insertion`)
  runs after AutodiffPass and, when the function is effect-annotated
  (per-arg `tessera.effect` from EffectAnnotationPass), synchronises only the
  cotangents whose argument carries a **memory-class** effect
  (`write` / `reduce_*` / `memory`) — a pure read-only input never gets a
  gradient collective. The sharding *kind* (`tessera.weight_sharding`) then
  selects the op: `dp` → `reduce_scatter`, `tp` → `all_gather`, `replicated`
  → `all_reduce`. When no effect annotation is present the pass falls back to a
  weight_sharding-only plan (recorded distinctly as `[sharding-only]` vs
  `[effect-gated]` in `tessera.adjoint_collective_plan`). Composed pipeline:
  `--tessera-autodiff-pipeline` (F4 → F2 → F5). Lit: `tests/tessera-ir/phase_f5/`.
* **Activation rematerialization — IR-pass form (Phase F2).**
  `ActivationRematerializationPass` (`--tessera-activation-rematerialization`)
  is the Graph-IR counterpart of the `tessera.autodiff.rematerialize` /
  `checkpoint` Python surface: it clones each `tessera.recompute`-tagged pure
  op to its backward consumers, shrinking the forward activation's live range at
  the cost of recompute (Decision #10 — budget-guided, pure region-free ops
  only; a tagged op with regions is a hard `[REMAT_NON_CLONABLE]` error). Lit:
  `tests/tessera-ir/phase_f2/`.

Still out of scope for the F4 first cut:

* Higher-order derivatives in Graph IR (run AutodiffPass twice +
  canonicalize). The Python reference F7 surface is shipped; this bullet only
  tracks lower-level IR support.
* Budget-guided *automatic* remat selection (greedy live-set scan). The IR pass
  honours explicit `tessera.recompute` markers today and records
  `--memory-budget-mb` as advisory; auto-selection under a budget is future.

## Cross-references

- Programming Guide Ch.7 (Autodiff) — describes the *full* Phase 5 epic.
  This spec implements the v1 first slice + F4 ODS scaffolding.
- `docs/audit/coverage/COVERAGE_AUDIT.md` — Theme 2.
- `python/tessera/nn/module.py` — `Module.zero_grad()` and `Parameter.grad`
  are in place from Tier 1.
- `python/tessera/autodiff/mixed_precision.py` — Phase F1 autocast + GradScaler.
- `python/tessera/autodiff/rematerialize.py` — Phase F2 activation checkpointing.
- `src/compiler/ir/include/Tessera/AdjointInterface.td` — Phase F4 ODS.
- `src/transforms/lib/AutodiffPass.cpp` — Phase F4 pass scaffold.
- `src/transforms/lib/AdjointCollectiveInsertionPass.cpp` — Phase F5 effect-aware
  adjoint collective insertion (`--tessera-adjoint-collective-insertion`).
- `src/transforms/lib/ActivationRematerializationPass.cpp` — Phase F2 IR-form
  activation rematerialization (`--tessera-activation-rematerialization`).
