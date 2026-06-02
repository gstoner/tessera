---
status: Tutorial
classification: Tutorial
last_updated: 2026-05-10
---

# Tessera Programming Guide
## Chapter 7: Autodiff

Autodiff is a **shipped** Tessera surface. The tape-based reverse-mode
implementation lives at `tessera.autodiff.tape()` / `tessera.autodiff.reverse()`,
with category-based VJP/JVP coverage across **188 VJPs and 140 JVPs** in the
standalone primitive registry. The transforms compose through Graph IR
(Phase F4), insert effect-aware adjoint collectives (Phase F5), and support
forward-mode (`jvp`/`jacfwd`), reverse-mode (`vjp`/`jacrev`/`grad`),
activation checkpointing (`rematerialize`), and JAX-style higher-order
derivatives.

This chapter is the user-facing tutorial; for the authoritative
implementation contract see `docs/spec/AUTODIFF_SPEC.md`, and for per-primitive
contract status see `docs/audit/coverage/COVERAGE_AUDIT.md`.

## 7.1 Tape-Based Reverse Mode

The canonical pattern: open a tape, run the forward, call `tape.backward(loss)`.

```python
import tessera as ts

with ts.autodiff.tape() as t:
    y = model(x)
    loss = ((y - target) ** 2).mean()
    t.backward(loss)

# .grad is now populated on every Parameter touched by the forward pass.
for p in model.parameters():
    p._data._data -= 0.01 * p.grad.numpy()
    p.zero_grad()
```

While a tape is active, every `tessera.ops.<name>` call with a registered
VJP is intercepted, the forward is computed, and a tape entry is appended.
`tape.backward(scalar)` walks the tape in reverse and propagates gradients.
Modules' `Parameter.grad` fields accumulate (PyTorch convention);
`Module.zero_grad()` resets them.

## 7.2 The `@tessera.autodiff.reverse` Decorator

```python
@ts.autodiff.reverse
def loss_fn(model, x, target):
    return ((model(x) - target) ** 2).mean()

loss, param_grads = loss_fn(model, x, target)
# param_grads: dict[str, np.ndarray] keyed by named_parameters()
```

`reverse(fn)` opens a tape, calls `fn`, runs the backward pass, and returns
`(loss, {name: grad})`. Modules referenced inside `fn` have their `.grad`
populated as a side effect.

## 7.3 Forward-Mode (JVP) and JAX-Style Transforms

Forward-mode is shipped alongside reverse-mode:

```python
# Forward-mode primal + tangent
primal, tangent = ts.autodiff.jvp(fn)(x, v)

# JAX-style transform helpers
grad_fn   = ts.autodiff.grad(fn)         # reverse-mode gradient
jacrev    = ts.autodiff.jacrev(fn)       # full Jacobian via reverse mode
jacfwd    = ts.autodiff.jacfwd(fn)       # full Jacobian via forward mode
hvp       = ts.autodiff.hvp(fn)          # Hessian-vector product
vmap      = ts.autodiff.vmap(fn)         # batched-axis transform
```

These compose: `grad(vmap(f))`, `vmap(grad(f))`, `grad(scan(f))`,
`remat(scan(f))`, and `shard_map(grad(f))` all behave consistently with the
JAX semantics they borrow vocabulary from. See `docs/spec/AUTODIFF_SPEC.md`
for the full transform composition matrix.

## 7.4 Custom VJP/JVP Rules

Use `custom_rule(op_name)` (or the lower-level `register_vjp` /
`register_jvp`) to register hand-written gradient rules:

```python
@ts.autodiff.custom_rule("flash_attn")
def _vjp_flash_attn(dout, Q, K, V, **kwargs):
    dQ, dK, dV = ...                           # hand-written backward
    return (dQ, dK, dV)
```

`custom_rule` registers (or overrides) the VJP for a named op. The Python
op wrapper is automatically tape-wired. Use this to opt op families into
autodiff incrementally — for example, the reasoning-model attention family
(`flash_attn`, `lightning_attention`, `deepseek_sparse_attention`,
`kimi_delta_attention`, etc.) all ship with hand-written `custom_rule`s.

## 7.5 Activation Checkpointing

`tessera.autodiff.rematerialize` (alias `checkpoint`) drops intermediate
activations from the tape and recomputes them on the backward pass:

```python
@ts.autodiff.rematerialize
def expensive_block(x, w):
    return ts.ops.gelu(ts.ops.gemm(x, w))

with ts.autodiff.tape() as t:
    y = expensive_block(x, w)
    loss = (y ** 2).mean()
    t.backward(loss)
```

The forward path runs normally; the backward path re-runs the wrapped
function under a nested tape, trading compute for memory. See
`tessera.autodiff.rematerialize` in the package docs.

## 7.6 Mixed Precision

`tessera.autodiff.autocast` and `tessera.autodiff.GradScaler` ship for
mixed-precision training:

```python
scaler = ts.autodiff.GradScaler(init_scale=2**16)

with ts.autodiff.autocast("bf16"):
    with ts.autodiff.tape() as t:
        y = model(x)
        loss = ((y - target) ** 2).mean()
        scaled = scaler.scale(loss)
        t.backward(scaled)

scaler.step(optimizer_step_fn)        # unscales + updates loss scale
```

`autocast` rewrites supported ops to the chosen dtype while keeping
reductions in fp32; `GradScaler` adapts the loss scale automatically.

## 7.7 Effects and Adjoint Collectives

Tessera tracks per-op effects, and the adjoint collective insertion pass
(Phase F5) uses them to maintain correctness under distribution:

| Effect | Sample ops | Adjoint behavior |
|--------|------------|------------------|
| `pure` | `gemm`, `softmax`, `gelu`, `relu` | Standard VJP; recompute-safe |
| `random` | `dropout`, `rng_*` samplers | Deterministic seed handling via `RNGKey.fold_in`; no gradient through the sample |
| `state` | `kv_cache_append`, `memory_write`, `selective_ssm` | State-effect boundary; gradient stops at the write |
| `collective` | `psum`, `all_reduce`, `moe_dispatch` | Adjoint is the dual collective (`psum^T = broadcast_to_axis`, etc.); inserted automatically |

Under `shard_map`, reverse-mode automatically inserts `reduce_scatter` /
`all_gather` at the right places so distributed gradients stay correct.
See `tessera.distributed.DDP` and `tessera.distributed.FSDP` for the
batteries-included wrappers.

## 7.8 Region Privileges and Gradient Lowering

Region annotations are preserved through gradient lowering:

```python
@tessera.jit
def accumulate_grad(
    X: tessera.Region["read"],
    W: tessera.Region["read"],
    G: tessera.Region["reduce_sum"],
):
    G[:] += tessera.ops.gemm(X, W)
```

The Graph IR adjoint pass respects `Region["reduce_sum"]` and emits the
matching reduction in the backward.

## 7.9 Coverage and Audit

Per-primitive VJP/JVP status is tracked in
`python/tessera/compiler/primitive_coverage.py` and rendered to
`docs/audit/standalone_primitive_coverage.md`. As of 2026-05-10:

- **188 VJPs registered**, 184 entries at `vjp = complete`.
- **140 JVPs registered**, 140 entries at `jvp = complete`.
- 137 entries marked `vjp = not_applicable` (RNG samplers / transforms /
  schedules / boolean-output / state-effect ops — non-differentiable by
  design).
- 138 entries marked `jvp = not_applicable` for the same reasons.

The remaining gaps (`vjp = planned` 53; `jvp = planned` 96) are niche
primitives in the spectral / sparse / linalg / quant-variant tail; see
`docs/audit/coverage/COVERAGE_AUDIT.md` for the per-category breakdown.

## 7.10 Common Patterns

**Train one step of a small MLP:**

```python
with ts.autodiff.tape() as t:
    logits = model(batch.x)
    loss = ts.losses.cross_entropy_loss(logits, batch.y)
    t.backward(loss)

new_params, new_state = ts.optim.adamw(
    params={name: p for name, p in model.named_parameters()},
    grads={name: p.grad for name, p in model.named_parameters()},
    state=opt_state,
    lr=1e-3,
)
```

**Reverse-mode through a `scan`:**

```python
@ts.autodiff.reverse
def total_loss(params, xs):
    def step(carry, x):
        return ts.nn.gru_cell(x, carry, params["W_ih"], params["W_hh"]), carry
    _, ys = ts.control.scan(step, init_carry, xs)
    return (ys ** 2).mean()
```

**JAX-style `grad`:**

```python
loss_fn = lambda params, x, y: ts.losses.mse_loss(model_apply(params, x), y)
grad_fn = ts.autodiff.grad(loss_fn)
g = grad_fn(params, x, y)
```

## 7.11 Summary

- Autodiff is **shipped**, not planned. `tessera.autodiff.tape()` and
  `tessera.autodiff.reverse(fn)` are the canonical entry points.
- 188 VJPs + 140 JVPs registered across the standalone primitive surface.
- Forward-mode (`jvp`/`jacfwd`), reverse-mode (`vjp`/`jacrev`/`grad`),
  activation checkpointing (`rematerialize`), and JAX-style transforms
  (`vmap`/`pmap`) all compose.
- Effect-aware adjoint collective insertion handles distributed gradients
  automatically.
- Custom VJP/JVP rules registered via `custom_rule(op_name)`; the
  reasoning-model attention family uses this pattern.
- Per-primitive coverage tracked in
  `docs/audit/standalone_primitive_coverage.md`; spec at
  `docs/spec/AUTODIFF_SPEC.md`.
