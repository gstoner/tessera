---
status: Normative (Tier 2 first-slice)
classification: Spec
authority: Programming Guide Ch.7 supersedes; this doc specifies the v1 implementation
last_updated: 2026-05-09
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

## Non-goals (deferred follow-ups)

| Item | Tracking |
|------|----------|
| Graph/Tile IR adjoint ops | Phase 5 follow-up — adds adjoint Graph IR ops + lowering |
| Effect-aware adjoint collective insertion (`reduce_scatter`/`all_gather` for distributed gradients) | Extends `GPUCollectiveInsertionPass` |
| Activation checkpointing / rematerialization pass | Phase 5 follow-up |
| Mixed-precision gradient master copy + loss scaling (fp32 master, fp8 forward) | Phase 5; depends on autocast (Theme 10) |
| Higher-order derivatives (HVP, jacrev, jacfwd) | Out of scope for v1 |
| `jax.vmap`-style batched transforms | Out of scope for v1 |
| Custom CUDA/Metal adjoint kernels | Each backend's later phase |
| Backward through `flash_attn` (fused) | Wired as a `custom_rule` once a kernel exists; v1 errors |
| Backward through `ops.moe`, `ops.fft/ifft`, `ops.spmm_*`, `ops.cholesky` | Same — error in v1 unless `custom_rule` registered |

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

## VJP coverage in v1

The first slice ships VJPs for these ops. Calling any other op inside an
active tape **raises `TesseraAutodiffError`** with a clear message naming
the op and pointing at `custom_rule`.

| Op | VJP |
|----|-----|
| `gemm` / `matmul` | `dA = dout @ B.T`, `dB = A.T @ dout` |
| `add` (incl. broadcast) | `dx = sum_to_shape(dout, x.shape)`, `dy = sum_to_shape(dout, y.shape)` |
| `mul` | `dx = sum_to_shape(dout * y, x.shape)`, `dy = sum_to_shape(dout * x, y.shape)` |
| `transpose` | `dx = transpose(dout, inverse_axes)` |
| `cast` | `dx = dout.astype(x.dtype)` |
| `relu` | `dx = dout * (x > 0)` |
| `silu` | `s = sigmoid(x); dx = dout * (s + x * s * (1 - s))` |
| `gelu` (tanh approx) | derivative of the tanh approx |
| `sigmoid` | `dx = dout * sig * (1 - sig)` |
| `tanh` | `dx = dout * (1 - tanh^2)` |
| `softmax(x, axis=-1)` | `dx = (dout - sum(dout * out, axis)) * out` |
| `layer_norm(x, eps)` | standard LN gradient |
| `rmsnorm(x, eps)` | standard RMSNorm gradient |
| `sum` / `reduce(op="sum")` | `dx = broadcast(dout, x.shape)` |
| `dropout` | `dx = dout * mask / (1-p)` (training); pass-through (eval) |

The VJPs are pure-numpy and live in `python/tessera/autodiff/vjp.py`. Adding
a new op is one VJP function + one `register_vjp` call.

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

## Phase F4 — Graph IR adjoint via `AdjointInterface` (design landed, build follow-up)

The numpy-tape implementation in this spec is the v1 surface that user code
binds to. The Phase F4 follow-up replaces the *internals* of that surface
with Graph-IR-level adjoints, while keeping `tape() / reverse(fn) /
custom_rule(name)` unchanged at the Python boundary.

Key pieces (landed 2026-05-09):

* **ODS interface** — `src/compiler/ir/include/Tessera/AdjointInterface.td`
  defines `Tessera_AdjointInterface` (`cppNamespace = ::tessera`) with three
  methods: `buildAdjoint(builder, outputCotangents) -> SmallVector<Value>`,
  `isDifferentiable() -> bool`, and `customAdjointName() -> StringRef`.

* **Pass body** — `src/transforms/lib/AutodiffPass.cpp` documents and codes
  the full four-step reverse walk (collect forward ops, identify scalar
  return as the loss seed, build cotangent map keyed by forward `Value`,
  reverse-walk dispatching `buildAdjoint` per op, accumulate via
  `arith.addf`). The interface-dispatch section is gated behind tablegen
  on the ODS file landing in the build — until then the pass registers as
  a no-op so opting into `--tessera-autodiff` doesn't break pipelines.

* **Build wiring** — `AutodiffPass.cpp` is in
  `src/transforms/lib/CMakeLists.txt`'s `TesseraPasses` library, registered
  via `createAutodiffPass()` declared in
  `src/transforms/include/Tessera/Transforms/Passes.h` and exposed to
  `tessera-opt` via `Passes.cpp`'s `registerTesseraPasses()`.

* **Lit smoke test** — `tests/tessera-ir/phase_f4/autodiff_pass_smoke.mlir`
  is currently `XFAIL` and flips to passing when the tablegen step
  produces `AdjointInterface.h.inc` and the per-op interface impls land.

* **Custom-rule bridge** — `tessera.autodiff.custom_rule(name)` continues to
  register Python VJPs at runtime. Ops that opt into a custom rule report the
  registry key via `customAdjointName()`; the AutodiffPass consults the
  bridge during its reverse walk via a `tessera.custom_adjoint_call`
  placeholder op (to be added once tablegen runs).

* **Remaining work to flip XFAIL** — (a) add the AdjointInterface ODS file
  to a tablegen target so `AdjointInterface.h.inc` is generated; (b) declare
  `Tessera_AdjointInterface` on each differentiable op in
  `src/compiler/ir/TesseraOps.td`; (c) implement `buildAdjoint` for
  `MatmulOp`, `AddOp`, `MulOp`, etc. (mirror the Python VJPs in
  `python/tessera/autodiff/vjp.py`); (d) flesh out the custom-rule
  placeholder op for ops carrying `customAdjointName()`.

Out of scope for the F4 first cut:

* Higher-order derivatives (run AutodiffPass twice + canonicalize). Tracked
  separately as F7 — currently 🔲 deferred.
* Effect-aware adjoint collective insertion. That's Phase F5 — runs after
  AutodiffPass and reads `Effect` attributes on adjoint values to insert
  `reduce_scatter` / `all_gather`.
* Activation rematerialization. The `tessera.autodiff.rematerialize` Python
  surface (Phase F2) is shipping today; the IR-pass form will arrive
  alongside F4.

## Cross-references

- Programming Guide Ch.7 (Autodiff) — describes the *full* Phase 5 epic.
  This spec implements the v1 first slice + F4 ODS scaffolding.
- `docs/audit/advanced_examples_capability_gap.md` — Theme 2.
- `python/tessera/nn/module.py` — `Module.zero_grad()` and `Parameter.grad`
  are in place from Tier 1.
- `python/tessera/autodiff/mixed_precision.py` — Phase F1 autocast + GradScaler.
- `python/tessera/autodiff/rematerialize.py` — Phase F2 activation checkpointing.
- `src/compiler/ir/include/Tessera/AdjointInterface.td` — Phase F4 ODS.
- `src/transforms/lib/AutodiffPass.cpp` — Phase F4 pass scaffold.
