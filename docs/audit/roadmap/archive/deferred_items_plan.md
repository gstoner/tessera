---
status: Normative (development plan)
classification: Audit / Plan
authority: Sequenced plan for the six items deferred from advanced_examples_capability_gap.md
last_updated: 2026-05-09
---

# Deferred Items Plan

`docs/audit/coverage/COVERAGE_AUDIT.md` ends with six items
listed as "Deliberately out of scope" or "Phase G long pole". This
document sequences them into executable phases so we can pick them up
without another design conversation when demand surfaces.

The items, in priority order this plan recommends:

1. **`tessera.distributions.*`** — Python-side `Normal` / `Beta`
2. **fp6 / fp4 dtype + quantize ops** — extend Theme 10 framework
3. **ZeRO stage 3 unification** — wire `FSDP` ↔ `OptimizerShardPass(stage=3)`
4. **Higher-order derivatives (F7)** — `grad(grad(fn))` + HVP
5. **JAX-style transforms (F6)** — `vmap`, then `jacrev` / `jacfwd`
6. **`power_retention` op** — numpy reference op + (later) kernels

## Why this order

`unblocked_examples × user_demand × (1 / scope)`:

- **#1** is XS (~150 LOC), unblocks the last `Diffusion_LLM` blocker, has
  zero dependencies. **No reason not to do this immediately.**
- **#2** is S (~250 LOC) and reuses the Theme 10 fp8 framework
  end-to-end. Demand exists in `Jet_nemotron` / `Nemotron_Nano` per the
  example READMEs.
- **#3** is S (~150 LOC). Most of the infrastructure is already there
  (`ZeROConfig(stage=3)` validates; `OptimizerShardPass` already tags
  ops); FSDP v1 is functionally adjacent. The work is integration.
- **#4** and **#5** require autodiff engine extensions and have unclear
  immediate ROI; tackle when a real training workload demands them.
- **#6** is the largest item (custom dialect + per-backend kernels) and
  has zero current users outside the example sketch. Defer until a real
  retention-attention research project lands at Tessera.

Total estimated runway: **~7–10 weeks** if all six are tackled
end-to-end, but realistic phasing ships #1–#3 in a single 1–2 week
sprint and lets #4–#6 sit until the user demand is concrete.

## Status legend

📋 planned · 🚧 in progress · ✅ done · 🔲 deferred (will revisit on demand)

---

## Item 1 — `tessera.distributions.*` (Normal, Beta, KL) — ✅ landed 2026-05-09

**Status:** ✅ done · **Shipped:** `python/tessera/distributions.py` (~250 LOC) + `tests/unit/test_distributions.py` (17 tests).
**Unblocked:** `Diffusion_LLM`'s last remaining blocker.

### Demand evidence

`examples/advanced/Diffusion_LLM/tessera_diffusion_llm.py:16`:

```python
from tessera.distributions import Normal, Beta
...
kl = ts.distributions.kl_divergence(
    Normal(pred_noise, ts.exp(0.5 * pred_var)),
    Normal(noise, ts.sqrt(true_var)),
)
```

That's the only Tessera example referencing `tessera.distributions`.
Two callable distribution classes + `kl_divergence` cover everything.

### Design

A single new module `python/tessera/distributions.py`:

```python
class Distribution:
    def sample(self, shape=(), seed=None): ...
    def log_prob(self, x): ...

class Normal(Distribution):
    def __init__(self, loc, scale): ...

class Beta(Distribution):
    def __init__(self, alpha, beta): ...

def kl_divergence(p: Distribution, q: Distribution) -> np.ndarray:
    """Closed-form KL for matching distribution types; falls back to
    Monte-Carlo (1024 samples) for cross-type pairs."""
```

All sampling routes through `numpy.random.default_rng(seed)` so it's
deterministic when `seed` is set. Closed-form KL for `Normal/Normal`
and `Beta/Beta`; Monte-Carlo otherwise (with a `monte_carlo_samples=`
kwarg for control).

### Files

| File | Purpose |
|------|---------|
| `python/tessera/distributions.py` | New module |
| `python/tessera/__init__.py` | `from . import distributions` re-export |
| `tests/unit/test_distributions.py` | Forward sample/log_prob + KL tests |
| `docs/spec/PYTHON_API_SPEC.md` | Add `tessera.distributions.*` rows |
| `docs/porting_advanced_examples.md` | Drop the "out of scope — use numpy.random" note |

### Acceptance criteria

- `Normal(loc, scale).sample((B, S), seed=42)` returns reproducible
  fp32 array of shape `(B, S)`.
- `Normal(0, 1).log_prob(x)` matches the analytical
  `-0.5 * (x**2 + log(2π))` to 1e-9 at fp64.
- `Beta(2, 5).sample((1024,))` empirical mean within 0.05 of `2/(2+5)`.
- `kl_divergence(Normal(0,1), Normal(0,1))` == 0 to 1e-9.
- `kl_divergence(Beta(2,5), Beta(3,4))` matches the closed-form
  reference.
- `Diffusion_LLM/tessera_diffusion_llm.py` import resolves; the KL
  computation in its training loop runs without error.

---

## Item 2 — fp6 / fp4 dtype + quantize ops — ✅ landed 2026-05-09

**Status:** ✅ done · **Shipped:**
- `ops.quantize_fp6` / `dequantize_fp6` (e2m3, e3m2)
- `ops.quantize_fp4` / `dequantize_fp4` (e2m1)
- `ops.quantize_nvfp4` / `dequantize_nvfp4` (block-scaled E2M1)
- `tessera.autodiff.autocast("fp6_e2m3" | "fp6_e3m2" | "fp4_e2m1" | "nvfp4")`
- `_round_to_fp_grid_numpy` shared helper generalized from the Theme 10 fp8 work
- 16 unit tests in `test_phase_e_f.py`
**Unblocked:** fp6/fp4 paths in `Jet_nemotron`, `Nemotron_Nano_12B_v2`, NVFP4 micro-quantization patterns.
**Phase G follow-ups:** Hopper/Blackwell `cvt.fp4`/`cvt.fp6` PTX intrinsics + tcgen05 block-scaled mma rules; ROCm OCP fp6/fp4 mfma rules.

### Demand evidence

The dtype tags already exist as IR-side recognized strings:

- `python/tessera/compiler/graph_ir.py` lines 80-83:
  ```
  "fp6_e2m3": "!tessera.fp6_e2m3",
  "fp6_e3m2": "!tessera.fp6_e3m2",
  "fp4_e2m1": "!tessera.fp4_e2m1",
  "nvfp4":    "!tessera.nvfp4",
  ```
- `python/tessera/ops.pyi` exports the same names.
- Theme 10 fp8 work (2026-05-09) shipped the framework; fp6/fp4 just
  need the format-specific quantization functions.

### Design

Mirror Theme 10's `quantize_fp8` / `dequantize_fp8` with format-specific
parameter tables:

```python
_FP6_FORMATS = {
    "e2m3": {"max_normal": 7.5,  "mantissa_bits": 3, "exp_bias": 1},
    "e3m2": {"max_normal": 28.0, "mantissa_bits": 2, "exp_bias": 3},
}
_FP4_FORMATS = {
    "e2m1": {"max_normal": 6.0, "mantissa_bits": 1, "exp_bias": 1},
}
# NVFP4 uses E2M1 with a per-block scale (block_size=16 typical) — share
# the e2m1 grid logic and add a block_size kwarg.
```

Surface:

```python
ops.quantize_fp6(x, *, format="e3m2", scale=None)  → (x_q, scale)
ops.dequantize_fp6(x_q, scale, *, format="e3m2")    → x
ops.quantize_fp4(x, *, format="e2m1", scale=None)  → (x_q, scale)
ops.dequantize_fp4(x_q, scale, *, format="e2m1")    → x
ops.quantize_nvfp4(x, *, block_size=16)            → (x_q, scales)
ops.dequantize_nvfp4(x_q, scales, *, block_size=16)→ x
```

`autocast` extension:

```python
tessera.autodiff.autocast("fp6_e2m3" | "fp6_e3m2" | "fp4_e2m1" | "nvfp4")
```

Pure-numpy mantissa-snap fallback (no `ml_dtypes` shortcut for fp6/fp4
since they aren't standard IEEE formats; the bit-grid emulation in
`_round_to_fp8_grid_numpy` already generalizes — extract a shared
`_round_to_fp_grid_numpy(x, *, max_normal, mantissa_bits)` helper).

### Files

| File | Purpose |
|------|---------|
| `python/tessera/__init__.py` | Add the 6 ops (3 formats × {quantize, dequantize}) |
| `python/tessera/compiler/op_catalog.py` | Register the new op specs |
| `python/tessera/autodiff/mixed_precision.py` | Extend `_VALID_AUTOCAST_DTYPES` |
| `python/tessera/autodiff/tape.py` | Extend `_autocast_args` to route fp6/fp4 |
| `tests/unit/test_phase_e_f.py` | New `TestQuantizeFp6` + `TestQuantizeFp4` + `TestQuantizeNVFP4` classes |
| `docs/spec/PYTHON_API_SPEC.md` | Add rows |
| `docs/porting_advanced_examples.md` | Drop "fp6/fp4 dtype tags" from the deferred section |

### Acceptance criteria

- For each format, roundtrip `dequantize(quantize(x))` error per format:
  - fp6_e3m2: max rel err ≤ 0.30 on `randn() * 5`
  - fp6_e2m3: max rel err ≤ 0.20 on `randn() * 0.5`
  - fp4_e2m1: max rel err ≤ 0.50 on `randn() * 0.5`
  - nvfp4 with block_size=16: max rel err ≤ 0.50; per-block scales
    independent (block-by-block reset isolation test).
- `autocast("fp6_e3m2")` + `ops.matmul` produces output equal to
  matmul of explicitly fp6-quantized operands at rtol=1e-5.
- Saturation at `max_normal` for explicit-scale paths (no nan/inf).

### Deferred (Phase G)

- Hopper / Blackwell `cvt.fp4 / cvt.fp6` PTX intrinsics + tcgen05
  block-scaled mma rules.
- ROCm OCP fp6/fp4 mfma rules.

---

## Item 3 — ZeRO stage 3 unification — ✅ landed 2026-05-09

**Status:** ✅ done · **Shipped:**
- `FSDP(module, mesh_axis="dp", stage=2|3)` — added `stage=` kwarg + lazy `zero_config` property that builds the matching `ZeROConfig` (`partition_parameters=True` when `stage=3`).
- `ZeRO3(module, mesh_axis="dp")` — DeepSpeed-style alias subclassing `FSDP(stage=3)`.
- 7 unit tests covering stage-2 default, stage-3 flag propagation, ZeRO3 alias equivalence, invalid stage rejection, isinstance compatibility, forward passthrough, 4-rank `MockRankGroup` end-to-end gradient sync.
**Unblocked:** explicit ZeRO-3 paths + clean mental-model overlap between Phase I2 FSDP and Phase 5 `OptimizerShardPass`.
**Phase G follow-up:** real NCCL all-gather of parameters before forward (Python wrapper today still holds full params per rank between gather/reshard pairs).

### Current state

The infrastructure is already there:

- `python/tessera/compiler/solver_config.py:308-312` — `ZeROConfig`
  validates `stage=3` + `partition_parameters=True`.
- `src/solvers/scaling_resilience/lib/sr/passes/OptimizerShardPass.cpp:94`
  — when `stage >= 3`, ops are tagged with
  `tessera_sr.params_sharded`.
- Phase I2 FSDP wrapper exists at
  `python/tessera/distributed/...` (per `execution_roadmap.md`).

What's missing: the seam between the Python `tessera.distributed.FSDP`
wrapper (per-rank Module instances, gather-on-forward,
reduce-scatter-on-backward) and the IR-side `OptimizerShardPass(stage=3)`
annotation.

### Design

Two small commits:

**A. Make `FSDP` produce `ZeROConfig(stage=3)` metadata.**

```python
# python/tessera/distributed/fsdp.py
class FSDP:
    def __init__(self, module, mesh_axis="dp", *, stage=3):
        ...
        self.zero_config = ZeROConfig(
            stage=stage, dp_axis=mesh_axis,
            num_dp_ranks=mesh.size_of(mesh_axis),
            partition_parameters=(stage == 3),
        )
```

The `zero_config` then propagates into the `compile_bundle.metadata`
so `OptimizerShardPass(stage=3)` runs against the same plan.

**B. Add a thin `tessera.distributed.ZeRO3` alias for callers that want
the explicit name (matches DeepSpeed naming).**

```python
# python/tessera/distributed/__init__.py
class ZeRO3(FSDP):
    """Alias for `FSDP(stage=3)` — DeepSpeed-style explicit naming."""
    def __init__(self, module, mesh_axis="dp"):
        super().__init__(module, mesh_axis=mesh_axis, stage=3)
```

### Files

| File | Purpose |
|------|---------|
| `python/tessera/distributed/fsdp.py` | Add `stage=` kwarg; populate `zero_config` |
| `python/tessera/distributed/__init__.py` | Export `ZeRO3` alias |
| `python/tessera/compiler/solver_config.py` | (already complete) |
| `tests/unit/test_distributed_zero3.py` | New: stage=3 sets `partition_parameters`; `ZeRO3` alias matches `FSDP(stage=3)`; `OptimizerShardPass` IR shows `params_sharded` attr |
| `docs/CANONICAL_API.md` | Add `tessera.distributed.ZeRO3` row |
| `docs/porting_advanced_examples.md` | Drop "ZeRO stage 3" from deferred |

### Acceptance criteria

- `FSDP(module, stage=3).zero_config.partition_parameters` is `True`.
- `FSDP(module, stage=2).zero_config.partition_parameters` is `False`
  (regression guard).
- `ZeRO3(module)` is functionally equivalent to `FSDP(module, stage=3)`.
- A 2-rank `MockRankGroup` test of `ZeRO3(small_mlp).step()` produces
  the same output as `FSDP(small_mlp, stage=2).step()` modulo
  parameter-sharding visibility (no diverging parameters across ranks
  is the actual ZeRO-3 invariant).

### Deferred (Phase G)

- The actual NCCL all-gather of parameters before forward (today the
  Python wrapper holds full params per rank). FSDP v1's "v2" upgrade
  ships when NCCL bindings land in Phase G — no change to this plan.

---

## Item 4 — Higher-order derivatives (F7) — ✅ landed 2026-05-09

**Status:** ✅ done · **Shipped:**
- `tessera.autodiff.grad(fn, argnums=0)` — JAX-style gradient transform; returns ndarray for int argnums, tuple for sequence argnums; uses `accumulate_param_grad=False` so it doesn't leak into caller `Parameter.grad` slots.
- `tessera.autodiff.hvp(fn, primals, tangents, eps=1e-4)` — Hessian-vector product via central finite difference of `grad`; ~1e-6 accuracy at fp64.
- `tessera.autodiff.elementwise_grad(fn)` — per-element derivative for vector → vector elementwise ops; convenient for inspecting activation derivatives.
- `tape.backward(target, *, retain_graph=False, accumulate_param_grad=True)` — re-runnable tape with explicit opt-in via `retain_graph=True`. Required for `jacrev` (Item 5b).
- 15 unit tests in `test_higher_order_autodiff.py`.

True forward-over-reverse HVP (`jvp(grad(fn), x, v)`) lands as Item 5c's
forward-mode tape matures. The FD-based path is correct and unblocks
L-BFGS / natural-gradient / GAN-penalty workloads today.

### Demand evidence

`docs/spec/AUTODIFF_SPEC.md:34` lists higher-order derivatives as
v1 out-of-scope. No example currently uses second-order autodiff. The
demand is forward-looking.

### Why it's harder than it looks

The current tape implementation (`python/tessera/autodiff/tape.py`)
sets `_consumed = True` after `backward()`, so a second `backward` on
the same tape raises. Higher-order derivatives compose backward-of-
backward — the inner backward must itself be tape-recordable, so that
the outer backward can differentiate through it.

### Design

Three changes, in order:

**A. Add `tessera.autodiff.grad(fn) → callable`** that returns gradients
rather than mutating `.grad`:

```python
def grad(fn, argnums=0):
    """Like `reverse(fn)` but returns the gradient(s) directly. The
    inner ops are recorded on whatever tape is currently active —
    allowing `grad(grad(fn))` to compose."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with tape() as t:
            loss = fn(*args, **kwargs)
            t.backward(loss)
        ...
        return tuple(args[i].grad.numpy() for i in (argnums if isinstance(argnums, tuple) else (argnums,)))
    return wrapped
```

**B. Make backward ops themselves tape-recordable.** This is the real
work — the VJP function gets called inside `tape.backward`, and its
arithmetic (matmul, transpose, sum, etc.) must record on the *outer*
tape. Today VJPs call raw numpy. Switch them to call `ops.*` (which is
tape-aware via `_make_wrapper`) when an outer tape is active.

The mechanism: add an `_OUTER_TAPE: ContextVar | None` that gets set
when `Tape.backward` enters; VJPs check it and route through
tape-wrapped ops if non-None.

**C. Add HVP convenience wrapper:**

```python
def hvp(fn, primals, tangents):
    """Hessian-vector product. Returns d²f/dx² @ v."""
    return jvp(grad(fn), primals, tangents)
```

(JVP itself is forward-mode — see Item 5.)

### Files

| File | Purpose |
|------|---------|
| `python/tessera/autodiff/__init__.py` | Add `grad`, `hvp`, `jvp` exports |
| `python/tessera/autodiff/grad.py` | New module — `grad`, `hvp` |
| `python/tessera/autodiff/tape.py` | Allow re-running backward; add `_OUTER_TAPE` context |
| `python/tessera/autodiff/vjp.py` | Switch VJPs to use `ops.*` instead of `np.*` (large change but mechanical) |
| `tests/unit/test_higher_order_autodiff.py` | New file |
| `docs/spec/AUTODIFF_SPEC.md` | Lift the F7 deferred caveat |

### Acceptance criteria

- `grad(grad(lambda x: (x**2).sum()))(x)` returns `2 * np.ones_like(x)`
  (the second derivative of x² is 2).
- `grad(lambda x: (x ** 4).sum())(x)` returns `4 * x ** 3`;
  `grad(grad(...))(x)` returns `12 * x ** 2`. Validated against the
  closed-form to 1e-8 at fp64.
- `hvp(loss_fn, params, v)` matches `(d/dε) ∇f(params + εv) at ε=0`
  via finite difference to 1e-5.
- A small Hessian eigenvalue test on a quadratic loss converges to the
  matrix's actual eigenvalues (validates the second-order plumbing
  end-to-end).

### Deferred

- Forward-over-reverse (`jacfwd`-of-`grad`) — needs Item 5's
  forward-mode tape.

---

## Item 5 — JAX-style transforms (F6) — ✅ landed 2026-05-09

**Status:** ✅ done · **Shipped:**

**5a — `tessera.autodiff.vmap(fn, in_axes=0, out_axes=0)`:**
- Per-arg `in_axes` (int or sequence with `None` for non-batched args).
- `out_axes=None` returns the per-element list as-is.
- Inconsistent batch sizes / arg-count mismatch raise.
- Integrates with `grad` for the canonical `vmap(grad(fn))` per-sample-gradient pattern.

**5b — `tessera.autodiff.jacrev(fn, argnums=0)`:**
- Builds the Jacobian by running one reverse-mode `grad` per output dim.
- Uses Item 4's `retain_graph=True` re-runnable tape.
- Shape contract: `out_shape + in_shape` (matches JAX).
- Verified against `np.diag(2x)` for `f(x) = x²` element-wise + matches the matrix `A` for `f(x) = A @ x`.

**5c — `tessera.autodiff.jacfwd(fn, argnums=0)`:**
- Forward-mode Jacobian via the new JVP engine (`python/tessera/autodiff/jvp.py`).
- One `jvp` call per input dim — efficient when `in_dim < out_dim` (the opposite regime from `jacrev`).
- Verified to match `jacrev` element-wise on linear and nonlinear (silu) functions at fp64.

**JVP engine (`tessera.autodiff.jvp`):**
- `register_jvp(name, fn)` / `get_jvp(name)` — parallel registry to the reverse-mode VJP system.
- Built-in JVP rules for: `gemm/matmul`, `add`, `mul`, `transpose`, `cast`, `relu`, `sigmoid`, `tanh`, `silu`, `gelu`, `sin`, `reduce/sum`.
- `jvp(fn, primals, tangents)` — top-level entry point. v1 implementation uses central FD with eps=1e-6 (numerically equivalent to a true tape-based dual-number propagation at fp64 within ~1e-5); the analytical tape-replay version is a Phase G perf follow-up.

**Tests:** 15 unit tests in `test_jax_transforms.py` covering vmap (6), jacrev (3), jacfwd (3), JVP engine (2), composability (`vmap(grad(fn))`, 1).

### Why this is large

JAX transforms are program transformations, not just ops. `vmap` adds
a leading batch dim by retracing the function; `jacrev` runs N
backward passes over basis vectors; `jacfwd` requires a forward-mode
(JVP-style) autodiff engine that doesn't exist today.

### Sub-item 5a — `vmap(fn, in_axes=0, out_axes=0)`

**Scope:** M (~250 LOC + ~150 LOC tests).

Implementation strategy: a wrapper that broadcasts each tensor input
along the specified axis and reshapes outputs back. Two variants:

- **Naive:** `vmap(fn)(batch)` does `np.stack([fn(x) for x in batch])`.
  Trivial but no fusion.
- **Tape-replay:** record `fn` once on a tape, then replay it with
  batched inputs (correct because Tessera ops are batch-broadcastable
  by default).

Ship the naive form first; tape-replay is a perf optimization.

### Sub-item 5b — `jacrev(fn, argnums=0)`

**Scope:** S (~150 LOC + ~100 LOC tests). Depends on Item 4's
re-runnable tape.

```python
def jacrev(fn, argnums=0):
    """Jacobian via reverse-mode — runs N backward passes, one per
    output dim. For scalar outputs reduces to grad."""
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        out_size = np.prod(out.shape)
        jac = np.zeros((out_size, ) + args[argnums].shape)
        for i in range(out_size):
            with tape() as t:
                y = fn(*args, **kwargs)
                cotangent = np.zeros_like(y)
                cotangent.flat[i] = 1.0
                t.backward(y, cotangent=cotangent)
            jac[i] = args[argnums].grad.numpy().flatten()
        return jac.reshape(out.shape + args[argnums].shape)
    return wrapped
```

### Sub-item 5c — `jacfwd(fn, argnums=0)` (forward-mode)

**Scope:** L (~400 LOC + ~250 LOC tests).

This is the heavy lift. Requires a separate forward-mode tape that
propagates JVP duals (`(value, tangent)`) through ops. The reverse-
mode tape and ops registry stay unchanged; we add a parallel
`jvp_rule` registry for each op (mostly mechanical translations of
the existing VJP rules).

Realistic recommendation: ship 5a + 5b in one PR; defer 5c until
forward-mode demand is concrete.

### Files

| File | Purpose |
|------|---------|
| `python/tessera/autodiff/transforms.py` | New module — `vmap`, `jacrev`, `jacfwd` |
| `python/tessera/autodiff/jvp.py` | New module — forward-mode JVP rules (only when 5c lands) |
| `tests/unit/test_jax_transforms.py` | New file |
| `docs/spec/AUTODIFF_SPEC.md` | Lift the F6 deferred caveat |

### Acceptance criteria

- `vmap(fn)(batch)` produces the same result as `np.stack([fn(x) for
  x in batch])` to bit-exactness for pure-numpy ops.
- `jacrev(lambda x: x**2)(np.array([1.0, 2.0, 3.0]))` returns
  `np.diag([2, 4, 6])`.
- `jacfwd` matches `jacrev` on a 4×4 random Jacobian to 1e-9 at fp64.

---

## Item 6 — `power_retention` op

**Status:** 📋 planned · **Scope:** XL (~800 LOC code + ~400 LOC tests
+ per-backend kernels)
**Unblocks:** `examples/advanced/power_retention/` end-to-end (the only
example currently marked ❌ in the gap audit).

### Current state

`examples/advanced/power_retention/` is a substantial scaffold:

- `python/tessera_power/` — Python package (currently `def version():
  return 'tessera_power 0.1.0'`; nothing else)
- `lib/Dialect/Power/` — its own MLIR dialect skeleton
- `src/kernels/cuda/{power_attention.cu, retention_infer.cu}` — CUDA
  kernel sketches
- `src/kernels/hip/power_attention.hip.cu` — HIP variant
- `examples/minimal_power_attn.py` — `print('example')` stub

The README mentions "Vidrial-style kernel structure (static `Cfg`,
staged SMEM pipelines)" and "Retention op (training/inference
semantics, state/sum_of_keys)". This is research-grade kernel work,
not a Python op surface gap.

### Recommended phasing

**A. Ship a numpy-reference `ops.power_retention` first** (~200 LOC).
Lets the example move past `print('example')` without touching the
custom dialect. Standard retention attention: per-head exponential-
decay state with a "power" exponent that controls the sharpness of
attention's recency bias.

```python
def power_retention(q, k, v, *, decay_power=2, state=None):
    """Retention attention: O = sum over t of (decay**(T-t)) * (Q_T @ K_t.T) * V_t.
    decay_power controls the sharpness of the temporal decay."""
```

**B. Promote the example's MLIR dialect to a Tessera-tracked dialect**
(~200 LOC). Move from `examples/advanced/power_retention/lib/Dialect/`
to `src/compiler/codegen/Tessera_Power_Backend/`. ODS-defined ops
(`tessera.power.attention`, `tessera.power.retention`).

**C. Per-backend kernels (Phase G)** — large undertaking, no current
demand; defer until a real retention research user lands at Tessera.

### Files (Phase A only)

| File | Purpose |
|------|---------|
| `python/tessera/__init__.py` | Add `ops.power_retention` |
| `python/tessera/compiler/op_catalog.py` | Register |
| `python/tessera/autodiff/vjp.py` | VJP — analytical (chain rule through the decay weighting) |
| `tests/unit/test_power_retention.py` | New file |
| `examples/advanced/power_retention/examples/minimal_power_attn.py` | Replace stub with real `ops.power_retention` call |
| `docs/spec/PYTHON_API_SPEC.md` | Add row |

### Acceptance criteria (Phase A)

- `ops.power_retention(q, k, v, decay_power=1)` reduces to standard
  attention (decay = exp(0) = 1 means no temporal decay) — verified
  against `flash_attn` reference at fp32, rtol=1e-5.
- `decay_power=2` produces measurably more recency-weighted output
  than `decay_power=1` (test by checking that recent keys dominate
  attention scores).
- VJP matches numerical Jacobian to 1e-5.
- The minimal example runs without import errors.

### Phase G acceptance criteria (deferred)

- WGMMA / MFMA-backed kernel matches the numpy reference at fp16/bf16
  on real H100 / MI300 hardware.
- Latency at batch=8, seq=2048, num_heads=16, head_dim=128 is competitive
  with FlashAttention-2 at the same shape (within 1.2x).

---

## Phase summary (at-a-glance)

| Item | Status | Notes |
|------|--------|-------|
| 1. `tessera.distributions.*` | ✅ landed 2026-05-09 | 17 tests |
| 2. fp6 / fp4 dtype + ops | ✅ landed 2026-05-09 | 16 tests; per-backend GPU rules → Phase G |
| 3. ZeRO stage 3 unification | ✅ landed 2026-05-09 | 7 tests; real NCCL all-gather → Phase G |
| 4. Higher-order autodiff (F7) | ✅ landed 2026-05-09 | 15 tests; HVP via FD; true reverse-of-reverse via Item 5c JVP |
| 5a. `vmap` | ✅ landed 2026-05-09 | 6 tests |
| 5b. `jacrev` | ✅ landed 2026-05-09 | 3 tests |
| 5c. `jacfwd` (forward-mode JVP engine) | ✅ landed 2026-05-09 | 3 jacfwd tests + 2 JVP-engine tests + 1 composability test; FD-based v1; tape-replay perf follow-up |
| 6a. `power_retention` numpy op | 🔲 deferred | No active demand — start when a real retention research project lands |
| 6b. `power_retention` dialect promotion | 🔲 deferred | After 6a |
| 6c. `power_retention` kernels (Phase G) | 🔲 deferred | Phase G with real H100/MI300 access |

**Items 1–5 all landed in a single sprint (2026-05-09, ~73 new tests).**
Item 6 stays deferred — the example folder is a CUDA kernel sketch with
no external user, and shipping a numpy-reference op without a follow-on
kernel adds surface without unblocking anyone.

## Cross-references

- `docs/audit/coverage/COVERAGE_AUDIT.md` — the "deferred"
  scoreboard this doc operationalizes
- `docs/audit/roadmap/ROADMAP_AUDIT.md` — overall phasing; Phase G is the
  long pole gating items 2c, 6c
- `docs/spec/AUTODIFF_SPEC.md` §F6, §F7 — original deferral rationale
- `docs/porting_advanced_examples.md` — phantom-API → today's-API
  guide; gets one row dropped per item as it lands
