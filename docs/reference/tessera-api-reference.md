---
status: Informative
classification: Informative
last_updated: 2026-05-19
---

# Tessera API Reference

This reference summarizes the current public API shape. The authoritative API
specification is `docs/spec/PYTHON_API_SPEC.md`; if this guide disagrees with
that spec, the spec wins. Tensor attribute and dtype vocabulary lives in
`docs/reference/tessera_tensor_attributes.md`.

> **Start here:** for a runnable, narrated tour of the compiler surface in
> ~80 lines — `@tessera.jit` → `fn(...)` → `fn.explain()` →
> `tessera.compiler.support(op)` → `tessera.from_text(...)` — see
> [`examples/getting_started/compile_and_explain.py`](../../examples/getting_started/compile_and_explain.py).

## Import Pattern

```python
import tessera as ts
```

Use the top-level `tessera` namespace (commonly aliased `ts`) for public
examples unless a spec explicitly names a submodule import.

## Decorators

| API | Status | Purpose |
|-----|--------|---------|
| `@tessera.jit` | Implemented | Compile a Python function to Graph IR; build Schedule / Tile / Target IR; return a `JitFn`. |
| `@tessera.kernel` | Implemented | Mark a tile-level function for `index_launch`. |
| `tessera.from_text(source, name=None, **jit_kwargs)` | Implemented (2026-05-19) | Notebook-safe factory: exec a source string and JIT the named function. Replaces the `exec(...) + ts.jit(..., source=...)` dance for REPL / Jupyter contexts. |

```python
@tessera.jit
def matmul_step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.gemm(A, B)

# Notebook-safe construction when @jit can't read source (REPL, heredoc):
fn = tessera.from_text("""
    def gelu_then_norm(x):
        return ts.ops.layer_norm(ts.ops.gelu(x))
""")
```

### `JitFn.explain()` — the inspection front door

Every JIT'd function carries an `.explain()` method that answers four
questions in one call:

1. **What ran?** (`execution_kind` — `native_cpu` / `reference_cpu` /
   `native_gpu` / `artifact_only` / `fallback_eager`).
2. **Was it native / reference / artifact / fallback?** (`is_native`,
   `is_reference`, `is_artifact_only`, `is_fallback` predicates).
3. **Why?** (typed `diagnostics` list — each entry carries a stable code
   from `tessera.compiler.JitDiagnosticCode` / `FallbackReason`).
4. **What should I do next?** (typed `next_actions` list with stable codes
   like `INSPECT_IR_LAYERS`, `USE_NATIVE_REQUIRED_TO_DIAGNOSE`,
   `PROVIDE_SOURCE_FOR_NOTEBOOK`).

```python
ex = matmul_step.explain()
print(ex)                       # 5-line opinionated summary
ex.execution_kind               # "reference_cpu"
ex.ir.graph                     # Graph IR as MLIR text
ex.ir.target                    # Target IR
ex.kernels                      # per-op resolution list
ex.diagnostics                  # typed Diagnostic list
ex.next_actions                 # typed NextAction list
ex.as_dict()                    # JSON-serializable
```

The legacy inspection methods (`ir_text()`, `schedule_ir`, `tile_ir`,
`target_ir`, `lowering_artifacts()`, `runtime_artifact()`,
`compile_report()`, `explain_lowering()`) remain as stable lower-level data
sources; `.explain()` consumes them under the hood.

### Strict native dispatch with `native_required=True`

Pass `native_required=True` to `@tessera.jit` to refuse the reference
fallback path.  Any condition that would otherwise silently drop to the
numpy backend raises `tessera.compiler.TesseraNativeRequiredError` with a
stable `FallbackReason` code:

```python
from tessera.compiler import FallbackReason, TesseraNativeRequiredError

@tessera.jit(target="apple_gpu", native_required=True)
def fast_matmul(a, b):
    return tessera.ops.matmul(a, b)

try:
    fast_matmul(a, b)
except TesseraNativeRequiredError as exc:
    if exc.reason is FallbackReason.NON_DARWIN_HOST:
        pytest.skip("Apple GPU only")
    raise
```

### Per-op readiness query

`tessera.compiler.support(op_name)` returns the same data the audit table
in `docs/audit/generated/support_table.md` renders, exposed as Python:

```python
info = tessera.compiler.support("matmul")
info.family                                   # "loop_nest"
info.best_tier                                # Tier.NATIVE_READY
info.for_target("apple_gpu").tier             # Tier.NATIVE_READY
info.for_target("cpu").tier                   # Tier.REFERENCE_ONLY

tessera.compiler.tier("matmul")               # best-tier rollup
tessera.compiler.tier("matmul", target="apple_gpu")  # per-target
tessera.compiler.is_native_supported("matmul", target="apple_gpu")  # True
```

Tier values: `NATIVE_READY` / `REFERENCE_ONLY` / `ARTIFACT_ONLY` / `PLANNED`.

## Region Privileges

`Region[...]` is a type annotation only. It lowers to privilege/effect attributes in Graph IR.

```python
@tessera.jit
def update(
    X: tessera.Region["read"],
    W: tessera.Region["read"],
    Y: tessera.Region["write"],
):
    Y[:] = tessera.ops.gemm(X, W)
```

Valid modes are:

| Mode | Meaning |
|------|---------|
| `read` | Read-only access |
| `write` | Exclusive write access |
| `reduce_sum` | Parallel sum reduction |
| `reduce_max` | Parallel max reduction |
| `reduce_min` | Parallel min reduction |

## Domains, Distributions, And Arrays

Domains describe shape. Distributions describe placement. Keep them separate.

```python
D = tessera.domain.Rect((4, 128, 256))
dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
```

`tessera.dist.Cyclic` exists as a Phase 4 planned distribution; in Phases 1-3 it raises `NotImplementedError` when shard specs are materialized.

Tensor attributes are split across logical shape (`shape`), storage dtype
(`dtype`), layout (`layout`), execution target (`target`), distribution
(`ShardSpec`), and numeric policy (`numeric_policy`). Dtype strings should use
the canonical names in `docs/reference/tessera_tensor_attributes.md`; aliases
such as `"f32"` normalize before storage.

## Index Launch

`index_launch` dispatches a `@tessera.kernel` function over shard lists.

```python
@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...], C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)

tessera.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"),
    B.parts("tp"),
    C.parts("tp"),
)
```

Phase 1 uses sequential/mock execution. Production NCCL/RCCL-backed distributed execution is Phase 4 planned.

## Constraints

Use `tessera.require(...)` inside `@tessera.jit` functions.

```python
@tessera.jit(bindings={"K": 128})
def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)
```

Implemented constraints:

| API | Meaning |
|-----|---------|
| `tessera.constraint.Divisible(dim, divisor)` | `dim % divisor == 0` |
| `tessera.constraint.Range(dim, lo, hi)` | `lo <= dim <= hi` |
| `tessera.constraint.Equal(dim_a, dim_b)` | `dim_a == dim_b` |

## Operations

Use `tessera.ops`.

| API | Status |
|-----|--------|
| `gemm`, `matmul` | Phase 1-3 implemented |
| `layer_norm`, `softmax`, `gelu`, `relu`, `transpose`, `cast` | Phase 1-3 implemented |
| `dropout` | Phase 1-3 implemented with random effect |
| `conv2d` | Implemented — NHWC + NCHW Module forms (`tessera.nn.Conv2d` / `Conv2dNCHW`); Graph IR op + VJP/JVP registered. |
| `flash_attn` | Phase 1 naive path; Phase 3 SM_90+ FA-4 lowering path |
| `all_reduce`, `reduce_scatter`, `all_gather` | Implemented — Phase 4 distributed lowering (`GPUCollectiveInsertionPass`); NCCL/RCCL adapters wired; VJP+JVP registered for all four collectives. |
| `fused_epilogue` | Phase 1-3 implemented where supported by canonicalization/lowering |

## GPU Targeting

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

@tessera.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True),
)
def flash_fwd(Q, K, V):
    return tessera.ops.flash_attn(Q, K, V, causal=True)
```

## Inspection

The compiler exposes all four IR layers as inspectable objects on the
JIT artifact:

```python
print(flash_fwd.graph_ir.to_mlir())
print(flash_fwd.schedule_ir)   # Schedule IR (mesh / pipeline)
print(flash_fwd.tile_ir)       # Tile IR (warps / TMA / async_copy)
print(flash_fwd.target_ir)     # per-target final IR
```

See `examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py`
for a runnable walkthrough that prints all four layers for a tiny MLP.
For static inspection without launching, `tessera-mlir
--mode=compile_artifact --symbol=<name>` reads the JIT artifact directly
(see `docs/guides/Tessera_Debugging_Tools_Guide.md`).

## Constrained-lane Graph IR views (Phase B, 2026-05-20)

The constrained math lanes (`@clifford_jit`, `@complex_jit`,
`@energy_jit`) keep their own narrower IR types but expose a
`to_graph_ir_view() -> GraphIRModule` adapter for tooling that
wants to consume any lane uniformly:

```python
from tessera.compiler.clifford_jit import lower_function_to_ir
program = lower_function_to_ir(my_clifford_fn)
view = program.to_graph_ir_view()
# Now view.functions[0].lane == "clifford_jit",
#     view.functions[0].verification_facts == {"ga_whitelisted"}
```

The view is a **1:1 projection** with the canonical op-name
vocabulary; it is **never** the source of truth for execution
(the constrained lanes keep that sovereignty).  Full contract in
[`docs/spec/COMPILER_REFERENCE.md`](../spec/COMPILER_REFERENCE.md)
§ "Constrained-lane Graph IR views".

## Optional IR metadata (Phase A, 2026-05-20)

Graph IR carries cross-cutting metadata as **optional** fields —
producers fill what they know; consumers tolerate missing values:

| Field | Type | Where it lives | What it means |
|---|---|---|---|
| `IROp.source_span` | `SourceSpan \| None` | per-op | line/col from the source frontend |
| `IROp.numeric_policy` | `NumericPolicy \| None` | per-op | storage / accum / scale (G3) |
| `IROp.value_kind` | `str \| None` | per-op | `tensor` / `multivector` / `complex` / `energy` |
| `IROp.verification_facts` | `frozenset[str]` | per-op | lane invariants (`holomorphic`, `ga_only`, ...) |
| `GraphIRFunction.lane` | `str` | per-fn | `tessera_jit` / `textual_dsl` / `clifford_jit` / `complex_jit` / `energy_jit` |
| `GraphIRFunction.verification_facts` | `frozenset[str]` | per-fn | function-level lane invariants |
| `GraphIRFunction.source_hash` | `str \| None` | per-fn | SHA-256 of the source text |

The drift-gate `tests/unit/test_optional_ir_metadata_contract.py`
asserts new fields stay optional unless explicitly grandfathered.
Adding a required IR field is a breaking change that fails CI.

Architecture rationale and the phased plan (A → B → C → D) live in
[`docs/architecture/frontend_substrate_plan.md`](../architecture/frontend_substrate_plan.md).

## Roadmap (formerly "Future APIs")

These rows were authored when Phases 4–6 were planned.  The status
below reflects the post-Phase-8 reality (Apple operational; Cerebras /
Metalium / Apple backends shipped under Phase 7-8; the S-series
standalone-compiler track shipped S0–S15 + autodiff coverage).  For
the current per-component picture, read
`docs/audit/generated/support_table.md` (drift-gated) rather than this
table.

| Area | Status |
|------|--------|
| NCCL/RCCL collectives + cluster execution | Implemented (Phase 4) — `GPUCollectiveInsertionPass`, NCCL/RCCL adapters, `ChunkPlanner`, `CollectiveScheduler`. |
| TPU StableHLO backend | Implemented — `tpu_target.py` + `Tessera_TPU_Backend/` (StableHLO + Shardy export); quantized dot lit-tested. |
| Autodiff transforms + custom VJP/JVP | Implemented — `tessera.autodiff` v1 ships 241 VJPs + 236 JVPs; `tessera.custom.custom_vjp` / `custom_jvp` user-facing. |
| Activation checkpointing + ZeRO sharding | Implemented — `tessera.autodiff.rematerialize` + ZeRO stage 2 via `OptimizerShardPass`. |
| Bayesian autotuning | Implemented — `tessera.autotune` + `compiler/autotune_v2.py` (Optuna TPE + Hyperband + SQLite cache v2). |
| Runtime Python wrapper | Implemented — `tessera.runtime.TesseraRuntime` over the C ABI. |
| ROCm MFMA + RubinCPX + Cerebras + Metalium + Apple backends | Implemented (Phase 7–8). |
