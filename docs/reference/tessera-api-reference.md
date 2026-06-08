---
status: Informative
classification: Informative
last_updated: 2026-05-20
---

# Tessera API Reference

This reference summarizes the current public API shape. The authoritative API
specification is `docs/spec/PYTHON_API_SPEC.md`; if this guide disagrees with
that spec, the spec wins. Tensor attribute and dtype vocabulary lives in
`docs/reference/tessera_tensor_attributes.md`.

> **Start here:** for a runnable, narrated tour of the compiler surface in
> ~80 lines — `@ts.jit` → `fn(...)` → `fn.explain()` →
> `ts.compiler.support(op)` → `ts.from_text(...)` — see
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
| `@ts.jit` | Implemented | Compile a Python function to Graph IR; build Schedule / Tile / Target IR; return a `JitFn`. |
| `@ts.kernel` | Implemented | Mark a tile-level function for `index_launch`. |
| `ts.from_text(source, name=None, **jit_kwargs)` | Implemented (2026-05-19) | Notebook-safe factory: exec a source string and JIT the named function. Replaces the `exec(...) + ts.jit(..., source=...)` dance for REPL / Jupyter contexts. |

```python
@ts.jit
def matmul_step(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]):
    return ts.ops.gemm(A, B)

# Notebook-safe construction when @jit can't read source (REPL, heredoc):
fn = ts.from_text("""
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

Pass `native_required=True` to `@ts.jit` to refuse the reference
fallback path.  Any condition that would otherwise silently drop to the
numpy backend raises `tessera.compiler.TesseraNativeRequiredError` with a
stable `FallbackReason` code:

```python
from tessera.compiler import FallbackReason, TesseraNativeRequiredError

@ts.jit(target="apple_gpu", native_required=True)
def fast_matmul(a, b):
    return ts.ops.matmul(a, b)

try:
    fast_matmul(a, b)
except TesseraNativeRequiredError as exc:
    if exc.reason is FallbackReason.NON_DARWIN_HOST:
        pytest.skip("Apple GPU only")
    raise
```

Unsupported targets should be handled by checking the support table first
when possible, then letting `native_required=True` turn accidental fallback
into a typed failure:

```python
support = ts.compiler.support("matmul").for_target("apple_gpu")
if support.tier is not ts.compiler.Tier.NATIVE_READY:
    print(f"apple_gpu matmul is {support.tier.value}; using reference path")

@ts.jit(target="apple_gpu", native_required=True)
def must_be_native(a, b):
    return ts.ops.matmul(a, b)

try:
    must_be_native(a, b)
except TesseraNativeRequiredError as exc:
    assert exc.reason in {
        FallbackReason.NON_DARWIN_HOST,
        FallbackReason.CAPABILITY_NOT_READY,
        FallbackReason.REFERENCE_FORCED,
    }
```

### Per-op readiness query

`ts.compiler.support(op_name)` returns the same data the audit table
in `docs/audit/generated/support_table.md` renders, exposed as Python:

```python
info = ts.compiler.support("matmul")
info.family                                   # "loop_nest"
info.best_tier                                # Tier.NATIVE_READY
info.for_target("apple_gpu").tier             # Tier.NATIVE_READY
info.for_target("cpu").tier                   # Tier.REFERENCE_ONLY

ts.compiler.tier("matmul")               # best-tier rollup
ts.compiler.tier("matmul", target="apple_gpu")  # per-target
ts.compiler.is_native_supported("matmul", target="apple_gpu")  # True
```

Tier values: `NATIVE_READY` / `REFERENCE_ONLY` / `ARTIFACT_ONLY` / `PLANNED`.

## Region Privileges

`Region[...]` is a type annotation only. It lowers to privilege/effect attributes in Graph IR.

```python
@ts.jit
def update(
    X: ts.Region["read"],
    W: ts.Region["read"],
    Y: ts.Region["write"],
):
    Y[:] = ts.ops.gemm(X, W)
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
D = ts.domain.Rect((4, 128, 256))
dist = ts.dist.Block(mesh_axes=("dp", "tp"))
X = ts.array.from_domain(D, dtype="bf16", distribution=dist)
```

`ts.dist.Cyclic` is part of the public distribution vocabulary.  Shard
materialization remains backend- and layout-gated, so unsupported cyclic
layouts fail explicitly instead of implying native execution.  Use
`ts.compiler.support(...)`, `docs/audit/generated/support_table.md`, and
`docs/audit/generated/e2e_op_coverage.md` for the current per-target truth.

Tensor attributes are split across logical shape (`shape`), storage dtype
(`dtype`), layout (`layout`), execution target (`target`), distribution
(`ShardSpec`), and numeric policy (`numeric_policy`). Dtype strings should use
the canonical names in `docs/reference/tessera_tensor_attributes.md`; aliases
such as `"f32"` normalize before storage.

## Index Launch

`index_launch` dispatches a `@ts.kernel` function over shard lists.

```python
@ts.kernel
def tp_gemm(A: ts.f16[..., ...], B: ts.f16[..., ...], C: ts.mut_f32[..., ...]):
    C[:] = ts.ops.gemm(A, B)

ts.index_launch(axis="tp")(tp_gemm)(
    A.parts("tp"),
    B.parts("tp"),
    C.parts("tp"),
)
```

`index_launch` supports local/mock execution paths and distributed lowering
surfaces.  Hardware-backed multi-rank execution through NCCL/RCCL is
capability- and validation-gated; consult `docs/spec/VALIDATION_SPINE.md` and
`docs/spec/CONFORMANCE.md` before treating a target as production-ready.

## Constraints

Use `ts.require(...)` inside `@ts.jit` functions.

```python
@ts.jit(bindings={"K": 128})
def aligned_gemm(A: ts.Tensor["M", "K"], B: ts.Tensor["K", "N"]):
    ts.require(ts.constraint.Divisible("K", 64))
    return ts.ops.gemm(A, B)
```

Implemented constraints:

| API | Meaning |
|-----|---------|
| `ts.constraint.Divisible(dim, divisor)` | `dim % divisor == 0` |
| `ts.constraint.Range(dim, lo, hi)` | `lo <= dim <= hi` |
| `ts.constraint.Equal(dim_a, dim_b)` | `dim_a == dim_b` |

## Operations

Use `ts.ops`.

| API | Status |
|-----|--------|
| `gemm`, `matmul` | Implemented with CPU reference execution and target-gated native/artifact paths. Apple GPU native dispatch is available only where the support table reports it. |
| `layer_norm`, `softmax`, `gelu`, `relu`, `transpose`, `cast` | Implemented with CPU reference execution; selected ops also have fused/native target paths reported by the generated support table. |
| `dropout` | Implemented with a random effect in the Python/compiler surface; native lowering remains target-specific. |
| `conv2d` | Implemented — NHWC + NCHW Module forms (`tessera.nn.Conv2d` / `Conv2dNCHW`); Graph IR op + VJP/JVP registered. |
| `flash_attn` | Implemented with reference execution and NVIDIA-oriented Tile/Target artifact paths; native execution is target-gated. |
| `all_reduce`, `reduce_scatter`, `all_gather` | Implemented distributed lowering (`GPUCollectiveInsertionPass`); NCCL/RCCL adapters wired; VJP+JVP registered for all four collectives. Production multi-rank execution is validation-gated. |
| `fused_epilogue` | Implemented where supported by canonicalization/lowering; target support varies by backend capability. |

## GPU Targeting

```python
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig

@ts.jit(
    target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4),
    attn_config=FlashAttnLoweringConfig(tile_q=64, tile_kv=64, causal=True),
)
def flash_fwd(Q, K, V):
    return ts.ops.flash_attn(Q, K, V, causal=True)
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

These rows were authored as a historical roadmap. The status below is a
human-readable summary; for the current per-component picture, read
`docs/audit/generated/support_table.md` and
`docs/audit/generated/e2e_op_coverage.md` rather than this table.

| Area | Status |
|------|--------|
| NCCL/RCCL collectives + cluster execution | Implemented lowering and adapter surfaces — `GPUCollectiveInsertionPass`, NCCL/RCCL adapters, `ChunkPlanner`, `CollectiveScheduler`; production multi-rank execution remains validation-gated. |
| Autodiff transforms + custom VJP/JVP | Implemented — `tessera.autodiff` v1 ships 241 VJPs + 236 JVPs; `tessera.custom.custom_vjp` / `custom_jvp` user-facing. |
| Activation checkpointing + ZeRO sharding | Implemented — `tessera.autodiff.rematerialize` + ZeRO stage 2 via `OptimizerShardPass`. |
| Bayesian autotuning | Implemented — `tessera.autotune` + `compiler/autotune_v2.py` (Optuna TPE + Hyperband + SQLite cache v2). |
| Runtime Python wrapper | Implemented — `tessera.runtime.TesseraRuntime` over the C ABI. |
| ROCm MFMA + Apple backends | Backend surfaces implemented to varying levels; native execution is target- and validation-gated. |
