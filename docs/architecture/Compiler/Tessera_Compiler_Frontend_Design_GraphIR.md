# Tessera Compiler Frontend Design — Python Surface to Graph IR
**Version:** 2.0  
**Date:** April 26, 2026  
**Status:** Informative — design rationale and implementation detail for `python/tessera/compiler/`  
**Audience:** Compiler engineers working on Phase 4+, contributors extending `@jit`

---

## 0. Goals and Non-Goals

**Goals**
- Document how the Python surface (`@tessera.jit`, `Region[...]`, `tessera.domain`, `index_launch`) lowers to Graph IR.
- Explain the `@jit` decoration sequence: ConstraintSolver, EffectLattice, GraphIRBuilder.
- Provide design rationale for key decisions (Python-as-frontend, annotation-not-wrapper, decoration-time-checking).
- Serve as the design reference for engineers extending the Python frontend in Phase 4+.

**Non-Goals**
- Schedule IR, Tile IR, Target IR lowering — covered in `docs/spec/LOWERING_PIPELINE_SPEC.md` and `docs/spec/TARGET_IR_SPEC.md`.
- Runtime execution, NCCL, CUDA Graphs — Phase 4+.
- Full public API reference — that is `docs/spec/PYTHON_API_SPEC.md`.

---

## 1. Frontend Architecture

### 1.1 What the frontend is

The Tessera frontend is **pure Python**. There is no separate language, no `.tss` file format, no Rust parsing layer, and no tracing JIT. Python's interpreter executes user code normally; `@tessera.jit` intercepts functions through Python's standard decorator protocol.

This is a deliberate, locked decision. The Python surface is a clean API with type annotations, not a DSL file format. The performance-critical stages (MLIR pass pipeline, x86 AMX kernels, WGMMA PTX emission) live in C++/MLIR. The frontend boundary is precisely: Python decorator → Graph IR MLIR text.

Key files:

| File | Responsibility |
|------|---------------|
| `python/tessera/compiler/jit.py` | `@tessera.jit`, `@tessera.kernel`, `JitFn`, `KernelFn`, `_ConstraintExtractor` |
| `python/tessera/compiler/constraints.py` | `ConstraintSolver`, `Divisible`, `Range`, `Equal`, `TesseraConstraintError` |
| `python/tessera/compiler/effects.py` | `Effect` enum, `EffectLattice`, `TesseraEffectError` |
| `python/tessera/compiler/graph_ir.py` | `GraphIRBuilder` — emits `tessera` dialect MLIR text |
| `python/tessera/compiler/gpu_target.py` | `GPUTargetProfile`, `ISA` enum |
| `python/tessera/compiler/attn_lower.py` | `FlashAttnLoweringConfig`, `SM90_DEFAULT` |

### 1.2 Pipeline overview

```text
@tessera.jit def fn(W: Region["read"], X: Region["read"], Y: Region["write"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    Y[:] = tessera.ops.gemm(X, W)
    │
    ├── Step 1: Annotation inspection     (fn.__annotations__ → RegionType objects)
    ├── Step 2: AST constraint extraction (_ConstraintExtractor → list[Constraint])
    ├── Step 3: Constraint checking       (ConstraintSolver.check_all(bindings))
    ├── Step 4: Effect inference          (EffectLattice.infer(fn) → Effect)
    ├── Step 5: Effect validation         (deterministic=True check)
    └── Step 6: Graph IR emission         (GraphIRBuilder.build(fn) → MLIR text)
         │
         ▼
    module @fn attributes {tessera.version = "1.0"} {
      func.func @fn(%W: tensor<256x256xbf16> {tessera.effect = "read"},
                    %X: tensor<128x256xbf16> {tessera.effect = "read"},
                    %Y: tensor<128x256xf32>  {tessera.effect = "write"}) {
        %r = tessera.matmul %X, %W : ...
        ...
      }
    }
```

---

## 2. `@tessera.jit` Decoration Sequence

### Step 1 — Annotation inspection

`jit.py` reads `fn.__annotations__` to discover parameter types. For each parameter:

- `Region["read"]` → `RegionType(mode="read", exclusive=False, reduces=False)`
- `Region["write"]` → `RegionType(mode="write", exclusive=True, reduces=False)`
- `Region["reduce_sum"]` → `RegionType(mode="reduce_sum", exclusive=False, reduces=True, op="sum")`
- `tessera.Tensor["M", "K"]` → `TensorType(dim_names=("M", "K"))`
- Any other annotation → passed through without tessera validation

Privilege conflict detection happens here: if two `Region["write"]` annotations appear on parameters that the body assigns simultaneously, `TesseraPrivilegeError` is raised immediately.

### Step 2 — AST constraint extraction

`_ConstraintExtractor` is an `ast.NodeVisitor` subclass that walks `fn`'s AST looking for calls of the form `tessera.require(...)`. For each such call, it extracts the `Constraint` object passed as the argument.

This is why `tessera.require(...)` is a **no-op at Python runtime** — the actual constraint checking never happens when the function body executes. The extractor finds the calls statically from the AST before any execution.

Supported constraint forms recognised by the extractor:
```python
tessera.require(tessera.constraint.Divisible("K", 64))
tessera.require(tessera.constraint.Range("M", 1, 8192))
tessera.require(tessera.constraint.Equal("D_in", "D_out"))
```

Non-tessera function calls in the body are ignored by the extractor.

### Step 3 — Constraint checking

If `bindings=` was supplied to `@jit` (e.g. `bindings={"K": 128}`), `ConstraintSolver.check_all(bindings)` is called immediately. Each extracted constraint is evaluated against the provided concrete dimension values.

- `Divisible("K", 64)` with `bindings={"K": 100}` → raises `TesseraConstraintError("K=100 is not divisible by 64")`
- If `bindings` is `None` (default), constraint checking is deferred to the first call where tensor shapes are known.

The `ConstraintSolver` stores the extracted constraints on the `JitFn` wrapper. They remain accessible via `fn.constraints` for downstream passes and diagnostics.

### Step 4 — Effect inference

`EffectLattice` walks the function body (again via AST) and infers the function's effect level by inspecting which `tessera.ops.*` are called:

| Op | Effect |
|----|--------|
| `gemm`, `matmul`, `layer_norm`, `softmax`, `gelu`, `relu`, `transpose`, `cast`, `flash_attn` | `pure` |
| `dropout` (when `training=True`) | `random` |
| `all_reduce`, `reduce_scatter`, `all_gather` | `io` |

The lattice join rule: `effect_a.join(effect_b) = max(a.value, b.value)`. A function calling both `gemm` (pure=0) and `dropout` (random=1) gets `random`.

The inferred `Effect` is stored on the `JitFn` wrapper as `fn.effect`.

### Step 5 — Effect validation

If `@jit(deterministic=True)` was set:
- Inferred effect is `random` AND `seed=None` → raises `TesseraEffectError`
- Inferred effect is `random` AND `seed=42` → allowed (RNG is seeded and deterministic)
- Inferred effect is `pure`, `memory`, or `io` → always allowed under `deterministic=True` (only `random` is restricted)

### Step 6 — Graph IR emission

`GraphIRBuilder.build(fn)` emits an MLIR module in tessera dialect text format. The emitter:

1. Opens a `module @<fn_name> attributes {tessera.version = "1.0"}` block.
2. Opens a `func.func @<fn_name>` with arguments typed from the function's annotations.
   - `Region["read"]` arg → `tensor<...> {tessera.effect = "read"}`
   - `Region["write"]` arg → `tensor<...> {tessera.effect = "write"}`
   - `DistributedArray` arg → `tensor<...> {tessera.shard = {axes=[...], dims=[...]}}`
3. Walks the function body calls and emits corresponding tessera ops:
   - `tessera.ops.gemm(X, W)` → `%r = tessera.matmul %X, %W : ...`
   - `tessera.ops.flash_attn(Q, K, V, causal=True)` → `%o = tessera.flash_attn %Q, %K, %V {head_dim=..., causal=true, ...} : ...`
   - `tessera.ops.gelu(x)` → `%g = tessera.gelu %x : ...` (canonicalized away later)
4. Attaches `tessera.effect = "<level>"` to the `func.func` based on the inferred effect.
5. Returns a `GraphIRBuilder` instance stored as `fn.graph_ir`. Call `.to_mlir()` to get the MLIR text string.

**Tile size attributes for FlashAttention:** If `attn_config` was provided (or auto-set from `SM90_DEFAULT` when `target.isa >= ISA.SM_90`), `FlashAttnLoweringConfig.to_mlir_attrs()` injects `tessera.tile_q`, `tessera.tile_kv`, and `tessera.pipeline_stages` onto the emitted `tessera.flash_attn` op. These attributes are read by `TileIRLoweringPass` and persist through to the autotuner.

---

## 3. `@tessera.kernel`

`@tessera.kernel` is a simpler decorator — it wraps the function in a `KernelFn` without running ConstraintSolver or EffectLattice. Kernel functions are not called directly; they are passed to `index_launch`.

```python
@tessera.kernel
def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...], C: tessera.mut_f32[..., ...]):
    C[:] = tessera.ops.gemm(A, B)
```

Dtype annotations (`f16`, `bf16`, `f32`, `mut_f32`) on kernel parameters carry both the element type and the read/write privilege. They use Python's `__class_getitem__` protocol; `tessera.f16[..., ...]` returns a `DtypeAnnotation` object.

---

## 4. ConstraintSolver Design

### 4.1 Purpose

The ConstraintSolver enforces structural properties of a function's type signature — dimension relationships that must hold for the computation to be correct — **at decoration time** rather than call time. This gives early, precise error messages before any execution.

### 4.2 Constraint types

All three constraint types are simple dataclasses:

```python
@dataclass
class Divisible:
    dim: str       # symbolic dimension name, e.g. "K"
    divisor: int   # e.g. 64

@dataclass
class Range:
    dim: str
    lo: int        # inclusive lower bound
    hi: int        # inclusive upper bound

@dataclass
class Equal:
    dim_a: str
    dim_b: str
```

### 4.3 Checking

`ConstraintSolver.check(bindings: dict[str, int])` evaluates one constraint against a concrete dim map. `check_all(bindings)` evaluates all added constraints in order, stopping at the first violation.

When `bindings` lacks a dimension referenced by a constraint, that constraint is skipped (deferred to a later call where the size is known).

### 4.4 Deferred checking

If `bindings=None` at decoration time, the `JitFn` stores the extracted constraints and checks them on the first call where the actual tensor shapes are available. This allows:

```python
@tessera.jit   # no bindings — checked at call time
def fn(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    tessera.require(tessera.constraint.Divisible("K", 64))
    return tessera.ops.gemm(A, B)

fn(A_128, B_128)   # K=128 → 128 % 64 == 0 → passes
fn(A_100, B_100)   # K=100 → 100 % 64 != 0 → TesseraConstraintError
```

---

## 5. EffectLattice Design

### 5.1 Lattice

Effects form a partial order with join (least upper bound):

```
pure(0) < random(1) < memory(2) < io(3) < top(4)
```

`join(a, b) = max(a.value, b.value)` — a function's effect is the maximum of all its sub-expressions' effects.

### 5.2 Why `random` is the restricted level

`deterministic=True` only restricts `random` effects, not `memory` or `io`. The reasoning:

- `random` effects produce different outputs on repeated calls with the same inputs. This directly violates reproducibility.
- `memory` effects (e.g. KV cache writes) are side-effecting but deterministic given the same execution order.
- `io` effects (collectives) are deterministic in practice for the same mesh configuration.
- Only unseeded `random` genuinely threatens reproducibility.

### 5.3 Inference from tessera.ops

The EffectLattice walks the function's AST and maps `tessera.ops.*` calls to their effect levels. Calls to non-tessera functions are treated as `io` (external, unknown). The walk is conservative: if a sub-function's effect level cannot be determined, `top` is assumed.

---

## 6. GraphIRBuilder Design

### 6.1 Role

`GraphIRBuilder` is a single-purpose emitter: it takes a Python function (post annotation-inspection, post constraint extraction) and emits the corresponding `tessera` MLIR dialect text. It does **not** parse existing MLIR — it writes new MLIR.

### 6.2 Op mapping

The builder maps Python-level `tessera.ops.*` calls to MLIR op names:

| Python | Graph IR op |
|--------|-------------|
| `tessera.ops.gemm(A, B)` | `tessera.matmul` |
| `tessera.ops.matmul(A, B)` | `tessera.matmul` (alias) |
| `tessera.ops.conv2d(x, w, ...)` | `tessera.conv2d_nhwc` |
| `tessera.ops.flash_attn(Q, K, V, ...)` | `tessera.flash_attn` |
| `tessera.ops.gelu(x)` | `tessera.gelu` (fused away by canonicalization) |
| `tessera.ops.relu(x)` | `tessera.relu` (fused away by canonicalization) |
| `tessera.ops.transpose(x)` | `tessera.transpose` (folded into matmul by canonicalization) |
| `tessera.ops.cast(x, dtype)` | `tessera.cast` |
| `tessera.ops.layer_norm(x)` | `tessera.layer_norm` |
| `tessera.ops.softmax(x)` | `tessera.softmax` |
| `tessera.ops.dropout(x, p)` | `tessera.dropout` |
| `tessera.ops.all_reduce(x)` | `tessera.collective.all_reduce` (Phase 4 stub) |

### 6.3 Shard attribute emission

When a parameter is a `DistributedArray`, `GraphIRBuilder` reads its `.shard_spec` and emits the corresponding `tessera.shard` attribute on the `func.func` argument:

```python
# Python
X = tessera.array.from_domain(Rect((128, 256)), dtype="bf16",
                               distribution=Block(mesh_axes=("dp",)))
@tessera.jit
def fn(X_arg: tessera.Region["read"]): ...

# Emitted Graph IR argument attribute
%X: tensor<128x256xbf16> {tessera.shard = {axes = ["dp"], dims = [0]},
                           tessera.effect = "read"}
```

### 6.4 GPU target routing

When `target=GPUTargetProfile(isa=ISA.SM_90)` is provided:
- `GraphIRBuilder` attaches `tessera.target_sm = 90 : i32` to the module.
- If `attn_config` is set (or auto-selected as `SM90_DEFAULT`), `FlashAttnLoweringConfig.to_mlir_attrs()` injects the tile size attributes onto `tessera.flash_attn` ops.
- These attributes are what `TileIRLoweringPass` reads when deciding tile sizes and whether to emit `causal_mask` / `dropout_mask` ops.

---

## 7. Distribution and Domain API

### 7.1 Separation of concerns

`tessera.domain.Rect` (shape) and `tessera.dist.Block/Replicated` (placement) are always separate objects. `tessera.array.from_domain` combines them into a `DistributedArray` carrying a `ShardSpec`.

This separation is intentional: algorithms should be written against logical shapes; placement strategies can be changed independently without touching the algorithm.

### 7.2 ShardSpec

`ShardSpec(partition=(0,), mesh_axes=("dp",))` encodes: "dim 0 is partitioned over the `dp` mesh axis." The `GraphIRBuilder` converts this to `tessera.shard = {axes = ["dp"], dims = [0]}` on the `func.func` argument attribute.

`DistributionLoweringPass` later reads these attributes and emits `schedule.mesh.define {dims=[N], axis_names=["dp"]}` + `schedule.mesh.region {mesh=@dp, axis="dp"}`.

### 7.3 `index_launch`

`index_launch(axis="tp")(my_kernel)(A.parts("tp"), B.parts("tp"), C.parts("tp"))` is the primary multi-rank dispatch mechanism. In Phase 1, this runs sequentially (one Python thread per rank). In Phase 3+, this maps to parallel GPU stream dispatch.

The three-step call pattern:

```python
tessera.index_launch(axis="tp")   # → IndexLauncher(axis="tp")
    (my_kernel)                   # → _ShardDispatcher(launcher, kernel)
    (shards_A, shards_B, shards_C) # → executes kernel once per rank
```

---

## 8. Error Taxonomy

### At decoration time

| Error | Cause |
|-------|-------|
| `TesseraConstraintError` | Constraint violated and `bindings=` provided concrete values. E.g. `Divisible("K", 64)` with `K=100`. |
| `TesseraEffectError` | `deterministic=True` + unseeded `random` op (e.g. `dropout` without `seed`). |
| `TesseraPrivilegeError` | Two `Region["write"]` annotations on the same tensor. |
| `TesseraTargetError` | Invalid `GPUTargetProfile` (e.g. `warps_per_cta` not a power of 2). |
| `TesseraAttnConfigError` | Invalid `FlashAttnLoweringConfig` (e.g. non-power-of-2 tile size). |
| `TesseraJitError` | `GraphIRBuilder` fails to emit valid IR (unsupported op, internal error). |

### At call time (deferred constraint checking)

| Error | Cause |
|-------|-------|
| `TesseraConstraintError` | Constraint violated with actual tensor shapes (when `bindings=None` at decoration time). |

### Common mistakes and fixes

| Symptom | Fix |
|---------|-----|
| `TesseraEffectError: deterministic=True but function has random effect` | Add `seed=42` to `@jit(deterministic=True, seed=42)`. |
| `TesseraConstraintError: K=100 not divisible by 64` | Use a K dimension that is a multiple of 64, or remove the `Divisible("K", 64)` constraint. |
| `TesseraPrivilegeError: conflicting write regions` | Change one `Region["write"]` to `Region["reduce_sum"]` if accumulation is intended. |
| `ValueError: invalid Region mode "readwrite"` | Use separate params: one `Region["read"]` and one `Region["write"]`. |

---

## 9. Inspection

The correct way to inspect Graph IR emitted by `@jit` is:

```python
@tessera.jit
def step(W: tessera.Region["read"], X: tessera.Region["read"], Y: tessera.Region["write"]):
    Y[:] = tessera.ops.gemm(X, W)

# Get MLIR text
ir_text = step.graph_ir.to_mlir()
print(ir_text)
```

Example output:
```mlir
module @step attributes {tessera.version = "1.0"} {
  func.func @step(
      %W: tensor<256x256xbf16> {tessera.effect = "read"},
      %X: tensor<128x256xbf16> {tessera.effect = "read"},
      %Y: tensor<128x256xf32>  {tessera.effect = "write"}
  ) attributes {tessera.effect = "memory"} {
    %r = tessera.matmul %X, %W : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xf32>
    return
  }
}
```

**Note:** The method `inspect_ir(stage)` shown in older documents does not exist. The `.graph_ir.to_mlir()` pattern is the canonical inspection mechanism.

---

## 10. Authoritative References

| Topic | Document |
|-------|----------|
| Full public API signatures | `docs/spec/PYTHON_API_SPEC.md` |
| Canonical API names | `docs/CANONICAL_API.md` |
| Graph IR op catalog | `docs/spec/GRAPH_IR_SPEC.md` |
| C++ pass pipeline | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Effect system design | `src/programming_model/docs/Tessera_Programming_Model_v1_1_Plan_20250917_212640.md §1.2` |
| Constraint design | Same doc §1.1 |
| `@jit` source | `python/tessera/compiler/jit.py` |
| `ConstraintSolver` source | `python/tessera/compiler/constraints.py` |
| `EffectLattice` source | `python/tessera/compiler/effects.py` |
| `GraphIRBuilder` source | `python/tessera/compiler/graph_ir.py` |
