# Tessera Style Guide v2

This updated guide consolidates style conventions across the Tessera programming model, IR layers, and kernel development. It integrates lessons from the Programming Model, Standard Operations, Target IR, and System Overview docs to ensure **consistency, clarity, and IR-awareness**.

---

## Chapter 1: General Principles

### 1.1 Predictability
- Every operation should behave consistently across layers (Python → Graph IR → Schedule IR → Tile IR → Target IR).
- Names and types in user code must map directly to identifiers in IR dumps.

### 1.2 Progressive Disclosure of Complexity
- **Tier 1**: Beginner – concise, defaults hidden.
- **Tier 2**: Intermediate – optional explicit precision, tiling, and scheduling.
- **Tier 3**: Expert – full control over IR lowering and numerical policies.

### 1.3 Composition over Configuration
- Prefer compositional programming patterns (`compose`, `pipe`, decorators) over config dicts.

---

## Chapter 2: Naming Conventions

### 2.1 Functions and Kernels
- Use snake_case for Python functions (`flash_attention_kernel`).
- Use CamelCase for classes (`MultiLatentAttention`).
- Kernel names must be unique and map identically into IR (`flash_attention_kernel` → Tile IR → Target IR).

### 2.2 Shapes and Dimensions
- Symbolic names: `B`, `H`, `S`, `D` (Batch, Heads, Sequence, Dimension).
- Composite names: `BH` → `B*H`.
- Dynamic dimensions: `?`.

### 2.3 Variables
- `q`, `k`, `v`, `o` for attention tensors.
- `smem_*`, `reg_*` for memory-scoped tiles.
- Accumulators use `_acc` suffix (`out_acc`).

---

## Chapter 3: Decorators and Order

### 3.1 Standard Order
Decorators must be ordered as follows:
1. `@tessera.function`
2. `@tessera.distribute` (if applicable)
3. `@tessera.checkpoint`
4. `@tessera.compile`

Example:
```python
@tessera.compile
@tessera.checkpoint
@tessera.distribute
@tessera.function
def transformer_layer(x):
    return attention(x) + mlp(x)
```

### 3.2 Enforcement
- Linters should validate decorator order.
- Violations must raise compile-time warnings.

---

## Chapter 4: Docstrings and Documentation

### 4.1 Public API Docstrings
- High-level description for end-users.
- Example:
```python
@tessera.function
def rms_norm(x: Tensor["B", "S", "D"], weight: Tensor["D"]) -> Tensor["B", "S", "D"]:
    """
    Root Mean Square Normalization (RMSNorm).
    Efficient alternative to LayerNorm.
    """
```

### 4.2 Internal Kernel Docstrings
- Document tiling, numerical policies, and IR hints.
```python
@tessera.kernel
def rms_norm_kernel(x: Tile["S", "D", bf16], weight: Tile["D", f32]):
    """
    Fused RMSNorm kernel.
    - Tiling: [S, D] across warps
    - Precision: bf16 compute, f32 accumulate
    - Maps to Tile IR reductions + vectorized load
    """
```

### 4.3 IR-Aware Notes
- If a kernel emits special IR constructs (e.g., `tessera_target.tensor_core`), docstring must reference them explicitly.

---

## Chapter 5: Numerical Policy Annotations

### 5.1 Explicit Annotation
Always annotate storage, compute, and accumulation types:
```python
x: Tensor[B, D, Policy(storage=bf16, compute=bf16, accumulate=f32)]
```

### 5.2 Defaults
- Training default: `storage=bf16, accumulate=f32`.
- Inference default: `storage=fp8_e4m3, accumulate=bf16`.

### 5.3 Docstring Notes
- Docstrings must state non-default precision policies.

---

## Chapter 6: Distributed & Parallel Constructs

### 6.1 Data Parallelism
```python
@tessera.data_parallel(devices=["gpu:0", "gpu:1"])
def train_step(batch: Tensor["B", ...]) -> Tensor:
    return model(batch)
```

### 6.2 Mesh Parallelism
```python
mesh = tessera.Mesh(
    devices=np.array(range(16)).reshape(2, 4, 2),
    axis_names=["data", "model", "pipeline"]
)

@tessera.on_mesh(mesh)
def mesh_reduce(x: MeshTensor):
    return tessera.mesh_reduce(x, axis="data", op="mean")
```

### 6.3 Expert Parallelism
```python
@tessera.expert_parallel
def moe_forward(x: Tensor) -> Tensor:
    return mixture_of_experts(x)
```

### 6.4 Style Rule
- Mesh and distributed constructs must use **explicit axis names**.
- Device lists must be explicit and reproducible.

---

## Chapter 7: IR-Aware Coding Style

### 7.1 Graph IR Awareness
- Function parameter names should map cleanly into Graph IR node names.
- Example Graph IR snippet should preserve symbolic names (`q`, `k`, `v`).

### 7.2 Schedule IR Awareness
- Kernels must document tile sizes and autotuning search spaces.
- Example:
```python
@tessera.autotune(tile_sizes=[(64,64), (128,64)], num_warps=[4,8])
def flash_attention(...):
    ...
```

### 7.3 Tile IR Awareness
- Explicitly declare memory hierarchy usage (`alloc_shared`, `alloc_register`).
- Use consistent naming: `smem_q`, `reg_acc`.

### 7.4 Target IR Awareness
- Explicit PTX/HIP ops must include constraints in docstrings.
```mlir
%result = tessera_target.ptx_instr "wgmma.mma_async.sync.m64n256k32.f16.f16.f16"
```

---

## Chapter 8: Examples of Good Style

### 8.1 Public Function
```python
@tessera.function
def attention(q: Tensor["B","H","S","D"],
             k: Tensor["B","H","S","D"],
             v: Tensor["B","H","S","D"],
             is_causal: bool = False) -> Tensor["B","H","S","D"]:
    """
    Flash Attention with online softmax.
    - Shape polymorphic
    - Precision: bf16 storage, f32 accumulate
    """
    return tessera.ops.flash_attention(q, k, v, is_causal=is_causal)
```

### 8.2 Kernel with Full Style
```python
@tessera.kernel
def flash_attention_kernel(Q: Tile["B*H","S","D", bf16],
                           K: Tile["B*H","S","D", bf16],
                           V: Tile["B*H","S","D", bf16],
                           O: Tile["B*H","S","D", bf16],
                           is_causal: bool = False,
                           scale: float = 1.0):
    """
    Flash Attention kernel.
    - Tiling: Q[K] blocks of 128
    - Precision: bf16 compute, f32 accumulate
    - Maps to Tile IR softmax + MMA ops
    - Emits Target IR PTX `wgmma` instruction for SM_90
    """
    ...
```

---

## Conclusion

This style guide enforces consistency across the Tessera stack. By making numerical policies, decorator order, distributed constructs, and IR mappings explicit, Tessera developers ensure **predictable lowering, reproducible performance, and high-quality documentation**.
