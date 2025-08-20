# Tessera vs PyTorch vs JAX

This section compares the **Tessera programming model** with two widely used frameworks: **PyTorch** and **JAX**. The goal is to highlight conceptual differences, execution strategies, and design trade‑offs.

---

## 1. Programming Model

### Tessera

- **Operator‑centric**: Operators (dense, factorized, spectral, recursive) are first‑class citizens.
- **Graph IR**: Computations are expressed as algebraic DAGs with explicit rewrites.
- **Hilbert Space Formalism**: Operators have adjoints, spectral decompositions, and algebraic equivalences.
- **Tile abstraction**: Execution is expressed via tiles `(BM, BN, BK)` mapped to hardware warps/blocks.
- **Determinism by default**: Numerics policy specifies precision, accumulation type, and reproducibility.

### PyTorch

- **Tensor‑centric**: Tensors and imperative APIs (`torch.nn.Module`) dominate.
- **Dynamic graph (eager mode)**: Graphs are built implicitly during execution.
- **Operators**: Rich but defined as functions over tensors, not algebraic objects.
- **Parallelism**: Mostly implicit; CUDA kernels are dispatched through ATen.
- **Determinism**: Available but opt‑in and not universal.

### JAX

- **Function‑centric**: Pure functions + transformations (`jit`, `grad`, `vmap`, `pmap`).
- **Tracing model**: Code is traced to XLA HLO for compilation.
- **Operators**: Primitive functions lowered to XLA ops; no explicit operator algebra.
- **Parallelism**: Explicit `pmap`/`pjit` for SPMD; XLA compiler handles tiling.
- **Determinism**: Stronger than PyTorch, but still subject to backend constraints.

---

## 2. Execution Flow

### Tessera

1. **Operator Graph IR** — DAG of algebraic operators.
2. **Rewrite/Planner** — algebraic simplifications (e.g., FFT∘IFFT → Identity).
3. **Scheduler/Tiler** — explicit tile‑level execution planning.
4. **Codegen** — lowers into backend kernels.

### PyTorch

1. Imperative ops execute directly in eager mode.
2. Optional TorchScript/FX for graph capture.
3. CUDA dispatch through ATen, cuDNN, or custom kernels.
4. No explicit tiling — left to libraries (CUTLASS, cuDNN).

### JAX

1. User function traced to XLA HLO.
2. XLA compiler applies fusions and optimizations.
3. Backend generates kernels.
4. Execution via device runtime.

---

## 3. Hardware Abstraction

### Tessera

- Explicit **ABI runtime**: `tessera_launch_tile`, memory handles, streams, events.
- Portable to CPUs, GPUs, or custom accelerators.
- Binary format similar to ELF with `.tessera.ops` + metadata.

### PyTorch

- Backend dispatch to CUDA, ROCm, XPU, or CPU.
- ABI hidden from users; C++/CUDA extension mechanism available.
- Execution closely tied to vendor libraries.

### JAX

- Single abstraction layer: XLA handles device codegen.
- Portable across TPU, GPU, CPU.
- No user‑visible ABI; relies on XLA runtime.

---

## 4. Differentiation & Training

### Tessera

- **Adjoint operators**: Differentiation is algebraically exact at IR level.
- **Custom differentiation rules** integrated into operator definitions.
- Supports **RLHF, MoE, spectral learning** as first‑class patterns.

### PyTorch

- **Autograd engine**: Dynamic tape‑based differentiation.
- Good for imperative programming and debugging.
- Less structured algebraic manipulation.

### JAX

- ``** / **``** tracing**: Transformation‑based autodiff.
- Strong composability with functional style.
- Algebraic reasoning limited to what XLA supports.

---

## 5. Strengths & Weaknesses

| Feature              | Tessera                    | PyTorch            | JAX                  |
| -------------------- | -------------------------- | ------------------ | -------------------- |
| Operator algebra     | ✅ explicit (Hilbert space) | ❌ tensor ops only  | ❌ primitives only    |
| Tiling control       | ✅ explicit                 | ❌ hidden           | ⚠️ implicit in XLA   |
| Determinism          | ✅ first‑class              | ⚠️ partial         | ⚠️ backend‑dependent |
| Eagerness            | ❌ graph‑first              | ✅ eager by default | ❌ compiled‑first     |
| Differentiation      | ✅ operator adjoints        | ✅ autograd         | ✅ tracing grad       |
| Hardware abstraction | ✅ ABI & IR                 | ⚠️ vendor libs     | ✅ XLA portable       |

---

## 6. Summary

- **Tessera**: Suited for research and deployment of **large, structured ML systems** requiring **operator algebra, determinism, and explicit control** over tiling and scheduling.
- **PyTorch**: Best for **developer productivity, prototyping, and flexible model authoring** with a rich ecosystem.
- **JAX**: Best for **functionally pure research workflows** and scaling to TPU/GPU clusters with **XLA compilation**.

Tessera can be seen as a **new layer beneath PyTorch/JAX**, providing algebraic rigor and hardware‑deterministic execution, while PyTorch and JAX provide higher‑level ergonomics and ecosystems.

---

## 7. Distributed Tensor Example in Tessera

Tessera supports **distributed tensors** as first-class objects in the Graph IR. A distributed tensor represents a logical array whose physical shards are partitioned across multiple devices or nodes.

### Definition

```python
from tessera import dist, op

# Create a distributed tensor of shape [8192, 8192] partitioned by rows across 4 GPUs
X = dist.tensor(shape=(8192, 8192), policy="row", devices=[0,1,2,3], dtype="bf16")

# Apply an operator (e.g., matmul) that respects distribution
Y = op.matmul(X, X.T)
```

### Lowering Flow

1. **Graph IR**: Marks `X` with distribution metadata (row-sharded).
2. **Rewrite**: Ensures that transpose/adjoint is communicated across shards.
3. **Tile IR**:
   ```mlir
   %x_shard = tile.load_shard %X {shard=row, device=di}
   %y_local = tile.matmul %x_shard, %x_shard^T
   %y = tile.allreduce %y_local {op=add, devices=[0,1,2,3]}
   ```
4. **ABI Mapping**:
   - `tessera_dist_handle` created for `X`.
   - `tessera_launch_tile` invoked per-device.
   - `tessera_allreduce` used for final result aggregation.

### Benefits

- Distribution strategy is **explicit and programmable**.
- Built-in collectives (`allreduce`, `broadcast`, `scatter`, `gather`) integrate into Tile IR.
- Ensures reproducible execution across devices with consistent numerics policies.

---

## 8. Physics-Informed Neural Network (PINN) Example in Tessera

Tessera’s operator algebra is well-suited to **PINNs**, which blend data loss with PDE residual loss.

### Problem: 1D Heat Equation

\(u_t = \alpha u_{xx}\)

### Tessera Implementation

```python
from tessera import op, graph

alpha = 0.01

with graph.module("pinn_heat") as G:
    # Inputs: spatial coords x, time t
    x = op.placeholder((B,1), name="x")
    t = op.placeholder((B,1), name="t")

    # Neural net approximation u(x,t)
    u = op.mlp([x,t], layers=[64,64,1], activation="tanh")

    # Derivatives via operator adjoints
    u_t  = op.grad(u, t)
    u_x  = op.grad(u, x)
    u_xx = op.grad(u_x, x)

    # PDE residual: u_t - alpha * u_xx ≈ 0
    residual = u_t - alpha * u_xx

    # Physics loss: mean squared residual
    L_phys = op.mean(residual**2)

    # Data loss (if supervised samples available)
    u_true = op.placeholder((B,1), name="u_true")
    L_data = op.mean((u - u_true)**2)

    # Total loss
    L_total = L_data + L_phys

    G.output("loss", L_total)
```

### Execution Flow

1. **Operator Graph IR**: Combines neural network with PDE residual operators.
2. **Rewrite**: Simplifies differential operator chains (e.g., grad∘grad → Laplacian).
3. **Tile IR**: Lowers network layers to fused matmuls; derivatives lower to automatic adjoint ops.
4. **ABI**: Runtime launches GEMM tiles + elementwise ops, with reproducible accumulation.

### Benefits of Tessera for PINNs

- **Adjoint operators** natively handle gradients, Laplacians, and PDE residuals.
- Physics terms integrate directly with data loss inside one graph.
- Deterministic tiling ensures reproducible training of physics-informed systems.



---

## 8. Physics-Informed Neural Networks (PINNs) in Tessera

This section shows how Tessera expresses a Physics-Informed Neural Network (PINN) by representing differential operators and PDE residuals as first-class operators. We implement a 1D viscous Burgers' equation example:

u\_t + u*u\_x - nu*u\_xx = 0 on (x,t) in [0,1]x[0,1], with u(x,0)=u0(x), u(0,t)=a(t), u(1,t)=b(t).

### 8.1 Model & Operators (User-Level API)

```python
from tessera import op, graph, train, runtime as rt

# Numerics policy for stable PDE terms
POLICY = op.numeric(dtype="bf16", accum="f32", deterministic=True)

# Neural field u(x,t; theta): MLP with sinusoidal features (SIREN optional)
Field = op.mlp(in_dim=2, hidden=[256,256,256], out_dim=1,
               activation="sine", policy=POLICY, name="u_theta")

# Differential operators (symbolic; lower to autodiff + fused kernels)
Dx  = op.grad(axis=0)        # d/dx
Dt  = op.grad(axis=1)        # d/dt
Dxx = op.grad(axis=0, order=2)  # d2/dx2 (Laplacian in 1D)

# PDE residual operator R[u] = u_t + u*u_x - nu*u_xx
nu = 0.01

def residual(x_t):
    u     = Field(x_t)                 # [N,1]
    ux    = Dx(Field)(x_t)             # du/dx evaluated via operator adjoint rules
    ut    = Dt(Field)(x_t)             # du/dt
    uxx   = Dxx(Field)(x_t)            # d2u/dx2
    R     = ut + u*ux - nu*uxx
    return R, u

# Boundary/initial condition operators
u0   = op.dataset("u0_init")           # provides u(x,0)
left  = op.dataset("u_left")           # u(0,t)
right = op.dataset("u_right")          # u(1,t)

BC_penalty = 10.0
IC_penalty = 10.0

@graph.training_step(module="PINN_Burgers")
def step(batch):
    # Collocation points for PDE interior, boundary, and initial line
    X_interior = batch["x_t_interior"]   # shape [Ni,2] (x,t)
    X_ic       = batch["x_t_ic"]         # [Nc,2], with t=0
    X_left     = batch["x_t_left"]       # [Nb,2], with x=0
    X_right    = batch["x_t_right"]      # [Nb,2], with x=1

    # PDE residual loss (interior)
    R, _  = residual(X_interior)
    L_pde = op.mean(op.square(R))

    # Initial condition loss
    _, u_ic = residual(X_ic)
    L_ic = op.mean(op.square(u_ic - nu0(X_ic)))

    # Boundary losses
    _, u_l = residual(X_left)
    _, u_r = residual(X_right)
    L_bc = op.mean(op.square(u_l - left(X_left))) \
         + op.mean(op.square(u_r - right(X_right)))

    # Total loss
    L = L_pde + IC_penalty*L_ic + BC_penalty*L_bc

    # AD on operator graph yields exact PDE gradients via adjoints
    grads = graph.backward(L)
    return grads, {"L": L, "L_pde": L_pde, "L_ic": L_ic, "L_bc": L_bc}

trainer = train.Trainer(optimizer=train.AdamW(lr=1e-3), numerics=POLICY)
trainer.fit(step, dataloader)
```

Notes:

- Dx, Dt, Dxx are operators; calling Dx(Field) returns a new operator whose adjoint rules drive automatic differentiation.
- The PDE residual is built symbolically and lowered to fused kernels.
- Determinism is ensured via the
