# Tessera Tutorials Volume
## Chapter 4 — Distributed Training

### 4.1 Defining a Distributed Mesh
Tessera uses **meshes** to define distributed device topologies.  
A mesh can be 1D (data parallel), 2D (tensor + data), or 3D (pipeline + tensor + data).

```python
import tessera as ts
from tessera import dist

# Create a 2D mesh: tensor-parallel × data-parallel
mesh = dist.Mesh(axes=["tp", "dp"], devices=[[0,1,2,3],[4,5,6,7]])
```

---

### 4.2 Sharding Tensors Across Devices
Tensors can be automatically sharded across the mesh with a **ShardSpec**.

```python
# Shard a large embedding table across tensor-parallel axis
E = dist.tensor(
    shape=(10_000_000, 4096),
    layout=dist.ShardSpec(partition=("row",), mesh_axes=("tp",)),
    mesh=mesh,
    dtype="bf16"
)
```

---

### 4.3 Collectives
Tessera provides optimized collectives (all-reduce, scatter, gather, pipeline send/recv) mapped to NVLink/NVSwitch or InfiniBand.

```python
from tessera import op

@op.kernel
def dp_step(x):
    # Local gradient
    grad = op.matmul(x, x.T)
    # Global sync across data-parallel axis
    grad = dist.allreduce(grad, axis="dp")
    return grad
```

---

### 4.4 Pipeline Parallelism
Pipeline stages can be defined explicitly with **graph partitioning**.

```python
from tessera import graph

@graph.stage("encoder")
def encoder(x): return op.transformer_block(x, depth=12)

@graph.stage("decoder")
def decoder(x): return op.transformer_block(x, depth=12)

@graph.pipeline(stages=[encoder, decoder], mesh=mesh, axis="pp")
def model(x): return decoder(encoder(x))
```

---

### 4.5 Training Loop
A distributed training loop looks the same as a single-device loop. Tessera automatically inserts communication.

```python
@graph.training_step(module="DistTransformer")
def step(batch):
    out = model(batch["input"])
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```

---

### 4.6 Debugging Distributed Execution
- Use `dist.inspect(tensor)` to see sharding layouts.  
- `graph.trace(model, batch)` shows IR lowering across devices.  
- `dist.profile()` reports communication/computation balance.  
