# Tessera Programming Guide  
## Chapter 10: Portability (Updated)

Tessera is designed for **performance portability**. The same program can run on NVIDIA GPUs today, and future extensions will target AMD, Intel, and other accelerators. Portability is achieved through a **multi-level IR stack**, a **Mapper API**, and integration with vendor libraries.

---

### 10.1 Multi-Level IR Stack

Tessera lowers code through multiple IR layers:

- **Graph IR**: high-level autodiff, algebra, and effect representation.  
- **Schedule IR**: tiling, fusion, pipelining, layout transforms.  
- **Tile IR**: explicit tile loads/stores, shared memory, mma intrinsics.  
- **Target IR**: backend-specific lowering (PTX, CUDA Tile IR, LLVM).  

This design ensures that high-level programs remain portable, while low-level optimizations adapt per-architecture.

---

### 10.2 NVIDIA GPUs

On NVIDIA accelerators (Ampere, Hopper, Blackwell):

- **PTX backend**: traditional lowering path.  
- **CUDA Tile IR backend**: preferred on Hopper/Blackwell.  
- **WMMA/WGMMA intrinsics**: tensor core operations.  
- **cp.async / TMA**: async memory movement.  
- **NCCL**: collectives across NVLink/NVSwitch.  

#### Example: Inspecting IR
```python
kernel = gemm_tile.jit(M=4096,N=4096,K=4096)
print(kernel.inspect_ir("tile"))     # Portable Tile IR
print(kernel.inspect_ir("target"))   # NVIDIA PTX / CUDA Tile IR
```

---

### 10.3 AMD GPUs (Future Work)

- **ROCm + XDLops**: target for tensor core equivalents.  
- **rccl**: collective backend.  
- **Unified memory (HMM)** support.  

---

### 10.4 Intel GPUs (Future Work)

- **oneAPI Level Zero + DPAS intrinsics**.  
- **oneCCL**: collective backend.  
- **SYCL-based interop** for kernel launches.  

---

### 10.5 Mapper API for Portability

Programmers can override placement and lowering decisions with a **Mapper**:

```python
class MyMapper(tessera.Mapper):
    def place(self, region, mesh):
        if region.axis == "tp":
            return "local_switch"
    def choose_collective(self, kind, size):
        return "sharp_ring" if size <= 72 else "tree"
    def choose_variant(self, op, arch):
        return "tile_ir" if arch.is_blackwell else "ptx"

tessera.runtime.set_mapper(MyMapper())
```

- Ensures optimal policies on NVL72.  
- Can be extended for AMD/Intel targets in the future.  

---

### 10.6 Domains and Distributions in Portability

Domains and distributions are portable constructs:

```python
D = tessera.domain.Rect((B, S, Dm))
dist = tessera.dist.Block(mesh_axes=("dp","tp"))
X = tessera.array.from_domain(D, dtype="bf16", distribution=dist)
```

- On NVIDIA: lowers to NCCL collectives.  
- On AMD: lowers to rccl.  
- On Intel: lowers to oneCCL.  

The same Tessera program runs across vendors.

---

### 10.7 Index Launch Portability

`index_launch` distributes kernels across mesh partitions:

```python
tessera.index_launch(axis="tp")(gemm_tile)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

- On NVIDIA: maps to NVLink/NVSwitch with NCCL.  
- On AMD: maps to xGMI/Infinity Fabric with rccl.  
- On Intel: maps to oneCCL collectives.  

---

### 10.8 NVL72 as a Portability Case Study

NVL72 demonstrates Tessera’s philosophy:

- Treats **72 GPUs as one device**.  
- Mapper co-locates tensor-parallel ranks on NVSwitch groups.  
- Collectives map to **NCCL with SHARP reductions**.  
- CUDA Graphs minimize per-launch overhead.  

The same code can run on **smaller NVIDIA clusters** or **future AMD/Intel systems** with no changes.

---

### 10.9 Summary

- Tessera achieves portability through **multi-level IR** and **Mapper API hooks**.  
- NVIDIA support is mature (PTX + CUDA Tile IR, NCCL).  
- AMD and Intel support planned (ROCm/XDLops/rccl, oneAPI/DPAS/oneCCL).  
- Domains, distributions, and index launches are portable abstractions.  
- NVL72 illustrates how Tessera adapts to extreme-scale NVIDIA systems.  

Programmers write once and run anywhere, with the compiler and runtime adapting to each backend.
