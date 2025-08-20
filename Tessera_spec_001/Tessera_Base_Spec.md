# Tessera Unified Specification (Base Spec)

This document defines the **Tessera programming model** and the **ABI runtime specification**. 
It is the normative base spec, without worked operator examples (those are in the separate Examples document).

---

# Volume I: Programming Model (Master Reference Manual)

## Chapter 1: Overview

The Tessera Programming Model defines a unified way to express deep learning and operator-based computation across heterogeneous hardware. Tessera emphasizes:

- **Operators** as first-class citizens that model linear, nonlinear, and spectral transforms.
- **Execution in Hilbert-space algebra** for mathematical clarity and symbolic reasoning.
- **Tile-based scheduling** for predictable, high-throughput execution.
- **Graph-level optimization** for fusion, spectral rewrites, and operator algebra transformations.

---

## Chapter 2: Execution Flow

The **execution model** is layered:

1. **Operator Graph IR**: Represents algebraic computation (dense, factorized, implicit operators).
2. **Planner/Rewrite Stage**: Applies algebraic simplifications (e.g., adjoint laws, composition rules).
3. **Scheduler/Tiler**: Selects schedules and tiling parameters.
4. **Codegen**: Produces target-specific kernels.

**Figure E.1: Execution Flow** — IR → Rewrites → Plans → Tiles → Codegen.

**Figure E.2: Factorized Operator Apply** — Factorized forms lower to multiple GEMMs with fusion opportunities.

---

## Chapter 3: Extended Diagrams

- **Figure F.1: Memory Hierarchy** — Global, shared (tile-local), private registers.
- **Figure F.2: Operator Graph Rewrite** — DAG with rewrite passes (e.g., FFT∘IFFT → Identity).
- **Figure F.3: Tile Scheduling Layout** — Mapping tiles to SMs/CU with warp/block subdivisions.

---

## Chapter 4: Algorithmic Flow Diagrams

- **Figure G.1: Spectral Decomposition Pipeline** — Operator → SVD/Eig → Truncation → Reconstruction.
- **Figure G.2: Mixture of Recursions Execution** — Recursive operator expansion and scheduling.

---

## Chapter 5: Core Concepts

- **Operator**: Encapsulates transformations, supporting dense, factorized, block-sparse, and implicit forms.
- **Spectral Operators**: Enable reasoning in eigen/singular value space, critical for spectral learning.
- **Tile**: Fixed compute unit with shape `(BM, BN, BK)`, bound to warp/block scheduling.
- **Graph IR**: DAG of operators with explicit dependencies, subject to rewrites.
- **Numerics Policy**: Declares precision, accumulation type, and determinism requirements.
- **Streams & Events**: Mechanism for concurrent execution and synchronization.
- **Adjoint**: A mathematically precise operator dual (A*) used in differentiation and reasoning.

---

## Chapter 7: Glossary

- **ABI**: Application Binary Interface – contract between compiled programs and runtime.
- **Adjoint**: Conjugate transpose of an operator.
- **Graph IR**: Intermediate Representation capturing operators and schedules.
- **Hilbert Space**: Complete vector space with inner product structure.
- **Operator**: First-class transformation object in Tessera.
- **Spectral Decomposition**: Factorization into eigen/singular forms.
- **Tile**: Minimum hardware execution unit.

---

## Chapter 8: Index

- Adjoint operator — Chapter 5
- Execution flow — Chapter 2
- Factorized apply — Chapter 2
- Graph IR — Chapter 5
- Hilbert space — Chapter 5
- Memory hierarchy — Chapter 3
- Operator — Chapter 5
- Tile scheduling — Chapter 3

---

# Volume II: ABI Runtime Specification

## Part I: Overview

The ABI defines the **low-level contract** between compiler-generated code, the Tessera runtime, and hardware accelerators. Goals:

- **Portability** — across hardware backends.
- **Stability** — forward/backward ABI compatibility.
- **Efficiency** — minimal overhead in hot paths.

---

## Part II: Process Model

**Figure ABI.1: Process Model**

- **Contexts** encapsulate operator graphs, memory, tiles, and streams.
- **Handles** reference ABI-visible objects (opaque).
- **Lifetimes** are context-owned, reference-counted, and deterministically released.

---

## Part III: Memory Model

**Figure ABI.2: Memory Model**

- **Global memory**: Host + device accessible.
- **Tile-local memory**: Shared scratch space per tile.
- **Private registers**: Thread-local storage.

Memory consistency: relaxed by default, sequential within a stream. Events enforce ordering across streams.

---

## Part IV: Execution Model

- **Launch descriptors**: Tile shape, warp count, stages, memory usage.
- **Scheduling hints**: Autotuned or manually specified.
- **Synchronization**: Events, waits, barriers.

---

## Part V: Operator ABI

- Operators serialized as binary **Graph IR**.
- ABI supports linking, relocation, and specialization.
- Special handling for adjoint, spectral, and recursion operators.

---

## Part VI: Error Handling

Error codes:

- `TESSERA_SUCCESS` (0)
- `TESSERA_ERROR_INIT_FAIL` (1)
- `TESSERA_ERROR_INVALID_HANDLE` (2)
- `TESSERA_ERROR_OUT_OF_MEMORY` (3)
- `TESSERA_ERROR_UNSUPPORTED_FEATURE` (4)

Errors propagate deterministically across operator graphs, ensuring reproducible failure states.

---

## Appendix A: C ABI Bindings

```c
typedef struct tessera_context_t* tessera_context_handle;

typedef struct tessera_memory_t* tessera_memory_handle;
typedef struct tessera_operator_t* tessera_operator_handle;

typedef struct {
  uint32_t bm, bn, bk;
  uint32_t warps;
  uint32_t stages;
  uint32_t flags;
} tessera_launch_desc;

int tessera_context_create(tessera_context_handle* out);
int tessera_context_destroy(tessera_context_handle ctx);

int tessera_memory_alloc(tessera_context_handle ctx, size_t bytes, tessera_memory_handle* out);
int tessera_memory_free(tessera_memory_handle mem);

int tessera_launch_tile(tessera_operator_handle op, const tessera_launch_desc* desc);
```

---

## Appendix B: Calling Convention

Tile entry functions follow a fixed ABI:

- **Parameters**: passed in registers.
- **Spillover**: uses tile-local memory.
- **Return values**: placed in designated registers or tile-local scratch.

---

## Appendix C: Example ABI Usage

Mapping a HuggingFace GPT model into Tessera runtime:

```c
tessera_context_handle ctx;
tessera_context_create(&ctx);

tessera_memory_handle mem;
tessera_memory_alloc(ctx, 1024*1024*1024, &mem);

tessera_operator_handle gpt_op = tessera_load_operator(ctx, "gpt_oss_120b.tessera.ops");

tessera_launch_desc desc = {64, 64, 64, 4, 2, 0};
tessera_launch_tile(gpt_op, &desc);

tessera_context_destroy(ctx);
```

---

## Appendix D: Binary Format Layout

**Figure ABI.3: Binary Format Layout**

Tessera binaries follow an ELF-like structure:

- `.tessera.ops` — serialized operator IR.
- `.tessera.meta` — metadata and tuning records.
- `.tessera.strtab` — string table.

Versioning encoded in ELF notes via `TESSERA_ABI_VERSION`. ABI compatibility requires bumping version on breaking changes.
