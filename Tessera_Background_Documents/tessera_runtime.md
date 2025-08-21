markdown
# Tessera ABI Runtime Specification (Draft)

## Part I. Overview

This document defines the **Application Binary Interface (ABI)** for the Tessera programming model. It specifies the low-level contract between:

- Compiler-generated code,
- The Tessera runtime,
- Underlying hardware accelerators.

The ABI ensures that binaries compiled for Tessera can run across different hardware implementations while preserving correctness and performance.

### Goals

- Portability across hardware vendors.
- Stability across compiler/runtime versions.
- Efficiency: minimal ABI overhead.

---

## Part II. Process Model

### 2.1 Contexts

- A **Tessera Context** is the runtime unit of execution.
- It encapsulates: operator graphs, memory allocations, tiles, and streams.

### 2.2 Handles

- ABI-visible objects are referenced through **opaque handles**.
- Examples: `tessera_context_handle`, `tessera_operator_handle`, `tessera_memory_handle`.

### 2.3 Lifetimes

- Handles are reference-counted.
- Contexts own subordinate resources; destroying a context destroys all subordinate objects.

---

## Part III. Memory Model

### 3.1 Memory Spaces

- **Global Memory**: Accessible by host and device. ABI guarantees 64-byte alignment.
- **Tile-local Memory**: ABI specifies maximum size per tile (queriable).
- **Private Registers**: ABI does not directly expose but specifies calling conventions.

### 3.2 Host–Device Interop

- ABI defines rules for host pointer import/export.
- Zero-copy semantics may be supported but are optional.

### 3.3 Vectorization Rules

- All memory access via ABI must honor natural vector alignment (e.g., 16B for float4).

---

## Part IV. Execution Model

### 4.1 Launch Descriptors

- Each tile launch is described by a **launch descriptor**:
  - Tile shape (BM, BN, BK)
  - Warp count
  - Shared memory usage
  - Stream association

### 4.2 Scheduling Hints

- ABI allows attaching hints (priority, affinity).
- Hints are advisory; runtime may ignore.

### 4.3 Synchronization

- ABI defines **events** and **wait primitives**:
  - `tessera_event_record`
  - `tessera_stream_wait_event`

---

## Part V. Operator ABI

### 5.1 Graph Representation

- Operator graphs are serialized into a binary **Graph IR** blob.
- ABI defines versioning and compatibility rules.

### 5.2 Linking and Relocation

- Operators can be linked into larger graphs.
- ABI specifies relocation entries for operator symbols.

### 5.3 Special Operators

- **Spectral Operators**: ABI exposes contracts for eigen/singular decomposition at binary level.
- **Adjoint Operators**: ABI defines metadata flagging an operator as adjoint.
- **Recursion Operators**: ABI specifies stack-frame size requirements.

---

## Part VI. Error Handling

### 6.1 Error Codes

- `TESSERA_SUCCESS` (0)
- `TESSERA_ERROR_INIT_FAIL` (1)
- `TESSERA_ERROR_INVALID_HANDLE` (2)
- `TESSERA_ERROR_OUT_OF_MEMORY` (3)
- `TESSERA_ERROR_UNSUPPORTED_FEATURE` (4)

### 6.2 Propagation

- Errors encountered in operator graphs propagate upstream.
- ABI guarantees deterministic error codes.

---

## Appendix A. C ABI Bindings

```c
// Example handle type
typedef struct tessera_context_t* tessera_context_handle;

// Context management
int tessera_context_create(tessera_context_handle* out);
int tessera_context_destroy(tessera_context_handle ctx);

// Memory management
int tessera_memory_alloc(tessera_context_handle ctx, size_t bytes, tessera_memory_handle* out);
int tessera_memory_free(tessera_memory_handle mem);

// Execution
int tessera_launch_tile(tessera_operator_handle op, const tessera_launch_desc* desc);
```

---

## Appendix B. Calling Convention

- Parameters passed via registers when possible.
- Tile entry functions use a fixed ABI prologue/epilogue.
- Spillover goes to tile-local memory.

---

## Appendix C. Example Usage

**Mapping HuggingFace GPT model into Tessera ABI:**

- Allocate memory for embeddings via `tessera_memory_alloc`.
- Upload operator graph (attention layers, MLPs).
- Launch tiled kernels with descriptors.
- Synchronize with events.

---

## Appendix D. Binary Format Layout

- Tessera binary is ELF-like.
- Sections:
  - `.tessera.ops` — operator IR blobs
  - `.tessera.meta` — metadata (tile sizes, graph topology)
  - `.tessera.strtab` — string table for symbol resolution

Versioning is encoded in ELF note section `TESSERA_ABI_VERSION`.

