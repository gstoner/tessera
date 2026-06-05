---
status: Normative
classification: Normative
authority: Runtime C ABI
last_updated: 2026-06-01
---

# Tessera Runtime ABI Specification

> **Canonical reference.** This document is grounded in the actual header files under
> `src/runtime/include/tessera/`. It is the only normative runtime ABI reference.
>
> **Current-state note (2026-06-01):** The C ABI is implemented and exercised
> by runtime-ABI smoke tests, sanitizer lanes, and the Python runtime wrapper.
> CPU execution is covered by the current smoke suite; CUDA/HIP behavior remains
> target- and hardware-gated. Do not infer production readiness from ABI
> presence alone — use `docs/spec/VALIDATION_SPINE.md` and
> `docs/spec/CONFORMANCE.md` for the validation spine. The concrete smoke
> binaries are `tessera-runtime-abi-smoke` and
> `tessera-collective-runtime-smoke`; Python wrappers live in
> `tests/unit/test_runtime_abi_smoke.py` and
> `tests/unit/test_sanitizer_smoke.py`.

---

## Documentation refresh (2026-06-01)

The 2026-05-06 audit asked this spec to cross-link the debugging guide
and to clarify that replay manifests are **not** part of the C ABI.
Resolution:

- **Replay manifests are Python-side developer contracts**, not C ABI
  surface. They are produced by `tessera.debug.save_replay_manifest`
  and consumed by `tessera.debug.replay_capture`. Their schema is in
  `python/tessera/debug.py` (526 LOC); user-facing documentation is
  `docs/guides/Tessera_Debugging_Tools_Guide.md`. The C ABI is
  unaffected by replay manifest presence.
- **Apple CPU + Apple GPU runtime symbols** are exported through the
  same C ABI. The generated ABI dashboard
  (`docs/audit/generated/runtime_abi.md`) is the drift-gated count source
  and currently reports 218 `extern "C" tessera_*` symbol entries, 207
  unique Apple symbols, and 84 Apple GPU kernel families:
  - `apple_cpu_runtime.cpp` exports `tessera_apple_cpu_gemm_{f32,f16,bf16}`
    plus `tessera_apple_cpu_gemm_f32_batched` (rank-3) — wired in
    Phase 8.2.
  - `apple_gpu_runtime` (Objective-C++ shim) exports the Apple GPU
    hardware-runtime surface: MPS/MPSGraph lanes, custom MSL kernels,
    GA/EBM/M7 fused kernels, Metal 4 matmul2d / epilogue / session /
    archive / conv lanes, and packaged `.mtlpackage` lifecycle symbols.
    Inventory: `docs/apple_gpu_kernel_inventory.md` and
    `docs/audit/generated/runtime_abi.md`.
- **Backend-kernel manifest** — `python/tessera/compiler/backend_manifest.py`
  synthesizes per-target × per-dtype kernel coverage from
  `capabilities.TARGET_CAPABILITIES` and is **observational metadata**
  on the registry — not a runtime ABI requirement. Valid statuses today:
  `fused` / `compileable` / `reference` / `artifact_only` / `planned` /
  `hardware_verified` / `packaged`. Packaged entries require
  `packaged_pipeline_path`; Apple packaged entries may also carry an
  `AppleKernelBindingSpec` built from `AppleTensorBindingSpec` rows for
  runtime reflection validation. Locked by backend-manifest and packaged
  ML tests under `tests/unit/`.
- **Collective adapter version pin** — `src/collectives/include/.../AdapterVersionPin.h`
  enforces NCCL ≥ 2.22 / RCCL ≥ 2.22 at C++ compile time via `#error`
  directives. The 8-symbol surface (`ncclAllReduce`, `ReduceScatter`,
  `AllGather`, `Send`, `Recv`, `CommInitRank`, `GetVersion`,
  `GetErrorString`) is probed by `scripts/probe_collective_libs.py`.
- **See also (developer tooling, not ABI):**
  `docs/guides/Tessera_Debugging_Tools_Guide.md` (replay, debug
  artifacts, debug traces),
  `docs/guides/Tessera_Profiling_And_Autotuning_Guide.md` (tprof
  telemetry, autotune cache),
  `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` (stable
  diagnostic codes).

---

## 1. Purpose and Scope

The Tessera Runtime ABI is a thin, stable C interface that decouples Tessera-compiled kernels
from the host toolchain, language bindings, and vendor drivers. It provides:

- **Device enumeration and capability queries.**
- **Memory allocation, transfer, and mapping.**
- **Asynchronous execution streams** with event-based synchronization.
- **Tile kernel launch** — the bridge between compiler-generated kernel functions and the
  hardware execution model.
- **Profiling timestamps** for performance measurement.

The ABI is versioned (`MAJOR.MINOR.PATCH`) and follows semantic versioning. Breaking changes
increment `MAJOR`. The current version is **0.2.0** (see `tsr_version.h`).

All public functions are declared `extern "C"`, making the ABI callable from both C and C++.
Python bindings wrap the runtime surface through `tessera.runtime.TesseraRuntime`
and related helpers. Backend coverage remains validation-gated.

---

## 2. Header Files

All public headers live under `src/runtime/include/tessera/`:

| Header | Contents |
|--------|----------|
| `tessera_runtime.h` | Master include; all public ABI functions |
| `tsr_types.h` | Opaque handle types, structs (`tsrLaunchParams`, `tsrTileCoord`, `tsrDeviceProps`) |
| `tsr_status.h` | `TsrStatus` enum, `tsrStatusString()`, `tsrGetLastError()` |
| `tsr_kernel.h` | Kernel callback signature (`tsrHostKernelFn`), `tsrKernelCtx` struct |
| `tsr_shape.h` | Launch validation helper (`tsrValidateLaunch`), tile size heuristic (`tsrSuggestTile`) |
| `tsr_version.h` | Version macros, `tsrGetVersion()`, profiling enable/timestamp |

Include only `tessera_runtime.h` — it pulls in all others.

---

## 3. Type System

### 3.1 Opaque Handle Types  (`tsr_types.h`)

All runtime objects are opaque pointer handles. Callers never dereference them directly.

```c
typedef struct tsrDevice_t* tsrDevice;   // logical device (CPU, CUDA, HIP)
typedef struct tsrStream_t* tsrStream;   // async execution queue
typedef struct tsrEvent_t*  tsrEvent;    // synchronization point / timestamp
typedef struct tsrBuffer_t* tsrBuffer;   // device-resident memory buffer
```

### 3.2 Device Kind Enum

```c
typedef enum {
  TSR_DEVICE_CPU  = 0,
  TSR_DEVICE_CUDA = 1,
  TSR_DEVICE_HIP  = 2
} TsrDeviceKind;
```

### 3.3 Device Properties

```c
typedef struct {
  TsrDeviceKind kind;
  char          name[128];
  uint32_t      logical_tile_threads_max;   // max threads per tile group
  uint32_t      concurrent_tiles_hint;      // scheduler hint for occupancy
} tsrDeviceProps;
```

### 3.4 Launch Parameters

```c
typedef struct { uint32_t x, y, z; } tsrDim3;

typedef struct {
  tsrDim3  grid;               // tile grid dimensions
  tsrDim3  tile;               // threads per tile group
  size_t   shared_mem_bytes;   // scratchpad per tile group
  uint32_t flags;              // reserved, set to 0
} tsrLaunchParams;
```

### 3.5 Tile / Thread Coordinates

Passed to each kernel invocation:

```c
typedef struct { uint32_t bx, by, bz; }            tsrTileCoord;    // tile position in grid
typedef struct { uint32_t tx, ty, tz, linear_tid; } tsrThreadCoord;  // thread in tile
```

### 3.6 Memcpy Direction Enum

```c
typedef enum {
  TSR_MEMCPY_HOST_TO_DEVICE,
  TSR_MEMCPY_DEVICE_TO_HOST,
  TSR_MEMCPY_DEVICE_TO_DEVICE,
  TSR_MEMCPY_HOST_TO_HOST
} TsrMemcpyKind;
```

---

## 4. Status and Error Model  (`tsr_status.h`)

Every ABI function returns `TsrStatus`. Callers must check the return value.

```c
typedef enum {
  TSR_STATUS_SUCCESS          = 0,
  TSR_STATUS_INVALID_ARGUMENT = 1,
  TSR_STATUS_NOT_FOUND        = 2,
  TSR_STATUS_ALREADY_EXISTS   = 3,
  TSR_STATUS_OUT_OF_MEMORY    = 4,
  TSR_STATUS_UNIMPLEMENTED    = 5,   // feature exists but backend not wired
  TSR_STATUS_INTERNAL         = 6,
  TSR_STATUS_DEVICE_ERROR     = 7
} TsrStatus;
```

**Diagnostic helpers:**

```c
const char* tsrStatusString(TsrStatus status);  // machine-stable string ("SUCCESS", etc.)
const char* tsrGetLastError(void);               // human-readable last error (thread-local)
void        tsrClearLastError(void);
```

`tsrGetLastError()` returns a pointer that is valid only until the next `tsr*` call on the
same thread. Copy it before calling any other ABI function.

**Convention:** When any function returns a non-SUCCESS status, `tsrGetLastError()` returns
a descriptive message. For `TSR_STATUS_UNIMPLEMENTED`, the message names the missing backend.

---

## 5. ABI Function Reference  (`tessera_runtime.h`)

### 5.1 Lifecycle

```c
TsrStatus tsrInit(void);
TsrStatus tsrIsInitialized(int* out);
TsrStatus tsrShutdown(void);
```

- `tsrInit` must be the first call. Initialises the backend registry and enumerates devices.
  Safe to call multiple times (idempotent after first success).
- `tsrIsInitialized` writes `1` to `out` after a successful `tsrInit` and `0`
  after `tsrShutdown`. It is intended for tests, embedding layers, and reload
  guards; callers still need to check every API return status.
- `tsrShutdown` releases all backend resources. All handles become invalid after this call.

### 5.2 Device Enumeration

```c
TsrStatus tsrGetDeviceCount(int* count);
TsrStatus tsrGetDevice(int index, tsrDevice* out);
TsrStatus tsrGetDeviceProps(tsrDevice dev, tsrDeviceProps* props);
```

- `index` is zero-based. Index 0 is always the CPU backend.
  CUDA/HIP devices follow if those backends are compiled in.
- `tsrGetDeviceProps` fills a `tsrDeviceProps` struct; useful for choosing tile sizes.
- The CPU backend always returns `TSR_STATUS_SUCCESS`; CUDA/HIP may return
  `TSR_STATUS_UNIMPLEMENTED` when not linked.

### 5.3 Streams

```c
TsrStatus tsrCreateStream(tsrDevice dev, tsrStream* out);
TsrStatus tsrDestroyStream(tsrStream s);
TsrStatus tsrStreamSynchronize(tsrStream s);
```

A stream is an ordered queue of kernel launches and memory operations. Operations submitted
to the same stream execute in submission order. Operations across streams are concurrent
(subject to backend capability).

On the CPU backend, a stream is backed by a `ThreadPool`; `tsrStreamSynchronize` waits
for all in-flight work on the pool.

### 5.4 Events

```c
TsrStatus tsrCreateEvent(tsrDevice dev, tsrEvent* out);
TsrStatus tsrRecordEvent(tsrEvent e, tsrStream s);
TsrStatus tsrWaitEvent(tsrEvent e, tsrStream s);
TsrStatus tsrEventSynchronize(tsrEvent e);
TsrStatus tsrDestroyEvent(tsrEvent e);
TsrStatus tsrEventGetTimestamp(tsrEvent e, uint64_t* ns_out);
```

- `tsrRecordEvent` marks the event as occurring at the current position in the stream.
- `tsrWaitEvent` blocks a stream at the event (inter-stream dependency).
- `tsrEventSynchronize` blocks the calling host thread until the event has fired.
- `tsrEventGetTimestamp` returns nanoseconds since process start (steady clock).
  Returns `TSR_STATUS_NOT_FOUND` if the event has not yet been recorded.

Events are the primary mechanism for measuring kernel latency:

```c
tsrEvent start, stop;
tsrCreateEvent(dev, &start);
tsrCreateEvent(dev, &stop);

tsrRecordEvent(start, stream);
tsrLaunchHostTileKernel(stream, &params, kernel_fn, payload);
tsrRecordEvent(stop, stream);
tsrStreamSynchronize(stream);

uint64_t t0, t1;
tsrEventGetTimestamp(start, &t0);
tsrEventGetTimestamp(stop,  &t1);
// elapsed = t1 - t0 nanoseconds
```

### 5.5 Memory Management

```c
TsrStatus tsrMalloc(tsrDevice dev, size_t bytes, tsrBuffer* out);
TsrStatus tsrFree(tsrBuffer b);
TsrStatus tsrMemset(tsrBuffer b, int value, size_t bytes);
TsrStatus tsrMemcpy(tsrBuffer dst, const tsrBuffer src, size_t bytes, TsrMemcpyKind kind);
TsrStatus tsrMap(tsrBuffer b, void** host_ptr, size_t* bytes);
TsrStatus tsrUnmap(tsrBuffer b);
```

- `tsrMalloc` allocates `bytes` bytes on the specified device. The buffer is uninitialized.
- `tsrMemset` fills with a byte value (`value` is cast to `unsigned char`).
- `tsrMemcpy` is synchronous with respect to the host (blocks until complete).
  Use streams + events for asynchronous transfers where the backend supports them.
- `tsrMap` / `tsrUnmap` provide host-accessible pointer to buffer contents.
  On the CPU backend this is a zero-copy pointer into the allocation.
  On GPU backends this maps into unified/pinned memory when supported.

**Ownership:** The caller owns the `tsrBuffer` handle and must call `tsrFree` when done.
Freeing a buffer that is in-flight on a stream is undefined behaviour.

### 5.6 Kernel Launch

```c
// Async: enqueues kernel on stream, returns immediately.
TsrStatus tsrLaunchHostTileKernel(
    tsrStream            s,
    const tsrLaunchParams* params,
    tsrHostKernelFn      kernel,
    void*                user_payload
);

// Sync: blocks host thread until all tiles complete.
TsrStatus tsrLaunchHostTileKernelSync(
    tsrDevice            dev,
    const tsrLaunchParams* params,
    tsrHostKernelFn      kernel,
    void*                user_payload
);
```

`tsrLaunchHostTileKernelSync` is a convenience wrapper equivalent to:
```c
tsrCreateStream(dev, &s);
tsrLaunchHostTileKernel(s, params, kernel, payload);
tsrStreamSynchronize(s);
tsrDestroyStream(s);
```

---

## 6. Tile Kernel Interface  (`tsr_kernel.h`)

Kernels launched via the ABI use a portable callback signature:

```c
typedef void (*tsrHostKernelFn)(
    void*                   user_ctx,    // cast to (tsrKernelCtx*)
    const tsrTileCoord*     tile,        // which tile in the grid
    const tsrThreadCoord*   thread       // which thread in the tile
);
```

The runtime calls this function once per (tile, thread) pair. The `user_ctx` is always a
`tsrKernelCtx*`:

```c
struct tsrKernelCtx {
  void*   user;           // the user_payload passed to tsrLaunchHostTileKernel
  void*   shared_mem;     // tile-group scratchpad (shared_mem_bytes in tsrLaunchParams)
  size_t  shared_bytes;
  void (*_tile_barrier)(tsrKernelCtx*);   // backend barrier implementation
  void*   _impl;                          // backend-internal, do not touch
};
```

**Accessor inlines** (use these instead of touching struct fields directly):

```c
void*  tsr_shared_mem(tsrKernelCtx* ctx);    // pointer to shared scratchpad
size_t tsr_shared_bytes(tsrKernelCtx* ctx);  // size of scratchpad
void   tsr_tile_barrier(tsrKernelCtx* ctx);  // synchronise all threads in the tile group
```

### 6.1 Kernel Example (CPU backend)

```c
typedef struct { float* A; float* B; float* C; int N; } GemmPayload;

void gemm_kernel(void* user_ctx,
                 const tsrTileCoord* tile,
                 const tsrThreadCoord* thread) {
    tsrKernelCtx* ctx = (tsrKernelCtx*)user_ctx;
    GemmPayload*  p   = (GemmPayload*)ctx->user;

    int row = tile->bx * 8 + thread->ty;
    int col = tile->by * 8 + thread->tx;
    if (row >= p->N || col >= p->N) return;

    float acc = 0.0f;
    for (int k = 0; k < p->N; ++k)
        acc += p->A[row * p->N + k] * p->B[k * p->N + col];
    p->C[row * p->N + col] = acc;
}

// Launch on CPU
tsrDevice dev; tsrGetDevice(0, &dev);
tsrStream stream; tsrCreateStream(dev, &stream);

tsrLaunchParams params = {
    .grid = {N/8, N/8, 1},
    .tile = {8, 8, 1},
    .shared_mem_bytes = 0,
    .flags = 0
};
GemmPayload payload = { A_buf, B_buf, C_buf, N };
tsrLaunchHostTileKernel(stream, &params, gemm_kernel, &payload);
tsrStreamSynchronize(stream);
```

### 6.2 Shared Memory Usage

```c
void reduction_kernel(void* user_ctx,
                      const tsrTileCoord* tile,
                      const tsrThreadCoord* thread) {
    tsrKernelCtx* ctx  = (tsrKernelCtx*)user_ctx;
    float*        smem = (float*)tsr_shared_mem(ctx);   // tile scratchpad

    smem[thread->linear_tid] = /* load from global */;
    tsr_tile_barrier(ctx);   // wait for all threads to write

    if (thread->linear_tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < tsr_shared_bytes(ctx) / sizeof(float); ++i)
            sum += smem[i];
        /* write sum to output */
    }
}
```

---

## 7. Launch Validation and Tile Size Helpers  (`tsr_shape.h`)

```c
// Validate params before launch. Returns INVALID_ARGUMENT with a reason string on failure.
TsrStatus tsrValidateLaunch(const tsrDeviceProps* props, const tsrLaunchParams* p);

// Suggest a tile geometry for a target thread count.
// Simple heuristic: tx = min(logical_threads, max_threads), ty = tz = 1.
void tsrSuggestTile(const tsrDeviceProps* props, uint32_t logical_threads, tsrDim3* out_tile);
```

These helpers are optional but recommended when writing portable kernels:

```c
tsrDeviceProps props;
tsrGetDeviceProps(dev, &props);

tsrDim3 tile;
tsrSuggestTile(&props, 128, &tile);   // -> { min(128, max_threads), 1, 1 }

tsrLaunchParams params = {
    .grid = {N / tile.x, 1, 1},
    .tile = tile,
    .shared_mem_bytes = tile.x * sizeof(float),
    .flags = 0
};
TsrStatus s = tsrValidateLaunch(&props, &params);
if (s != TSR_STATUS_SUCCESS) {
    fprintf(stderr, "bad params: %s\n", tsrGetLastError());
    return 1;
}
```

---

## 8. Versioning and Profiling  (`tsr_version.h`)

```c
#define TESSERA_VERSION_MAJOR  0
#define TESSERA_VERSION_MINOR  2
#define TESSERA_VERSION_PATCH  0

void tsrGetVersion(int* major, int* minor, int* patch);

// Profiling
void     tsrEnableProfiling(int enable);     // 1 = on, 0 = off
uint64_t tsrTimestampNowNs(void);            // nanoseconds since process start (steady clock)
```

**ABI stability guarantee:** Functions with `TESSERA_VERSION_MAJOR == 0` may change in
minor releases. Once MAJOR reaches 1, any function present at 1.0.0 is stable for all 1.x
releases.

---

## 9. Backend Architecture

The public C ABI is designed around a `Backend` abstract class
(`src/runtime/src/backend/base_backend.h`). Each production backend is expected to be an
independent shared library:

| Backend | Active file | Status |
|---------|-------------|--------|
| CPU (thread pool) | `src/runtime/src/backend/cpu_backend.cpp` | implemented / mock-runtime |
| CUDA | `src/runtime/src/backend/cuda_backend.cpp` | hardware-runtime when built with `TESSERA_ENABLE_CUDA` and a CUDA device is present; otherwise unavailable |
| HIP (AMD) | `src/runtime/src/backend/hip_backend.cpp` | hardware-runtime when built with `TESSERA_ENABLE_HIP` and a HIP device is present; otherwise unavailable |

The `Backend` interface mirrors the C ABI 1:1 (every `tsr*` function calls the corresponding
`Backend` virtual method). Selecting a backend at runtime:

1. `tsrInit` scans for available backends in priority order: CUDA > HIP > CPU.
2. `tsrGetDevice(0, ...)` always returns the CPU backend; higher indices are accelerators.
3. Python distributed tests use `MockRankGroup` for multi-rank behavior. That
   mock does not demonstrate NCCL/RCCL/MPI runtime conformance.

---

## 10. Python Wrapper

The `TesseraRuntime` Python class in `python/tessera/runtime.py` is the current
Python wrapper over the C ABI when a shared runtime library is available, and a
deterministic mock-runtime fallback otherwise:

```python
from tessera.runtime import TesseraRuntime

rt = TesseraRuntime()
rt.init()
assert rt.get_device_count() >= 1   # always >= 1 (CPU/mock)

dev = rt.get_device(0)
stream = rt.create_stream(dev)

buf = rt.malloc(dev, 4096)
rt.memset(buf, 0, 4096)
rt.stream_sync(stream)

rt.destroy_stream(stream)
rt.free(buf)
rt.shutdown()
```

The Python wrapper maps `TsrStatus` return codes to `TesseraRuntimeError` exceptions. It
does **not** add a garbage-collected handle layer — callers must explicitly destroy objects in
reverse creation order.

---

## 11. Phase Coverage

| Feature | ABI headers | CPU impl | CUDA/HIP impl | Python wrapper |
|---------|-------------|----------|---------------|----------------|
| `tsrInit` / `tsrShutdown` | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| Device enumeration | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| Streams | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| Events + timestamps | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| `tsrMalloc` / `tsrFree` | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| `tsrMemcpy` | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| `tsrMap` / `tsrUnmap` | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| `tsrLaunchHostTileKernel` | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |
| Artifact compile/load/get-kernel/launch | implemented | scaffolded / implemented subset | scaffolded | scaffolded / implemented subset |
| Launch validation helpers | implemented | implemented / mock-runtime | n/a | implemented / mock-runtime |
| Profiling timestamps | implemented | implemented / mock-runtime | hardware-runtime when built and device-present | implemented / mock-runtime |

**Note:** `python/tessera/testing/mock_collective.py` (`MockRankGroup`) remains
the current multi-rank test mechanism. It does not go through the C ABI; it uses
Python threads and in-process numpy buffers.

Replay manifests, structured debug traces, and compiler artifact bundles are
Python/developer-tool contracts described in
`docs/guides/Tessera_Debugging_Tools_Guide.md`. They are intentionally not part
of the stable C ABI unless a future runtime replay API promotes them.

---

## 12. Compiled-Function ABI (Production MLIR/LLVM Lane)

> **Status: Implemented — Phase 0 (CPU, total elementwise).** Landed 2026-06-05.
> The boundary is exercised end-to-end by `tools/tessera-jit` (the experimental
> `libtessera_jit` CPU JIT) and `tests/unit/test_production_jit_add.py`:
> `tessera.add` lowers `tessera → linalg → bufferize → llvm`, JITs via
> `mlir::ExecutionEngine`, and writes a caller-allocated output through
> `_mlir_ciface_tessera_jit_add(a*, b*, out*)` (void, DPS). The oracle test asserts
> numerical match vs numpy **and** an unfakeable JIT invocation-counter advance, so
> a silent numpy fallback fails the suite. Scope today is the Phase-0 total-op class
> (§12.6); broader coverage is Phase 1. Design ref: `PRODUCTION_COMPILER_PLAN.md` D3.

### 12.0 Why this is a separate ABI

§5–§6 define the **tile-kernel launch ABI**: the runtime calls a user/compiler
callback once per `(tile, thread)` via `tsrHostKernelFn`, threading state through
an opaque `void* user_payload`. That is the *imperative host-kernel* contract used
by the Python/eager lane and the x86 host-tile path.

The **Compiled-Function ABI** is different in kind. It is the boundary between a
caller and a **whole MLIR-compiled function** — the unit produced by lowering a
`tessera` Graph IR function through `linalg → bufferize → llvm` and JIT-compiling
it with `mlir::ExecutionEngine`. There is no per-thread callback; the caller
hands over typed buffers and the compiled function runs to completion. This is the
contract every production phase inherits, so it is pinned before any nontrivial op
is lowered.

The two ABIs coexist. The tile-kernel ABI remains the device-memory / stream /
event substrate (`tsrMalloc`, `tsrCreateStream`, …); the Compiled-Function ABI is
the leaf that the runtime ultimately invokes. In Phase 0 the compiled function
operates on **host memory only** — `tsr*` device integration arrives in Phase 3.

### 12.1 Calling convention

A production-accepted Tessera function lowers to an LLVM-dialect `func.func`
emitted with **C-interface wrappers** (`-llvm-request-c-wrappers`). For a function
symbol `S`, the runtime invokes:

```c
void _mlir_ciface_S(/* descriptors, see 12.3 */);
```

- Every tensor operand and result is passed as a **pointer to a memref
  descriptor** (§12.2). The C wrapper is what stabilizes the layout — the bare
  (non-`ciface`) entry uses an unpacked argument expansion that is *not* part of
  this ABI.
- The function returns `void`. Results are **not** returned by value or via `sret`
  in v1 — they are destination operands (§12.4). This keeps every wrapper
  parameter uniform (`Descriptor*`) and avoids struct-return ABI variance.

### 12.2 The memref descriptor (the wire format)

For a rank-`N` buffer of element type `T`, the descriptor is MLIR's standard
lowered struct (stable across LLVM targets):

```c
typedef struct {
  T*        allocated_ptr;   // base of the allocation (for free); = aligned_ptr in v1
  T*        aligned_ptr;     // element-0 pointer the kernel reads/writes
  intptr_t  offset;          // element offset from aligned_ptr; MUST be 0 in v1
  intptr_t  sizes[N];        // extent per dimension
  intptr_t  strides[N];      // element stride per dimension (row-major in v1)
} TsrMemRefDescriptor_<T>_<N>;
```

`intptr_t` is 64-bit on supported hosts. `N` is fixed per compiled symbol (the
function is monomorphic in rank). The descriptor is **caller-owned stack/heap
memory**; the compiled function never retains a pointer to the descriptor past
the call.

### 12.3 Argument ordering — Destination-Passing Style (D3)

Operands appear in source order, **inputs first, then outputs** (`outs`),
mirroring `linalg`'s DPS convention so the boundary composes with bufferization
end-to-end:

```
func @S(%in0: memref<...>, ..., %inK: memref<...>,
        %out0: memref<...>, ..., %outM: memref<...>)
  ⇒  void _mlir_ciface_S(Desc* in0, ..., Desc* inK, Desc* out0, ..., Desc* outM)
```

- **The caller allocates all buffers, including outputs.** The compiled function
  writes results into the caller-provided `out*` descriptors. It performs **no
  allocation or free of boundary memory** in v1.
- Callee-allocated results (e.g. data-dependent output shapes) are a recognized
  future need; they will be added as an **explicit, opt-in** descriptor variant,
  never by silently changing v1 ownership. This is the ABI rule that prevents
  callee allocation from becoming a permanent wart.

### 12.4 Layout and contiguity contract

In v1, every boundary memref is **identity-layout, C-contiguous**:
`offset == 0` and `strides` are the row-major strides of `sizes`. A caller
holding a non-contiguous / non-zero-offset / non-identity-layout buffer (e.g. a
transposed numpy view) **must materialize a contiguous copy at the boundary**
before the call. Non-identity layouts inside the compiled function are
unconstrained — this rule governs only the boundary.

### 12.5 Dtype contract (bf16 rule)

Element types are the canonical Tessera dtypes (`python/tessera/dtype.py`). The
boundary maps them to fixed byte representations:

| Tessera dtype | Boundary representation | Phase 0 |
|---|---|---|
| `fp32` / `fp16` | IEEE-754 binary32 / binary16 | f32 yes; f16 P1 |
| `bf16` | **raw 16-bit** (`uint16` storage; no native host type) | P1 |
| `int8/16/32/64`, `bool` | native two's-complement / 1-byte bool | i32/i64 yes |
| fp8 / fp6 / fp4 / nvfp4 | packed per `dtype.py`; boundary-opaque | later |

**bf16 ABI rule (ratified):** the Python lane uses `ml_dtypes.bfloat16`; the
MLIR/runtime boundary uses **raw 16-bit** storage; mismatched producers/consumers
**copy/convert at the boundary**. This is an ABI rule, not an implementation
accident — no path may reinterpret bf16 bits as fp16 or vice versa.

### 12.6 Error and effect model

**`void` return is Phase 0 only.** It is valid solely because Phase 0 admits a
strictly-total op class:

- elementwise / structured math with **no failure mode** (add/mul/relu-style);
- shapes and dtypes **prevalidated by the caller** before invocation;
- **caller-allocated outputs** (§12.3), so no allocation occurs inside the
  compiled function;
- **no dynamic dispatch** that could fail inside the function.

Under those conditions a compiled function cannot fail, so `void` is correct and
v1 callers may assume success. **This guarantee does not extend past Phase 0.** Any
op that can fail at runtime (bounds, device errors, callee allocation in later
phases, dynamic dispatch) requires the status mechanism below; introducing such an
op without it is an ABI violation.

**Reserved future status mechanism.** When a fallible op first lands, the
signature gains an explicit status channel. The **preferred** primary contract is a
status **return value** or a trailing `TsrStatus* status_out` parameter — *not*
hidden reliance on thread-local `tsrGetLastError()` as the primary signal
(`tsrGetLastError` may still carry the human-readable detail, but must not be the
sole success/failure indicator). The exact form is chosen by the phase that first
needs it and recorded here. The signature section above (§12.1) is written so this
addition is additive, not a breaking re-shape of the total-op case.

### 12.7 Relationship to `tsrCompileArtifact`

`tsrCompileArtifact` (§5, `tessera_runtime.h`) currently interprets `module_ir` as
a comma-separated list of pre-registered host-kernel names — the tile-kernel path.
Its artifact shape is **the wrong fit for MLIR `ExecutionEngine` proof work**, and
forcing integration now would couple Phase 0 to legacy runtime decisions.

The Compiled-Function ABI is therefore reached through a **separate, experimental
production-lane JIT surface** — working name **`tessera_jit`** (a.k.a.
`mlir_cpu_jit`): compile MLIR module → `ExecutionEngine` → look up
`_mlir_ciface_S`, bound from Python by `canonical_compile(target="cpu")`. It is a
standalone dylib loaded via `ctypes`.

- **Naming guardrail:** this is *experimental production-lane plumbing*, **not
  "runtime v2."** Do not market or document it as a runtime replacement. Its job in
  Phase 0 is narrow and provable: MLIR lowers, the C-ABI wrapper exists, Python can
  call it, the oracle passes.
- **Compatibility note (Phase 1 decision, open):** Phase 1 must explicitly decide
  whether `tessera_jit` (a) becomes the real-codegen behavior behind
  `tsrCompileArtifact`, (b) is exposed as a sibling TSR API, or (c) remains a
  separate CPU-JIT subsystem. Phase 0 deliberately does not pre-judge this.

---

## 13. Authoritative References

| Topic | Where to look |
|-------|--------------|
| C ABI header (master include) | `src/runtime/include/tessera/tessera_runtime.h` |
| Opaque types and structs | `src/runtime/include/tessera/tsr_types.h` |
| Status codes | `src/runtime/include/tessera/tsr_status.h` |
| Kernel callback + KernelCtx | `src/runtime/include/tessera/tsr_kernel.h` |
| Launch validation | `src/runtime/include/tessera/tsr_shape.h` |
| Version macros | `src/runtime/include/tessera/tsr_version.h` |
| Backend abstract class | `src/runtime/src/backend/base_backend.h` |
| Runtime implementation notes | `CLAUDE.md` architecture-decision log |
| Compiled-Function ABI design + phasing (§12) | `docs/spec/PRODUCTION_COMPILER_PLAN.md` |
| Benchmark integration | `benchmarks/` |
| Runtime ABI smoke binary | `src/runtime/tools/tessera-runtime-abi-smoke/runtime_abi_smoke.cpp` |
| Collective runtime smoke binary | `src/collectives/tools/tessera-collective-runtime-smoke/runtime_smoke.cpp` |
| Python smoke wrappers | `tests/unit/test_runtime_abi_smoke.py`, `tests/unit/test_sanitizer_smoke.py` |
