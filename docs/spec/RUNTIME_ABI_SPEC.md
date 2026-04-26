---
status: Normative
classification: Normative
authority: Runtime C ABI
last_updated: 2026-04-26
---

# Tessera Runtime ABI Specification

> **Canonical reference.** This document is grounded in the actual header files under
> `src/runtime/include/tessera/`. It is the only normative runtime ABI reference.
>
> **Phase status:** The C ABI headers are complete and well-defined (Phase 6 target for full
> GPU backend wiring). The CPU backend (`tessera_runtime_cpu.cpp`) is the reference
> implementation used in all current tests. The Python `TesseraRuntime` wrapper is Phase 6
> planned — not yet implemented.

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
Python bindings (Phase 6) will wrap these via ctypes/cffi.

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
TsrStatus tsrShutdown(void);
```

- `tsrInit` must be the first call. Initialises the backend registry and enumerates devices.
  Safe to call multiple times (idempotent after first success).
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
  Use streams + events for asynchronous transfers (Phase 4+ GPU backend).
- `tsrMap` / `tsrUnmap` provide host-accessible pointer to buffer contents.
  On the CPU backend this is a zero-copy pointer into the allocation.
  On GPU backends (Phase 6) this maps into unified/pinned memory.

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

The public C ABI is implemented over a `Backend` abstract class
(`src/runtime/src/backend/base_backend.h`). Each backend is an independent shared library:

| Backend | File | Status |
|---------|------|--------|
| CPU (thread pool) | `tessera_runtime_cpu.cpp` | **Phase 6 planned** (header complete) |
| CUDA | `tessera_runtime_cuda.cpp` | **Phase 6 planned** |
| HIP (AMD) | `tessera_runtime_hip.cpp` | **Phase 6 planned** |

The `Backend` interface mirrors the C ABI 1:1 (every `tsr*` function calls the corresponding
`Backend` virtual method). Selecting a backend at runtime:

1. `tsrInit` scans for available backends in priority order: CUDA > HIP > CPU.
2. `tsrGetDevice(0, ...)` always returns the CPU backend; higher indices are accelerators.
3. Phase 1–3 tests use a mock thread-pool implementation that satisfies the `Backend`
   interface without the runtime C ABI wiring.

---

## 10. Python Wrapper (Planned — Phase 6)

The `TesseraRuntime` Python class (to be implemented in `python/tessera/runtime.py`) is a
thin ctypes/cffi binding over the C ABI:

```python
from tessera.runtime import TesseraRuntime   # Phase 6

rt = TesseraRuntime()
assert rt.device_count() >= 1   # always >= 1 (CPU)

ctx = rt.create_context(device_id=0)
stream = rt.create_stream(ctx)

buf = rt.malloc(ctx, 4096)
rt.memset(buf, 0, 4096)
rt.synchronize(stream)

rt.destroy_stream(stream)
rt.destroy_context(ctx)
```

The Python wrapper maps `TsrStatus` return codes to `TesseraRuntimeError` exceptions. It
does **not** add a garbage-collected handle layer — callers must explicitly destroy objects in
reverse creation order.

---

## 11. Phase Coverage

| Feature | ABI headers | CPU impl | CUDA impl | Python wrapper |
|---------|------------|----------|-----------|----------------|
| `tsrInit` / `tsrShutdown` | ✅ | Phase 6 | Phase 6 | Phase 6 |
| Device enumeration | ✅ | Phase 6 | Phase 6 | Phase 6 |
| Streams | ✅ | Phase 6 | Phase 6 | Phase 6 |
| Events + timestamps | ✅ | Phase 6 | Phase 6 | Phase 6 |
| `tsrMalloc` / `tsrFree` | ✅ | Phase 6 | Phase 6 | Phase 6 |
| `tsrMemcpy` | ✅ | Phase 6 | Phase 6 | Phase 6 |
| `tsrMap` / `tsrUnmap` | ✅ | Phase 6 | Phase 6 | Phase 6 |
| `tsrLaunchHostTileKernel` | ✅ | Phase 6 | Phase 6 | Phase 6 |
| Launch validation helpers | ✅ | Phase 6 | n/a | Phase 6 |
| Profiling timestamps | ✅ | Phase 6 | Phase 6 | Phase 6 |

**Note:** Phases 1–5 use `python/tessera/testing/mock_collective.py` (`MockRankGroup`) for
multi-rank tests. That mock does not go through the C ABI — it uses Python threads and
in-process numpy buffers. The C ABI becomes the execution path starting in Phase 6.

---

## 12. Authoritative References

| Topic | Where to look |
|-------|--------------|
| C ABI header (master include) | `src/runtime/include/tessera/tessera_runtime.h` |
| Opaque types and structs | `src/runtime/include/tessera/tsr_types.h` |
| Status codes | `src/runtime/include/tessera/tsr_status.h` |
| Kernel callback + KernelCtx | `src/runtime/include/tessera/tsr_kernel.h` |
| Launch validation | `src/runtime/include/tessera/tsr_shape.h` |
| Version macros | `src/runtime/include/tessera/tsr_version.h` |
| Backend abstract class | `src/runtime/src/backend/base_backend.h` |
| Phase 6 deliverables | `CLAUDE.md` §Phase 6 |
| Benchmark integration | `benchmarks/` (Phase 6) |
