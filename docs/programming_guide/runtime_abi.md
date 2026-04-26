---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Runtime ABI — Programmer's Guide

> **Scope:** This document is the programmer-facing guide to Tessera's runtime C ABI —
> how to use it, when to use it, and what to expect from each function. For the normative
> function-by-function specification (exact signatures, all type definitions, error codes,
> backend architecture) see [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md).
>
> **Phase status:** The C ABI headers are fully defined. The CPU backend implementation and
> Python wrapper are planned for Phase 6. Phases 1–5 use `MockRankGroup` for multi-rank
> tests, which does not go through this ABI.

---

## What the Runtime ABI Is

The Tessera runtime ABI is a thin, stable C interface that sits between compiler-generated
kernels and the hardware. It handles five things:

1. **Device discovery** — which accelerators are available and what they can do.
2. **Memory management** — allocate, zero, copy, and map device buffers.
3. **Asynchronous execution** — streams and events for overlap and timing.
4. **Tile kernel launch** — run a compiled kernel across a tile grid.
5. **Profiling** — nanosecond timestamps for measuring kernel latency.

The ABI is `extern "C"`, so it is callable from C, C++, and any language with a C FFI
(Python ctypes/cffi, Rust, Swift, etc.). Every function returns a `TsrStatus` code — check
it every time.

The single header to include is:

```c
#include "tessera/tessera_runtime.h"
```

---

## Object Lifecycle

Every runtime object follows a strict create-before-use, destroy-after-use lifecycle:

```
tsrInit()
  │
  ├─ tsrGetDevice()       → tsrDevice  (no explicit destroy)
  │
  ├─ tsrCreateStream()    → tsrStream
  │     │
  │     ├─ tsrMalloc()    → tsrBuffer  ── tsrFree()
  │     ├─ tsrMemset()
  │     ├─ tsrMemcpy()
  │     │
  │     ├─ tsrCreateEvent() → tsrEvent
  │     │     ├─ tsrRecordEvent()
  │     │     ├─ tsrWaitEvent()
  │     │     └─ tsrDestroyEvent()
  │     │
  │     └─ tsrLaunchHostTileKernel()
  │
  └─ tsrDestroyStream()
  
tsrShutdown()
```

**Rule:** Never use a buffer or event after its stream has completed and been destroyed.
Never call any `tsr*` function after `tsrShutdown()`.

---

## Step 1 — Initialize and Discover Devices

```c
#include "tessera/tessera_runtime.h"
#include <stdio.h>

int main(void) {
    // Initialize (always first call)
    if (tsrInit() != TSR_STATUS_SUCCESS) {
        fprintf(stderr, "tsrInit failed: %s\n", tsrGetLastError());
        return 1;
    }

    // How many devices are available?
    int count = 0;
    tsrGetDeviceCount(&count);
    printf("Tessera sees %d device(s)\n", count);
    // Index 0 is always CPU. CUDA/HIP devices follow if linked.

    // Inspect a device
    tsrDevice dev;
    tsrGetDevice(0, &dev);   // 0 = CPU backend

    tsrDeviceProps props;
    tsrGetDeviceProps(dev, &props);
    printf("Device: %s  max_tile_threads=%u  concurrent_tiles=%u\n",
           props.name,
           props.logical_tile_threads_max,
           props.concurrent_tiles_hint);

    tsrShutdown();
    return 0;
}
```

`tsrDeviceProps` tells you the device name, the maximum number of logical threads per tile
group (`logical_tile_threads_max`), and a scheduling hint for occupancy
(`concurrent_tiles_hint`). Use these to choose tile sizes at runtime.

**Device kinds:**

| `TsrDeviceKind` | Value | Notes |
|-----------------|-------|-------|
| `TSR_DEVICE_CPU` | 0 | Always present; index 0 |
| `TSR_DEVICE_CUDA` | 1 | Present if CUDA backend compiled in |
| `TSR_DEVICE_HIP` | 2 | Present if HIP backend compiled in |

---

## Step 2 — Allocate Device Memory

```c
tsrDevice dev;
tsrGetDevice(0, &dev);

// Allocate 1024 floats on device
tsrBuffer buf_a, buf_b, buf_c;
tsrMalloc(dev, 1024 * sizeof(float), &buf_a);
tsrMalloc(dev, 1024 * sizeof(float), &buf_b);
tsrMalloc(dev, 1024 * sizeof(float), &buf_c);

// Zero-initialise output buffer
tsrMemset(buf_c, 0, 1024 * sizeof(float));

// Copy host data into device buffer
float host_data[1024] = { /* ... */ };
// Wrap host pointer in a buffer handle:
tsrBuffer host_buf = /* ... host-side buffer handle ... */;
tsrMemcpy(buf_a, host_buf, 1024 * sizeof(float), TSR_MEMCPY_HOST_TO_DEVICE);
```

**`tsrMemcpy` direction flags:**

| Flag | Direction |
|------|-----------|
| `TSR_MEMCPY_HOST_TO_DEVICE` | CPU RAM → device buffer |
| `TSR_MEMCPY_DEVICE_TO_HOST` | device buffer → CPU RAM |
| `TSR_MEMCPY_DEVICE_TO_DEVICE` | device → device (same or different device) |
| `TSR_MEMCPY_HOST_TO_HOST` | CPU → CPU (useful for testing) |

**Mapping buffers for host inspection:**

```c
void*  host_ptr;
size_t nbytes;
tsrMap(buf_c, &host_ptr, &nbytes);
float* results = (float*)host_ptr;
// read/write results[i] ...
tsrUnmap(buf_c);
```

On the CPU backend `tsrMap` is a zero-copy pointer into the allocation. On GPU backends
(Phase 6) it maps into pinned/unified memory. Always `tsrUnmap` before the next kernel
launch that touches the buffer.

**When you are done:**

```c
tsrFree(buf_a);
tsrFree(buf_b);
tsrFree(buf_c);
```

---

## Step 3 — Define a Tile Kernel

A tile kernel is a plain C function with this signature:

```c
void my_kernel(void*                   user_ctx,
               const tsrTileCoord*     tile,
               const tsrThreadCoord*   thread);
```

The runtime calls it once per (tile, thread) pair. Cast `user_ctx` to `tsrKernelCtx*`
to access shared memory and call barriers.

### Coordinate helpers

```c
// tile->bx, tile->by, tile->bz  — tile position in the launch grid
// thread->tx, thread->ty, thread->tz — thread position in the tile
// thread->linear_tid — flat thread index (tx + ty*tile_w + tz*tile_w*tile_h)
```

### Kernel context

```c
#include "tessera/tsr_kernel.h"

void kernel_fn(void* user_ctx,
               const tsrTileCoord* tile,
               const tsrThreadCoord* thread) {
    tsrKernelCtx* ctx = (tsrKernelCtx*)user_ctx;
    
    // Access shared scratchpad (shared_mem_bytes set in tsrLaunchParams)
    float* smem = (float*)tsr_shared_mem(ctx);
    size_t smem_floats = tsr_shared_bytes(ctx) / sizeof(float);
    
    // Access caller's payload
    MyPayload* p = (MyPayload*)ctx->user;
    
    // Do work...
    smem[thread->linear_tid] = p->input[tile->bx * /* stride */ + thread->linear_tid];
    
    // Synchronise all threads in the tile group before reading smem
    tsr_tile_barrier(ctx);
    
    // Reduction (thread 0 only)
    if (thread->linear_tid == 0) {
        float sum = 0.0f;
        for (size_t i = 0; i < smem_floats; ++i) sum += smem[i];
        p->output[tile->bx] = sum;
    }
}
```

**Three accessor inlines (prefer these over touching the struct fields directly):**

| Inline | What it returns |
|--------|----------------|
| `tsr_shared_mem(ctx)` | `void*` — tile-group scratchpad |
| `tsr_shared_bytes(ctx)` | `size_t` — size of scratchpad |
| `tsr_tile_barrier(ctx)` | (void) — barrier: all threads wait here |

---

## Step 4 — Configure and Launch

### Choose tile dimensions

Use `tsrSuggestTile` to get a reasonable starting point, then validate:

```c
tsrDeviceProps props;
tsrGetDeviceProps(dev, &props);

tsrDim3 tile_dim;
tsrSuggestTile(&props, 128, &tile_dim);   // suggest tile for 128 logical threads

tsrLaunchParams params;
params.grid             = (tsrDim3){ n_tiles_x, n_tiles_y, 1 };
params.tile             = tile_dim;
params.shared_mem_bytes = tile_dim.x * sizeof(float);   // smem per tile group
params.flags            = 0;

// Always validate before launching
TsrStatus s = tsrValidateLaunch(&props, &params);
if (s != TSR_STATUS_SUCCESS) {
    fprintf(stderr, "Invalid params: %s\n", tsrGetLastError());
    return 1;
}
```

`tsrValidateLaunch` checks that tile dimensions don't exceed device limits and that
`shared_mem_bytes` is within the device's on-chip budget.

### Async launch (returns immediately)

```c
tsrStream stream;
tsrCreateStream(dev, &stream);

MyPayload payload = { .input = input_ptr, .output = output_ptr, .n = 1024 };

TsrStatus s = tsrLaunchHostTileKernel(stream, &params, kernel_fn, &payload);
if (s != TSR_STATUS_SUCCESS) {
    fprintf(stderr, "Launch failed: %s\n", tsrGetLastError());
}

// Do other work while kernel executes...

// Block until all stream work completes
tsrStreamSynchronize(stream);
tsrDestroyStream(stream);
```

### Synchronous launch (convenience wrapper)

For simple cases where you don't need an explicit stream:

```c
tsrLaunchHostTileKernelSync(dev, &params, kernel_fn, &payload);
// Returns only after all tiles complete — no stream management needed
```

---

## Step 5 — Events and Timing

Events are the primary tool for measuring kernel latency and creating inter-stream
dependencies.

### Measuring kernel time

```c
tsrEvent ev_start, ev_stop;
tsrCreateEvent(dev, &ev_start);
tsrCreateEvent(dev, &ev_stop);

tsrRecordEvent(ev_start, stream);
tsrLaunchHostTileKernel(stream, &params, kernel_fn, &payload);
tsrRecordEvent(ev_stop, stream);

tsrStreamSynchronize(stream);

uint64_t t0_ns, t1_ns;
tsrEventGetTimestamp(ev_start, &t0_ns);
tsrEventGetTimestamp(ev_stop,  &t1_ns);

double elapsed_ms = (t1_ns - t0_ns) / 1e6;
printf("Kernel time: %.3f ms\n", elapsed_ms);

tsrDestroyEvent(ev_start);
tsrDestroyEvent(ev_stop);
```

Timestamps are nanoseconds since process start using a steady clock — they are
monotonically increasing and safe to subtract.

### Inter-stream dependencies

When stream B must wait for work in stream A to complete:

```c
tsrRecordEvent(sync_event, stream_a);   // mark a point in A
tsrWaitEvent(sync_event, stream_b);     // B waits at that point
// stream_b now executes after stream_a has passed the event
```

This is the building block for overlapping computation and data transfer: one stream
does compute while another prefetches the next batch.

---

## Complete Example — Tiled Reduction

```c
#include "tessera/tessera_runtime.h"
#include "tessera/tsr_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N       1024
#define TILE_W  128

typedef struct { float* in; float* out; int n; } Payload;

void reduce_kernel(void* user_ctx,
                   const tsrTileCoord* tile,
                   const tsrThreadCoord* thread) {
    tsrKernelCtx* ctx  = (tsrKernelCtx*)user_ctx;
    Payload*      p    = (Payload*)ctx->user;
    float*        smem = (float*)tsr_shared_mem(ctx);
    
    int g   = (int)tile->bx;
    int tid = (int)thread->linear_tid;
    int off = g * TILE_W + tid;
    
    smem[tid] = (off < p->n) ? p->in[off] : 0.0f;
    tsr_tile_barrier(ctx);
    
    // Tree reduction
    for (int s = TILE_W / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        tsr_tile_barrier(ctx);
    }
    if (tid == 0) p->out[g] = smem[0];
}

int main(void) {
    tsrInit();
    
    tsrDevice dev;
    tsrGetDevice(0, &dev);   // CPU backend
    
    // Allocate and fill input
    tsrBuffer in_buf, out_buf;
    tsrMalloc(dev, N * sizeof(float), &in_buf);
    tsrMalloc(dev, (N / TILE_W) * sizeof(float), &out_buf);
    
    void* in_ptr;  size_t in_sz;
    tsrMap(in_buf, &in_ptr, &in_sz);
    float* in = (float*)in_ptr;
    for (int i = 0; i < N; ++i) in[i] = (float)i;
    tsrUnmap(in_buf);
    
    void* out_ptr; size_t out_sz;
    tsrMap(out_buf, &out_ptr, &out_sz);
    
    // Configure launch
    tsrDeviceProps props;
    tsrGetDeviceProps(dev, &props);
    tsrDim3 tile_dim; tsrSuggestTile(&props, TILE_W, &tile_dim);
    
    tsrLaunchParams params = {
        .grid             = { N / TILE_W, 1, 1 },
        .tile             = tile_dim,
        .shared_mem_bytes = TILE_W * sizeof(float),
        .flags            = 0
    };
    tsrValidateLaunch(&props, &params);
    
    // Time the launch
    tsrStream  stream; tsrCreateStream(dev, &stream);
    tsrEvent   ev_a, ev_b;
    tsrCreateEvent(dev, &ev_a);
    tsrCreateEvent(dev, &ev_b);
    
    Payload payload = { .in = (float*)in_ptr, .out = (float*)out_ptr, .n = N };
    
    tsrRecordEvent(ev_a, stream);
    tsrLaunchHostTileKernel(stream, &params, reduce_kernel, &payload);
    tsrRecordEvent(ev_b, stream);
    tsrStreamSynchronize(stream);
    
    uint64_t t0, t1;
    tsrEventGetTimestamp(ev_a, &t0);
    tsrEventGetTimestamp(ev_b, &t1);
    printf("Reduction time: %.3f µs\n", (t1 - t0) / 1e3);
    
    // Check result
    float total = 0.0f;
    float* out = (float*)out_ptr;
    for (int g = 0; g < N / TILE_W; ++g) total += out[g];
    printf("Sum = %.0f (expected %.0f)\n", total, (float)(N - 1) * N / 2.0f);
    
    tsrUnmap(out_buf);
    tsrDestroyEvent(ev_a); tsrDestroyEvent(ev_b);
    tsrDestroyStream(stream);
    tsrFree(in_buf); tsrFree(out_buf);
    tsrShutdown();
    return 0;
}
```

---

## Error Handling Pattern

Every `tsr*` function returns `TsrStatus`. The recommended pattern:

```c
#define TSR_CHECK(call)                                          \
    do {                                                         \
        TsrStatus _s = (call);                                   \
        if (_s != TSR_STATUS_SUCCESS) {                          \
            fprintf(stderr, "[tessera] %s:%d  %s: %s\n",        \
                    __FILE__, __LINE__,                          \
                    tsrStatusString(_s), tsrGetLastError());     \
            exit(1);                                             \
        }                                                        \
    } while (0)

// Usage:
TSR_CHECK(tsrInit());
TSR_CHECK(tsrGetDevice(0, &dev));
TSR_CHECK(tsrMalloc(dev, 4096, &buf));
```

**Status codes:**

| Code | Meaning |
|------|---------|
| `TSR_STATUS_SUCCESS` | OK |
| `TSR_STATUS_INVALID_ARGUMENT` | Bad parameter (check `tsrGetLastError()` for details) |
| `TSR_STATUS_NOT_FOUND` | Device or resource not present |
| `TSR_STATUS_ALREADY_EXISTS` | Re-initialization after success |
| `TSR_STATUS_OUT_OF_MEMORY` | Device memory exhausted |
| `TSR_STATUS_UNIMPLEMENTED` | Backend not compiled in (e.g., CUDA on CPU-only build) |
| `TSR_STATUS_INTERNAL` | Unexpected internal error — file a bug |
| `TSR_STATUS_DEVICE_ERROR` | Hardware/driver error |

`tsrGetLastError()` returns a thread-local string valid until the next `tsr*` call on
the same thread. Copy it before making any further API calls.

---

## Versioning

```c
int major, minor, patch;
tsrGetVersion(&major, &minor, &patch);
// Currently: 0.2.0
```

Defined as macros in `tsr_version.h`:

```c
#define TESSERA_VERSION_MAJOR  0
#define TESSERA_VERSION_MINOR  2
#define TESSERA_VERSION_PATCH  0
```

While `MAJOR == 0`, API may change across minor versions. At 1.0.0, any function present
at that release is stable for all 1.x releases.

---

## Profiling

```c
tsrEnableProfiling(1);                    // enable at program start
uint64_t now = tsrTimestampNowNs();       // nanoseconds since process start
```

`tsrEventGetTimestamp` uses the same clock, so you can mix event timestamps with
`tsrTimestampNowNs()` readings for host+device timing correlation.

---

## Phase 6: Python Wrapper

A Python wrapper (`python/tessera/runtime.py`) is planned for Phase 6. It will be a
thin ctypes/cffi binding that raises `TesseraRuntimeError` on any non-SUCCESS status:

```python
# Phase 6 API — not yet implemented
from tessera.runtime import TesseraRuntime

rt = TesseraRuntime()
ctx = rt.create_context(device_id=0)
stream = rt.create_stream(ctx)
buf = rt.malloc(ctx, 4096)
rt.memset(buf, 0, 4096)
rt.synchronize(stream)
rt.destroy_stream(stream)
rt.destroy_context(ctx)
```

---

## Where to Go Next

| Goal | Document |
|------|----------|
| Full normative function reference | [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md) |
| Kernel programming model | [Chapter 5: Kernel Programming](Tessera_Programming_Guide_Chapter5_Kernel_Programming.md) |
| Memory model (tiers, sharding, async) | [Chapter 3: Memory Model](Tessera_Programming_Guide_Chapter3_Memory_Model.md) |
| Execution model (tiles, pipelines) | [Chapter 4: Execution Model](Tessera_Programming_Guide_Chapter4_Execution_Model.md) |
| All `tsr*` type definitions | `src/runtime/include/tessera/tsr_types.h` |
| All `TsrStatus` codes | `src/runtime/include/tessera/tsr_status.h` |
| Kernel callback interface | `src/runtime/include/tessera/tsr_kernel.h` |
