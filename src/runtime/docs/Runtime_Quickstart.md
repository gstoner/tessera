# Tessera Runtime (tsr) – Quickstart (v0.2)

This drops in a minimal but practical runtime under `tessera/runtime` with:

- **CPU backend** with **tile shared memory** + **`tsr_tile_barrier()`** emulation
- **C API** (devices, streams, events, buffers)
- **Profiling**: record-event timestamps + `tsrTimestampNowNs()`
- **Error model**: thread-local `tsrGetLastError()`
- **Shape helpers**: `tsrValidateLaunch()` + `tsrSuggestTile()`
- **CUDA/HIP stubs** if you build with `-DTESSERA_ENABLE_CUDA/HIP=ON`
- **Tests**: basic runtime + **tile-local reduction** correctness

## Build & Test

```bash
mkdir -p build && cd build
cmake -S ../tessera/runtime -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

Expected:
```
Basic runtime test passed.
Tile reduction test passed.
```

## Host Portable Kernels

Include `tessera/tsr_kernel.h` and write kernels as:

```c
static void kernel(void* user_ctx, const tsrTileCoord* tile, const tsrThreadCoord* thr) {
  tsrKernelCtx* kctx = (tsrKernelCtx*)user_ctx;
  float* smem = (float*)tsr_shared_mem(kctx);
  // ...
  tsr_tile_barrier(kctx);
}
```

The runtime passes a `tsrKernelCtx` per **logical thread**. On CPU, a real thread is spawned for each logical thread inside a tile so that barriers behave as expected.

## Profiling

- `tsrRecordEvent()` stores a monotonic timestamp (ns) retrievable with `tsrEventGetTimestamp()`.
- `tsrTimestampNowNs()` gives a runtime-relative monotonic timestamp for your own breadcrumbs.

## Error Model

- Functions set a thread-local error message; read with `tsrGetLastError()` if a call fails.
- `tsrStatusString()` maps status codes to constant strings.

## Shape Helpers

- `tsrValidateLaunch()` checks tile/grid sanity vs device limits.
- `tsrSuggestTile()` gives a simple heuristic to fit a desired logical thread count.

## CUDA/HIP

- Stubs compile when `-DTESSERA_ENABLE_{CUDA,HIP}=ON`; they currently return unimplemented/no-ops.
- Replace the stubs with real implementations and `CreateCudaBackend()/CreateHipBackend()` will add devices at `tsrInit()`.
