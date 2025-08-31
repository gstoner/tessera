#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declared in public API for host portable kernels.
typedef struct tsrKernelCtx tsrKernelCtx;

// Host tile kernel signature (portable).
// user_ctx should be a (tsrKernelCtx*).
typedef void (*tsrHostKernelFn)(void* user_ctx,
                                const struct tsrTileCoord* tile,
                                const struct tsrThreadCoord* thread);

// Runtime-provided context passed to each logical thread.
struct tsrKernelCtx {
  void* user;           // user payload (set by caller)
  void* shared_mem;     // tile-shared scratch
  size_t shared_bytes;  // size of shared_mem
  void (*_tile_barrier)(tsrKernelCtx*); // backend-provided barrier
  void* _impl;          // backend-internal
};

// Accessors/helpers
static inline void* tsr_shared_mem(tsrKernelCtx* ctx) { return ctx->shared_mem; }
static inline size_t tsr_shared_bytes(tsrKernelCtx* ctx) { return ctx->shared_bytes; }
static inline void tsr_tile_barrier(tsrKernelCtx* ctx) { ctx->_tile_barrier(ctx); }

#ifdef __cplusplus
} // extern "C"
#endif
