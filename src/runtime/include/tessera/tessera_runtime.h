#ifndef TESSERA_RUNTIME_H
#define TESSERA_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define TSR_API

typedef struct tsrContext__* tsrContext;
typedef struct tsrModule__*  tsrModule;
typedef struct tsrMem__*     tsrMem;
typedef struct tsrGraph__*   tsrGraph;
typedef struct tsrEvent__*   tsrEvent;

typedef enum {
  TSR_SUCCESS = 0,
  TSR_ERROR_GENERIC = 1,
  TSR_ERROR_INVALID_ARGUMENT = 2,
  TSR_ERROR_OUT_OF_MEMORY = 3,
  TSR_ERROR_UNSUPPORTED = 4
} tsrStatus;

// version
TSR_API int tsrGetVersion(int* major, int* minor, int* patch);

// context
TSR_API tsrStatus tsrContextCreate(tsrContext* out);
TSR_API void      tsrContextDestroy(tsrContext ctx);

// modules
TSR_API tsrStatus tsrModuleLoad(tsrContext ctx, const void* data, size_t len, tsrModule* out);
TSR_API void      tsrModuleUnload(tsrModule mod);

// memory
TSR_API tsrStatus tsrMemAlloc(tsrContext ctx, size_t bytes, tsrMem* out);
TSR_API tsrStatus tsrMemFree(tsrMem mem);
TSR_API tsrStatus tsrMemcpy(tsrMem dst, const void* src, size_t bytes);

// execution
TSR_API tsrStatus tsrTileGraphCreate(tsrContext ctx, const void* tile_descs, size_t n, tsrGraph* out);
TSR_API tsrStatus tsrLaunch(tsrGraph graph, const void* launch_params);
TSR_API tsrStatus tsrSynchronize(tsrGraph graph);

// events
TSR_API tsrStatus tsrEventCreate(tsrContext ctx, tsrEvent* out);
TSR_API tsrStatus tsrEventRecord(tsrEvent ev);
TSR_API tsrStatus tsrEventWait(tsrEvent ev);
TSR_API tsrStatus tsrEventDestroy(tsrEvent ev);

#ifdef __cplusplus
}
#endif
#endif // TESSERA_RUNTIME_H