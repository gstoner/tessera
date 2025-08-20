#include "tessera_runtime.h"
#include <stdlib.h>
#include <string.h>

/* Opaque structs */
struct tessContext_s  { int _dummy; };
struct tessMesh_s     { int _dummy; };
struct tessStream_s   { int _dummy; };
struct tessEvent_s    { int _dummy; };
struct tessModule_s   { int _dummy; const void* image; size_t size; };
struct tessKernel_s   { int _dummy; const char* name; };
struct tessProfiler_s { int _dummy; };

static struct tessNumericsPolicy_s g_policy = {1,1,1,0};

const char* tessStatusString(tessStatus_t s) {
  switch (s) {
    case TESS_OK: return "TESS_OK";
    case TESS_ERR_INVALID_VALUE: return "TESS_ERR_INVALID_VALUE";
    case TESS_ERR_OUT_OF_MEMORY: return "TESS_ERR_OUT_OF_MEMORY";
    case TESS_ERR_NOT_INITIALIZED: return "TESS_ERR_NOT_INITIALIZED";
    case TESS_ERR_LAUNCH_FAILED: return "TESS_ERR_LAUNCH_FAILED";
    case TESS_ERR_ARCH_MISMATCH: return "TESS_ERR_ARCH_MISMATCH";
    case TESS_ERR_UNSUPPORTED: return "TESS_ERR_UNSUPPORTED";
    case TESS_ERR_COLLECTIVE_MISMATCH: return "TESS_ERR_COLLECTIVE_MISMATCH";
    case TESS_ERR_DETERMINISM_VIOLATION: return "TESS_ERR_DETERMINISM_VIOLATION";
    case TESS_ERR_TIMEOUT: return "TESS_ERR_TIMEOUT";
    case TESS_ERR_ABI_VERSION_MISMATCH: return "TESS_ERR_ABI_VERSION_MISMATCH";
    case TESS_ERR_CACHE_CORRUPT: return "TESS_ERR_CACHE_CORRUPT";
    default: return "UNKNOWN_STATUS";
  }
}

/* Init/Shutdown/Version */
tessStatus_t tessInit(tessContext_t* outCtx) {
  if (!outCtx) return TESS_ERR_INVALID_VALUE;
  *outCtx = (tessContext_t)malloc(sizeof(**outCtx));
  return *outCtx ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessShutdown(tessContext_t ctx) {
  if (ctx) free(ctx);
  return TESS_OK;
}
tessStatus_t tessGetVersion(int* major, int* minor, int* patch) {
  if (major) *major = TESSERA_ABI_MAJOR;
  if (minor) *minor = TESSERA_ABI_MINOR;
  if (patch) *patch = 0;
  return TESS_OK;
}

/* Mesh */
tessStatus_t tessMeshCreate(tessContext_t ctx, const int* deviceIds, int nDevices,
                            tessMeshAxes axes, tessMesh_t* outMesh) {
  (void)ctx; (void)deviceIds; (void)nDevices; (void)axes;
  if (!outMesh) return TESS_ERR_INVALID_VALUE;
  *outMesh = (tessMesh_t)malloc(sizeof(**outMesh));
  return *outMesh ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessMeshDestroy(tessMesh_t mesh) { if (mesh) free(mesh); return TESS_OK; }

/* Streams/Events */
tessStatus_t tessStreamCreate(tessMesh_t mesh, int priority, tessStream_t* out) {
  (void)mesh; (void)priority;
  if (!out) return TESS_ERR_INVALID_VALUE;
  *out = (tessStream_t)malloc(sizeof(**out));
  return *out ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessStreamDestroy(tessStream_t s) { if (s) free(s); return TESS_OK; }
tessStatus_t tessStreamSynchronize(tessStream_t s) { (void)s; return TESS_OK; }

tessStatus_t tessEventCreate(tessEvent_t* out) {
  if (!out) return TESS_ERR_INVALID_VALUE;
  *out = (tessEvent_t)malloc(sizeof(**out));
  return *out ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessEventDestroy(tessEvent_t e) { if (e) free(e); return TESS_OK; }
tessStatus_t tessEventRecord(tessEvent_t e, tessStream_t s) { (void)e; (void)s; return TESS_OK; }
tessStatus_t tessStreamWaitEvent(tessStream_t s, tessEvent_t e) { (void)s; (void)e; return TESS_OK; }

/* Memory */
tessStatus_t tessMalloc(tessMesh_t mesh, size_t bytes, void** devPtr) {
  (void)mesh;
  if (!devPtr) return TESS_ERR_INVALID_VALUE;
  *devPtr = malloc(bytes > 0 ? bytes : 1);
  return *devPtr ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessFree(tessMesh_t mesh, void* devPtr) {
  (void)mesh; free(devPtr); return TESS_OK;
}
tessStatus_t tessHostAlloc(size_t bytes, void** hostPtr, tessMemKind kind) {
  (void)kind;
  if (!hostPtr) return TESS_ERR_INVALID_VALUE;
  *hostPtr = malloc(bytes > 0 ? bytes : 1);
  return *hostPtr ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessHostFree(void* hostPtr) { free(hostPtr); return TESS_OK; }

tessStatus_t tessMemcpyAsync(void* dst, const void* src, size_t bytes,
                             tessCopyKind kind, tessStream_t stream) {
  (void)kind; (void)stream;
  if (!dst || !src) return TESS_ERR_INVALID_VALUE;
  memcpy(dst, src, bytes);
  return TESS_OK;
}

/* Modules & Kernels */
tessStatus_t tessModuleLoad(tessContext_t ctx, const void* image, size_t size,
                            tessModule_t* outModule) {
  (void)ctx;
  if (!outModule) return TESS_ERR_INVALID_VALUE;
  tessModule_t m = (tessModule_t)malloc(sizeof(*m));
  if (!m) return TESS_ERR_OUT_OF_MEMORY;
  m->image = image;
  m->size = size;
  *outModule = m;
  return TESS_OK;
}
tessStatus_t tessModuleUnload(tessModule_t module) { if (module) free(module); return TESS_OK; }

tessStatus_t tessKernelGet(tessModule_t module, const char* name, tessKernel_t* outKernel) {
  (void)module;
  if (!outKernel || !name) return TESS_ERR_INVALID_VALUE;
  tessKernel_t k = (tessKernel_t)malloc(sizeof(*k));
  if (!k) return TESS_ERR_OUT_OF_MEMORY;
  k->name = name;
  *outKernel = k;
  return TESS_OK;
}

tessStatus_t tessLaunch(tessKernel_t kernel, tessMesh_t mesh, tessLaunchConfig cfg,
                        const void* argBuffer, size_t argSize, tessStream_t stream) {
  (void)kernel; (void)mesh; (void)cfg; (void)argBuffer; (void)argSize; (void)stream;
  /* mock does nothing */
  return TESS_OK;
}

/* Collectives */
tessStatus_t tessAllReduce(tessMesh_t mesh, void* buffer, size_t count, int dtype,
                           tessCollOp op, const char* axis, tessStream_t stream) {
  (void)mesh; (void)buffer; (void)count; (void)dtype; (void)op; (void)axis; (void)stream;
  return TESS_OK;
}
tessStatus_t tessReduceScatter(tessMesh_t mesh, void* inout, size_t count, int dtype,
                               tessCollOp op, const char* axis, tessStream_t stream) {
  (void)mesh; (void)inout; (void)count; (void)dtype; (void)op; (void)axis; (void)stream;
  return TESS_OK;
}
tessStatus_t tessAllGather(tessMesh_t mesh, void* inout, size_t count, int dtype,
                           const char* axis, tessStream_t stream) {
  (void)mesh; (void)inout; (void)count; (void)dtype; (void)axis; (void)stream;
  return TESS_OK;
}
tessStatus_t tessBroadcast(tessMesh_t mesh, void* buffer, size_t count, int dtype,
                           int root, tessStream_t stream) {
  (void)mesh; (void)buffer; (void)count; (void)dtype; (void)root; (void)stream;
  return TESS_OK;
}

/* Numerics policy */
tessStatus_t tessSetNumericsPolicy(tessContext_t ctx, const tessNumericsPolicy* p) {
  (void)ctx;
  if (!p) return TESS_ERR_INVALID_VALUE;
  g_policy = *p;
  return TESS_OK;
}
tessStatus_t tessGetNumericsPolicy(tessContext_t ctx, tessNumericsPolicy* p) {
  (void)ctx;
  if (!p) return TESS_ERR_INVALID_VALUE;
  *p = g_policy;
  return TESS_OK;
}

/* Profiling */
tessStatus_t tessProfilerStart(tessContext_t ctx, tessProfiler_t* out) {
  (void)ctx;
  if (!out) return TESS_ERR_INVALID_VALUE;
  *out = (tessProfiler_t)malloc(sizeof(**out));
  return *out ? TESS_OK : TESS_ERR_OUT_OF_MEMORY;
}
tessStatus_t tessProfilerStop(tessProfiler_t prof, const char* outJsonPath) {
  (void)outJsonPath;
  if (prof) free(prof);
  return TESS_OK;
}
