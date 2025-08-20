#ifndef TESSERA_RUNTIME_H
#define TESSERA_RUNTIME_H
/*
 * Tessera Runtime & ABI — C Header
 * Normative API matching Tessera_Runtime_ABI_Spec.md
 */
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ----------------------------- Versioning ----------------------------- */
#define TESSERA_ABI_MAJOR 1
#define TESSERA_ABI_MINOR 0
#define TESSERA_VERSION_MAKE(maj, min, patch) (((maj) << 16) | ((min) << 8) | (patch))

/* ------------------------------ Opaques ------------------------------- */
typedef struct tessContext_s*  tessContext_t;
typedef struct tessMesh_s*     tessMesh_t;
typedef struct tessStream_s*   tessStream_t;
typedef struct tessEvent_s*    tessEvent_t;
typedef struct tessModule_s*   tessModule_t;
typedef struct tessKernel_s*   tessKernel_t;
typedef struct tessProfiler_s* tessProfiler_t;

/* ------------------------------- Status ------------------------------- */
typedef enum tessStatus_e {
  TESS_OK = 0,
  TESS_ERR_INVALID_VALUE,
  TESS_ERR_OUT_OF_MEMORY,
  TESS_ERR_NOT_INITIALIZED,
  TESS_ERR_LAUNCH_FAILED,
  TESS_ERR_ARCH_MISMATCH,
  TESS_ERR_UNSUPPORTED,
  TESS_ERR_COLLECTIVE_MISMATCH,
  TESS_ERR_DETERMINISM_VIOLATION,
  TESS_ERR_TIMEOUT,
  TESS_ERR_ABI_VERSION_MISMATCH,
  TESS_ERR_CACHE_CORRUPT
} tessStatus_t;

/* ------------------------------- DTypes ------------------------------- */
typedef enum tessDType_e {
  TESS_DTYPE_FP8_E4M3 = 1,
  TESS_DTYPE_FP8_E5M2 = 2,
  TESS_DTYPE_FP16     = 3,
  TESS_DTYPE_BF16     = 4,
  TESS_DTYPE_FP32     = 5,
  TESS_DTYPE_FP64     = 6,
  TESS_DTYPE_INT8     = 7,
  TESS_DTYPE_INT16    = 8,
  TESS_DTYPE_INT32    = 9,
  TESS_DTYPE_INT64    = 10,
  TESS_DTYPE_BOOL     = 11,
  TESS_DTYPE_COMPLEX64  = 12,
  TESS_DTYPE_COMPLEX128 = 13
} tessDType_t;

/* ------------------------------- Mesh --------------------------------- */
typedef struct tessMeshAxes_s { int tp, pp, dp, ep; } tessMeshAxes;

/* --------------------------- Streams/Events ---------------------------- */
tessStatus_t tessStreamCreate (tessMesh_t mesh, int priority, tessStream_t* out);
tessStatus_t tessStreamDestroy(tessStream_t stream);
tessStatus_t tessStreamSynchronize(tessStream_t stream);

tessStatus_t tessEventCreate(tessEvent_t* out);
tessStatus_t tessEventDestroy(tessEvent_t ev);
tessStatus_t tessEventRecord(tessEvent_t ev, tessStream_t stream);
tessStatus_t tessStreamWaitEvent(tessStream_t stream, tessEvent_t ev);

/* ----------------------------- Init/Shutdown --------------------------- */
tessStatus_t tessInit(tessContext_t* outCtx);
tessStatus_t tessShutdown(tessContext_t ctx);
tessStatus_t tessGetVersion(int* major, int* minor, int* patch);

tessStatus_t tessMeshCreate(tessContext_t ctx, const int* deviceIds, int nDevices,
                            tessMeshAxes axes, tessMesh_t* outMesh);
tessStatus_t tessMeshDestroy(tessMesh_t mesh);

/* ---------------------------- Memory API ------------------------------- */
typedef enum tessMemKind_e { TESS_MEM_GLOBAL = 0, TESS_MEM_PINNED_HOST = 1 } tessMemKind;
tessStatus_t tessMalloc   (tessMesh_t mesh, size_t bytes, void** devPtr);
tessStatus_t tessFree     (tessMesh_t mesh, void* devPtr);
tessStatus_t tessHostAlloc(size_t bytes, void** hostPtr, tessMemKind kind);
tessStatus_t tessHostFree (void* hostPtr);

typedef enum tessCopyKind_e { TESS_COPY_H2D = 0, TESS_COPY_D2H = 1, TESS_COPY_D2D = 2 } tessCopyKind;
tessStatus_t tessMemcpyAsync(void* dst, const void* src, size_t bytes,
                             tessCopyKind kind, tessStream_t stream);

/* -------------------------- Modules & Kernels -------------------------- */
tessStatus_t tessModuleLoad  (tessContext_t ctx, const void* image, size_t size, tessModule_t* outModule);
tessStatus_t tessModuleUnload(tessModule_t module);
tessStatus_t tessKernelGet   (tessModule_t module, const char* name, tessKernel_t* outKernel);

typedef struct tessLaunchConfig_s {
  int    grid[3];
  int    block[3];
  size_t shmemBytes;
  unsigned flags;
} tessLaunchConfig;

/* Launch flags */
#define TESS_LAUNCH_DEFAULT        0u
#define TESS_LAUNCH_DETERMINISTIC  (1u << 0)
#define TESS_LAUNCH_CAPTURE        (1u << 1)
#define TESS_LAUNCH_PERSISTENT     (1u << 2)
#define TESS_LAUNCH_LOW_LATENCY    (1u << 3)

tessStatus_t tessLaunch(tessKernel_t kernel, tessMesh_t mesh, tessLaunchConfig cfg,
                        const void* argBuffer, size_t argSize, tessStream_t stream);

/* ------------------------------- Collectives -------------------------- */
typedef enum tessCollOp_e { TESS_COLL_SUM = 0, TESS_COLL_MAX = 1, TESS_COLL_MIN = 2 } tessCollOp;

tessStatus_t tessAllReduce    (tessMesh_t mesh, void* buffer, size_t count, int dtype,
                               tessCollOp op, const char* axis, tessStream_t stream);
tessStatus_t tessReduceScatter(tessMesh_t mesh, void* inout,  size_t count, int dtype,
                               tessCollOp op, const char* axis, tessStream_t stream);
tessStatus_t tessAllGather    (tessMesh_t mesh, void* inout,  size_t count, int dtype,
                               const char* axis, tessStream_t stream);
tessStatus_t tessBroadcast    (tessMesh_t mesh, void* buffer, size_t count, int dtype,
                               int root, tessStream_t stream);

/* --------------------------- Numerics Policy --------------------------- */
typedef struct tessNumericsPolicy_s {
  int stableReductions; /* 1 = pairwise/Kahan enforced */
  int deterministic;    /* 1 = fixed collective order */
  int rngStateless;     /* 1 = stateless RNG */
  int allowFastMath;    /* 0 = strict; 1 = opt-in (non-normative) */
} tessNumericsPolicy;

tessStatus_t tessSetNumericsPolicy(tessContext_t ctx, const tessNumericsPolicy*);
tessStatus_t tessGetNumericsPolicy(tessContext_t ctx,       tessNumericsPolicy*);

/* ------------------------------ Profiling ----------------------------- */
tessStatus_t tessProfilerStart(tessContext_t ctx, tessProfiler_t* out);
tessStatus_t tessProfilerStop (tessProfiler_t prof, const char* outJsonPath);

/* ----------------------------- Error utils ---------------------------- */
const char* tessStatusString(tessStatus_t s);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* TESSERA_RUNTIME_H */
