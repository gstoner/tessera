#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsrDevice_t* tsrDevice;
typedef struct tsrStream_t* tsrStream;
typedef struct tsrEvent_t* tsrEvent;
typedef struct tsrBuffer_t* tsrBuffer;
typedef struct tsrArtifact_t* tsrArtifact;
typedef struct tsrKernel_t* tsrKernel;

typedef enum {
  TSR_DEVICE_CPU  = 0,
  TSR_DEVICE_CUDA = 1,
  TSR_DEVICE_HIP  = 2
} TsrDeviceKind;

typedef struct { uint32_t x, y, z; } tsrDim3;

typedef struct {
  tsrDim3 grid;
  tsrDim3 tile;
  size_t shared_mem_bytes;
  uint32_t flags;
} tsrLaunchParams;

typedef struct {
  uint32_t bx, by, bz;
} tsrTileCoord;

typedef struct {
  uint32_t tx, ty, tz;
  uint32_t linear_tid;
} tsrThreadCoord;

typedef struct {
  TsrDeviceKind kind;
  char name[128];
  uint32_t logical_tile_threads_max;
  uint32_t concurrent_tiles_hint;
} tsrDeviceProps;

typedef enum {
  TSR_MEMCPY_HOST_TO_DEVICE,
  TSR_MEMCPY_DEVICE_TO_HOST,
  TSR_MEMCPY_DEVICE_TO_DEVICE,
  TSR_MEMCPY_HOST_TO_HOST
} TsrMemcpyKind;

typedef enum {
  TSR_PROFILE_RUNTIME_API = 0,
  TSR_PROFILE_DEVICE_ACTIVITY = 1
} TsrProfileEventKind;

// Profiling callback ABI. ``payload_json`` is a transient, null-terminated JSON
// object owned by the runtime and valid only for the duration of the callback.
// ``value`` is feature-specific: runtime API events use 0.0 in v1; device
// activity events use elapsed microseconds so tprof-style collectors can map it
// directly to a duration.
typedef void (*tsrProfileEventFn)(TsrProfileEventKind kind,
                                  const char* name,
                                  const char* payload_json,
                                  double value,
                                  void* user);

typedef struct {
  const char* target;
  const char* options_json;
} tsrCompileOptions;

// G7 — GPU kernel launch params for the C-ABI launch bridge.
//
// A GPU artifact carries kernel *names* (e.g. "tessera_apple_gpu_mps_matmul_f32")
// rather than host fn-pointers. ``tsrLaunchKernel`` on such a kernel hands a
// registered backend launcher (see ``tsrRegisterGpuLauncher``) the kernel name
// plus ordered buffer pointers + scalar dims; the launcher maps the name to its
// native symbol and executes it. The buffer/dim order is the kernel's own ABI
// (a GEMM is buffers={A,B,C}, dims={M,N,K}). Keeping this generic lets the core
// runtime stay backend-agnostic — no hardcoded dlopen, no Apple/CUDA dependency.
typedef struct {
  void** buffers;        // ordered I/O pointers, kernel-defined (e.g. A,B,C)
  size_t num_buffers;
  const int64_t* dims;   // ordered scalar dims, kernel-defined (e.g. M,N,K)
  size_t num_dims;
} tsrGpuLaunchParams;

#ifdef __cplusplus
} // extern "C"
#endif
