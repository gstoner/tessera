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

#ifdef __cplusplus
} // extern "C"
#endif
