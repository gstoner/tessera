#pragma once
#include <stddef.h>
#include <stdint.h>
#include "tsr_status.h"
#include "tsr_types.h"
#include "tsr_version.h"
#include "tsr_kernel.h"
#include "tsr_shape.h"

#ifdef __cplusplus
extern "C" {
#endif

TsrStatus tsrInit(void);
TsrStatus tsrShutdown(void);

TsrStatus tsrGetDeviceCount(int* count);
TsrStatus tsrGetDevice(int index, tsrDevice* out);
TsrStatus tsrGetDeviceProps(tsrDevice dev, tsrDeviceProps* props);

TsrStatus tsrCreateStream(tsrDevice dev, tsrStream* out);
TsrStatus tsrDestroyStream(tsrStream s);
TsrStatus tsrStreamSynchronize(tsrStream s);

TsrStatus tsrCreateEvent(tsrDevice dev, tsrEvent* out);
TsrStatus tsrRecordEvent(tsrEvent e, tsrStream s);
TsrStatus tsrWaitEvent(tsrEvent e, tsrStream s);
TsrStatus tsrEventSynchronize(tsrEvent e);
TsrStatus tsrDestroyEvent(tsrEvent e);

// Profiling: timestamp in nanoseconds since process start (steady clock).
TsrStatus tsrEventGetTimestamp(tsrEvent e, uint64_t* ns_out);

TsrStatus tsrMalloc(tsrDevice dev, size_t bytes, tsrBuffer* out);
TsrStatus tsrFree(tsrBuffer b);
TsrStatus tsrMemset(tsrBuffer b, int value, size_t bytes);
TsrStatus tsrMemcpy(tsrBuffer dst, const tsrBuffer src, size_t bytes, TsrMemcpyKind kind);
TsrStatus tsrMap(tsrBuffer b, void** host_ptr, size_t* bytes);
TsrStatus tsrUnmap(tsrBuffer b);

// Host portable tile kernel launch with shared memory/barrier support.
// The user_ctx passed to kernel is a (tsrKernelCtx*).
TsrStatus tsrLaunchHostTileKernel(tsrStream s,
                                  const tsrLaunchParams* params,
                                  tsrHostKernelFn kernel,
                                  void* user_payload);

TsrStatus tsrLaunchHostTileKernelSync(tsrDevice dev,
                                      const tsrLaunchParams* params,
                                      tsrHostKernelFn kernel,
                                      void* user_payload);

#ifdef __cplusplus
} // extern "C"
#endif
