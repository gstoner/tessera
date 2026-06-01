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
/// Inspect whether the runtime currently has any backends initialized.
/// Writes 1 to ``*out`` when ``tsrInit`` has been called without a
/// subsequent ``tsrShutdown``, 0 otherwise.  Safe to call before
/// ``tsrInit``.  Mainly intended for tests / embedded-runtime
/// lifecycles where the caller needs to query state across a
/// shutdown/reinit cycle (see ``src/runtime/src/tessera_runtime.cpp``
/// for the lifecycle rationale).
TsrStatus tsrIsInitialized(int* out);

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

// G5 — Artifact lifecycle for the CPU host-kernel ABI (wired end-to-end).
//
// `tsrCompileArtifact` interprets `module_ir` as a comma-separated list of
// kernel names previously registered via `tsrRegisterHostKernel`. The artifact
// captures those (name -> host fn) entries and serializes a canonical payload
// for round-trip through `tsrLoadArtifact`. Non-CPU codegen JIT is a separate
// gap — an unregistered name returns TSR_STATUS_UNIMPLEMENTED with a precise
// error. `tsrLaunchKernel(args, nargs)` convention for the CPU lane:
//   args[0] = const tsrLaunchParams* (required);
//   args[1] = void* user_payload (optional);
//   nargs >= 1.
// The launch is performed via the CPU backend's existing host-kernel ABI
// (tsrLaunchHostTileKernel).
TsrStatus tsrCompileArtifact(const char* module_ir,
                             const tsrCompileOptions* options,
                             tsrArtifact* out);
TsrStatus tsrLoadArtifact(const void* bytes, size_t bytes_len, tsrArtifact* out);
TsrStatus tsrDestroyArtifact(tsrArtifact artifact);
TsrStatus tsrGetKernel(tsrArtifact artifact, const char* name, tsrKernel* out);
TsrStatus tsrLaunchKernel(tsrStream s, tsrKernel kernel, void** args, size_t nargs);

// Register a CPU host-kernel function under a name so `tsrCompileArtifact` can
// bundle it. Idempotent for the same (name, fn); conflicting re-registration
// returns TSR_STATUS_INVALID_ARGUMENT. Required before compiling an artifact
// that references this kernel.
TsrStatus tsrRegisterHostKernel(const char* name, tsrHostKernelFn fn);

// G6 — destroy a kernel handle obtained via tsrGetKernel. Pairs with
// tsrDestroyArtifact: kernels are owned by the caller, the artifact is the
// backing store. Safe to call with NULL.
TsrStatus tsrDestroyKernel(tsrKernel kernel);

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

// Native CPU execution probes used by the Python JIT fast path and runtime
// micro-tests. Matrices are row-major: A[M,K], B[K,N], C[M,N].
TsrStatus tsrNativeGemmF32(tsrDevice dev,
                           const float* a,
                           const float* b,
                           float* c,
                           int32_t m,
                           int32_t n,
                           int32_t k);

TsrStatus tsrGetWorkerThreadCount(tsrDevice dev, uint32_t* out);

#ifdef __cplusplus
} // extern "C"
#endif
