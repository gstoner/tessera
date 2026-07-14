// Shipped NVIDIA PTX launch bridge — C-ABI (COMPILER_REFACTOR_PLAN C2 tail).
//
// The counterpart to Apple's apple_gpu_runtime.mm launcher: it takes Tessera's
// *emitted* PTX (from python .../compiler/ptx_emit.py — e.g. the sm_120
// mma.sync m16n8k16 bf16 kernel), driver-JITs it (cuModuleLoadDataEx, cached by
// kernel name), and launches it (cuLaunchKernel) over ordered host buffers.
//
// Two entry surfaces over one shared launch body:
//   * the direct C-ABI below (register PTX, then invoke) — dlopen-able from
//     Python/ctypes and standalone tests with NO core-runtime dependency, so the
//     bridge is live-testable on its own;
//   * a tsrGpuLauncherFn registered via tsrRegisterGpuLauncher (see the .cpp),
//     the backend-agnostic seam tsrLaunchKernel routes GPU kernels through.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

// Register PTX text for a kernel entry name (the "serialize" input from
// ptx_emit). Returns 0 on success, nonzero on a null argument. Overwrites any
// prior PTX for the name and invalidates its cached module so a re-register
// recompiles.
int tessera_nvidia_ptx_register(const char* kernel_name, const char* ptx);

// JIT-load (cached) the module for kernel_name and launch it over the ordered
// host buffers + scalar dims, per the kernel's ABI (buffer sizes / directions /
// launch config keyed by name — the Apple-launcher pattern). Copies inputs H2D,
// launches, syncs, copies outputs D2H. Returns 0 ok; nonzero rc: 4 = no PTX
// registered for the name, 5 = unknown kernel ABI / bad shape, 2 = no usable
// GPU, 3 = a device op failed.
int tessera_nvidia_ptx_invoke(const char* kernel_name,
                              void** buffers, size_t num_buffers,
                              const int64_t* dims, size_t num_dims);

// Benchmark a registered Tile GEMM with device-resident buffers. Host inputs are
// copied once before warmup; CUDA events time only ``repetitions`` kernel
// launches. Returns the mean device latency in ``latency_ms``.
int tessera_nvidia_ptx_benchmark(const char* kernel_name,
                                 void** buffers, size_t num_buffers,
                                 const int64_t* dims, size_t num_dims,
                                 int warmup, int repetitions,
                                 float* latency_ms);

// Register this bridge as the process-wide GPU launcher (tsrRegisterGpuLauncher),
// so tsrLaunchKernel routes ("nvidia*", kernel_name) here. Returns 0 on success.
// Requires linking against the core runtime (libtessera_runtime); the direct
// register/invoke pair above does not.
int tessera_nvidia_register_ptx_launcher(void);

}  // extern "C"
