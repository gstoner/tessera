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

// Compiler-verified static matmul package owner. dtype: f32=1,f16=2,bf16=3.
// Views carry bytes, exact physical shape and byte strides; all are checked
// before upload. prepare pins its module/context; synchronous calls share context scratch.
struct TesseraNvidiaMatmulHostView {
  void *data;
  size_t bytes;
  int32_t dtype, rank;
  int64_t shape[2], strides[2];
};
int tessera_nvidia_matmul_prepare(const void *image, size_t image_bytes,
    const char *entry, const int64_t *mnk, int storage, int bias, int residual,
    int row_b, int half_output, uint64_t *handle);
// Bind immutable capacities to a checked dynamic strided kernel before use.
// Axis mask: 1=M, 2=N, 4=K. Unset axes retain exact capacity extents.
int tessera_nvidia_matmul_set_dynamic_axes(uint64_t handle, int axes);
// Add a verified RMSNorm/LayerNorm/softmax producer. Its output is
// private native scratch; synchronous stream completion retires both kernels.
int tessera_nvidia_matmul_attach_producer(uint64_t handle, const void *image,
    size_t image_bytes, const char *entry, int cooperative);
// Append another checked shape-preserving producer before first invocation.
int tessera_nvidia_matmul_append_producer(uint64_t handle, const void *image,
    size_t imageBytes, const char *entry, int cooperative);

// Append a checked KxN row producer; its two scratch buffers stay native-owned.
// append=0 attaches the first RHS stage; append=1 extends it before invocation.
int tessera_nvidia_matmul_attach_rhs_producer(uint64_t handle, const void *image,
    size_t image_bytes, const char *entry, int cooperative, int append);

// Resolve the live checked native context for context-scoped portable owners.
int tessera_nvidia_matmul_context_identity(uint64_t *identity);
int tessera_nvidia_matmul_invoke(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count);
// Synchronous resident variant. Views contain device pointers; the last view
// is the disjoint source-shaped intermediate. All buffers and the nondefault
// stream must belong to the handle context. Earlier views follow consumer ABI.
int tessera_nvidia_matmul_invoke_resident(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count, void *stream);
// Resident DAG invocation has no caller-supplied intermediate. The C++ owner
// allocates and retains both operand edges; views contain roots and output.
int tessera_nvidia_matmul_invoke_dag_resident(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count, void *stream);
// Ordered synchronous DAG roots: one declared producer stream per input view.
// Streams, root allocations and the output belong to the checked context.
// Events order differing streams before native producers/consumer; completion
// retires all borrowed reads. Default-stream sentinels 1/2 follow CUDA semantics.
int tessera_nvidia_matmul_invoke_dag_resident_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producer_streams, size_t producer_count, void *stream);
// Ordered borrowed roots with an independent completed host result. Native
// ownership retains the bounded device result and uses its own launch stream.
// The invocation mutex covers execution, completion and download atomically.
int tessera_nvidia_matmul_invoke_dag_resident_to_host_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots, size_t count,
    const uint64_t *producer_streams, size_t producer_count,
    const TesseraNvidiaMatmulHostView *output);
// Profile the current borrowed resident frame while holding the same lease.
// Full-program and grouped-stage event windows are independent, not additive.
// Producer event waits precede the program start timestamp.
int tessera_nvidia_matmul_profile_dag_resident_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producer_streams, size_t producer_count, void *stream,
    int repeats, float *stage_ms, size_t stage_count, float *program_ms);
// Single-sided row-major raw RHS contracts. The native owner materializes
// only the LHS producer chain; all external reads retain producer ordering.
int tessera_nvidia_matmul_invoke_lhs_resident_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producer_streams, size_t producer_count, void *stream);
int tessera_nvidia_matmul_invoke_lhs_resident_to_host_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots, size_t count,
    const uint64_t *producer_streams, size_t producer_count,
    const TesseraNvidiaMatmulHostView *output);
int tessera_nvidia_matmul_profile_lhs_resident_ordered(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producer_streams, size_t producer_count, void *stream,
    int repeats, float *stage_ms, size_t stage_count, float *program_ms);
// CUDA-event profiles of the last successful host frame. Rejects stale shared
// arena leases. Stage windows group repeated kernels and are not additive.
int tessera_nvidia_matmul_profile(uint64_t handle, int repeats,
    float *stage_ms, size_t stage_count, float *program_ms,
    TesseraNvidiaMatmulHostView *output);
int tessera_nvidia_matmul_close(uint64_t handle);
int tessera_nvidia_matmul_scratch_stats(uint64_t handle,
    size_t *capacity, size_t *allocations);
const char *tessera_nvidia_matmul_last_error();


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
// Launch one approved resident package kernel using caller-owned CUDA device
// pointers and stream. Supported only for RMSNorm and scheduled fp16 matmul.
int tessera_nvidia_ptx_benchmark_resident(const char* kernel_name,
                                          void** device_buffers,
                                          size_t num_buffers,
                                          const int64_t* dims,
                                          size_t num_dims,
                                          void* stream,
                                          int warmup, int repetitions,
                                          float* latency_ms);

int tessera_nvidia_ptx_invoke_resident(const char* kernel_name,
                                       void** device_buffers,
                                       size_t num_buffers,
                                       const int64_t* dims,
                                       size_t num_dims,
                                       void* stream);

int tessera_nvidia_ptx_invoke(const char* kernel_name,
                              void** buffers, size_t num_buffers,
                              const int64_t* dims, size_t num_dims);

// Dynamic-shared-memory variant. ``dynamic_shared_bytes`` is passed directly
// to cuLaunchKernel after the kernel ABI validates the compiler-owned launch
// expression.
int tessera_nvidia_ptx_invoke_v2(const char* kernel_name,
                                 void** buffers, size_t num_buffers,
                                 const int64_t* dims, size_t num_dims,
                                 size_t dynamic_shared_bytes);

// Benchmark a registered Tile GEMM with device-resident buffers. Host inputs are
// copied once before warmup; CUDA events time only ``repetitions`` kernel
// launches. Returns the mean device latency in ``latency_ms``.
int tessera_nvidia_ptx_benchmark(const char* kernel_name,
                                 void** buffers, size_t num_buffers,
                                 const int64_t* dims, size_t num_dims,
                                 int warmup, int repetitions,
                                 float* latency_ms);

int tessera_nvidia_ptx_benchmark_v2(const char* kernel_name,
                                    void** buffers, size_t num_buffers,
                                    const int64_t* dims, size_t num_dims,
                                    size_t dynamic_shared_bytes,
                                    int warmup, int repetitions,
                                    float* latency_ms);

// Query live driver/JIT resources for a registered kernel and the descriptor's
// launch geometry. ``local_bytes`` is per-thread local memory (including
// spills); ``active_blocks_per_sm`` includes dynamic shared-memory pressure.
int tessera_nvidia_ptx_resources(const char* kernel_name, int block_size,
                                 size_t dynamic_shared_bytes,
                                 int* registers_per_thread,
                                 int* static_shared_bytes,
                                 int* local_bytes,
                                 int* active_blocks_per_sm);

// Why the last call on this thread returned a nonzero rc: the CUDA driver call
// that failed, its CUresult name/number/text, and for a JIT failure the driver's
// JIT log. Empty after a successful call. Valid until the next bridge call on
// the same thread; copy it out before calling again.
const char* tessera_nvidia_ptx_last_error(void);

// Query the exact CUDA device memory envelope associated with the bridge's
// retained primary context. Both outputs are required.
int tessera_nvidia_ptx_device_memory(size_t* total_bytes, size_t* free_bytes);

// Register this bridge as the process-wide GPU launcher (tsrRegisterGpuLauncher),
// so tsrLaunchKernel routes ("nvidia*", kernel_name) here. Returns 0 on success.
// Requires linking against the core runtime (libtessera_runtime); the direct
// register/invoke pair above does not.
int tessera_nvidia_register_ptx_launcher(void);


// Native prepared saved-LSE attention product; host fp32 storage, frontend
// primals followed by requested physical tangent roles. Compiler-owned images.
int tessera_nvidia_attention_jvp_prepare(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* tangent_image, size_t tangent_bytes, const char* tangent_entry,
    const char* sizer_path, const char* sizer_entry, const int64_t* dims,
    const int* frontend_mapping, const int* active_roles, size_t active_count,
    uint64_t* handle);
// Explicit rank-four physical bias: four frontend primals and tangent role 3.
int tessera_nvidia_attention_jvp_prepare_bias(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* tangent_image, size_t tangent_bytes, const char* tangent_entry,
    const char* sizer_path, const char* sizer_entry, const int64_t* dims,
    const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, uint64_t* handle);
int tessera_nvidia_attention_jvp_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, void* const* outputs, const size_t* output_bytes,
    float* device_milliseconds);
int tessera_nvidia_attention_jvp_close(uint64_t handle);
const char* tessera_nvidia_attention_jvp_last_error(void);

// Synchronous static f32 saved-LSE reverse product. Requested role order is
// retained while native compact kernels receive their sorted physical outputs.
int tessera_nvidia_attention_vjp_prepare(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const int64_t* dims, const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, uint64_t* handle);
int tessera_nvidia_attention_vjp_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, void* const* outputs, const size_t* output_bytes,
    size_t output_count, float* device_milliseconds);
int tessera_nvidia_attention_vjp_close(uint64_t handle);
const char* tessera_nvidia_attention_vjp_last_error(void);

}  // extern "C"
