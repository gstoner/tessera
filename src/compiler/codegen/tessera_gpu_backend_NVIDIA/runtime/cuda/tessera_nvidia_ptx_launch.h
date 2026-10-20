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
// Sealed static RHS-linear paired Graph program. Buffer IDs and ordered members
// are compiler-authored. Three readonly roots precede private SSA intermediates
// and two f32 returned outputs. No matmul/bias argument is repurposed as a seed.
struct TesseraNvidiaTensorProgramStep {
  const void *image;
  size_t image_bytes;
  const char *entry;
  int64_t inputs[2], input_count, output;
  int32_t kind, mapping; // kind: row=0/matmul=1; row mapping: serial/cooperative/flat.
};
int tessera_nvidia_tensor_program_prepare(const TesseraNvidiaMatmulHostView *buffers,
    size_t buffer_count, const int64_t *roles, const TesseraNvidiaTensorProgramStep *steps,
    size_t step_count, uint64_t *handle);
int tessera_nvidia_tensor_program_invoke(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t view_count);
int tessera_nvidia_tensor_program_profile(uint64_t handle, int repeats, int captured,
    float *stage_ms, size_t stage_count, float *program_ms,
    const TesseraNvidiaMatmulHostView *outputs, size_t output_count);
int tessera_nvidia_tensor_program_scratch_stats(uint64_t handle,
    size_t *capacity, size_t *allocations);
int tessera_nvidia_tensor_program_close(uint64_t handle);

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
int tessera_nvidia_matmul_supports_flat_activation(void);
int tessera_nvidia_matmul_profile(uint64_t handle, int repeats,
    float *stage_ms, size_t stage_count, float *program_ms,
    TesseraNvidiaMatmulHostView *output);
// Per-call CUDA graphs of the same checked frame, at most 4096 repeats and
// 65536 member nodes. Graph setup/prime/retirement is outside event timing.
int tessera_nvidia_matmul_profile_captured(uint64_t handle, int repeats,
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
// Retained native forward image, static checked storage ABI and private O/LSE.
// storage/result_storage: 1=f32, 2=f16, 3=bf16. Half O matches input storage;
// saved row LSE requires f32 Q/K/V/O. Null producer streams select
// host staging, otherwise every input needs an explicit ordered CUDA stream.
int tessera_nvidia_attention_forward_prepare(
    const void* image, size_t image_bytes, const char* entry, const int64_t* dims,
    const int64_t* bias_shape, int storage, int result_storage, int saved_lse, int bias_scalars,
    uint64_t* handle);
// Static homogeneous half RMSNorm(Q) -> saved-LSE attention. The producer
// output is private, reused across calls and never exposed as a caller buffer.
// cooperative: 0=serial norm rows, 1=cooperative_128. Results: half O, f32 LSE.
// Original-source RMSNorm -> saved-LSE attention reverse. The native owner
// retains normalized Q, O/LSE and the normalization pullback on one stream.
int tessera_nvidia_source_attention_reverse_prepare(
    const void* norm_image, size_t norm_bytes, const char* norm_entry, int cooperative,
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const void* pullback_image, size_t pullback_bytes, const char* pullback_entry,
    const int64_t* dims, const int* frontend_mapping, const int* cotangent_indices,
    const int* selected_roles, size_t selected_count, int storage, uint64_t* handle);
int tessera_nvidia_source_attention_reverse_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes, size_t input_count,
    const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count);
int tessera_nvidia_source_attention_reverse_last_times(
    uint64_t handle, float* milliseconds, size_t count);
int tessera_nvidia_source_attention_reverse_close(uint64_t handle);
// Versioned five-member source reverse: private FP32 row producer before compact backward.
int tessera_nvidia_source_attention_reverse_rows_prepare(
    const void* norm_image, size_t norm_bytes, const char* norm_entry, int cooperative,
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* row_image, size_t row_bytes, const char* row_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const void* pullback_image, size_t pullback_bytes, const char* pullback_entry,
    const int64_t* dims, const int* frontend_mapping, const int* cotangent_indices,
    const int* selected_roles, size_t selected_count, int storage, uint64_t* handle);
int tessera_nvidia_source_attention_reverse_rows_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes, size_t input_count,
    const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count);
int tessera_nvidia_source_attention_reverse_rows_last_times(
    uint64_t handle, float* milliseconds, size_t count);

// Bounded source profiles keep B/H/D fixed and admit only Sq/Sk changes.
// Prepare owns maximum capacities once; invoke validates the complete actual frame.
int tessera_nvidia_source_attention_reverse_bounded_prepare(
    const void* norm_image, size_t norm_bytes, const char* norm_entry, int cooperative,
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const void* pullback_image, size_t pullback_bytes, const char* pullback_entry,
    const int64_t* dims, const int* frontend_mapping, const int* cotangent_indices,
    const int* selected_roles, size_t selected_count, int storage, uint64_t* handle);
int tessera_nvidia_source_attention_reverse_bounded_invoke(
    uint64_t handle, const int64_t *actualDims, const void* const* inputs, const size_t* input_bytes, size_t input_count,
    const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count);
int tessera_nvidia_source_attention_reverse_rows_bounded_prepare(
    const void* norm_image, size_t norm_bytes, const char* norm_entry, int cooperative,
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* row_image, size_t row_bytes, const char* row_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const void* pullback_image, size_t pullback_bytes, const char* pullback_entry,
    const int64_t* dims, const int* frontend_mapping, const int* cotangent_indices,
    const int* selected_roles, size_t selected_count, int storage, uint64_t* handle);
int tessera_nvidia_source_attention_reverse_rows_bounded_invoke(
    uint64_t handle, const int64_t *actualDims, const void* const* inputs, const size_t* input_bytes, size_t input_count,
    const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count);



int tessera_nvidia_norm_attention_prepare(
    const void* norm_image, size_t norm_bytes, const char* norm_entry, int cooperative,
    const void* image, size_t image_bytes, const char* entry, const int64_t* dims,
    int storage, uint64_t* handle);
int tessera_nvidia_norm_attention_last_times(
    uint64_t handle, float* norm_milliseconds, float* attention_milliseconds);
int tessera_nvidia_attention_forward_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count,
    float* device_milliseconds);
int tessera_nvidia_attention_forward_close(uint64_t handle);
const char* tessera_nvidia_attention_forward_last_error(void);

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
// Borrowed compact device roots are ordered then snapshotted into the private
// arena. Both products and independent host outputs complete before return.
int tessera_nvidia_attention_jvp_invoke_resident_ordered(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, float* device_milliseconds);
// Paired O/LSE/dO/dLSE spans; preserves the legacy two-output ABI.
int tessera_nvidia_attention_jvp_prepare_lse(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* tangent_image, size_t tangent_bytes, const char* tangent_entry,
    const char* sizer_path, const char* sizer_entry, const int64_t* dims,
    const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, uint64_t* handle);
// Typed static JVP owner; storage IDs: 1=f32, 2=f16, 3=bf16.
// Q/K/V and their tangents share storage; O/dO may use f32 or data storage.
// Bias, bias tangent and row LSE remain f32. Legacy prepare ABIs stay f32.
int tessera_nvidia_attention_jvp_prepare_typed(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* tangent_image, size_t tangent_bytes, const char* tangent_entry,
    const char* sizer_path, const char* sizer_entry, const int64_t* dims,
    const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, int storage,
    int output_storage, int saved_lse, uint64_t* handle);
// Bounded owner keeps capacity arenas; each invocation validates concrete dimensions.
// Shape symbols use MLIR kDynamic only for Sq/Sk. Bias symbols preserve broadcast ones.
// bias_shape_arguments=1 requires explicitly shaped dynamic bias and four native
// bias scalars; 0 requires static physical or implicit logical bias.
int tessera_nvidia_attention_jvp_prepare_bounded(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* tangent_image, size_t tangent_bytes, const char* tangent_entry,
    const char* sizer_path, const char* sizer_entry, const int64_t* symbolic_dims,
    const int64_t* shape_bounds, const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, int storage,
    int output_storage, int saved_lse, int bias_shape_arguments, uint64_t* handle);
int tessera_nvidia_attention_jvp_invoke_bounded(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, const int64_t* runtime_dims, const int64_t* runtime_bias_shape,
    const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count,
    float* device_milliseconds);
int tessera_nvidia_attention_jvp_invoke_lse(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, void* const* outputs, const size_t* output_bytes,
    size_t output_count, float* device_milliseconds);
int tessera_nvidia_attention_jvp_invoke_lse_resident_ordered(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count,
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
// Additive typed registration: storage/output_storage use f32=1,f16=2,bf16=3.
// LSE/bias/auxiliary seeds stay f32; lse_cotangent adds a rank-three seed input.
// All flags are checked against the exact native entry spelling before loading.
// Additive static half reverse owner with an explicit row residual member.
// Value-only activity omits rimage/rbytes/rentry and allocates no row workspace.
// Invocation timings require exactly four floats: forward,row,backward,total
// native device launch window. nullptr with timing_count=0 disables output.
int tessera_nvidia_attention_vjp_rows_prepare(
    const void *forward_image,size_t forward_bytes,const char *forward_entry,
    const void *backward_image,size_t backward_bytes,const char *backward_entry,
    const void *row_image,size_t row_bytes,const char *row_entry,
    const int64_t *dims,const int64_t *bias_shape,const int *mapping,
    const int *roles,size_t active_count,int storage,int output_storage,int lse_cotangent,uint64_t *handle);
int tessera_nvidia_attention_vjp_rows_invoke(
    uint64_t handle,const void *const *inputs,const size_t *input_bytes,size_t input_count,
    void *const *outputs,const size_t *output_bytes,size_t output_count,float *timings,size_t timing_count);
int tessera_nvidia_attention_vjp_rows_invoke_resident_ordered(
    uint64_t handle,const void *const *inputs,const size_t *input_bytes,size_t input_count,
    const uint64_t *producer_streams,size_t producer_count,
    void *const *outputs,const size_t *output_bytes,size_t output_count,float *timings,size_t timing_count);
int tessera_nvidia_attention_vjp_rows_info(uint64_t handle,size_t *row_bytes,size_t *stages);
// Last successful complete native device window, for either reverse ABI.
int tessera_nvidia_attention_vjp_last_device_window(uint64_t handle,float *milliseconds);

int tessera_nvidia_attention_vjp_prepare_typed(
    const void* forward_image, size_t forward_bytes, const char* forward_entry,
    const void* backward_image, size_t backward_bytes, const char* backward_entry,
    const int64_t* dims, const int64_t* bias_shape, const int* frontend_mapping,
    const int* active_roles, size_t active_count, int storage, int output_storage,
    int lse_cotangent, uint64_t* handle);
int tessera_nvidia_attention_vjp_invoke(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, void* const* outputs, const size_t* output_bytes,
    size_t output_count, float* device_milliseconds);
// Ordered borrowed FP32 roots and cotangent are snapshotted before saved-LSE
// forward/backward. Requested host gradients complete independently on return.
int tessera_nvidia_attention_vjp_invoke_resident_ordered(
    uint64_t handle, const void* const* inputs, const size_t* input_bytes,
    size_t input_count, const uint64_t* producer_streams, size_t producer_count,
    void* const* outputs, const size_t* output_bytes, size_t output_count,
    float* device_milliseconds);
int tessera_nvidia_attention_vjp_close(uint64_t handle);
const char* tessera_nvidia_attention_vjp_last_error(void);

// Additive checked ordered-FP32 scalar contraction. types uses existing host
// view codes 1=f32, 2=f16, 3=bf16 independently for A/B/output. Host invocation,
// profile and close use the existing retained matmul owner APIs.
int tessera_nvidia_scalar_matmul_prepare_v1(
    const void *image, size_t image_bytes, const char *entry,
    const int64_t *dimensions, const int32_t *types,
    int transpose_a, int transpose_b, uint64_t *handle);
int tessera_nvidia_scalar_matmul_invoke_resident_v1(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t view_count,
    const uint64_t *producer_streams, size_t producer_count);

}  // extern "C"
