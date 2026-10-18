// Additive static source-active row JVP ABI. Existing RHS v7 frames unchanged.
#pragma once
#include "tessera_nvidia_ptx_launch.h"
#include <cstddef>
struct TesseraNvidiaActiveSourceProgramStep {
  const void *image;
  size_t image_bytes;
  const char *entry;
  int64_t inputs[2], outputs[2];
  int64_t input_count, output_count, dimensions[3];
  int32_t schema, kind, storage, reserved;
};

static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8,
              "source-active program requires a 64-bit native host ABI");
static_assert(sizeof(TesseraNvidiaActiveSourceProgramStep) == 112 &&
              offsetof(TesseraNvidiaActiveSourceProgramStep, inputs) == 24 &&
              offsetof(TesseraNvidiaActiveSourceProgramStep, dimensions) == 72 &&
              offsetof(TesseraNvidiaActiveSourceProgramStep, schema) == 96,
              "source-active program step ABI changed");
// Exactly seven disjoint native buffers and three ordered compiler-authored
// stages: source/seed -> two private edges; each edge and shared RHS -> f32.
// Metadata views have null data; invoke binds three compact readonly host
// roots followed by two distinct compact f32 returned outputs.
extern "C" int tessera_nvidia_active_source_program_prepare(
    const TesseraNvidiaMatmulHostView *buffers, size_t buffer_count,
    const int64_t *roles, const TesseraNvidiaActiveSourceProgramStep *steps,
    size_t step_count, uint64_t *handle);
extern "C" int tessera_nvidia_active_source_program_invoke(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views, size_t view_count);
// Profile reuses the last successful native input snapshot, with a live
// generation lease. Bind the same three root addresses plus two outputs so
// alias and root-identity validation precede every profile invocation.
extern "C" int tessera_nvidia_active_source_program_profile(uint64_t handle,
    int repeats, int captured, float *stage_ms, size_t stage_count,
    float *program_ms, const TesseraNvidiaMatmulHostView *views, size_t view_count);
extern "C" int tessera_nvidia_active_source_program_scratch_stats(uint64_t handle,
    size_t *capacity, size_t *allocations);
extern "C" int tessera_nvidia_active_source_program_close(uint64_t handle);

// Completed-call variant: the first three views are borrowed compact CUDA
// roots and the final two are independent host F32 results. Producer streams
// (including CUDA legacy/per-thread handles 1/2) are ordered by native events.
// Callers retain roots/streams until return. Private edges remain owner-owned.
extern "C" int tessera_nvidia_active_source_program_invoke_resident_inputs(
    uint64_t handle,const TesseraNvidiaMatmulHostView *views,size_t view_count,
    const uint64_t *producer_streams,size_t producer_count);
// Refresh/warm up resident roots before separate stage/composed event windows.
// Root waits, warmup and host result readback are excluded from those windows.
extern "C" int tessera_nvidia_active_source_program_profile_resident_inputs(
    uint64_t handle,const TesseraNvidiaMatmulHostView *views,size_t view_count,
    const uint64_t *producer_streams,size_t producer_count,
    int repeats,int captured,float *stage_ms,size_t stage_count,float *program_ms);

// Additive generation-owned result transport for validated source/RHS programs.
// Reserve allocates private outputs and a stream before submission. Query/read
// admit completed generations only. Submission accepts either three host views
// (null producer_streams, zero count) or three resident CUDA views with explicit
// producer streams. Completed generations are immutable and independently owned.
extern "C" int tessera_nvidia_tensor_program_reserve_result(
    uint64_t program, uint64_t *result);
extern "C" int tessera_nvidia_tensor_program_submit_result(
    uint64_t program, uint64_t result, const TesseraNvidiaMatmulHostView *views,
    size_t view_count, const uint64_t *producer_streams, size_t producer_count);
extern "C" int tessera_nvidia_tensor_result_query(
    uint64_t result, size_t ordinal, TesseraNvidiaMatmulHostView *view,
    uint64_t *stream, uint64_t *context_identity);
extern "C" int tessera_nvidia_tensor_result_read(
    uint64_t result, size_t ordinal, const TesseraNvidiaMatmulHostView *host_output);
// Record after enqueueing an external consumer's final read. Close waits these
// events, restores the recorded CUDA context for cleanup, and preserves the
// calling thread's context. No device pointers are serialized in artifacts.
extern "C" int tessera_nvidia_tensor_result_record_consumer(
    uint64_t result, uint64_t consumer_stream);
extern "C" int tessera_nvidia_tensor_result_close(uint64_t result);
