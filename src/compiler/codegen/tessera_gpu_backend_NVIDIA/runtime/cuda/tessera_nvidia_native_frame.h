// Compiler-owned static native program frame ABI. No numerical family routing.
#pragma once
#include "tessera_nvidia_ptx_launch.h"
extern "C" {
struct TesseraNvidiaNativeFrameStepV1 {
  const void *image;
  size_t image_bytes;
  const char *entry;
  uint32_t input_count, scalar_count;
  uint32_t buffers[9]; // input_count read buffers followed by one written buffer
  int64_t scalars[8];
  uint32_t grid[3], block[3];
};
// Generic expressions selected by the compiler, evaluated at active row M.
// A zero multiplier retains the capacity constant. Otherwise scalars become
// M*multiplier and grids become ceil(M*multiplier/divisor). No operation tags.
struct TesseraNvidiaNativeFrameRowPlanV2 {
  int64_t scalar_multipliers[8];
  int64_t grid_multipliers[3], grid_divisors[3];
};
// Buffer roles: 0=original read-only input, 1=private, 2=returned output.
// Descriptors have null data and exact compact physical rank-two storage.
int tessera_nvidia_native_frame_prepare_v1(
    const TesseraNvidiaMatmulHostView *buffers, const uint8_t *roles, size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps, size_t stage_count, uint64_t *handle);
// Capacity descriptors are immutable. Flagged buffers have active rows equal
// to the original input buffer row_root, in [1,row_bound], on every call.
// Prepared bounded owners share the checked invoke/resident/profile/close
// handle API below; static v1 preparation never admits dynamic row metadata.
int tessera_nvidia_native_frame_prepare_v2(
    const TesseraNvidiaMatmulHostView *buffers, const uint8_t *roles, size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps, size_t stage_count,
    const uint8_t *row_buffers, uint32_t row_root, int64_t row_bound,
    const TesseraNvidiaNativeFrameRowPlanV2 *plans, uint64_t *handle);
// Compiler-generated MLIR/LLVM shape computation. The retained caller keeps
// the code image alive until frame close succeeds. Nonzero status must leave
// all plan words untouched. No Python callback or numerical dispatcher.
typedef int32_t (*TesseraNvidiaNativeShapeFunctionV3)(const int64_t *, int64_t *);
int tessera_nvidia_native_frame_prepare_v3(
    const TesseraNvidiaMatmulHostView *buffers, const uint8_t *roles, size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps, size_t stage_count,
    TesseraNvidiaNativeShapeFunctionV3 shape_function,
    size_t root_words, size_t plan_words, uint64_t *handle);
// Pure shape query. Returns every active buffer descriptor with null data;
// does not allocate, enqueue work or change the last successful frame.
int tessera_nvidia_native_frame_shapes_v3(uint64_t handle,
    const int64_t *root_dimensions, size_t root_words,
    TesseraNvidiaMatmulHostView *active, size_t buffer_count);
int tessera_nvidia_native_frame_invoke_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots, size_t root_count,
    const TesseraNvidiaMatmulHostView *outputs, size_t output_count);
// Ordered borrowed roots, native-owned intermediates/results and completed host
// outputs. Every root declares a CUDA producer stream (sentinels 1/2 permitted).
int tessera_nvidia_native_frame_resident_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots, size_t root_count,
    const uint64_t *producer_streams, size_t producer_count,
    const TesseraNvidiaMatmulHostView *outputs, size_t output_count);
// Profiles the last successful frame. Resident allocations retain buffer IDs
// and their declared producer streams; changed producers require rebinding.
int tessera_nvidia_native_frame_profile_v1(uint64_t handle, int repeats, int captured,
    float *stage_ms, size_t stage_count, float *program_ms);
// Six fields: device capacity, allocations, loaded modules, stage count,
// frame generation, required bytes. No private device addresses are exported.
int tessera_nvidia_native_frame_stats_v1(uint64_t handle, uint64_t *values, size_t count);
int tessera_nvidia_native_frame_close_v1(uint64_t handle);
// Errors use tessera_nvidia_matmul_last_error() from the same provider.
}
