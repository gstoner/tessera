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
// Buffer roles: 0=original read-only input, 1=private, 2=returned output.
// Descriptors have null data and exact compact physical rank-two storage.
int tessera_nvidia_native_frame_prepare_v1(
    const TesseraNvidiaMatmulHostView *buffers, const uint8_t *roles, size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps, size_t stage_count, uint64_t *handle);
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
