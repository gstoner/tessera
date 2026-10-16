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
