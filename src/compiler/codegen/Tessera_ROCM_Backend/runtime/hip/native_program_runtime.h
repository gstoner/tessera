// Checked native ownership for compiler-exported static SSA programs.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All kernel buffer arguments use the flattened rank-one memref ABI:
// allocated, aligned, offset=0, elements, stride=1; scalars follow buffers.
// Slots are SSA IDs: arguments first, then one fresh output for each step.
typedef struct {
  uint64_t bytes;
  uint64_t elements;
  int64_t first_write;
  int64_t last_read;
  uint32_t ownership; // 0 readonly argument, 1 private scratch, 2 returned output.
  uint32_t reserved;
} TesseraRocmProgramBuffer;
typedef struct {
  const void *image;
  uint64_t image_bytes;
  const char *entry;
  uint32_t input_count;
  uint32_t scalar_count;
  uint32_t inputs[6];
  uint32_t output;
  uint32_t geometry[6]; // grid x/y/z, block x/y/z.
  int64_t scalars[8];
} TesseraRocmProgramStep;
// Status matches other prepared HIP owners:
// 1 contract, 2 identity, 3 image, 4 allocation, 5 copy, 6 launch,
// 7 completion, 8 lease, 9 cleanup, 10 state, 12 exception.
// Any nonzero prepare status may retain a nonzero handle: caller must close.
int tessera_rocm_program_prepare(
    const char *architecture, uint32_t argument_count, uint32_t buffer_count,
    const TesseraRocmProgramBuffer *buffers, uint32_t step_count,
    const TesseraRocmProgramStep *steps, const void *const *inputs,
    const uint64_t *input_bytes, uint64_t *handle);
// Pack a checked positive-stride read-only host view into compact borrowed
// destination storage. This performs byte movement only, before HIP upload.
// Source/destination spans must be disjoint. Rank and strides are byte facts.
int tessera_rocm_program_pack_host_view(
    const void *source, uint64_t source_span, uint32_t rank,
    const uint64_t *shape, const uint64_t *byte_strides, uint32_t item_bytes,
    void *destination, uint64_t destination_bytes);
// Upload all arguments atomically with respect to contract admission.
// A failed device operation poisons the owner; close is then required.
int tessera_rocm_program_update(uint64_t handle, const void *const *inputs,
                               const uint64_t *input_bytes);
// Complete sequence and repetition loops execute in native C++ on a private
// stream. Optional HIP event interval includes native enqueue gaps, not Python.
int tessera_rocm_program_invoke(uint64_t handle, uint32_t repeats,
                               uint64_t *generation, float *elapsed_ms);
// Per-member captured HIP graph windows, averaged over repeats (1..65536).
// Profiling groups repetitions of each pure SSA member in dependency order.
// Includes device graph dispatch; excludes capture/instantiate and host copies.
// Failure quarantines the owner until close. Success publishes a generation.
int tessera_rocm_program_profile_members(
    uint64_t handle, uint32_t repeats, uint32_t member_count,
    uint64_t *generation, float *elapsed_ms);
int tessera_rocm_program_read(uint64_t handle, uint32_t slot,
                             uint64_t generation, void *output, uint64_t bytes);
// Distinct returned-output slots and disjoint writable host destinations.
// Exact byte contracts and generation are checked before copying; all outputs
// are published only after a single successful private-stream completion.
int tessera_rocm_program_read_many(
    uint64_t handle, uint64_t generation, uint32_t count,
    const uint32_t *slots, void *const *outputs, const uint64_t *bytes);
int tessera_rocm_program_close(uint64_t handle);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
int tessera_rocm_program_cache_stats(uint64_t *, uint64_t *, uint64_t *, uint64_t *);
int tessera_rocm_program_cache_clear(void);
#ifdef __cplusplus
}
#endif
