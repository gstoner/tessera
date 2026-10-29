// Checked native ownership for compiler-exported static SSA programs.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All kernel buffer arguments use the flattened rank-one memref ABI:
// allocated, aligned, offset=0, elements, stride=1; scalars follow buffers.
// Slots are SSA IDs: arguments first, then each step's fresh results.
// The original preparation API below requires exactly one result per step.
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
// Additive multi-result ABI. Inputs precede every output in the flattened
// kernel memref arguments. Output slots follow the compiler's SSA result order;
// the runtime independently derives and checks all write/read lifetimes.
typedef struct {
  const void *image;
  uint64_t image_bytes;
  const char *entry;
  uint32_t input_count;
  uint32_t scalar_count;
  uint32_t inputs[6];
  uint32_t output_count;
  uint32_t outputs[7];
  uint32_t geometry[6];
  int64_t scalars[8];
} TesseraRocmProgramMultiOutputStep;
// Status matches other prepared HIP owners:
// 1 contract, 2 identity, 3 image, 4 allocation, 5 copy, 6 launch,
// 7 completion, 8 lease, 9 cleanup, 10 state, 12 exception.
// Any nonzero prepare status may retain a nonzero handle: caller must close.
int tessera_rocm_program_prepare(
    const char *architecture, uint32_t argument_count, uint32_t buffer_count,
    const TesseraRocmProgramBuffer *buffers, uint32_t step_count,
    const TesseraRocmProgramStep *steps, const void *const *inputs,
    const uint64_t *input_bytes, uint64_t *handle);
// At most seven total memrefs per step, with one or more fresh outputs.
// The original one-result ABI above retains its layout and admission behavior.
int tessera_rocm_program_prepare_multi_output(
    const char *architecture, uint32_t argument_count, uint32_t buffer_count,
    const TesseraRocmProgramBuffer *buffers, uint32_t step_count,
    const TesseraRocmProgramMultiOutputStep *steps, const void *const *inputs,
    const uint64_t *input_bytes, uint64_t *handle);
// Additive bounded-row ABI. All allocations retain capacity_m rows. Only the
// named original activation/K16-scale inputs and one returned consumer output
// vary their active byte spans. Slots are A,SA,output; scalar M is index0 and
// the grid row axis is1 in the A/B/SA/SB/output + M/N/K ABI. The last consumer's M scalar and grid axis follow active_m.
typedef struct {
  int64_t capacity_m;
  uint32_t slot_count;
  uint32_t slots[7];
  uint64_t row_bytes[7];
  uint32_t consumer_step;
  uint32_t m_scalar;
  uint32_t grid_axis;
  uint32_t tile_m;
} TesseraRocmProgramRows;
int tessera_rocm_program_prepare_multi_output_rows(
    const char *, uint32_t, uint32_t, const TesseraRocmProgramBuffer *, uint32_t,
    const TesseraRocmProgramMultiOutputStep *, const void *const *,
    const uint64_t *, const TesseraRocmProgramRows *, int64_t active_m,
    uint64_t *handle);
// Additive multi-step row frame. Every prepared scalar, grid and byte span is
// the capacity value. A row step scales one scalar to active_m*scalar_per_row
// and one grid axis to ceil(active_m*grid_per_row/grid_tile). The runtime
// derives the row closure from the SSA plan: a step is a row step exactly when
// it reads or writes a row slot, and every output of a row step is a row slot.
// Invariant steps (weight expansion) therefore never observe active_m.
typedef struct {
  uint32_t step;
  uint32_t scalar;
  int64_t scalar_per_row;
  uint32_t grid_axis;
  uint32_t grid_tile;
  int64_t grid_per_row;
} TesseraRocmProgramRowStep;
typedef struct {
  int64_t capacity_m;
  uint32_t slot_count; // 1..32 distinct slots, at least one original argument.
  uint32_t step_count; // 1..16 distinct row steps.
  const uint32_t *slots;
  const uint64_t *row_bytes;
  const TesseraRocmProgramRowStep *steps;
} TesseraRocmProgramRowFrame;
int tessera_rocm_program_prepare_multi_output_row_frame(
    const char *, uint32_t, uint32_t, const TesseraRocmProgramBuffer *, uint32_t,
    const TesseraRocmProgramMultiOutputStep *, const void *const *,
    const uint64_t *, const TesseraRocmProgramRowFrame *, int64_t active_m,
    uint64_t *handle);
// Additive two-variable shape frame over M rows and G K16 groups. Every
// frame byte span, scaled scalar and scaled grid field is coef * M^a * G^b
// with a,b in {0,1} (powers bit 0: M, bit 1: G; never zero). Every prepared
// value is the capacity value. An entry scales one scalar and, when
// grid_axis < 3, one grid axis to ceil(coef_grid * M^a * G^b / grid_tile);
// grid_axis == 3 scales no grid field and then grid_tile and grid_coef are 0.
// The frame closure is derived from the SSA plan as for the row frame.
typedef struct {
  uint32_t step;
  uint32_t scalar;
  int64_t scalar_coef;
  uint32_t powers;
  uint32_t grid_axis;
  uint32_t grid_tile;
  int64_t grid_coef;
} TesseraRocmProgramShapeStep;
typedef struct {
  int64_t capacity_m;
  int64_t capacity_g;
  uint32_t slot_count; // 1..32 distinct slots, at least one original argument.
  uint32_t step_count; // 1..32 entries.
  const uint32_t *slots;
  const uint64_t *unit_bytes; // bytes at M = G = 1
  const uint32_t *powers;
  const TesseraRocmProgramShapeStep *steps;
} TesseraRocmProgramShapeFrame;
int tessera_rocm_program_prepare_multi_output_shape_frame(
    const char *, uint32_t, uint32_t, const TesseraRocmProgramBuffer *, uint32_t,
    const TesseraRocmProgramMultiOutputStep *, const void *const *,
    const uint64_t *, const TesseraRocmProgramShapeFrame *, int64_t active_m,
    int64_t active_g, uint64_t *handle);
// Additive three-variable axis frame: the shape frame plus N output columns
// (powers bit 2). A variable with capacity 0 is not framed and its active
// extent must be 1; a power on it refuses. M is always framed.
typedef struct {
  int64_t capacity_m;
  int64_t capacity_g;
  int64_t capacity_n;
  uint32_t slot_count;
  uint32_t step_count;
  const uint32_t *slots;
  const uint64_t *unit_bytes; // bytes at M = G = N = 1
  const uint32_t *powers;
  const TesseraRocmProgramShapeStep *steps;
} TesseraRocmProgramAxisFrame;
int tessera_rocm_program_prepare_multi_output_axis_frame(
    const char *, uint32_t, uint32_t, const TesseraRocmProgramBuffer *, uint32_t,
    const TesseraRocmProgramMultiOutputStep *, const void *const *,
    const uint64_t *, const TesseraRocmProgramAxisFrame *, int64_t active_m,
    int64_t active_g, int64_t active_n, uint64_t *handle);
// A zero extent keeps that variable at its active value.
int tessera_rocm_program_update_axes(
    uint64_t handle, int64_t active_m, int64_t active_g, int64_t active_n,
    uint32_t count, const uint32_t *slots, const void *const *inputs,
    const uint64_t *bytes);
// capacity/active: three extents each (M, G, N); an absent variable reports
// capacity 0 and active 1.
int tessera_rocm_program_axes_info(
    uint64_t handle, int64_t *capacity, int64_t *active,
    uint64_t *allocation_bytes, uint64_t *allocation_count);
// A change of M or G requires every original slot depending on it in this
// atomic update and invalidates every private slot depending on it.
int tessera_rocm_program_update_shape(
    uint64_t handle, int64_t active_m, int64_t active_g, uint32_t count,
    const uint32_t *slots, const void *const *inputs, const uint64_t *bytes);
int tessera_rocm_program_shape_info(
    uint64_t handle, int64_t *capacity_m, int64_t *capacity_g, int64_t *active_m,
    int64_t *active_g, uint64_t *allocation_bytes, uint64_t *allocation_count);
// Row changes require all row-dependent input slots in this atomic update.
// A row change invalidates every private row slot and its SSA dependents.
int tessera_rocm_program_update_rows(
    uint64_t handle, int64_t active_m, uint32_t count, const uint32_t *slots,
    const void *const *inputs, const uint64_t *bytes);
int tessera_rocm_program_rows_info(
    uint64_t handle, int64_t *capacity_m, int64_t *active_m,
    uint64_t *allocation_bytes, uint64_t *allocation_count);
// Pack a checked positive-stride read-only host view into compact borrowed
// destination storage. This performs byte movement only, before HIP upload.
// Source/destination spans must be disjoint. Rank and strides are byte facts.
int tessera_rocm_program_pack_host_view(
    const void *source, uint64_t source_span, uint32_t rank,
    const uint64_t *shape, const uint64_t *byte_strides, uint32_t item_bytes,
    void *destination, uint64_t destination_bytes);
// Host-only E4M3FN admission. Each complete projection returns its smallest
// and largest positive encoded scale. All-zero projections return (127,0).
// Input is compact row-major bytes; offsets cover [0,rows]. Bounds must be
// disjoint from inputs and commit only after complete admission. Status:
// 0 success,1 frame,2 nonfinite/negative encoding,12 exception. No HIP access.
int tessera_rocm_program_nvfp4_scale_bounds(
    const uint8_t *scales, uint64_t bytes, uint64_t rows, uint64_t columns,
    const int64_t *offsets, uint64_t offset_count,
    uint8_t *bounds, uint64_t bounds_bytes);
// Upload all arguments atomically with respect to contract admission.
// A failed device operation poisons the owner; close is then required.
int tessera_rocm_program_update(uint64_t handle, const void *const *inputs,
                               const uint64_t *input_bytes);
// Update distinct original roots only. Transitive SSA dependents become invalid;
// unaffected private results stay resident. Malformed frames change no state.
int tessera_rocm_program_update_slots(
    uint64_t handle, uint32_t count, const uint32_t *slots,
    const void *const *inputs, const uint64_t *bytes);
// Execute an explicit contiguous stage range only when every external
// dependency is initialized. Successful completion publishes a new generation;
// readback additionally requires all declared returned outputs to be ready.
int tessera_rocm_program_invoke_range(
    uint64_t handle, uint32_t first, uint32_t count, uint32_t repeats,
    uint64_t *generation, float *elapsed_ms);
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
