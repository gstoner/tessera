#ifndef TESSERA_LAYOUT_ALGEBRA_H
#define TESSERA_LAYOUT_ALGEBRA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define TESSERA_LAYOUT_EXPORT __declspec(dllexport)
#else
#define TESSERA_LAYOUT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum TesseraLayoutStatus {
  TESSERA_LAYOUT_OK = 0,
  TESSERA_LAYOUT_INVALID_ARGUMENT = 1,
  TESSERA_LAYOUT_MALFORMED_SPEC = 2,
  TESSERA_LAYOUT_DYNAMIC_UNRESOLVED = 3,
  TESSERA_LAYOUT_BUFFER_TOO_SMALL = 4,
  TESSERA_LAYOUT_OVERFLOW = 5,
};

// Build an exact reshape/transpose/reshape plan for a named-axis regrouping.
// Groups are written with parentheses, e.g. "b h s d -> b s (h d)".  The
// inverse supplies factorizations in bindings_csv, e.g. "h=2,d=8".
TESSERA_LAYOUT_EXPORT int tessera_layout_rearrange_plan_v1(
    const char *spec, const int64_t *input_shape, size_t input_rank,
    const char *bindings_csv, int64_t *expanded_shape,
    size_t expanded_capacity, int64_t *permutation,
    size_t permutation_capacity, int64_t *output_shape,
    size_t output_capacity, size_t *atomic_rank, size_t *output_rank,
    char *error, size_t error_capacity);

TESSERA_LAYOUT_EXPORT int tessera_layout_size_v1(
    const int64_t *shape, size_t rank, int64_t *result);
TESSERA_LAYOUT_EXPORT int tessera_layout_cosize_v1(
    const int64_t *shape, const int64_t *stride, size_t rank, int64_t *result);
TESSERA_LAYOUT_EXPORT int tessera_layout_crd2idx_v1(
    const int64_t *shape, const int64_t *stride, const int64_t *coord,
    size_t rank, int64_t *result);
TESSERA_LAYOUT_EXPORT int tessera_layout_idx2crd_v1(
    const int64_t *shape, size_t rank, int64_t index, int64_t *coord,
    size_t coord_capacity);
TESSERA_LAYOUT_EXPORT const char *tessera_layout_algebra_version_v1(void);

#ifdef __cplusplus
}
#endif

#endif
