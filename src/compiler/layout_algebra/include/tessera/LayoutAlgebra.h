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

// Product and divide are one address construction with distinct, semantically
// observable groupings.  Keep the grouping choice explicit in the ABI instead
// of asking callers to re-bracket native results themselves.
enum TesseraLayoutVariantV1 {
  TESSERA_LAYOUT_LOGICAL = 0,
  TESSERA_LAYOUT_ZIPPED = 1,
  TESSERA_LAYOUT_TILED = 2,
  TESSERA_LAYOUT_FLAT = 3,
  TESSERA_LAYOUT_BLOCKED = 4,
  TESSERA_LAYOUT_RAKED = 5,
};

// Physical rank-2 order exported for source emitters that cannot include the
// C++ Rank2Index.h authority directly.  The returned coordinate indices are
// the same plan consumed by Tile IR and native target kernels.
enum TesseraLayoutRank2OrderV1 {
  TESSERA_LAYOUT_ROW_MAJOR = 0,
  TESSERA_LAYOUT_COLUMN_MAJOR = 1,
};

typedef struct TesseraLayoutRank2IndexPlanV1 {
  uint8_t major_coordinate;
  uint8_t minor_coordinate;
} TesseraLayoutRank2IndexPlanV1;

// A layout tree is carried across the ABI as a preorder sequence.  A leaf has
// child_count == 0 and value is its extent (shape tree) or stride (stride
// tree).  A group has child_count > 0 and value == 0.  Shape and stride trees
// must have identical child-count structure.  This is deliberately a
// structured transport: layout syntax is not embedded in a type string.
typedef struct TesseraLayoutNodeV1 {
  int64_t value;
  uint32_t child_count;
} TesseraLayoutNodeV1;

// A slice retains an address offset in addition to its residual layout.  The
// offset is deliberately not folded into strides: fixed coordinates are a
// composed-layout boundary, not a lossy reshape.
typedef struct TesseraLayoutSliceV1 {
  int64_t offset;
} TesseraLayoutSliceV1;

// Exact host-free evidence used by the locality/residency decision boundary.
// factorizes is one only when every address visible through `read` is in the
// producer partition's image.  The two cosizes are returned so Schedule IR can
// bind the materialization footprint rather than repeating the calculation.
typedef struct TesseraLayoutFactorizationV1 {
  int factorizes;
  int64_t read_cosize;
  int64_t partition_cosize;
} TesseraLayoutFactorizationV1;

typedef struct TesseraLayoutResidencyV1 {
  int admitted;
  int64_t elements;
  int64_t bytes;
  int64_t capacity_bytes;
} TesseraLayoutResidencyV1;

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

// Canonicalize a nested layout.  The operation preserves the layout function
// and returns a structurally canonical tree. Dynamic leaves (-1) are retained
// as explicit, non-mergeable residues; operations requiring their value remain
// fail-closed.
TESSERA_LAYOUT_EXPORT int tessera_layout_coalesce_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count);

// Construct the coordinate inverse of a compact bijective layout.  Right and
// left inverse coincide for the admitted bijective subset; they remain two ABI
// names so callers do not erase the directionality of their proof obligation.
TESSERA_LAYOUT_EXPORT int tessera_layout_right_inverse_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count);
TESSERA_LAYOUT_EXPORT int tessera_layout_left_inverse_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count);

// Manufacture the gaps needed for source + complement to tile a cotarget.
// cotarget <= 0 selects the documented cosize-derived form.
TESSERA_LAYOUT_EXPORT int tessera_layout_complement_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count, int64_t cotarget,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count);

// Product forms disjoint copies of lhs at rhs coordinates.  All six variants
// have the same leaf address construction but retain their distinct grouping.
TESSERA_LAYOUT_EXPORT int tessera_layout_product_v1(
    const TesseraLayoutNodeV1 *lhs_shape, size_t lhs_shape_count,
    const TesseraLayoutNodeV1 *lhs_stride, size_t lhs_stride_count,
    const TesseraLayoutNodeV1 *rhs_shape, size_t rhs_shape_count,
    const TesseraLayoutNodeV1 *rhs_stride, size_t rhs_stride_count,
    int variant, TesseraLayoutNodeV1 *output_shape,
    size_t output_shape_capacity, TesseraLayoutNodeV1 *output_stride,
    size_t output_stride_capacity, size_t *output_shape_count,
    size_t *output_stride_count);

// Divide a static layout by a compact rectangular tiler.  Logical/zipped/
// tiled/flat select the documented bracketing; blocked/raked are product-only.
// Non-compact tilers need general composition and fail closed in this slice.
TESSERA_LAYOUT_EXPORT int tessera_layout_divide_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count,
    const TesseraLayoutNodeV1 *tiler_shape, size_t tiler_shape_count,
    const TesseraLayoutNodeV1 *tiler_stride, size_t tiler_stride_count,
    int variant, TesseraLayoutNodeV1 *output_shape,
    size_t output_shape_capacity, TesseraLayoutNodeV1 *output_stride,
    size_t output_stride_capacity, size_t *output_shape_count,
    size_t *output_stride_count);

// Slice by one coordinate per flattened mode.  A coordinate of -1 retains a
// mode; a non-negative coordinate fixes it and contributes to result.offset.
// Dynamic modes may be retained but cannot be fixed before materialization.
TESSERA_LAYOUT_EXPORT int tessera_layout_slice_v1(
    const TesseraLayoutNodeV1 *shape, size_t shape_count,
    const TesseraLayoutNodeV1 *stride, size_t stride_count,
    const int64_t *coordinates, size_t coordinate_count,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count,
    TesseraLayoutSliceV1 *result);

// Materialize A(B(c)) for fully-static nested layouts.  The result retains B's
// grouping, splitting a B mode only where an A mixed-radix boundary makes that
// split semantically necessary.  Inputs outside this initial affine/material-
// izable subset fail closed rather than returning a flattened approximation.
TESSERA_LAYOUT_EXPORT int tessera_layout_compose_v1(
    const TesseraLayoutNodeV1 *a_shape, size_t a_shape_count,
    const TesseraLayoutNodeV1 *a_stride, size_t a_stride_count,
    const TesseraLayoutNodeV1 *b_shape, size_t b_shape_count,
    const TesseraLayoutNodeV1 *b_stride, size_t b_stride_count,
    TesseraLayoutNodeV1 *output_shape, size_t output_shape_capacity,
    TesseraLayoutNodeV1 *output_stride, size_t output_stride_capacity,
    size_t *output_shape_count, size_t *output_stride_count);

// Decide `read ⊑ partition`.  Compact bijective partitions are proved by
// their interval image for any representable size.  General affine layouts use
// an exact finite image proof up to `enumeration_limit`; larger non-compact
// cases fail closed with DYNAMIC_UNRESOLVED rather than approximate legality.
TESSERA_LAYOUT_EXPORT int tessera_layout_factorizes_v1(
    const int64_t *read_shape, const int64_t *read_stride, size_t read_rank,
    const int64_t *partition_shape, const int64_t *partition_stride,
    size_t partition_rank, int64_t enumeration_limit,
    TesseraLayoutFactorizationV1 *result);

// Prove that a materialized layout fits its declared capacity.  `cosize`, not
// logical size, owns the footprint so holes/padding cannot be ignored.
TESSERA_LAYOUT_EXPORT int tessera_layout_residency_v1(
    const int64_t *shape, const int64_t *stride, size_t rank,
    int64_t element_bytes, int64_t capacity_bytes,
    TesseraLayoutResidencyV1 *result);
TESSERA_LAYOUT_EXPORT int tessera_layout_rank2_index_plan_v1(
    int order, TesseraLayoutRank2IndexPlanV1 *result);
TESSERA_LAYOUT_EXPORT const char *tessera_layout_algebra_version_v1(void);

#ifdef __cplusplus
}
#endif

#endif
