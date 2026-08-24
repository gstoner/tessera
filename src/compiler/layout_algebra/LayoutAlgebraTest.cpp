#include "tessera/LayoutAlgebra.h"
#include "tessera/Rank2Index.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
  using tessera::layout::linearIndex2D;
  using tessera::layout::Rank2Order;
  static_assert(linearIndex2D<Rank2Order::RowMajor>(2, 3, 7) == 17);
  static_assert(linearIndex2D<Rank2Order::ColumnMajor>(2, 3, 7) == 23);
  for (int64_t rows = 1; rows <= 19; ++rows) {
    for (int64_t columns = 1; columns <= 23; ++columns) {
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
          assert(linearIndex2D<Rank2Order::RowMajor>(row, column, columns) ==
                 row * columns + column);
          assert(linearIndex2D<Rank2Order::ColumnMajor>(row, column, rows) ==
                 column * rows + row);
        }
      }
    }
  }

  assert(std::strcmp(tessera_layout_algebra_version_v1(),
                     "tessera.layout_algebra.v1") == 0);
  int64_t shape[] = {2, 3, 4};
  int64_t stride[] = {12, 4, 1};
  int64_t coord[] = {1, 2, 3};
  int64_t value = 0;
  assert(tessera_layout_size_v1(shape, 3, &value) == TESSERA_LAYOUT_OK);
  assert(value == 24);
  assert(tessera_layout_cosize_v1(shape, stride, 3, &value) == TESSERA_LAYOUT_OK);
  assert(value == 24);
  assert(tessera_layout_crd2idx_v1(shape, stride, coord, 3, &value) ==
         TESSERA_LAYOUT_OK);
  assert(value == 23);

  int64_t expanded[8] = {}, permutation[8] = {}, output[8] = {};
  size_t atomicRank = 0, outputRank = 0;
  char error[256] = {};
  assert(tessera_layout_rearrange_plan_v1(
             "b h s d -> b s (h d)", shape, 3, "", expanded, 8,
             permutation, 8, output, 8, &atomicRank, &outputRank, error,
             sizeof(error)) == TESSERA_LAYOUT_MALFORMED_SPEC);
  int64_t gqaShape[] = {2, 3, 5, 7};
  assert(tessera_layout_rearrange_plan_v1(
             "b h s d -> b s (h d)", gqaShape, 4, "", expanded, 8,
             permutation, 8, output, 8, &atomicRank, &outputRank, error,
             sizeof(error)) == TESSERA_LAYOUT_OK);
  assert(atomicRank == 4 && outputRank == 3);
  assert(output[0] == 2 && output[1] == 5 && output[2] == 21);
  assert(permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1 &&
         permutation[3] == 3);

  int64_t dynamicInput[] = {-1, 3, 4};
  assert(tessera_layout_rearrange_plan_v1(
             "a b c -> a (b c)", dynamicInput, 3, "", expanded, 8,
             permutation, 8, output, 8, &atomicRank, &outputRank, error,
             sizeof(error)) == TESSERA_LAYOUT_OK);
  assert(atomicRank == 3 && outputRank == 2);
  assert(output[0] == -1 && output[1] == 12);

  // The structured ABI preserves nesting; coalesce canonicalizes only after
  // validating matching shape/stride tree profiles.
  TesseraLayoutNodeV1 nestedShape[] = {{0, 2}, {2, 0}, {0, 2}, {1, 0}, {6, 0}};
  TesseraLayoutNodeV1 nestedStride[] = {{0, 2}, {1, 0}, {0, 2}, {6, 0}, {2, 0}};
  TesseraLayoutNodeV1 treeOutShape[8] = {}, treeOutStride[8] = {};
  size_t treeShapeCount = 0, treeStrideCount = 0;
  assert(tessera_layout_coalesce_v1(
             nestedShape, 5, nestedStride, 5, treeOutShape, 8, treeOutStride,
             8, &treeShapeCount, &treeStrideCount) == TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 1 && treeStrideCount == 1);
  assert(treeOutShape[0].value == 12 && treeOutShape[0].child_count == 0);
  assert(treeOutStride[0].value == 1 && treeOutStride[0].child_count == 0);

  // (6,2):(8,2) o (4,3):(3,1) must split B's first mode at A's
  // mixed-radix boundary, retaining ((2,2),3) rather than flattening it.
  TesseraLayoutNodeV1 aShape[] = {{0, 2}, {6, 0}, {2, 0}};
  TesseraLayoutNodeV1 aStride[] = {{0, 2}, {8, 0}, {2, 0}};
  TesseraLayoutNodeV1 bShape[] = {{0, 2}, {4, 0}, {3, 0}};
  TesseraLayoutNodeV1 bStride[] = {{0, 2}, {3, 0}, {1, 0}};
  assert(tessera_layout_compose_v1(
             aShape, 3, aStride, 3, bShape, 3, bStride, 3, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 5 && treeStrideCount == 5);
  assert(treeOutShape[0].child_count == 2 && treeOutShape[1].child_count == 2);
  assert(treeOutShape[2].value == 2 && treeOutShape[3].value == 2 &&
         treeOutShape[4].value == 3);
  assert(treeOutStride[2].value == 24 && treeOutStride[3].value == 2 &&
         treeOutStride[4].value == 8);

  // The compact bijective subset has a coordinate inverse.  Keep both ABI
  // directions explicit even though they coincide for this subset.
  TesseraLayoutNodeV1 inverseShape[] = {{0, 3}, {2, 0}, {4, 0}, {6, 0}};
  TesseraLayoutNodeV1 inverseStride[] = {{0, 3}, {4, 0}, {1, 0}, {8, 0}};
  assert(tessera_layout_right_inverse_v1(
             inverseShape, 4, inverseStride, 4, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 4 && treeOutShape[1].value == 4 &&
         treeOutShape[2].value == 2 && treeOutShape[3].value == 6);
  assert(treeOutStride[1].value == 2 && treeOutStride[2].value == 1 &&
         treeOutStride[3].value == 8);
  assert(tessera_layout_left_inverse_v1(
             inverseShape, 4, inverseStride, 4, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);

  // Complement fills the codomain holes. The no-cotarget form intentionally
  // retains the documented cosize-derived trailing extent of one.
  TesseraLayoutNodeV1 complementShape[] = {{0, 2}, {2, 0}, {2, 0}};
  TesseraLayoutNodeV1 complementStride[] = {{0, 2}, {4, 0}, {1, 0}};
  assert(tessera_layout_complement_v1(
             complementShape, 3, complementStride, 3, 24, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 3 && treeOutShape[1].value == 2 &&
         treeOutShape[2].value == 3 && treeOutStride[1].value == 2 &&
         treeOutStride[2].value == 8);

  // Product and divide are one leaf-address construction with distinct native
  // bracketings. Pin one structural representative from each family here;
  // ctypes checks every documented variant.
  TesseraLayoutNodeV1 productShape[] = {{0, 2}, {3, 0}, {4, 0}};
  TesseraLayoutNodeV1 productStride[] = {{0, 2}, {4, 0}, {1, 0}};
  TesseraLayoutNodeV1 productTilerShape[] = {{0, 2}, {2, 0}, {5, 0}};
  TesseraLayoutNodeV1 productTilerStride[] = {{0, 2}, {1, 0}, {2, 0}};
  assert(tessera_layout_product_v1(
             productShape, 3, productStride, 3, productTilerShape, 3,
             productTilerStride, 3, TESSERA_LAYOUT_BLOCKED, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 7 && treeOutShape[1].child_count == 2 &&
         treeOutShape[2].value == 3 && treeOutShape[3].value == 2 &&
         treeOutShape[4].child_count == 2 && treeOutShape[5].value == 4 &&
         treeOutShape[6].value == 5);

  TesseraLayoutNodeV1 divideShape[] = {{0, 2}, {6, 0}, {8, 0}};
  TesseraLayoutNodeV1 divideStride[] = {{0, 2}, {8, 0}, {1, 0}};
  TesseraLayoutNodeV1 divideTilerShape[] = {{0, 2}, {3, 0}, {4, 0}};
  TesseraLayoutNodeV1 divideTilerStride[] = {{0, 2}, {1, 0}, {3, 0}};
  assert(tessera_layout_divide_v1(
             divideShape, 3, divideStride, 3, divideTilerShape, 3,
             divideTilerStride, 3, TESSERA_LAYOUT_LOGICAL, treeOutShape, 8,
             treeOutStride, 8, &treeShapeCount, &treeStrideCount) ==
         TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 7 && treeOutShape[1].child_count == 2 &&
         treeOutShape[2].value == 3 && treeOutShape[3].value == 2 &&
         treeOutStride[2].value == 8 && treeOutStride[3].value == 24);

  // General composition has no proof-table cardinality ceiling and preserves
  // the dynamic suffix after statically required radix splits.
  TesseraLayoutNodeV1 generalOuterShape[] = {{0, 2}, {6, 0}, {2, 0}};
  TesseraLayoutNodeV1 generalOuterStride[] = {{0, 2}, {8, 0}, {2, 0}};
  TesseraLayoutNodeV1 dynamicInnerShape[] = {{-1, 0}};
  TesseraLayoutNodeV1 dynamicInnerStride[] = {{3, 0}};
  assert(tessera_layout_compose_v1(
             generalOuterShape, 3, generalOuterStride, 3, dynamicInnerShape,
             1, dynamicInnerStride, 1, treeOutShape, 8, treeOutStride, 8,
             &treeShapeCount, &treeStrideCount) == TESSERA_LAYOUT_OK);
  assert(treeShapeCount == 3 && treeOutShape[1].value == 2 &&
         treeOutShape[2].value == -1 && treeOutStride[1].value == 24 &&
         treeOutStride[2].value == 2);

  // Slice is an explicit composed-layout boundary: fixing mode 1 contributes
  // offset 10 but leaves the two residual modes unchanged.
  TesseraLayoutNodeV1 sliceShape[] = {{0, 3}, {3, 0}, {4, 0}, {5, 0}};
  TesseraLayoutNodeV1 sliceStride[] = {{0, 3}, {20, 0}, {5, 0}, {1, 0}};
  int64_t sliceCoordinates[] = {-1, 2, -1};
  TesseraLayoutSliceV1 sliceResult{};
  assert(tessera_layout_slice_v1(
             sliceShape, 4, sliceStride, 4, sliceCoordinates, 3, treeOutShape,
             8, treeOutStride, 8, &treeShapeCount, &treeStrideCount,
             &sliceResult) == TESSERA_LAYOUT_OK);
  assert(sliceResult.offset == 10 && treeShapeCount == 3 &&
         treeOutShape[1].value == 3 && treeOutShape[2].value == 5 &&
         treeOutStride[1].value == 20 && treeOutStride[2].value == 1);

  // Exhaust every compact rank-1/2/3 layout whose domain is at most 64.
  // This is the native side of the same corpus driven through ctypes.
  for (int64_t a = 1; a <= 64; ++a) {
    for (int64_t b = 1; a * b <= 64; ++b) {
      for (int64_t c = 1; a * b * c <= 64; ++c) {
        int64_t compactShape[] = {a, b, c};
        int64_t compactStride[] = {1, a, a * b};
        int64_t total = 0;
        assert(tessera_layout_size_v1(compactShape, 3, &total) ==
               TESSERA_LAYOUT_OK);
        assert(total == a * b * c);
        for (int64_t linear = 0; linear < total; ++linear) {
          int64_t compactCoord[3] = {};
          int64_t recovered = -1;
          assert(tessera_layout_idx2crd_v1(compactShape, 3, linear,
                                           compactCoord, 3) ==
                 TESSERA_LAYOUT_OK);
          assert(tessera_layout_crd2idx_v1(
                     compactShape, compactStride, compactCoord, 3,
                     &recovered) == TESSERA_LAYOUT_OK);
          assert(recovered == linear);
        }
      }
    }
  }
  return 0;
}
