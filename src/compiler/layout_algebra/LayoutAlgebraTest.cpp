#include "tessera/LayoutAlgebra.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
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
