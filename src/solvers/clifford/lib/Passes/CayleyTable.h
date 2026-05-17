//===- CayleyTable.h --------------------------------------------*- C++ -*-===//
//
// Compile-time Cayley-table builder for Cl(p, q, r) signatures.  Mirrors
// `tessera.ga.signature._product_table` in Python so the GA8 lowering
// passes (ExpandProductTable / GradeFusion / RotorSandwichFold) emit
// identical numerical results to the Python reference surface.
//
// The table is a `dim × dim` array where `table[i][j] == (result_mask,
// sign)` with `sign ∈ {-1, 0, +1}`. `sign == 0` means the product
// is identically zero (any null generator squared); otherwise
// `result_mask` is the bit-mask of the result blade and `sign` is the
// reordering sign times the product of squared-generator signs.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace tessera {
namespace clifford {

struct ProductEntry {
  int64_t result_mask;
  int8_t sign;  // -1, 0, or +1
};

// Compute the geometric product of two basis blades in Cl(p, q, r).
// Returns (result_mask, sign). `sign = 0` if a null generator squared.
inline ProductEntry bladeProduct(int64_t mask_a, int64_t mask_b,
                                  int64_t p, int64_t q, int64_t r) {
  int64_t n = p + q + r;
  int sign = 1;
  // Reordering sign: for each generator in B (bit i in mask_b), count
  // generators in A with higher index. Each crossing flips the sign.
  for (int64_t i = 0; i < n; ++i) {
    if ((mask_b >> i) & 1) {
      int64_t higher_a = mask_a >> (i + 1);
      // popcount
      int crossings = 0;
      while (higher_a) {
        crossings += int(higher_a & 1);
        higher_a >>= 1;
      }
      if (crossings & 1) sign = -sign;
    }
  }
  int64_t common = mask_a & mask_b;
  int64_t result_mask = mask_a ^ mask_b;
  if (common) {
    // Per-generator signature contributions.
    int64_t q_lo = p;
    int64_t q_hi = p + q;
    for (int64_t i = 0; i < n; ++i) {
      if ((common >> i) & 1) {
        if (i < p) {
          // +1 square — no change.
        } else if (i < q_hi) {
          sign = -sign;
        } else {
          return {0, 0};  // null generator squared → product is zero
        }
      }
    }
  }
  return {result_mask, static_cast<int8_t>(sign)};
}

// Full `dim × dim` Cayley table for Cl(p, q, r). `dim = 2^(p+q+r)`.
inline std::vector<std::vector<ProductEntry>>
buildCayleyTable(int64_t p, int64_t q, int64_t r) {
  int64_t n = p + q + r;
  int64_t dim = int64_t(1) << n;
  std::vector<std::vector<ProductEntry>> table(dim);
  for (int64_t i = 0; i < dim; ++i) {
    table[i].resize(dim);
    for (int64_t j = 0; j < dim; ++j) {
      table[i][j] = bladeProduct(i, j, p, q, r);
    }
  }
  return table;
}

// Grade (popcount) of a blade mask.
inline int gradeOfMask(int64_t mask) {
  int g = 0;
  while (mask) {
    g += int(mask & 1);
    mask >>= 1;
  }
  return g;
}

}  // namespace clifford
}  // namespace tessera
