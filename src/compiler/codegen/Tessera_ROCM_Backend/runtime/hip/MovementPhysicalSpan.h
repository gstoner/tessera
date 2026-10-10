#pragma once
#include <climits>
#include <cstddef>
#include <cstdint>

namespace tessera::rocm {
// The GPU uses element strides; copy/retain the complete addressed byte span.
// Read-only page views may alias their own elements, but every addressed element
// must fit in the caller's reported allocation and signed memref offset domain.
inline bool pagedPhysicalSpan(const int64_t *dimensions, size_t &bytes) {
  constexpr size_t limit = size_t(INT64_MAX) / sizeof(float);
  const unsigned axes[] = {0, 2, 3, 4};
  size_t elements = 1;
  for (unsigned i = 0; i < 4; ++i) {
    int64_t extent = dimensions[axes[i]], stride = dimensions[7 + i];
    if (extent <= 0 || stride <= 0) return false;
    uint64_t count = uint64_t(extent - 1);
    if (count && uint64_t(stride) > (limit - elements) / count) return false;
    elements += size_t(count) * size_t(stride);
  }
  bytes = elements * sizeof(float);
  return true;
}
} // namespace tessera::rocm
